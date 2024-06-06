import torch
import torchvision

import numpy as np
from torch.nn import functional as F
from torchvision.transforms.functional import resize

from functions.param_loader import Params
from functions.lens_OFT_propagator import LensOFT
from functions.time_multiplexed_multifocals_propagator import TimeMulMultifocalsPropagator

class FarFieldHolographicPropagator(torch.nn.Module):

    def __init__(self, 
                 focal_stack_propagator : TimeMulMultifocalsPropagator, 
                 params : Params = Params()):

        """
        This class propagates an input field to a 3D volume given a quantized depth map, 
        then generates a focal stack given the camera focus settings for the capture.

        Args:

            focal_stack_propagator (TimeMulMultifocalsPropagator) : propagator used to generate the focal stack dataset, 
                                                              needed for coordinates purposes
            params (Params)                       : simulation parameters

        """

        super().__init__()

        # set parameters
        self.params = params
        self.device = params.device
        self.focal_stack_propagator = focal_stack_propagator

        # initialize the optical system
        self._initialize_lenses_()


    def _initialize_lenses_(self):

        self.dx_slm = self.focal_stack_propagator.dx2
        self.dy_slm = self.focal_stack_propagator.dy2
        self.cx_slm = self.focal_stack_propagator.cx2
        self.cy_slm = self.focal_stack_propagator.cy2
        self.Nx_slm = self.focal_stack_propagator.Nx2
        self.Ny_slm = self.focal_stack_propagator.Ny2

        self.lens1_oft = LensOFT(focal_length=self.params.focal_length, 
                                             previous_focal_length=self.params.focal_length, 
                                             input_spacing_x=self.dx_slm, 
                                             input_spacing_y=self.dy_slm, 
                                             input_shift_x=self.cx_slm, 
                                             input_shift_y=self.cy_slm, 
                                             input_shape_x=self.Nx_slm, 
                                             input_shape_y=self.Ny_slm, 
                                             wavelengths=self.params.wavelengths).to(self.device)
        self.X1, self.Y1, self.dx1, self.dy1, self.cx1, self.cy1, self.Nx1, self.Ny1 = self.lens1_oft.get_output_dimensions()

        self.eyepc_oft = LensOFT(focal_length=self.params.eyepiece_focal_length, 
                                             previous_focal_length=self.params.focal_length, 
                                             input_spacing_x=self.dx1, 
                                             input_spacing_y=self.dy1, 
                                             input_shift_x=self.cx1, 
                                             input_shift_y=self.cy1, 
                                             input_shape_x=self.Nx1, 
                                             input_shape_y=self.Ny1, 
                                             wavelengths=self.params.wavelengths).to(self.device)
        X_eye, Y_eye, dx2, dy2, cx2, cy2, Nx2, Ny2 = self.eyepc_oft.get_output_dimensions()
        self.X_eye, self.Y_eye = X_eye.to(self.device), Y_eye.to(self.device)

        self.eye2retina_oft = LensOFT(focal_length=self.params.eye_retina_distance, 
                                                  previous_focal_length=self.params.eyepiece_focal_length, 
                                                  input_spacing_x=dx2, 
                                                  input_spacing_y=dy2, 
                                                  input_shift_x=cx2, 
                                                  input_shift_y=cy2, 
                                                  input_shape_x=Nx2, 
                                                  input_shape_y=Ny2, 
                                                  wavelengths=self.params.wavelengths).to(self.device)
        X_focalstack, Y_focalstack, dx3, dy3, self.cx3, self.cy3, Nx3, Ny3 = self.eye2retina_oft.get_output_dimensions()
        self.X_focalstack, self.Y_focalstack = X_focalstack.to(self.device), Y_focalstack.to(self.device)

    def sample_output_image(self, focal_stack):

        ### Set the input sample coordinates
        x_sensor_normalized = self.params.x_sensor / np.abs(self.cx3)
        y_sensor_normalized = self.params.y_sensor / np.abs(self.cy3)
        X_sensor_normalized, Y_sensor_normalized = torch.meshgrid(x_sensor_normalized, 
                                                              y_sensor_normalized, 
                                                              indexing='xy')        
        grid = torch.stack([X_sensor_normalized, Y_sensor_normalized], dim=-1).unsqueeze(0)
        grid = grid.repeat(focal_stack.shape[0], 1, 1, 1)

        ### Sample the input images to the simulation coordinates
        images_sampled = F.grid_sample(focal_stack,                     # F x C x H x W
                                       grid, 
                                       mode='nearest', 
                                       padding_mode='zeros', 
                                       align_corners=True)
        
        return images_sampled

    def postprocess(self, focal_stack : torch.Tensor) -> torch.Tensor:
        """ Assume that texture_image is 4D tensor
        
        Can be either grayscale or color as input

        Args:
            focal_stack (torch.Tensor): focal stack, (1, num_focuses, 1, C, H, W)

        Returns:
            capture_stack_abs (torch.Tensor): abs focal stack, (1, num_focuses, 1, C, H_input, W_input)
            capture_stack_phase (torch.Tensor): phase of focal stack, (1, num_focuses, 1, C, H_input, W_input)
        """

        capture_stack_abs = focal_stack.squeeze().abs()         # (num_focuses, C, H, W)
        capture_stack_phase = focal_stack.squeeze().angle()         # (num_focuses, C, H, W)
        capture_stack_abs = torch.nn.functional.interpolate(input=capture_stack_abs, 
                            size=[self.Nx_slm, self.Ny_slm], 
                            mode='bilinear', 
                            align_corners=True,
                            antialias=True)
        capture_stack_phase = torch.nn.functional.interpolate(input=capture_stack_phase, 
                            size=[self.Nx_slm, self.Ny_slm], 
                            mode='bilinear', 
                            align_corners=True,
                            antialias=True)
        F, C, H, W = capture_stack_abs.shape
        capture_stack_abs = capture_stack_abs.view(1, F, 1, C, H, W)
        capture_stack_phase = capture_stack_phase.view(1, F, 1, C, H, W)

        return capture_stack_abs, capture_stack_phase
    
    def forward(self, 
                u_slm               : torch.Tensor, 
                eye_focal_lengths   : torch.Tensor = None) -> torch.Tensor:
        """
        Propagates the slm field and generates a focal stack.

        Args:
            u_slm (torch.complex64)               : input slm field tensor (B x 1 x 1 x C x H1 x W1).
            eye_focal_lengths (torch.float32)   : diopters to focus the eye to (num_focuses).

        Returns:
            focal_stack (torch.complex128)      : Complex output field tensor (B x num_focuses x 1 x C x H8 x W8)

        """

        ### Setup parameters
        B, _, _, C, H, W = u_slm.shape
        if u_slm.device != self.device:
            u_slm = u_slm.to(self.device)
        wavelength = self.params.wavelengths
        dz_half_max = self.params.dz_max / 2
        feyepiece = self.params.eyepiece_focal_length
        num_focuses = len(eye_focal_lengths)
        torch.cuda.empty_cache()

        ### Propagate from the slm plane to the 3D volume nominal plane
        uout = self.lens1_oft(u_slm)

        ### Propagate from the 3D volume nominal plane to the eye plane
        uout = self.eyepc_oft(uout)
        d = feyepiece - dz_half_max
        term1 = (1j * 2 * torch.pi / (wavelength * 2 * feyepiece)) * (1 - d / feyepiece)
        eye_quadratic_coordinates = self.X_eye ** 2 + self.Y_eye ** 2
        uout = uout * torch.exp(term1 * eye_quadratic_coordinates)
        eye_aperture_mask = eye_quadratic_coordinates <= self.params.eye_diameter / 2
        uout = uout * eye_aperture_mask
        torch.cuda.empty_cache()

        ### Apply different eye focus settings
        uout = uout.repeat(1, num_focuses, 1, 1, 1, 1)
        Peye = 1 / eye_focal_lengths
        Pvirtual = Peye - 1 / self.params.eye_retina_distance
        term2 = - 1j * 2 * torch.pi / (wavelength * 2 * (1 / Pvirtual))
        term2 = term2[None, ..., None, None, None, None].to(self.device)
        uout = uout * torch.exp(term2 * eye_quadratic_coordinates)
        torch.cuda.empty_cache()

        ### Propagate from the eye plane to the retina plane
        uout = self.eye2retina_oft(uout)
        term3 = 1j * 2 * torch.pi / (wavelength * 2 * self.params.eye_retina_distance)
        uout = uout * torch.exp(term3 * (self.X_focalstack ** 2 + self.Y_focalstack ** 2))

        focal_stack_abs, focal_stack_phase = self.postprocess(uout)
        torch.cuda.empty_cache()

        return focal_stack_abs, focal_stack_phase

    def __str__(self):
        """
        Print function.
        """
        
        # mystr = super().__str__()

        mystr = "=============================================================\n"
        mystr += "CLASS Name: Holographic_Focal_Stack_Propagator"
        mystr += "\nFunctionality: Propagates slm field and gets a focal stack.\n"
        mystr += "-------------------------------------------------------------\n"

        mystr += "\nComponent 1: the eyepiece \n"
        mystr += self.eyepc_oft.__str__()
        mystr += "\nComponent 2: the eye\n"
        mystr += self.eye2retina_oft.__str__()
        mystr += "\n"

        return mystr


    def __repr__(self):
        return self.__str__()
