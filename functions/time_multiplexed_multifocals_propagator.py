import torch
import numpy as np
from torch.nn import functional as F

from functions.param_loader import Params
from functions.lens_OFT_propagator import LensOFT

class TimeMulMultifocalsPropagator(torch.nn.Module):

    def __init__(self, params : Params = Params()):

        """
        This class propagates an input field to a 3D volume given a quantized depth map.
        It uses the time-multiplexed multifocal displays approach to generates a focal stack given the camera focus settings for the capture.

        Args:

            params (Params)   : simulation parameters

        """

        super().__init__()

        # set parameters
        self.params = params
        self.device = params.device

        # initialize the optical system
        self._initialize_lenses_()

 
    def _initialize_lenses_(self):

        self.lens1_oft = LensOFT(focal_length=self.params.focal_length, 
                                             previous_focal_length=None, 
                                             input_spacing_x=self.params.d1_sim, 
                                             input_spacing_y=self.params.d1_sim, 
                                             input_shift_x=self.params.c1_sim, 
                                             input_shift_y=self.params.c1_sim, 
                                             input_shape_x=self.params.N1_sim, 
                                             input_shape_y=self.params.N1_sim, 
                                             wavelengths=self.params.wavelengths, 
                                             device=self.device).to(self.device)
        X2, Y2, dx2, dy2, cx2, cy2, Nx2, Ny2 = self.lens1_oft.get_output_dimensions()

        self.lens2_oft = LensOFT(focal_length=self.params.focal_length, 
                                             previous_focal_length=self.params.focal_length, 
                                             input_spacing_x=dx2, 
                                             input_spacing_y=dy2, 
                                             input_shift_x=cx2, 
                                             input_shift_y=cy2, 
                                             input_shape_x=Nx2, 
                                             input_shape_y=Ny2, 
                                             wavelengths=self.params.wavelengths, 
                                             device=self.device).to(self.device)
        X3, Y3, dx3, dy3, cx3, cy3, Nx3, Ny3 = self.lens2_oft.get_output_dimensions()

        self.eyepc_oft = LensOFT(focal_length=self.params.eyepiece_focal_length, 
                                             previous_focal_length=self.params.focal_length, 
                                             input_spacing_x=dx3, 
                                             input_spacing_y=dy3, 
                                             input_shift_x=cx3, 
                                             input_shift_y=cy3, 
                                             input_shape_x=Nx3, 
                                             input_shape_y=Ny3, 
                                             wavelengths=self.params.wavelengths, 
                                             device=self.device).to(self.device)
        X_eye, Y_eye, dx4, dy4, cx4, cy4, Nx4, Ny4 = self.eyepc_oft.get_output_dimensions()

        self.eye2retina_oft = LensOFT(focal_length=self.params.eye_retina_distance, 
                                                  previous_focal_length=self.params.eyepiece_focal_length, 
                                                  input_spacing_x=dx4, 
                                                  input_spacing_y=dy4, 
                                                  input_shift_x=cx4, 
                                                  input_shift_y=cy4, 
                                                  input_shape_x=Nx4, 
                                                  input_shape_y=Ny4, 
                                                  wavelengths=self.params.wavelengths, 
                                                  device=self.device).to(self.device)
        X_focalstack, Y_focalstack, dx5, dy5, self.cx5, self.cy5, Nx5, Ny5 = self.eye2retina_oft.get_output_dimensions()

    def sample_output_image(self, focal_stack):

        ### Set the input sample coordinates
        x_sensor_normalized = self.params.x_sensor / np.abs(self.cx5)
        y_sensor_normalized = self.params.y_sensor / np.abs(self.cy5)
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

    def forward(self, 
                u1                  : torch.Tensor, 
                diopter_map         : torch.Tensor,
                diopter_bins        : torch.Tensor = None,
                eye_focal_lengths   : torch.Tensor = None) -> torch.Tensor:
        """
        Propagates the input field given the depth map and generates a focal stack.

        Args:
            u1 (torch.float32)                  : input field mpi tensor (B x num_depths x 1 x C x H1 x W1).
            diopter_map (torch.float32)         : diopter map mpi tensor (B x num_depths x 1 x 1 x H1 x W1).
            diopter_bins (torch.float32)        : the diopter values in the diopter map (num_depths).
            eye_focal_lengths (torch.float32)   : diopters to focus the eye to (num_focuses).

        Returns:
            u_result (torch.complex128)         : Complex output field tensor (B x num_focuses x 1 x C x H8 x W8)

        """

        ### Setup parameters
        B, _, _, C, H, W = u1.shape
        if u1.device != self.device:
            u1 = u1.to(self.device)
        if diopter_map.device != self.device:
            diopter_map = diopter_map.to(self.device)
        wavelength = self.params.wavelengths
        dz_half_max = self.params.dz_max / 2
        feyepiece = self.params.eyepiece_focal_length
        if eye_focal_lengths is None:
            num_depths = len(diopter_bins)
            diopter_idx_increment = int(np.round(num_depths/self.params.num_focuses))
            focus_diopter_idxes = torch.arange(0, num_depths, diopter_idx_increment)
            focus_diopter_idxes = focus_diopter_idxes[:self.params.num_focuses]
            focal_stack_diopters = diopter_bins[focus_diopter_idxes]
            eye_focal_lengths = 1 / (focal_stack_diopters + 1 / self.params.eye_retina_distance)
        num_focuses = len(eye_focal_lengths)
        torch.cuda.empty_cache()

        ### Propagate: from the input plane to the focus-tunable lens plane
        uout = self.lens1_oft(u1)

        ### Modulate: apply focus-tunable lens
        coefficient = (feyepiece ** 2) / (self.params.focal_length ** 2)
        f1_power_map = coefficient * (self.params.working_range / 2 - diopter_map)
        f1_map = 1 / f1_power_map
        f1_lens_phase_map = - 1j * 2 * torch.pi / (wavelength * 2 * f1_map)
        X2, Y2, dx2, dy2, cx2, cy2, Nx2, Ny2 = self.lens1_oft.get_output_dimensions()
        uout = uout * torch.exp(f1_lens_phase_map * (X2 ** 2 + Y2 ** 2))
        torch.cuda.empty_cache()

        ### Propagate: from the focus-tunable lens plane to the 3D volume nominal plane
        uout = self.lens2_oft(uout)

        ### Put together the mpi output to 1 layer
        uout = torch.sum(uout, dim=1, keepdim=True)

        ### Propagate: from the 3D volume nominal plane to the eye plane
        uout = self.eyepc_oft(uout)
        term1 = (1j * 2 * torch.pi / (wavelength * 2 * feyepiece)) * (dz_half_max / feyepiece)
        X_eye, Y_eye, dx4, dy4, cx4, cy4, Nx4, Ny4 = self.eyepc_oft.get_output_dimensions()
        eye_quadratic_coordinates = X_eye ** 2 + Y_eye ** 2
        uout = uout * torch.exp(term1 * eye_quadratic_coordinates)
        eye_aperture_mask = eye_quadratic_coordinates <= self.params.eye_diameter / 2
        uout = uout * eye_aperture_mask
        torch.cuda.empty_cache()

        ### Apply different eye focus settings
        uout = uout.repeat(1, num_focuses, 1, 1, 1, 1)
        Peye = 1 / eye_focal_lengths
        Pvirtual = Peye - 1 / self.params.eye_retina_distance
        term2 = - 1j * 2 * torch.pi / (wavelength * 2 * (1/Pvirtual))
        term2 = term2[None, ..., None, None, None, None].to(self.device)
        uout = uout * torch.exp(term2 * eye_quadratic_coordinates)
        torch.cuda.empty_cache()

        ### Propagate: from the eye plane to the retina plane
        uout = self.eye2retina_oft(uout)
        term3 = 1j * 2 * torch.pi / (wavelength * 2 * self.params.eye_retina_distance)
        X_focalstack, Y_focalstack, dx5, dy5, cx5, cy5, Nx5, Ny5 = self.eye2retina_oft.get_output_dimensions()
        uout = uout * torch.exp(term3 * (X_focalstack ** 2 + Y_focalstack ** 2))
        torch.cuda.empty_cache()

        return uout

    def __str__(self):
        """
        Print function.
        """
        
        # mystr = super().__str__()

        mystr = "=============================================================\n"
        mystr += "CLASS Name: FocalStackPropagator"
        mystr += "\nFunctionality: Propagates and generates a focal stack.\n"
        mystr += "-------------------------------------------------------------\n"

        mystr += "\nComponent 1: relay lens 1\n"
        mystr += self.lens1_oft.__str__()
        mystr += "\nComponent 2: relay lens 2\n"
        mystr += self.lens2_oft.__str__()
        mystr += "\nComponent 3: the eyepiece \n"
        mystr += self.eyepc_oft.__str__()
        mystr += "\nComponent 4: the eye\n"
        mystr += self.eye2retina_oft.__str__()
        mystr += "\n"

        return mystr


    def __repr__(self):
        return self.__str__()
