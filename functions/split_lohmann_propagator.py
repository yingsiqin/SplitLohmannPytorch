import torch
import numpy as np
from torch.nn import functional as F

from functions.param_loader import Params
from functions.lens_OFT_propagator import LensOFT

class SplitLohmannPropagator(torch.nn.Module):

    def __init__(self, params : Params = Params()):

        """
        This class is the Split-Lohmann propagator that propagates an input field to 
        a 3D volume given a quantized depth map.
        It generates a focal stack given the camera focus settings for the capture.

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

        self.lens3_oft = LensOFT(focal_length=self.params.focal_length, 
                                             previous_focal_length=self.params.focal_length, 
                                             input_spacing_x=dx3, 
                                             input_spacing_y=dy3, 
                                             input_shift_x=cx3, 
                                             input_shift_y=cy3, 
                                             input_shape_x=Nx3, 
                                             input_shape_y=Ny3, 
                                             wavelengths=self.params.wavelengths, 
                                             device=self.device).to(self.device)
        X4, Y4, dx4, dy4, cx4, cy4, Nx4, Ny4 = self.lens3_oft.get_output_dimensions()


        self.lens4_oft = LensOFT(focal_length=self.params.focal_length, 
                                             previous_focal_length=self.params.focal_length, 
                                             input_spacing_x=dx4, 
                                             input_spacing_y=dy4, 
                                             input_shift_x=cx4, 
                                             input_shift_y=cy4, 
                                             input_shape_x=Nx4, 
                                             input_shape_y=Ny4, 
                                             wavelengths=self.params.wavelengths, 
                                             device=self.device).to(self.device)
        X5, Y5, dx5, dy5, cx5, cy5, Nx5, Ny5 = self.lens4_oft.get_output_dimensions()

        self.eyepc_oft = LensOFT(focal_length=self.params.eyepiece_focal_length, 
                                             previous_focal_length=self.params.focal_length, 
                                             input_spacing_x=dx5, 
                                             input_spacing_y=dy5, 
                                             input_shift_x=cx5, 
                                             input_shift_y=cy5, 
                                             input_shape_x=Nx5, 
                                             input_shape_y=Ny5, 
                                             wavelengths=self.params.wavelengths, 
                                             device=self.device).to(self.device)
        X_eye, Y_eye, dx_eye, dy_eye, cx_eye, cy_eye, Nx_eye, Ny_eye = self.eyepc_oft.get_output_dimensions()
        self.X_eye, self.Y_eye = X_eye.to(self.device), Y_eye.to(self.device)

        self.eye2retina_oft = LensOFT(focal_length=self.params.eye_retina_distance, 
                                                  previous_focal_length=self.params.eyepiece_focal_length, 
                                                  input_spacing_x=dx_eye, 
                                                  input_spacing_y=dy_eye, 
                                                  input_shift_x=cx_eye, 
                                                  input_shift_y=cy_eye, 
                                                  input_shape_x=Nx_eye, 
                                                  input_shape_y=Ny_eye, 
                                                  wavelengths=self.params.wavelengths, 
                                                  device=self.device).to(self.device)
        X_out, Y_out, dx_out, dy_out, cx_out, cy_out, Nx_out, Ny_out = self.eye2retina_oft.get_output_dimensions()
        self.X_focalstack, self.Y_focalstack = X_out.to(self.device), Y_out.to(self.device)
        self.cx_focalstack, self.cy_focalstack = cx_out, cy_out
        self.Nx_focalstack = Nx_out
        self.Ny_focalstack = Ny_out

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
                binary_mask_sampled : torch.Tensor, 
                diopter_bins        : torch.Tensor = None,
                eye_focal_lengths   : torch.Tensor = None) -> torch.Tensor:
        """
        Propagates the input field given the depth map and generates a focal stack.

        Args:
            u1 (torch.float32)                  : input field mpi tensor (B x 1 x 1 x C x H1 x W1).
            diopter_map (torch.float32)         : diopter map mpi tensor (B x 1 x 1 x 1 x H1 x W1).
            binary_mask_sampled (torch.float32) : binary SLM region mask tensor (B x 1 x 1 x 1 x H1 x W1).
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

        ### PROPAGATE: from the input plane to the first bpp plane
        uout = self.lens1_oft(u1)
        torch.cuda.empty_cache()

        ### MODULATE: apply the first bpp
        X2, Y2, dx2, dy2, cx2, cy2, Nx2, Ny2 = self.lens1_oft.get_output_dimensions()
        T1 = torch.exp((-1j*2*torch.pi/(wavelength*self.params.C0)) * (X2 ** 3 + Y2 ** 3))
        uout = uout * T1 * ((X2 ** 2 + Y2 ** 2) < (self.params.aperture_T1/2) ** 2)

        ### Propagate: from the first bpp plane to the slm plane
        uout = self.lens2_oft(uout)
        torch.cuda.empty_cache()

        ### MODULATE: apply the slm modulation
        diopter_map_flipped = torch.flip(torch.flip(diopter_map,[-1,]),[-2,])
        Delta = ((self.params.eyepiece_focal_length ** 2) / (self.params.focal_length ** 2)) * (self.params.C0/6) * (self.params.working_range/2 - diopter_map_flipped)
        X3, Y3, dx3, dy3, cx3, cy3, Nx3, Ny3 = self.lens2_oft.get_output_dimensions()
        phase_slm_mask = torch.exp(1j * (2 * torch.pi / (wavelength * self.params.focal_length)) * Delta * (X3 + Y3))
        uout = uout * binary_mask_sampled * phase_slm_mask
        torch.cuda.empty_cache()

        ### Propagate: from the slm plane to the second bpp plane
        uout = self.lens3_oft(uout)
        torch.cuda.empty_cache()

        ### MODULATE: apply the second bpp
        X4, Y4, dx4, dy4, cx4, cy4, Nx4, Ny4 = self.lens3_oft.get_output_dimensions()
        T2 = torch.exp((1j*2*torch.pi/(wavelength*self.params.C0)) * -(X4 ** 3 + Y4 ** 3))
        uout = uout * T2 * ((X4 ** 2 + Y4 ** 2) < (self.params.aperture_T2/2) ** 2)
        torch.cuda.empty_cache()

        ### Propagate: from the second bpp plane to the 3D volume nominal plane
        uout = self.lens4_oft(uout)
        torch.cuda.empty_cache()

        ### Put together the mpi output to 1 layer
        uout = torch.sum(uout, dim=1, keepdim=True)
        torch.cuda.empty_cache()

        ### Propagate: from the 3D volume nominal plane to the eye plane
        uout = self.eyepc_oft(uout)
        term1 = (1j * 2 * torch.pi / (wavelength * 2 * feyepiece)) * (dz_half_max / feyepiece)
        eye_quadratic_coordinates = self.X_eye ** 2 + self.Y_eye ** 2
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
        uout = uout * torch.exp(term3 * (self.X_focalstack ** 2 + self.Y_focalstack ** 2))
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
