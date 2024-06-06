import torch
import numpy as np
from torch.nn import functional as F


class LensOFT(torch.nn.Module):

    def __init__(self,
                 focal_length           : float,
                 previous_focal_length  : float or None,
                 input_spacing_x        : torch.Tensor,
                 input_spacing_y        : torch.Tensor,
                 input_shift_x          : torch.Tensor,
                 input_shift_y          : torch.Tensor,
                 input_shape_x          : torch.Tensor,
                 input_shape_y          : torch.Tensor,
                 wavelengths            : float = 532e-09,
                 device                 : str = 'cuda'
                 ):

        """
        This class takes a field in the rear focal plane and computes the (properly sampled) field in the Fourier plane.
        This implementation performs the Optical Discrete Fourier Transform (not a DFT).

        Adapted from: Split-Lohmann Multifocal Displays, Qin et al.
        Source: https://github.com/Image-Science-Lab-cmu/SplitLohmann/blob/main/functions/lens_fourier_propagate.m

        Args:

            focal_length (float)                        : Focal length of lens (m)
            previous_focal_length (float)               : Focal length of the previous lens that performed an Optical DTFT (m)
            input_spacing_dx (torch.Tensor or float)    : Spacing dx of the input field, float or shape 1x1x1xCx1x1
            input_spacing_dy (torch.Tensor or float)    : Spacing dy of the input field, float or shape 1x1x1xCx1x1
            input_shift_x (torch.Tensor or float)       : Shift cx1 of the input to center the origin, float or shape 1x1x1xCx1x1
            input_shift_y (torch.Tensor or float)       : Shift cy1 of the input to center the origin, float or shape 1x1x1xCx1x1
            input_shape_x (torch.Tensor or int)         : Shape Nx1 of the input field, float or shape 1x1x1xCx1x1
            input_shape_y (torch.Tensor or int)         : Shape Ny1 of the input field, float or shape 1x1x1xCx1x1
            wavelengths (float)                         : wavelength in meters, float (TODO (Yingsi): add support for multiple wavelengths)

        """

        super().__init__()

        self.f = focal_length
        self.f_previous = previous_focal_length if previous_focal_length is not None else focal_length
        self.dx1 = input_spacing_x
        self.dy1 = input_spacing_y
        self.cx1 = input_shift_x
        self.cy1 = input_shift_y
        self.Nx1 = input_shape_x
        self.Ny1 = input_shape_y
        self.lambdas = wavelengths
        self.device = device

        self._compute_dimensions()


    def _compute_dimensions(self):
        """
        Creates coordinate system for the Optical DFT.

        Currently supports one wavelength only.
        Need to add support for multiple wavelengths (different wavelengths will have different OFT coordinates scaling).
        """

        self.dx2 = self.f / self.f_previous * self.dx1          # (1, 1, 1, C, 1, 1) or float
        self.dy2 = self.f / self.f_previous * self.dy1          # (1, 1, 1, C, 1, 1) or float

        ### set up input and output coordinate grids
            
        self.Nx2 = int(np.abs(np.round(self.lambdas * self.f / (self.dx1 * self.dx2))))         
        self.Ny2 = int(np.abs(np.round(self.lambdas * self.f / (self.dy1 * self.dy2))))         
        self.dx2 = self.lambdas * self.f / (self.dx1 * self.Nx2)                                
        self.dy2 = self.lambdas * self.f / (self.dy1 * self.Ny2)                                
        self.cx2 = -self.Nx2 * self.dx2 / 2                                                     
        self.cy2 = -self.Ny2 * self.dy2 / 2

        X2, Y2 = torch.meshgrid(torch.arange(0, self.Nx2) * self.dx2 + self.cx2, 
                                torch.arange(0, self.Ny2) * self.dy2 + self.cy2, 
                                indexing='xy')                                                     
        self.X2 = X2[None, None, None, None, ...].to(self.device)   # (1, 1, 1, 1, Ny2, Nx2)
        self.Y2 = Y2[None, None, None, None, ...].to(self.device)   # (1, 1, 1, 1, Ny2, Nx2)

        assert isinstance(int(self.Nx1), int)
        assert isinstance(int(self.Ny1), int)
        nx1, ny1 = torch.meshgrid(torch.arange(0, self.Nx1), 
                                    torch.arange(0, self.Ny1), 
                                    indexing='xy')
        nx1 = nx1[None, None, None, None, ...]         # (1, 1, 1, 1, Ny1, Nx1)
        ny1 = ny1[None, None, None, None, ...]         # (1, 1, 1, 1, Ny1, Nx1)

        ### compute Optical DFT terms
                                                  
        term1 = (self.dx1 * self.cx2 * nx1) + (self.dy1 * self.cy2 * ny1)
        self.u1_antialiasing_phase_term = torch.exp(-1j*2*torch.pi*(term1)/(self.lambdas*self.f)).to(self.device)
            
        coefficient = (self.dy1 * self.dx1 / (1j * self.lambdas * self.f))
        term2 = (self.cx1 * self.X2) + (self.cy1 * self.Y2)
        self.h = coefficient * torch.exp(-1j * 2 * torch.pi * term2 / (self.lambdas * self.f))

    def get_output_dimensions(self):

        return self.X2, self.Y2, self.dx2, self.dy2, self.cx2, self.cy2, self.Nx2, self.Ny2


    def forward(self, u1 : torch.Tensor) -> torch.Tensor:
        """
        Performs the Optcial Discrete Fourier Transform.

        Args:
            u1 (torch.complex128) : Complex input field tensor (B x F x P x C x H x W).

        Returns:
            u2 (torch.complex128) : Complex output field tensor (B x F x P x C x Ny2 x Nx2)

        """

        # Perform DFT
        u1_2 = u1 * self.u1_antialiasing_phase_term
        u1_2_hat = torch.fft.fft2(u1_2, s=[self.Ny2, self.Nx2], dim=[-2, -1])            
        u2 = self.h * u1_2_hat
        
        return u2

    def __str__(self, ):
        """
        Print function.
        """
        
        # mystr = super().__str__()
        B, F, P, C, H, W = self.X2.shape
        mm = 10e-3
        um = 10e-6

        mystr = "=============================================================\n"
        mystr += "CLASS Name: LensOFT"
        mystr += "\nFunctionality: Performs the Optical DFT.\n"
        mystr += "-------------------------------------------------------------\n"
        mystr += "Focal Length: " + str(self.f/mm) + " mm \n"

        if C == 1:
            mystr += "Spacing dx1: " + str(self.dx1/um) + " um; Spacing dy1: " + str(self.dy1/um) + " um \n"
            mystr += "Spacing dx2: " + str(self.dx2/um) + " um; Spacing dy2: " + str(self.dy2/um) + " um \n"
            mystr += "Offset cx1: " + str(self.cx1/mm) + " mm; Offset cy1: " + str(self.cy1/mm) + " mm \n"
            mystr += "Offset cx2: " + str(self.cx2/mm) + " mm; Offset cy2: " + str(self.cy2/mm) + " mm \n"
            mystr += "N x1: " + str(self.Nx1) + "; N y1: " + str(self.Ny1) + " \n"
            mystr += "N x2: " + str(self.Nx2) + "; N y2: " + str(self.Ny2) + " \n"

        else:
            mystr += "Spacing dx1: " + str(self.dx1.squeeze()/um) + " um; Spacing dy1: " + str(self.dy1.squeeze()/um) + " um \n"
            mystr += "Spacing dx2: " + str(self.dx2.squeeze()/um) + " um; Spacing dy2: " + str(self.dy2.squeeze()/um) + " um \n"
            mystr += "Offset cx1: " + str(self.cx1.squeeze()/mm) + " mm; Offset cy1: " + str(self.cy1.squeeze()/mm) + " mm \n"
            mystr += "Offset cx2: " + str(self.cx2.squeeze()/mm) + " mm; Offset cy2: " + str(self.cy2.squeeze()/mm) + " mm \n"
            mystr += "N x1: " + str(self.Nx1.squeeze()) + "; N y1: " + str(self.Ny1.squeeze()) + " \n"
            mystr += "N x2: " + str(self.Nx2.squeeze()) + "; N y2: " + str(self.Ny2.squeeze()) + " \n"

        return mystr


    def __repr__(self):
        return self.__str__()
