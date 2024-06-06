import torch
import numpy as np

from enum import Enum

class Params():


    def __init__(self) -> None:
        super(Params).__init__()
        
        self.focal_length           = 100e-03
        self.aperture               = 25.4e-3
        self.eyepiece_focal_length  = 40e-03
        self.eye_diameter           = 5e-03
        self.eye_retina_distance    = 25e-03
        self.grayscale              = False
        self.working_range          = 4.0
        self.C0                     = 1.9337e-02
        self.d_slm_to_3dnominal     = self.focal_length

        self.simulation_resolution  = 6.0e-06
        self.oled_pixel_pitch       = 4.0e-06
        self.slm_pixel_pitch        = 4.0e-06
        self.sensor_pixel_pitch     = 4.0e-06
        self.sensor_shape           = [2160, 3840]

        self.num_depths             = 50        # number of depths to quantize the depth map to
        self.num_focuses            = 10        # number of focuses to generate for focal stack
        self.num_iters_per_round    = 4
        self.num_rounds             = 5

        self.wavelengths            = 530e-09   # torch.Tensor([619e-09, 530e-09, 461e-09]) # R, G, B
        self.num_channels           = 3

        self.device                 = 'cuda'

        self.calculate_simulation_coordinates()

    def calculate_simulation_coordinates(self):

        self.N = np.round(self.wavelengths * self.focal_length / self.simulation_resolution ** 2)
        self.simulation_resolution = np.sqrt(self.wavelengths * self.focal_length / self.N)
        self.dz_max = self.working_range * (self.eyepiece_focal_length ** 2)

        self.max_shift_by_slm = self.wavelengths * self.focal_length/(2*self.slm_pixel_pitch)
        dbpp_max = self.dz_max / 2 * (self.C0/(6*self.focal_length ** 2))
        self.aperture_T2 = self.eye_diameter
        self.aperture_T1 = self.aperture_T2 + 2 * np.sqrt(2) * dbpp_max
        
        ### Define simulation input coordinates
        d1_sim = self.simulation_resolution
        N1_sim = self.N
        W1_sim = N1_sim * d1_sim
        c1_sim = - W1_sim / 2
        y1_sim = c1_sim + torch.arange(0, N1_sim) * d1_sim
        self.d1_sim = d1_sim
        self.c1_sim = c1_sim
        self.N1_sim = int(N1_sim)
        self.x1_sim = y1_sim
        self.y1_sim = y1_sim

        ### Define viewing coordinates
        self.d_sensor = self.sensor_pixel_pitch
        self.Ny_sensor, self.Nx_sensor = self.sensor_shape
        Wy_sensor, Wx_sensor = self.Ny_sensor * self.d_sensor, self.Nx_sensor * self.d_sensor
        self.cy_sensor, self.cx_sensor = - Wy_sensor / 2, - Wx_sensor / 2
        self.y_sensor = self.cy_sensor + torch.arange(0, self.Ny_sensor) * self.d_sensor
        self.x_sensor = self.cx_sensor + torch.arange(0, self.Nx_sensor) * self.d_sensor


    def print(self):

        vars_dict = vars(self)
        for key in vars_dict:
            var = vars_dict[key]
            if not isinstance(var, torch.Tensor): 
                print(key, ':', var)


class PropagatorMode(Enum):
    SPLIT_LOHMANN       = 0
    TIMEMUL_MULTIFOCALS = 1


class SLMInitMode(Enum):
    RANDOM  = 0
    ZEROS   = 1
    ONES    = 2