import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import kornia

from functions.param_loader import Params

class SceneLoader(torch.nn.Module):

    def __init__(self, params : Params) -> None:
        
        super().__init__()

        self.num_pixel_y, self.num_pixel_x = params.sensor_shape
        self.grayscale = params.grayscale
        self.num_depths = params.num_depths
        self.params = params
        
    @staticmethod
    def normalize(input : torch.Tensor, method="max"):
        """
        Args:
            method(str) : mean | max

        """
        if method == "mean":
            # compute the mean from only the valid portion
            normalization_factor = torch.mean(input.abs())
        elif method == "max":
            # compute the max from only the valid portion
            normalization_factor = torch.max(input)

        out = input / normalization_factor

        return out
    
    @staticmethod
    def compute_aspect_ratio(height, width):
        aspect = width / height
        return aspect

    @staticmethod
    def crop_to_aspect_ratio(ideal_aspect, img):

        height, width = img.shape[-2], img.shape[-1]
        aspect_ratio = SceneLoader.compute_aspect_ratio(height=height, width=width)

        if aspect_ratio > ideal_aspect:
            # Then crop the left and right edges:
            new_width = int(height * ideal_aspect)
            offset = (width - new_width) / 2
            resize = (0, offset, height, width - offset)
        else:
            # ... crop the top and bottom:
            new_height = int(width / ideal_aspect)
            offset = (height - new_height) / 2
            resize = (offset, 0, height - offset, width)

        resize = np.array(resize,dtype=np.uint32)

        img_new = img[:,resize[0]:resize[2],resize[1]:resize[3]]
        
        return img_new

    def discretize(self, image: torch.Tensor, diopter_bins: torch.Tensor):

        diopter_indexes = torch.bucketize(image, diopter_bins)
        diopter_image_discretized = diopter_bins[diopter_indexes]

        # print('num depths:', len(torch.unique(diopter_image_discretized)))
        
        return diopter_image_discretized
    
    def sample_input_image(self, image):

        ### Define input image coordinates
        dy1 = self.params.oled_pixel_pitch
        Ny1 = image.shape[-2]
        wy1 = Ny1 * dy1
        self.cy1 = - wy1 / 2
        self.y1 = self.cy1 + torch.arange(0, Ny1) * dy1
        dx1 = self.params.oled_pixel_pitch
        Nx1 = image.shape[-1]
        wx1 = Nx1 * dx1
        self.cx1 = - wx1 / 2
        self.x1 = self.cx1 + torch.arange(0, Nx1) * dx1

        ### Set the input sample coordinates
        x1_sim_normalized = self.params.x1_sim / np.abs(self.cx1)    # torch.max(x1)
        y1_sim_normalized = self.params.y1_sim / np.abs(self.cy1)    # torch.max(y1)
        X1_sim_normalized, Y1_sim_normalized = torch.meshgrid(x1_sim_normalized, 
                                                              y1_sim_normalized, 
                                                              indexing='xy')        
        grid = torch.stack([X1_sim_normalized, Y1_sim_normalized], dim=-1).unsqueeze(0)

        ### Sample the input images to the simulation coordinates
        image_sampled = F.grid_sample(image.unsqueeze(0),               # 1 x C x H x W
                                      grid, 
                                      mode='nearest', 
                                      padding_mode='zeros', 
                                      align_corners=True)
        
        ### Crop out apertured regions
        X1_sim, Y1_sim = torch.meshgrid(self.params.x1_sim, self.params.y1_sim, indexing='xy')
        X1_sim = X1_sim[None, None, ...]                                # 1 x 1 x H x W
        Y1_sim = Y1_sim[None, None, ...]                                # 1 x 1 x H x W
        mask = (X1_sim ** 2 + Y1_sim ** 2) <= self.params.aperture / 2
        image_sampled = image_sampled * mask                            # 1 x C x H x W

        return image_sampled

    def forward(self,
                input               : torch.Tensor, 
                normalize           : bool = False,
                discretize          : bool = False,
                resize              : bool = False,
                isDepth             : bool = False):
        '''
        Args:

            input (torch.Tensor)                : input image, (C, H, W) or (H, W)
            normalize (bool)                    : flag to normalize image or not
            discretize (bool)                   : flag to discretize the depth image or not
            resize (bool)                       : flag to resize image or not
        '''

        ### Process input shape
        if input.ndim == 2:
            input = input.unsqueeze(0)
        if input.shape[0] == 4:
            input = input[:3]
        if self.grayscale and input.shape[0] != 1: 
            input = kornia.color.rgb_to_grayscale(input)
        
        ### Crop input to the sensor aspect ratio
        sensor_aspect = SceneLoader.compute_aspect_ratio(height=self.num_pixel_y, 
                                                              width=self.num_pixel_x)
        image = SceneLoader.crop_to_aspect_ratio(sensor_aspect, input)
        
        ### Resize the image
        if resize:
            if isDepth:
                op_resize = torchvision.transforms.Resize(size=[self.num_pixel_y, self.num_pixel_x], interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT)
                image = op_resize(image)
            else:
                op_resize = torchvision.transforms.Resize(size=[self.num_pixel_y, self.num_pixel_x])
                image = op_resize(image)

        ### Normalize the image
        if normalize:
            image = SceneLoader.normalize(input = image, method =  "max")
        
        ### Sample to simulation coordinates
        image_sampled = self.sample_input_image(image)  # (1, C, H, W) or (1, 1, H, W)
        
        ### Discretize the image
        diopter_bins = None
        if discretize:
            diopter_bins = torch.linspace(torch.min(image), torch.max(image), self.num_depths)
            image_sampled = self.discretize(image_sampled, diopter_bins)
            image = self.discretize(image, diopter_bins)

        # Convert to 5D format (1, 1, C, H, W) pr (1, 1, 1, H, W)
        image = image[None]
        image_sampled = image_sampled[None]

        return image, image_sampled, diopter_bins
