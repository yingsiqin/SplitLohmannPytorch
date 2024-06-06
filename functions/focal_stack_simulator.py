import numpy as np
import cv2
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import resize

import pathlib
from tqdm import tqdm

from functions.scene_loader import SceneLoader
from functions.time_multiplexed_multifocals_propagator import TimeMulMultifocalsPropagator
from functions.split_lohmann_propagator import SplitLohmannPropagator
from functions.param_loader import Params, SLMInitMode, PropagatorMode

class FocalStackSimulator(torch.utils.data.Dataset):

    def __init__(self, 
                 scene_path             : pathlib.Path or str = None, 
                 texture_path           : pathlib.Path or str = None, 
                 diopter_path           : pathlib.Path or str = None, 
                 params                 : Params = Params(), 
                 resize                 : bool = True, 
                 initiate_focal_stack   : bool = True, 
                 propagator_type        : PropagatorMode = PropagatorMode.SPLIT_LOHMANN, 
                 alpha                  : float = 1.89, 
                 gamma                  : float = 1.1, 
                 cropimage              : bool = False):

        super().__init__()

        self.params = params
        self.alpha = alpha
        self.gamma = gamma

        ### Read scene
        assert (scene_path is not None) or ((texture_path is not None) and (diopter_path is not None))
        if scene_path is not None:
            self.texture_image, self.diopter_image, focal_stack_diopters = self._load_scene(scene_path)
        else:
            self.texture_path = texture_path
            self.diopter_path = diopter_path
            assert (scene_path is not None) or ((texture_path is not None) and (diopter_path is not None))
        if scene_path is not None:
            self.texture_image, self.diopter_image, focal_stack_diopters = self._load_scene(scene_path)
        else:
            self.texture_path = texture_path
            self.diopter_path = diopter_path
            self.texture_image = self._read_texture_image(texture_path)                             # (C, H, W)
            self.diopter_image = self._read_diopter_image(diopter_path)                             # (H, W)

        assert self.texture_image.shape[-2:] == self.diopter_image.shape[-2:]

        ### Crop, resize, and sample scene files to simulation coordinates
        self.transform = SceneLoader(params=params)
        self.num_scenes = 1
        mask = torch.ones(self.diopter_image.shape)
        self.texture_processed, self.texture_image_sampled, _ = self.transform(self.texture_image, resize=resize)   # (1, 1, C, H, W)
        self.diopter_processed, self.diopter_image_sampled, diopter_bins = self.transform(self.diopter_image, 
                                                                                          resize=resize, 
                                                                                          isDepth=True, 
                                                                                          discretize=True)          # (1, 1, 1, H, W) 
        self.mask_processed, self.mask_sampled, _ = self.transform(mask, isDepth=True, resize=resize)               # (1, 1, 1, H, W)
        self.diopter_image_sampled = self.diopter_image_sampled * self.mask_sampled
        print(self.texture_image_sampled.shape, self.diopter_image_sampled.shape)
        
        ### Set input image and simulation coordinates
        self.set_input_coordinates()
        self.diopter_bins = diopter_bins
        self.diopter_bins = diopter_bins
        
        ### Define camera / eye focus settings
        if scene_path is not None:
            self.focal_stack_diopters = focal_stack_diopters
        else:
            diopter_idx_increment = np.round(len(diopter_bins)/self.params.num_focuses).astype(np.uint8)
            focus_diopter_idxes = torch.arange(1, len(diopter_bins), diopter_idx_increment)
            focus_diopter_idxes = focus_diopter_idxes[:self.params.num_focuses]
            self.focal_stack_diopters = diopter_bins[focus_diopter_idxes]
        self.eye_focal_lengths = 1 / (self.focal_stack_diopters + 1 / self.params.eye_retina_distance)

        ### Define the propagator
        self.propagator_type = propagator_type
        if propagator_type == PropagatorMode.SPLIT_LOHMANN:
            self.propagator = SplitLohmannPropagator(params=params)
        elif propagator_type == PropagatorMode.TIMEMUL_MULTIFOCALS:
            self.propagator = TimeMulMultifocalsPropagator(params=params)

        ### Generate the focal stack
        if initiate_focal_stack:
            focal_stack = self.create_focal_stack(self.texture_image_sampled, 
                                                  self.diopter_image_sampled, 
                                                  self.mask_sampled, 
                                                  self.propagator, 
                                                  propagator_type, 
                                                  cropimage)     # (num_focuses, C, H, W)
            self.F, self.C, self.H, self.W = focal_stack.shape
            self.focal_stack = focal_stack.view(self.F, 1, self.C, self.H, self.W)

            self.set_train_focuses()

    def set_input_coordinates(self):

        self.x1 = self.transform.x1
        self.y1 = self.transform.y1
        self.cx1 = self.transform.cx1
        self.cy1 = self.transform.cy1
        self.d1_sim = self.params.d1_sim
        self.c1_sim = self.params.c1_sim
        self.N1_sim = self.params.N1_sim
        self.x1_sim = self.params.y1_sim
        self.y1_sim = self.params.y1_sim
    
    def get_input_coordinates(self):

        return self.x1_sim, self.y1_sim, self.d1_sim, self.c1_sim, self.N1_sim

    def get_input_images_processed(self):

        return self.texture_processed, self.diopter_processed
    
    def get_focal_stack_propagator(self):

        return self.propagator
    
    def get_eye_focal_lengths(self):

        return self.eye_focal_lengths_train

    def _load_scene(self, path):
        '''
        Read texture and diopter images as well as eye focus diopters from a .mat scene file.
        '''
        scene = loadmat(path)
        texture_image = scene['textureMap'].astype(np.float32)
        diopter_image = scene['diopterMap'].astype(np.float32)
        if len(diopter_image.shape)==2:
            diopter_image = np.stack([diopter_image,diopter_image,diopter_image], axis=2)
        focal_stack_diopters = scene['example_focus_diopters'][0].astype(np.float32)

        texture_image = torch.tensor(texture_image)
        texture_image = texture_image.permute(2,0,1)
        diopter_image = torch.tensor(diopter_image)
        diopter_image = diopter_image.permute(2,0,1)

        return texture_image, diopter_image, focal_stack_diopters

    def _load_scene(self, path):
        '''
        Read texture and diopter images as well as eye focus diopters from a .mat scene file.
        '''
        scene = loadmat(path)
        texture_image = scene['textureMap'].astype(np.float32)
        diopter_image = scene['diopterMap'].astype(np.float32)
        if len(diopter_image.shape)==2:
            diopter_image = diopter_image[..., None]
        focal_stack_diopters = scene['example_focus_diopters'][0].astype(np.float32)

        texture_image = torch.tensor(texture_image)
        texture_image = texture_image.permute(2,0,1)
        diopter_image = torch.tensor(diopter_image)
        diopter_image = diopter_image.permute(2,0,1)

        return texture_image, diopter_image, focal_stack_diopters

    def _read_texture_image(self, path):
        '''
        Returns a (C, H, W) torch.float32 tensor for color images
        or (H, W) torch.float32 tensor for grayscale images.
        '''

        try:
            image = torchvision.io.read_image(path) / 255.0
        except:
            image = cv2.imread(filename=path)
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.tensor(image)
                image = image.permute(2,0,1)
            image = torch.tensor(image)
            image = image / 255.0

        return image
    
    def _read_diopter_image(self, path):

        try:
            image = torchvision.io.read_image(path)
            image = image / 255.0 * self.params.working_range
        except:
            image = cv2.imread(filename=path)
            if image.shape[-1] == 3:
                image = image[:,:,1]
            image = torch.tensor(image)
            image = image / 255.0 * self.params.working_range

        return image

    def create_mpi_intensity(self,
                             u1                     : torch.Tensor, 
                             diopter_map_sampled    : torch.Tensor, 
                             focal_stack_diopters   : torch.Tensor):
        '''
        Separates the input field tensor into multi-plane image tensor with each plane containing
        only intensity corresponding to 1 depth.

        Args:
            u1 (torch.float32)                  : input field intensity tensor (1 x 1 x C x H1 x W1).
            diopter_map_sampled (torch.float32) : diopter map tensor (1 x 1 x 1 x H1 x W1).
            focal_stack_diopters (torch.float32): diopter map tensor (num_focuses).

        Returns:
            mpi_u1 (torch.float32)              : (num_depths, 1, C, H, W)
            mpi_diopter (torch.float32)         : (num_depths, 1, 1, H, W)
            mpi_mask (torch.float32)            : (num_depths, 1, C, H, W)
            mpi_mask_focus (torch.float32)      : (num_focuses, 1, 1, H, W)
        '''

        _, _, C, H, W = u1.shape
        num_depths = len(self.diopter_bins)
        num_focuses = len(focal_stack_diopters)
        diopter_map_sampled = diopter_map_sampled.repeat(1, 1, C, 1, 1)
        mpi_u1 = torch.zeros(num_depths, 1, C, H, W)
        mpi_diopter = torch.zeros(num_depths, 1, 1, H, W)
        mpi_mask = torch.zeros(num_depths, 1, C, H, W)
        mpi_mask_focus = []

        for j in range(num_depths):
            diopter = self.diopter_bins[j]
            u1_layer = torch.where(diopter_map_sampled==diopter, u1, 0.0)       # (1, 1, C, H, W)
            mask_layer = torch.where(diopter_map_sampled==diopter, 1.0, 0.0)    # (1, 1, C, H, W)
            if diopter in focal_stack_diopters:
                mask_layer_focus = torch.where(diopter_map_sampled==diopter, 1.0, 0.0)  # (1, 1, C, H, W)
                mpi_mask_focus.append(mask_layer_focus[0])                      # (1, C, H, W)
            mpi_u1[j, :, :, :, :] = u1_layer[0]                             # (1, C, H, W)
            mpi_diopter[j, :, :, :, :] = torch.zeros(1, 1, H, W) + diopter  # (1, 1, H, W)
            mpi_mask[j, :, :, :, :] = mask_layer[0]                         # (1, C, H, W)
        
        if len(mpi_mask_focus)>0:
            mpi_mask_focus = torch.cat(mpi_mask_focus, dim=0)                  # (F, C, H, W)
            mpi_mask_focus = mpi_mask_focus[:, 1, :, :]                      # (F, H, W)
            F, H, W = mpi_mask_focus.shape
            mpi_mask_focus = mpi_mask_focus.view(F, 1, 1, H, W)                 # (F, 1, 1, H, W)

        return mpi_u1, mpi_diopter, mpi_mask, mpi_mask_focus

    def create_focal_stack(self, 
                            texture_image : torch.Tensor, 
                            diopter_image : torch.Tensor,
                            mask_sampled  : torch.Tensor, 
                            prop2focalstack : torch.nn.Module, 
                            propagator_type : PropagatorMode = PropagatorMode.SPLIT_LOHMANN, 
                            cropimage       : bool = False):
        """ Assume that texture_image is 3D tensor
        
        Can be either grayscale or color as input

        Args:
            texture_image (torch.Tensor): intensity having shape 1 x 1 x C x H x W
            diopter_image (torch.Tensor): discretized depth having shape 1 x 1 x 1 x H x W
            mask_sampled  (torch.Tensor): valid region binary mask having shape 1 x 1 x 1 x H x W

        Returns:
            focal_stack (torch.Tensor): intensity having shape num_focuses x C x H x W
        """

        num_iters_per_round = self.params.num_iters_per_round
        num_rounds = self.params.num_rounds
        num_iters_total = num_rounds * num_iters_per_round
        H, W = self.params.N1_sim, self.params.N1_sim

        mpi_texture, mpi_diopter, mpi_mask, mpi_mask_focus = self.create_mpi_intensity(
                                                                texture_image, 
                                                                diopter_image, 
                                                                self.focal_stack_diopters)
        mpi_texture = mpi_texture.unsqueeze(0)                  # (1, num_depths, 1, C, H, W)
        mpi_diopter = mpi_diopter.unsqueeze(0)                  # (1, num_depths, 1, 1, H, W)
        mpi_mask = mpi_mask.unsqueeze(0)                        # (1, num_depths, 1, C, H, W)
        
        if propagator_type == PropagatorMode.TIMEMUL_MULTIFOCALS:
            focal_stack = 0
            for i in tqdm(range(num_rounds), desc="Generating the focal stack"):
                initial_phases = torch.exp(1j * 2 * torch.pi * torch.rand(num_iters_per_round, 1, 1, 1, H, W))
                u1 = mpi_texture * initial_phases * mpi_mask        # (num_iterations, num_depths, 1, C, H, W)
                u1 = u1.to(self.params.device)
                diopter_map = mpi_diopter.to(self.params.device)
                focal_stack += prop2focalstack(u1, diopter_map, self.diopter_bins, self.eye_focal_lengths).abs()
            focal_stack = torch.sum(focal_stack, dim=0, keepdim=True) / num_iters_total

        elif propagator_type == PropagatorMode.SPLIT_LOHMANN:
            focal_stack = 0
            for i in tqdm(range(num_rounds), desc="Generating the focal stack"):
                initial_phases = torch.exp(1j * 2 * torch.pi * torch.rand(num_iters_per_round, 1, 1, 1, H, W))
                u1 = texture_image.unsqueeze(0) * initial_phases                # (num_iterations, 1, 1, C, H, W)
                u1 = u1.to(self.params.device)
                diopter_map = diopter_image.unsqueeze(0).to(self.params.device) # (1, 1, 1, 1, H, W)
                binary_mask = mask_sampled.unsqueeze(0).to(self.params.device)  # (1, 1, 1, 1, H, W)
                focal_stack += prop2focalstack(u1, diopter_map, binary_mask, self.diopter_bins, self.eye_focal_lengths).abs()
            focal_stack = torch.sum(focal_stack, dim=0, keepdim=True) / num_iters_total
            focal_stack = torch.flip(torch.flip(focal_stack, [-1,]), [-2,])

        focal_stack = focal_stack.detach().cpu().squeeze()                     # (num_focuses, C, H, W)
        focal_stack = focal_stack / torch.max(focal_stack)

        focal_stack = self.postprocess(focal_stack=focal_stack, alpha=self.alpha, gamma=self.gamma, cropimage=cropimage)
        self.mpi_mask_focus = mpi_mask_focus                    # (num_focuses, 1, 1, H, W)

        torch.cuda.empty_cache()

        return focal_stack

    def gamma_correction(self, 
                         focal_stack : torch.Tensor, 
                         alpha       : float = 1.9, 
                         gamma        : float = 1.1,) -> torch.Tensor:

        focal_stack = torch.clip(alpha * focal_stack ** gamma, 0, 1)

        return focal_stack
    
    def postprocess(self, 
                    focal_stack : torch.Tensor, 
                    alpha       : float = 2.2, 
                    gamma       : float = 1.1,
                    cropimage   : bool = False) -> torch.Tensor:
        """ Assume that texture_image is 4D tensor
        
        Can be either grayscale or color as input

        Args:
            focal_stack (torch.Tensor): focal stack, (num_focuses, 1, C, H, W)

        Returns:
            capture_stack (torch.Tensor): processed and resized focal stack, (num_focuses, 1, C, H_input, W_input)
        """

        capture_stack = self.gamma_correction(focal_stack, alpha=alpha, gamma=gamma)

        capture_stack = resize(img=capture_stack, 
                               size=[self.texture_image_sampled.shape[-2],self.texture_image_sampled.shape[-1]], 
                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                               antialias=True)
        capture_stack = capture_stack / torch.max(capture_stack)

        if cropimage:
            coords = torch.nonzero(self.mask_sampled.squeeze())
            y0, x0 = coords.min(dim=0)[0]
            y1, x1 = coords.max(dim=0)[0] + 1  # add 1 to include the max point
            capture_stack = capture_stack[:, :, y0:y1, x0:x1]

        return capture_stack
    
    def show_focalstack(self, focal_stack : torch.Tensor, cropimage: bool = False):
        """ Assume that texture_image is 4D tensor
        
        Can be either grayscale or color as input

        Args:
            focal_stack (torch.Tensor): intensity having shape num_focuses x C x H x W
        """

        if cropimage:
            coords = torch.nonzero(self.mask_sampled.squeeze())
            y0, x0 = coords.min(dim=0)[0]
            y1, x1 = coords.max(dim=0)[0] + 1  # add 1 to include the max point
            focal_stack = focal_stack[:, :, y0:y1, x0:x1]

        capture_stack = focal_stack.permute(0, 2, 3, 1).detach().cpu().numpy()

        B, H, W, C = capture_stack.shape

        num_rows = len(np.arange(0, B, 3))

        plt.figure(figsize=(20,10 * (num_rows/3)))

        for i in range(0, B, 3):
            img1 = capture_stack[i]
            if i + 1 < B:
                img2 = capture_stack[i+1]
            if i + 2 < B:
                img3 = capture_stack[i+2]
            
            plt.subplot(num_rows,3,i+1)
            plt.imshow(img1)
            plt.axis('off')
            plt.title('Focal stack image '+str(i))
            if i + 1 < B:
                plt.subplot(num_rows,3,i+2)
                plt.imshow(img2)
                plt.axis('off')
                plt.title('Focal stack image '+str(i+1))
            if i + 2 < B:
                plt.subplot(num_rows,3,i+3)
                plt.imshow(img3)
                plt.axis('off')
                plt.title('Focal stack image '+str(i+2))

        plt.show()

        return capture_stack
    
    def show_mp_mask(self):
        """ 
        Shows the multiplane mask for in focus regions.

        Args:
            mpi_mask_focus (torch.Tensor): mask having shape num_focuses x H x W
        """

        mpi_mask_focus = self.mpi_mask_focus.squeeze().detach().cpu().numpy()

        B, H, W = mpi_mask_focus.shape

        num_rows =len(np.arange(0, B, 4))

        plt.figure(figsize=(20,20 * (num_rows/3)))

        for i in range(0, B, 4):
            img1 = mpi_mask_focus[i]
            if i + 1 < B:
                img2 = mpi_mask_focus[i+1]
            if i + 2 < B:
                img3 = mpi_mask_focus[i+2]
            if i + 3 < B:
                img4 = mpi_mask_focus[i+3]
            plt.subplot(num_rows,4,i+1)
            plt.imshow(img1)
            plt.axis('off')
            plt.title('Multiplane mask '+str(i))
            if i + 1 < B:
                plt.subplot(num_rows,4,i+2)
                plt.imshow(img2)
                plt.axis('off')
                plt.title('Multiplane mask '+str(i+1))
            if i + 2 < B:
                plt.subplot(num_rows,4,i+3)
                plt.imshow(img3)
                plt.axis('off')
                plt.title('Multiplane mask '+str(i+2))
            if i + 3 < B:
                plt.subplot(num_rows,4,i+4)
                plt.imshow(img4)
                plt.axis('off')
                plt.title('Multiplane mask '+str(i+3))

        plt.show()

        return mpi_mask_focus
    
    def simulate(self, cropimage : bool = False):
        
        focal_stack = self.create_focal_stack(self.texture_image_sampled, 
                                                self.diopter_image_sampled, 
                                                self.mask_sampled, 
                                                self.propagator, 
                                                self.propagator_type, 
                                                cropimage)     # (num_focuses, C, H, W)
        self.F, self.C, self.H, self.W = focal_stack.shape
        self.focal_stack = focal_stack.view(self.F, 1, self.C, self.H, self.W)
        self.set_train_focuses()

        return self.focal_stack

    def get_mpi_mask_focus(self):

        return self.mpi_mask_focus
    
    def _get_single_batch(self):
        return self.texture_image_sampled, self.diopter_image_sampled, self.focal_stack_train, self.mpi_mask_focus_train
    
    def set_train_focuses(self, focus_idxes=None):

        if focus_idxes is None:
            self.focal_stack_train = self.focal_stack
            self.mpi_mask_focus_train = self.mpi_mask_focus
            self.eye_focal_lengths_train = self.eye_focal_lengths

        else:
            self.focal_stack_train = self.focal_stack[focus_idxes]
            self.mpi_mask_focus_train = self.mpi_mask_focus[focus_idxes]
            self.eye_focal_lengths_train = self.eye_focal_lengths[focus_idxes]

    def __getitem__(self, idx):
        '''
        Returns:

        texture_image   : (1, 1, C, H, W), torch.float32, range [0, 1]
        diopter_image   : (1, 1, 1, H, W), torch.float32, range [0, params, working_range]
        focal_stack     : (F, 1, C, H, W), torch.float32, range [0, 1]
        mpi_mask_focus  : (F, 1, 1, H, W), torch.float32, range [0, 1]
        '''
        
        if idx >= len(self): raise IndexError

        texture_image, diopter_image, focal_stack, mpi_mask_focus = self._get_single_batch()
                
        self.current_batch = texture_image, diopter_image, focal_stack, mpi_mask_focus

        return texture_image, diopter_image, focal_stack, mpi_mask_focus
                
    def __len__(self):
        return self.num_scenes

    # def pre_load_dataset(self):
    #     pass

    # def set_test_dataset(self):
    #     pass
    
    # def set_train_dataset(self):
    #     pass
        
    # def show_sample(self):
    #     pass