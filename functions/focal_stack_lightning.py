import os
import numpy as np
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import pytorch_lightning as L
import matplotlib.pyplot as plt

import kornia
import kornia.metrics

from functions.focal_stack_simulator import FocalStackSimulator
from functions.param_loader import Params, SLM_INIT_MODE
from functions.far_field_holographic_focal_stack_propagator import FarFieldHolographicPropagator
from functions.slm_loader import SLMLoader

class FocusStackLightning(L.LightningModule):
    
    def __init__(self, 
                 focal_stack_dataset    : FocalStackSimulator, 
                 slm_propagator         : FarFieldHolographicPropagator, 
                 params                 : Params, 
                 verbose                : bool = False,
                 lr_slm                 : float = 0.01, 
                 lr_scale               : float = 1e-1, 
                 lr_conv                : float = 1e-3,
                 slm_init_type          : SLM_INIT_MODE = SLM_INIT_MODE.RANDOM, 
                 slm_init_variance      : float = 0.1*np.pi, 
                 l2_loss_scale          : float = 100.0, 
                 l1_loss_scale          : float = 0.5, 
                 regularization_on      : bool = True, 
                 enable_conv_layer      : bool = False, 
                 supervise_mpfocus_only : bool = True):
        
        """
        This class creates a Pytorch Lightning module for traning a hologram to display on the SLM using a focus stack for supervision.

        Part of the code was adapted from Holotorch (https://github.com/facebookresearch/holotorch).

        """
        
        super().__init__()

        self.focal_stack_dataset = focal_stack_dataset
        self.focus_stack_propagtor = focal_stack_dataset.get_focus_stack_propagator()
        self.eye_focus_lengths = focal_stack_dataset.get_eye_focus_lengths()
        self.slm_propagator = slm_propagator
        self.params = params

        slm_tensor_shape = (1, 1, self.params.num_channels, self.focus_stack_propagtor.Ny3, self.focus_stack_propagtor.Nx3)
        self.slm_model = SLMLoader(init_type=slm_init_type, 
                                   tensor_shape=slm_tensor_shape, 
                                   init_variance=slm_init_variance, 
                                   device=self.params.device).to(self.params.device)
        
        self.lr_slm = lr_slm
        self.lr_scale = lr_scale
        self.lr_conv = lr_conv
        self.l2_loss_scale = l2_loss_scale
        self.l1_loss_scale = l1_loss_scale
        self.regularization_on = regularization_on
        self.enable_conv_layer = enable_conv_layer
        self.supervise_mpfocus_only = supervise_mpfocus_only
        self.verbose = verbose

        if self.enable_conv_layer:
            self.cnn = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, padding=2), 
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 32, kernel_size=5, padding=2), 
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 3, kernel_size=5, padding=2))

        # self.configure_optimizers()

        self.list_loss = []
        self.list_psnr = []

    @staticmethod
    def compute_laplacian(img):
        '''
        Assuming img is a PyTorch tensor with shape (B, F, P, C, H, W)
        '''

        # Extend img borders by 1 pixel to mimic 'nearest' padding
        img_padded = F.pad(img, (1, 1, 1, 1), mode='constant')
        center = img_padded[:, :, :, :, 1:-1, 1:-1]
        top = img_padded[:, :, :, :, :-2, 1:-1]
        bottom = img_padded[:, :, :, :, 2:, 1:-1]
        left = img_padded[:, :, :, :, 1:-1, :-2]
        right = img_padded[:, :, :, :, 1:-1, 2:]
        diagonal_tl = img_padded[:, :, :, :, :-2, :-2]
        diagonal_tr = img_padded[:, :, :, :, :-2, 2:]
        diagonal_bl = img_padded[:, :, :, :, 2:, :-2]
        diagonal_br = img_padded[:, :, :, :, 2:, 2:]

        # Compute the Laplacian
        laplacian = top + bottom + left + right + \
                    diagonal_tl + diagonal_tr + diagonal_bl + diagonal_br - 8 * center
        return laplacian

    def quantized_voltage(self, voltage_map, bit_depth=8):
        '''
        Perform quantization of the slm phase pattern into 8 bit with the same dtype.
        '''

        voltage_map = torch.round((2.**bit_depth-1) * voltage_map) / (2.**bit_depth-1)

        return voltage_map    
    
    def training_step(self, batch, batch_idx):

        self.train()

        ### Obtain the SLM field
        phases, scale = self.slm_model.forward(quantize=False)

        if self.enable_conv_layer:
            phases = self.cnn(phases[0])
            phases = phases.unsqueeze(0)

        slm_field = torch.exp(1j*phases)
        slm_field = slm_field[None, ...]

        ### Obtain focus stack and mask layers targets
        texture, diopter, focus_stack_target, mpi_mask_focus = batch

        ### Propagate the SLM field to predict focus stack
        output_abs, output_phase = self.slm_propagator(slm_field, self.eye_focus_lengths)

        ### Compute loss
        if self.supervise_mpfocus_only:
            output_abs = output_abs * mpi_mask_focus * scale[:, :, None, :, None, None]
            target_abs = focus_stack_target * mpi_mask_focus
        else:
            output_abs = output_abs * scale[:, :, None, :, None, None]
            target_abs = focus_stack_target

        loss = nn.functional.mse_loss(output_abs, target_abs) * self.l2_loss_scale
        psnr = self.compute_psnr(target=target_abs, input=output_abs)
            
        ### Regularize in focus regions to have flat phase
        if self.regularization_on:
            output_phase_laplacian = self.compute_laplacian(output_phase) * mpi_mask_focus
            reg = torch.nn.functional.l1_loss(
                output_phase_laplacian, 
                torch.zeros(output_phase_laplacian.shape).to(output_phase.device)) * self.l1_loss_scale
            if self.verbose:
                print('l2 loss:', loss.cpu().item(), '; l1 reg loss:', reg.cpu().item())
            loss = loss + reg
        else:
            if self.verbose:
                print('l2 loss:', loss.cpu().item())

        ### Log metrics
        self.list_loss.append(loss.detach().cpu().numpy())
        self.list_psnr.append(psnr.detach().cpu().numpy())

        torch.cuda.empty_cache()

        return loss

    @staticmethod
    def compute_psnr(
        target : torch.Tensor,
        input : torch.Tensor,
        max_value = 255.0,
    ):

        amax = target.amax(dim=[-2,-1])[...,None,None] 
        target = ( target / amax ) * max_value # 255 represents the bitlevel
        input  = ( input / amax ) * max_value 

        psnr = kornia.metrics.psnr(image = input, target = target, max_val = max_value)

        return psnr
    
    @staticmethod
    def compute_ssim(
        target : torch.Tensor,
        input : torch.Tensor,
        max_value = 255.0,
    ):

        amax = target.amax(dim=[-2,-1])[...,None,None] 
        target = ( target / amax ) * max_value # 255 represents the bitlevel
        input  = ( input / amax ) * max_value 

        psnr = kornia.metrics.ssim(image = input, target = target, max_val = max_value)

        return psnr
    
    def configure_optimizers(self):
        
        if self.enable_conv_layer:
            optimizer = torch.optim.Adam([{"params" : self.cnn.parameters(), "lr" : self.lr_conv}, 
                                          {"params" : self.slm_model.scale, "lr" : self.lr_scale}])
        else:
            optimizer = torch.optim.Adam([{"params" : self.slm_model.phase, "lr" : self.lr_slm}, 
                                          {"params" : self.slm_model.scale, "lr" : self.lr_scale}])

        return optimizer

    def visualize_loss(self):

        mse = np.array(self.list_loss)
        psnr = np.array(self.list_psnr)

        # visualize loss
        fontsize = 17
        fig, ax = plt.subplots()
        plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

        # plot MSE/PSNR
        ax.plot(mse,color='blue')
        ax.set_ylabel('Training Loss', fontsize=fontsize)
        ax.set_xlabel('Iteration #', fontsize=fontsize)
        ax = plt.gca()
        ax.yaxis.label.set_color('blue')        #setting up X-axis label color to yellow
        ax.tick_params(axis='y', colors='blue')  #setting up Y-axis tick color to black
        
        # Generate a new Axes instance, on the twin-X axes (same position)
        ax2 = ax.twinx()
        ax2.plot(psnr, color='red')
        ax2.set_ylabel('PSNR (dB)', fontsize=fontsize, color='red')
        ax2.spines['right'].set_color('red')  
        ax2.spines['left'].set_color('blue')  
        ax2.xaxis.label.set_color('red')        #setting up X-axis label color to yellow
        ax2.tick_params(axis='y', colors='red')  #setting up Y-axis tick color to black

        plt.show()

    def show_focusstack(self, focus_stack : torch.Tensor, name='focus stack'):
        '''
        Args:
            focus_stack (torch.Tensor): intensity having shape num_focuses x H x W X C
        '''

        capture_stack = focus_stack.permute(0, 2, 3, 1).detach().cpu().numpy()

        B, H, W, C = capture_stack.shape

        num_rows =len(np.arange(0, B, 3))

        plt.figure(figsize=(20,20 * (num_rows/3)))

        for i in range(0, B, 3):
            img1 = capture_stack[i]
            if i + 1 < B:
                img2 = capture_stack[i+1]
            if i + 2 < B:
                img3 = capture_stack[i+2]
            
            plt.subplot(num_rows,3,i+1)
            plt.imshow(img1)
            plt.axis('off')
            plt.title(name + ' image '+str(i))
            if i + 1 < B:
                plt.subplot(num_rows,3,i+2)
                plt.imshow(img2)
                plt.axis('off')
                plt.title(name + ' image '+str(i+1))
            if i + 2 < B:
                plt.subplot(num_rows,3,i+3)
                plt.imshow(img3)
                plt.axis('off')
                plt.title(name + ' image '+str(i+2))

        plt.show()

        return capture_stack
    
    def visualize_slm(self):

        phases, scale = self.slm_model.forward(quantize=False)

        slm_img = phases.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        slm_img = slm_img / np.max(slm_img)

        plt.figure(figsize=(20,4))
        plt.subplot(1,3,1)
        plt.imshow(slm_img[:,:,0], 'gray')
        plt.colorbar()
        plt.title('Red')
        plt.subplot(1,3,2)
        plt.imshow(slm_img[:,:,1], 'gray')
        plt.colorbar()
        plt.title('Green')
        plt.subplot(1,3,3)
        plt.imshow(slm_img[:,:,2], 'gray')
        plt.colorbar()
        plt.title('Blue')
        plt.show()

    def load_checkpoint(self, path):
        '''
        example path: ./lightning_logs/version_8/checkpoints/epoch=399-step=400.ckpt
        '''

        model = FocusStackLightning.load_from_checkpoint(path, 
                                          focal_stack_dataset = self.focal_stack_dataset, 
                                          slm_propagator = self.slm_propagator, 
                                          params = self.params, 
                                          lr_slm = self.lr_slm, 
                                          regularization_on = self.regularization_on, 
                                          supervise_mpfocus_only = self.supervise_mpfocus_only)
        model.eval()
        
        return model
    