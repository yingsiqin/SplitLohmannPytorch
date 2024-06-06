import torch

from functions.param_loader import SLM_INIT_MODE

class SLMLoader(torch.nn.Module):
    def __init__(self, 
                 init_type      : SLM_INIT_MODE, 
                 tensor_shape   : tuple = None, 
                 init_variance  : float = 1.0, 
                 device         : str = 'cuda', 
                 flag_complex   : bool = False):
        
        super(SLMLoader, self).__init__()

        self.device = device
        slm_tensor = self.compute_init_tensor(init_type=init_type, 
                                              tensor_shape=tensor_shape, 
                                              init_variance=init_variance, 
                                              flag_complex=flag_complex, 
                                              device=device)
        self.phase = torch.nn.Parameter(slm_tensor)
        self.scale = torch.nn.Parameter(torch.ones(tensor_shape[:-2], requires_grad=True, device=device))

    def compute_init_tensor(self, 
                            init_type       : SLM_INIT_MODE, 
                            tensor_shape    : tuple = None, 
                            init_variance   : float = 1.0, 
                            flag_complex    : bool = False, 
                            device          : str = 'cuda') -> torch.Tensor:
        """ Computes the data tensor for one value depending on the init-type

        Returns:
            torch.Tensor: [description]
        """        

        if init_type == SLM_INIT_MODE.RANDOM:           
            slm_tensor = init_variance*torch.rand(tensor_shape, requires_grad=True, device=device)
            if flag_complex:
                slm_tensor = slm_tensor + 1j * init_variance*torch.rand(tensor_shape, requires_grad=True, device=device)
                
        elif init_type == SLM_INIT_MODE.ZEROS:
            slm_tensor = torch.zeros(tensor_shape, requires_grad=True, device=device)
            if flag_complex:
                slm_tensor += 0j

        elif init_type == SLM_INIT_MODE.ONES:
            slm_tensor = torch.ones(tensor_shape, requires_grad=True, device=device)
            if flag_complex:
                slm_tensor += 0j
        
        return slm_tensor
    
    def forward(self, quantize : bool = False) -> torch.Tensor:

        if quantize:
            bit_depth = 8
            self.phase = torch.round((2.**bit_depth-1) * self.phase) / (2.**bit_depth-1)

        return self.phase, self.scale
