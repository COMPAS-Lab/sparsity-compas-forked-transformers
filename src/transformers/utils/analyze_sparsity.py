import torch
import numpy as np

def get_mat_sparsity(dat):
    if type(dat) == torch.Tensor:
        return (1. - torch.count_nonzero(dat) / torch.numel(dat))
    else:
        return (1. - np.count_nonzero(dat) / dat.size)