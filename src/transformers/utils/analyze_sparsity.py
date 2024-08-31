import torch
import numpy as np

def get_mat_sparsity(dat, causal_mask = False, per_layer=False):
    dat = dat.cpu()
    if type(dat) == torch.Tensor:
        if causal_mask:
            if per_layer is True:
                attn_list = [get_mat_sparsity(l, causal_mask) for l in dat]
                attn_list = torch.tensor(attn_list)
                print(f"attn list shape: {attn_list.size()}")
                return attn_list
            else:
                total_num_elems = sum(np.arange(1, dat.size()[-1]+1, 1))
                total_num_elems *= dat.view(-1, dat.size()[-1], dat.size()[-1]).size()[0]
                nonzeros_per_row = torch.count_nonzero(dat, dim=-1)
                proposed_nonzeros = torch.tensor(np.arange(1, dat.size()[-1]+1, 1))
                actual_zeros = torch.sum(proposed_nonzeros - nonzeros_per_row).item()
                return float(actual_zeros) / total_num_elems
        else:
            return (1. - torch.count_nonzero(dat) / torch.numel(dat))
    else:
        return (1. - np.count_nonzero(dat) / dat.size)