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
                nonzeros_per_row = torch.count_nonzero(torch.tril(dat), dim=-1)
                proposed_nonzeros = torch.tensor(np.arange(1, dat.size()[-1]+1, 1))
                actual_zeros = torch.sum(proposed_nonzeros - nonzeros_per_row).item()
                return float(actual_zeros) / total_num_elems
        else:
            return (1. - torch.count_nonzero(dat) / torch.numel(dat))
    else:
        return (1. - np.count_nonzero(dat) / dat.size)
    
def to_bco_format(dat: torch.tensor, block_size = [3, 20]):
    flatten_dat = dat.view(-1, dat.size()[-2], dat.size()[-1])
    assert flatten_dat.size()[-2] == flatten_dat.size()[-1], "attn size must be in square"
    
    brow_idx, bcol_idx = [], []
    sparse_val = []
    for head_dat in flatten_dat:
        # padding
        required_padding_size = \
            (block_size[0] - head_dat.size()[0] % block_size[0], \
             block_size[1] - head_dat.size()[1] % block_size[1],)
        if sum(required_padding_size) > 0:
            head_dat = torch.nn.functional.pad(
                head_dat, 
                pad=(0, required_padding_size[1], 0, required_padding_size[0]), 
                mode="constant", 
                value=0.0)
        hdat_rblocks = torch.split(head_dat, block_size[0], dim=0)
        for rbidx, rblock in enumerate(hdat_rblocks):
            cblocks = torch.split(rblock, block_size[1], dim=-1)
            for cbidx, cblock in enumerate(cblocks):
                if torch.sum(cblock).item() > 0:
                    brow_idx.append(rbidx)
                    bcol_idx.append(cbidx)
                    sparse_val.append(cblock)

        brow_idx.append(-1)
        bcol_idx.append(-1)
        sparse_val.append(torch.ones(block_size))

    return (brow_idx, bcol_idx, sparse_val) 
