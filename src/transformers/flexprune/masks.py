import torch
from torch.nn.attention.flex_attention import _score_mod_signature

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx