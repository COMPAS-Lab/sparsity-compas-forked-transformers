import torch
from torch.nn.attention.flex_attention import _score_mod_signature

def generate_causal() -> _score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """

    def causal_mod(score, b, h, q_idx, kv_idx):
        return torch.where(q_idx >= kv_idx, score, -float("inf"))

    return causal_mod