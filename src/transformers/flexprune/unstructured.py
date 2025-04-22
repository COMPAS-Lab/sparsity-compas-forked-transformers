import torch
from torch.nn.attention.flex_attention import _score_mod_signature


def generate_unstructured_mod() -> _score_mod_signature:
    def unstructured_mod(score, b, h, q_idx, kv_idx):
        return torch.where(score < 0.6, -float("inf"), score)

    return unstructured_mod