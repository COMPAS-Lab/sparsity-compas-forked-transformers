import torch
from torch.nn.attention.flex_attention import _score_mod_signature


def generate_unstructured_mod(seq_len) -> _score_mod_signature:
    def unstructured_mod(score, b, h, q_idx, kv_idx):
        # return score
        return torch.where(score < (0.6 / seq_len), -float("inf"), score)

    return unstructured_mod