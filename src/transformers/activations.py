import math

import torch
import torch.nn.functional as F

from .utils import logging


logger = logging.get_logger(__name__)


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def gelu_plan(x):
    """Implementation of the gelu by piecewise linear approximation
    """
    # define threhold, factors and offsets here for the linear transformation
    linear_lut = [-2.625, -2.25, -1.875, -1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5, 1.875, 2.25, 2.625]
    factors = [-0.04345971561780215, 
                -0.07946560965665517, 
                -0.11580649391038977,
                -0.12359143782706698,
                -0.062038050010177294,
                0.0995924410601643,
                0.35384589680886097, 
                0.646154103191139,
                0.9004075589398356, 
                1.0620380500101776,
                1.1235914378270666,
                1.1158064939103898,
                1.0794656096566555,
                1.0434597156178025]

    offsets = [-0.12498564006433087, 
                -0.20599890165175014,
                -0.27413805962750254,
                -0.28581547550251835,
                -0.21656791420851745,
                -0.09534504590576126,
                0.0,
                0.0,
                -0.0953450459057612,
                -0.21656791420851762,
                -0.28581547550251796,
                -0.2741380596275025,
                -0.20599890165175072,
                -0.12498564006433144]
    # apply to inputs
    gelu_res = torch.zeros(x.shape)
    if gelu_res.get_device() != x.get_device():
        gelu_res = gelu_res.to(x.get_device())
    for lut_range in range(len(linear_lut)-1):
        res = torch.zeros(x.shape)
        tmp = x * (x > linear_lut[lut_range])
        res = tmp * (tmp <= linear_lut[lut_range+1])
        trans_mask = (res != 0.000)
        res = trans_mask * (res * factors[lut_range] + offsets[lut_range])
        gelu_res += res

    gelu_res += x * (x > linear_lut[-1])
    return gelu_res

if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


ACT2FN = {
    "relu": F.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "gelu_plan": gelu_plan,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))
