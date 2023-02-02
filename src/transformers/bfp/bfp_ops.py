# Tianchu ji: copy bfp_ops.py directly from https://compas.cs.stonybrook.edu:1121/jason.m.cheung/bfp-transformers/raw/mytransformers/examples/pytorch/question-answering/bfp_ops.py
# consider adding it as submodule if in the future
# we need to trace its history
# Jason Cheung: Commented the important functions below

# Copyright (c) 2021, Parallel Systems Architecture Laboratory (PARSA), EPFL & 
# Machine Learning and Optimization Laboratory (MLO), EPFL. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the PARSA, EPFL & MLO, EPFL
#    nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pdb
import itertools as it
import logging
import unittest
import os.path
from torch.autograd.function import InplaceFunction, Function


class rounding_modes:
    """
    When converting fp32 tensors to bfp, the rounding mode can be chosen.
    STOC: Stochastic rounding
    DETERM: Deterministic rounding
    """
    STOC, DETERM = 'stoc', 'determ'
    modes = [STOC, DETERM]
    
def fp_range(exp_bits, mant_bits,): 
    maximum = (2**(2**(exp_bits - 1)-1)) * (2-2**-mant_bits) 
    minimum = (2**-(2**(exp_bits - 1)- 2)) * (2**-mant_bits)
    print(maximum, minimum)
    return((2**(2**(exp_bits - 1)-1)) * (2-2**-mant_bits) - (2**-(2**(exp_bits - 1)- 2)) * (2**-mant_bits))

def round_tensor(t, mode, device):
    """
    Perform the rounding of the tensor t by using selected mode
    """
    #if mode == rounding_modes.STOC:
    #    if device == "cpu":
    #        sampled = torch.FloatTensor(t.size(), device = device).uniform_(-0.5, 0.5)
    #    else:
    #        sampled = torch.cuda.FloatTensor(t.size()).uniform_(-0.5, 0.5)
    #    return sampled.add_(t).round()
    if mode == rounding_modes.DETERM:
        return t.round()
    else:
        raise NotImplementedError("Rounding mode %s is not implemented", mode)

def get_exponent(t, epsilon):
    """
    Find the shared exponent of the tensor t.
    The exponent of the largest tensor value is selected as the shared exponent.
    """
    #Exponent is independent of the sign
    t = t.abs()
    #Find the maximum element of the tensor t
    max_v, _ = t.max(dim=1, keepdim=True)

    #Get the exponent of that element (We use ceil because in bfp format, we convert using 0.mantissa_bits instead of fp32's 1.mantissa_bits)
    maximum = (max_v + epsilon).log2().ceil()
    return maximum #- maximum%(2**(8 - exp_bits))


def _float_to_bfp(t,  mant_bits, epsilon = 1e-30, rounding_mode = 'determ', device = 'cpu',  prune = False):
    """
    Convert float tensor t to bfp
    """
    
    
    #t = t / 2**((t.abs() + epsilon).log2().ceil()%(2**(8-exp_bits)))    
    #PRUNE BLOCKt[(torch.quantile(t.abs(), 0.8, dim = 1) < torch.quantile(t.abs(), 0.5)).flatten()] = torch.zeros(t.shape[1]) 
    #if prune:    
    #    k = 3
    #    idx = np.argpartition(t.abs(),k,axis = 1)
    #    t[np.arange(t.shape[0])[:,None],idx[:,:k]] = 0
    #t[t==10000] = 0
    exp = get_exponent(t, epsilon)
        
    #The interval between two consecutive numbers with that exponent value
    interval = torch.pow(2.0, exp-mant_bits)
    #The maximum representable value with exp
    max_v = torch.pow(2.0, exp) - interval

    # To ensure that we preserve the interval
    t = t/interval
    rounded = round_tensor(t, rounding_mode, device)
    rounded *=  interval

    #To ensure that there is no underflow or overflow
    return torch.min(torch.max(rounded, -max_v), max_v)


def float_to_bfp_batched(t, mant_bits, epsilon, rounding_mode, device, bfp_tile_size=25,
                         num_format='', weight_mant_bits=''):
    """
    Convert a batch of fp32 tensor t to bfp
    """
    assert num_format == 'bfp'
    orig_shape = t.size()

    t = t.view(t.size()[0], -1)
    o = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
    return o.view(orig_shape)


def tensor_to_tiled(t, orig_shape, height_tile_size, width_tile_size):
    """
    Handle the tiling process.
    Output: the tiled tensor, the number of tiles in each dimension, the dimensions before and after the tiling to help 'untiling'
    """
    t = t.view(orig_shape[0], -1)
    matrix_h, matrix_w = t.size()

    numberOf_h_tiles = (matrix_h + height_tile_size - 1) // height_tile_size
    numberOf_w_tiles = (matrix_w + width_tile_size - 1) // width_tile_size

    matrix_h_pad = numberOf_h_tiles*height_tile_size
    matrix_w_pad = numberOf_w_tiles*width_tile_size

    h_pad = matrix_h_pad - matrix_h
    w_pad = matrix_w_pad - matrix_w

    t = F.pad(t, (0, w_pad, 0, h_pad),'constant',)
    # t <-numberOf_h_tiles, tile_h, matrix_w
    t = t.view(numberOf_h_tiles, height_tile_size, matrix_w_pad)
    # t <- numberOf_h_tiles, matrix_w, tile_h,
    t.transpose_(1, 2)
    return (t.contiguous().view(numberOf_h_tiles*numberOf_w_tiles, -1),
            numberOf_h_tiles, numberOf_w_tiles,
            matrix_h, matrix_w,
            matrix_h_pad, matrix_w_pad)

def tiled_to_tensor(t, orig_shape, height_tile_size, width_tile_size,
                    numberOf_h_tiles, numberOf_w_tiles,
                    matrix_h, matrix_w,
                    matrix_h_pad, matrix_w_pad):
    """
    Turn the tensor back to its shape before tiling
    """
    # t <- numberOf_h_tiles, numberOf_w_tiles, tile_w, tile_h
    t = t.view(numberOf_h_tiles, numberOf_w_tiles, width_tile_size, height_tile_size)
    # t <- numberOf_h_tiles, numberOf_w_tiles, tile_h, tile_w
    t.transpose_(2, 3)
    # t <- numberOf_h_tiles, tile_h, numberOf_w_tiles, tile_w
    t.transpose_(1, 2)
    t = t.contiguous().view(matrix_h_pad, matrix_w_pad)
    return t.narrow(0, 0, matrix_h).narrow(1, 0, matrix_w).view(orig_shape)

def float_to_bfp_tiled(t, mant_bits, height_tile_size, width_tile_size, prune = False, save_blocks = False, filename = '', epsilon = 1e-30, rounding_mode = 'determ', device = 'cpu', exp_bits = 8, 
                       num_format='bfp', weight_mant_bits=0,
                       sgd_update=False, mant_bits_pow=None):
    """
    Convert fp32 tensor t to bfp with tiling.
    Used for weights (which are handled in the optimizer)
    """
    
    #if prune:
    #    t[t.abs() < torch.quantile(t.abs(), 0.2)] = 0
        
    assert num_format == 'bfp'
    if sgd_update:
        mant_bits = weight_mant_bits

    orig_shape = t.size()
    if height_tile_size == 0 or width_tile_size == 0:
        return _float_to_bfp(t.view(1, -1), mant_bits, epsilon, rounding_mode, device).view(orig_shape)

    (t, numberOf_h_tiles, numberOf_w_tiles, matrix_h, matrix_w,
        matrix_h_pad, matrix_w_pad) = tensor_to_tiled(t, orig_shape, height_tile_size, width_tile_size)
    #print(t)
    
    # OLD
    if save_blocks and not os.path.exists(filename + '_fullblocks.pt'):
        with open(filename + '_fullblocks.pt', 'wb') as f:
            torch.save(t, f)
            
    
    t = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device, prune)
    
    # OLD
    if save_blocks and not os.path.exists(filename + '_bfpblocks.pt'):
        with open(filename + '_bfpblocks.pt', 'wb') as f:
            torch.save(t, f)
            
    return tiled_to_tensor(t, orig_shape, height_tile_size, width_tile_size,
                           numberOf_h_tiles, numberOf_w_tiles,
                           matrix_h, matrix_w,
                           matrix_h_pad, matrix_w_pad)



#######################################################################################
# My stuff is here
#######################################################################################

# function that handles actual bfp conversion: uses division as in the EPFL code
# I rewrote the blocking since I found the original code hard to understand
def my_float_to_bfp(t,  mant_bits, vec_size, entire = 0, epsilon = 6e-5, rounding_mode = 'determ', device = 'cpu',  prune = False):
    """
    Convert float tensor t to bfp
    """
    # experimental: share exponent across the entire tensor, not just a row 
    if entire:
        orig_shape = t.shape
        reshaped_t = t.reshape(1, -1)
        
        exp = get_exponent(reshaped_t, epsilon)
    
        interval = torch.pow(2.0, exp-mant_bits)
        #The maximum representable value with exp
        max_v = torch.pow(2.0, exp) - interval
    
        # To ensure that we preserve the interval
        reshaped_t = reshaped_t/interval
        rounded = round_tensor(reshaped_t, rounding_mode, device)
        rounded *=  interval
        
        result = torch.min(torch.max(rounded, -max_v), max_v).reshape(orig_shape)
    
    # usually do this    
    else:
        if t.shape[-1]%vec_size == 0:
            pad = 0
        else:
            pad = vec_size - t.shape[-1]%vec_size
        pad = vec_size - t.shape[-1]%vec_size
        padded_t = F.pad(t, (0, pad))
        padded_t_shape = padded_t.shape
        reshaped_t = padded_t.reshape(-1, vec_size)
        
        exp = get_exponent(reshaped_t, epsilon)
    
        interval = torch.pow(2.0, exp-mant_bits)
        #The maximum representable value with exp
        max_v = torch.pow(2.0, exp) - interval
        # To ensure that we preserve the interval
        reshaped_t = reshaped_t/interval
        rounded = round_tensor(reshaped_t, rounding_mode, device)
        rounded *= interval
        
        result = torch.min(torch.max(rounded, -max_v), max_v).reshape(padded_t_shape)
        result = result[..., :-pad]
        #To ensure that there is no underflow or overflow
    return result
    

# pytorch function that wraps my_float_to_bfp to handle gradients during fine tuning
# I define the backward pass as the straight through estimator
class bfp(Function):
    @classmethod
    def forward(cls, ctx, t, mant_bits, width_tile_size, entire=0, prune = False, save_blocks = False, filename = '', epsilon = 1e-30, rounding_mode = 'determ', device = 'cpu', exp_bits = 8, 
                           num_format='bfp', weight_mant_bits=0,
                           sgd_update=False, mant_bits_pow=None):
        return my_float_to_bfp(t, mant_bits, width_tile_size, entire = entire)

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None

# call this function in modeling_bert.py on activation tensors
def convert_bfp(t, mant_bits, width_tile_size, entire = 0, rounding_mode = 'determ', device = 'cpu'):
    return bfp().apply(t, mant_bits, width_tile_size, entire, rounding_mode, device)

# replace nn.Linear layers with BFPLinear layers
class BFPLinear(nn.Linear):
    def __init__(self, in_features : int, out_features : int, bias : bool = True):
        super(BFPLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
    def forward(self, input, mant_bits, width_tile_size, toggle = False, entire = 0, rounding_mode = 'determ', device = 'cpu'):
        
        if toggle:
            bfp_input = convert_bfp(input, mant_bits, width_tile_size, entire = entire, rounding_mode=rounding_mode, device=device)
            bfp_weight = convert_bfp(self.weight, mant_bits, width_tile_size, entire = entire, rounding_mode=rounding_mode, device=device)
            result = nn.functional.linear(bfp_input, bfp_weight, self.bias)
        else:
            result = nn.functional.linear(input, self.weight, self.bias)

        return result