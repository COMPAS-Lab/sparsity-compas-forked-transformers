# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Masked Version of BERT. It replaces the `torch.nn.Linear` layers with
:class:`~emmental.MaskedLinear` and add an additional parameters in the forward pass to
compute the adaptive mask.
Built on top of `transformers.models.bert.modeling_bert`"""

from transformers.bfp import bfp_ops
import os
import torch.nn.functional as F
import skimage.measure
import numpy as np

import logging
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from emmental import MaskedBertConfig
from emmental.modules import MaskedLinear
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.models.bert.modeling_bert import ACT2FN, load_tf_weights_in_bert


logger = logging.getLogger(__name__)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = MaskedLinear(
            config.hidden_size,
            self.all_head_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
            mask_block_rows = 3,
            mask_block_cols = 20,
        )
        self.key = MaskedLinear(
            config.hidden_size,
            self.all_head_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
            mask_block_rows = 3,
            mask_block_cols = 20,
        )
        self.value = MaskedLinear(
            config.hidden_size,
            self.all_head_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
            mask_block_rows = 3,
            mask_block_cols = 20,
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        threshold=None,
        
        mant_bits = 0,
        vec_size = 0, 
        Wqkv = 0, 
        qk = 0,
        av = 0, 
        index = -1,
        samplepath = '',
        sample = 0,
        stats = 0
    ):
    
        if stats:
            statspath = 'share_qkv/stats/'
            if not os.path.exists(statspath):
                os.makedirs(statspath)
            filepath = 'stats_count.txt'
            if not os.path.exists(statspath + filepath):
                with open(statspath + filepath, 'w') as f:
                    f.write(str(0) + '\n')
                    f.write(str(0) + '\n')
                    f.write(str(0))
                    f.close()
            with open(statspath + filepath, 'r') as f:
                lines = f.read().splitlines()
                count = lines[0]
                accum_seq_len = lines[1]
                accum_seq_len_2 = lines[2]
                f.close()
                
        if sample:
        
            if not os.path.exists(samplepath):
                os.makedirs(samplepath)
            if not os.path.exists(samplepath + 'weights/'):
                os.makedirs(samplepath + 'weights/')
            if not os.path.exists(samplepath + 'act/'):
                os.makedirs(samplepath + 'act/')
            if not os.path.exists(samplepath + 'act/seqlen/'):
                os.makedirs(samplepath + 'act/seqlen/')            
            if not os.path.exists(samplepath + 'act/query/'):
                os.makedirs(samplepath + 'act/query/')
            if not os.path.exists(samplepath + 'act/key/'):
                os.makedirs(samplepath + 'act/key/')
            if not os.path.exists(samplepath + 'act/value/'):
                os.makedirs(samplepath + 'act/value/')
            if not os.path.exists(samplepath + 'act/attscores/'):
                os.makedirs(samplepath + 'act/attscores/')
            if not os.path.exists(samplepath + 'act/mask_attscores/'):
                os.makedirs(samplepath + 'act/mask_attscores/')
            if not os.path.exists(samplepath + 'act/attprobs/'):
                os.makedirs(samplepath + 'act/attprobs/')
            if not os.path.exists(samplepath + 'act/bfp_attprobs/'):
                os.makedirs(samplepath + 'act/bfp_attprobs/')
            if not os.path.exists(samplepath + 'act/context/'):
                os.makedirs(samplepath + 'act/context/')
            
        if sample:
            filepath = 'sample_count.txt'
            if not os.path.exists(samplepath + filepath):
                with open(samplepath + filepath, 'w') as f:
                    f.write(str(0))
                    f.close()
            with open(samplepath + filepath, 'r') as f:
                count = f.readline()
                f.close()
                
        mixed_query_layer = self.query(hidden_states, threshold=threshold, filepath = samplepath + 'weights/query-' + str(index), mant_bits = mant_bits, vec_size = vec_size, sample=sample, toggle_bfp=Wqkv)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states, threshold=threshold, filepath = samplepath + 'weights/key-' + str(index), mant_bits = mant_bits, vec_size = vec_size, sample=sample, toggle_bfp=Wqkv)
            mixed_value_layer = self.value(encoder_hidden_states, threshold=threshold, filepath = samplepath + 'weights/value-' + str(index), mant_bits = mant_bits, vec_size = vec_size, sample=sample, toggle_bfp=Wqkv)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states, threshold=threshold, filepath = samplepath + 'weights/key-' + str(index), mant_bits = mant_bits, vec_size = vec_size, sample = sample, toggle_bfp=Wqkv)
            mixed_value_layer = self.value(hidden_states, threshold=threshold, filepath = samplepath + 'weights/value-' + str(index), mant_bits = mant_bits, vec_size = vec_size, sample=sample, toggle_bfp=Wqkv)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        if qk: 
            BFP_query_layer = bfp_ops.convert_bfp(query_layer, mant_bits, query_layer.shape[-1], )
            BFP_key_layer = bfp_ops.convert_bfp(key_layer, mant_bits, key_layer.shape[-1], )       
    
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(BFP_query_layer, BFP_key_layer.transpose(-1, -2))
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))     
                    
        
        
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
            



        #JASON
        mask_attention_scores = attention_scores.clone()
        #mask_attention_scores [mask_attention_scores < 0] = -10000
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(mask_attention_scores, dim=-1)



        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        if av:
            # must transpose
            transpose_value_layer = value_layer.transpose(2,3)

            #BFP_attention_probs = bfp_ops.convert_bfp(attention_probs, mant_bits, vec_size, )
            BFP_attention_probs = bfp_ops.convert_bfp(attention_probs, mant_bits, attention_probs.shape[-1], entire = 0 )
            BFP_value_layer = bfp_ops.convert_bfp(transpose_value_layer, mant_bits, transpose_value_layer.shape[-1], ).transpose(2,3) #transpose back
            seq_len = torch.argmin(attention_mask, dim = -1, keepdim=True)

            mask_att_prob = BFP_attention_probs.clone()
            mask_att_prob[BFP_attention_probs <= ((1 / seq_len) + 1e-4)] = 0

            context_layer = torch.matmul(mask_att_prob, BFP_value_layer)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)
    
        
        #print(BFP_attention_probs[BFP_attention_probs!=0].min())

        if stats:
            sparse_path = statspath + str(index) + '.pt'
            #print(attention_mask.shape)
            #print(attention_mask)
            #seq_len = np.argmin(attention_mask)
            seq_len = torch.argmin(attention_mask)
            #print(seq_len)
            
            if seq_len > 0:
                #print(seq_len)
                if not os.path.exists(sparse_path):
                    sparsity = torch.zeros(12, BFP_attention_probs.shape[1])
                    torch.save(sparsity, sparse_path)
                #sparsity = torch.load(sparse_path, map_location = torch.device('cpu'))
                sparsity = torch.load(sparse_path, map_location = torch.device('cuda:0'))
                
                actual_att_probs = BFP_attention_probs[0,:,:seq_len,:seq_len]
                flatten_att_probs = actual_att_probs.reshape(actual_att_probs.shape[0], -1) #flatten by head
                
                # sum sparsity percent
                sample_sparsity = (flatten_att_probs == 0).sum(axis = -1) / flatten_att_probs.shape[1] #number of 0s divide by total elements per head
                sparsity[0] += sample_sparsity
            
                # sum sparsity percent squared
                sparsity[1] += sample_sparsity**2
            
                #pad = vec_size - actual_att_probs.shape[-1]%vec_size
                #padded_att_probs = F.pad(actual_att_probs, (0, pad))
                #reshaped_att_probs = padded_att_probs.reshape(padded_att_probs.shape[0], padded_att_probs.shape[1], padded_att_probs.shape[-1] // vec_size, vec_size) # pad column
                
                reshaped_att_probs = torch.from_numpy(skimage.measure.block_reduce(actual_att_probs.cpu(), (1, 3, vec_size), np.count_nonzero)).cuda()
                
               
                count_nzs = reshaped_att_probs.count_nonzero(axis=-1) # number of nonzeros per block of 20
                
                #print(vec_size)
                #print(reshaped_att_probs.shape)
                #print(count_nzs.shape)
                #print(count_nzs)
                avg_nzs_per_block = torch.stack([arr[arr!=0].mean() for arr in reshaped_att_probs.float()]) # avg number of nzs per blocks that aren't completely 0 
                sparsity [2] += avg_nzs_per_block
                sparsity [3] += avg_nzs_per_block**2
                
                avg_nz_blocks_per_row = count_nzs.float().mean(axis=-1) # number of nonzero blocks per row averaged over the number of rows)
                print(avg_nz_blocks_per_row)            
                sparsity [4] += avg_nz_blocks_per_row
                sparsity[5] += avg_nz_blocks_per_row**2
            
                avg_nz_blocks_per_row_percent = (count_nzs / reshaped_att_probs.shape[-1]).mean(axis=-1) # number of nonzero blocks per row divided by the total number of blocks per row averaged over the rows
     
                sparsity [6] += avg_nz_blocks_per_row_percent
                sparsity[7] += avg_nz_blocks_per_row_percent**2
                
                #for row sparsity
                reshaped_att_probs=actual_att_probs
                count_nzs = reshaped_att_probs.count_nonzero(axis=-1)

                avg_nzs_per_row = torch.stack([arr.mean() for arr in count_nzs.float()]) #avg number of nzs per row that aren't completely 0 
                sparsity [8] += avg_nzs_per_row
                sparsity [9] += avg_nzs_per_row**2
                            
                nz_row_percent = (count_nzs!=0).sum(axis=-1) / reshaped_att_probs.shape[1] #number of nonzero rows divided by the number of rows
                            
                sparsity[10] += nz_row_percent
                sparsity[11] += nz_row_percent**2
                
                torch.save(sparsity,sparse_path)
                #print(sparsity)
                
                if index == 11:
                    with open(statspath + filepath, 'w') as f:
                        f.write(str(int(count)+BFP_attention_probs.shape[0]) + '\n')
                        f.write(str(int(accum_seq_len) + seq_len.item()) + '\n')
                        f.write(str(int(accum_seq_len_2) + (seq_len.item())**2))
                        f.close()
        
    
        if sample:
            #print(str(index) + '-' + str(count))
            seq_len = torch.argmin(attention_mask, dim = -1, keepdim=True)
            seqlen_path = samplepath + 'act/seqlen/' + str(index) + '-' + str(count) + '.pt'
            query_path = samplepath + 'act/query/' + str(index) + '-' + str(count) + '.pt'
            key_path = samplepath + 'act/key/' + str(index) + '-' + str(count) + '.pt'
            value_path = samplepath + 'act/value/' + str(index) + '-' + str(count) + '.pt'
            att_score_path = samplepath + 'act/attscores/' + str(index) + '-' + str(count) + '.pt'
            mask_att_score_path = samplepath + 'act/mask_attscores/' + str(index) + '-' + str(count) + '.pt'
            att_prob_path = samplepath + 'act/attprobs/' + str(index) + '-' + str(count) + '.pt'
            bfp_att_prob_path = samplepath + 'act/bfp_attprobs/' + str(index) + '-' + str(count) + '.pt'
            context_path = samplepath + 'act/context/' + str(index) + '-' + str(count) + '.pt'

            with open(seqlen_path, 'wb') as f:
                torch.save(seq_len, f)
            with open(query_path, 'wb') as f:
                torch.save(BFP_query_layer, f)
            with open(key_path, 'wb') as f:
                torch.save(BFP_key_layer, f)
            with open(value_path, 'wb') as f:
                torch.save(BFP_value_layer, f)
            #with open(att_score_path, 'wb') as f:
            #    torch.save(attention_scores, f)
            #with open(mask_att_score_path, 'wb') as f:
            #    torch.save(mask_attention_scores, f)
            #with open(att_prob_path, 'wb') as f:
            #    torch.save(attention_probs, f)
            with open(bfp_att_prob_path, 'wb') as f:
                torch.save(BFP_attention_probs, f)
            #with open(context_path, 'wb') as f:
            #    torch.save(context_layer, f)
            if index == 11:
                with open(samplepath + filepath, 'w') as f:
                    f.write(str(int(count)+1))
                    f.close()


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = MaskedLinear(
            config.hidden_size,
            config.hidden_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
            mask_block_rows = 3,
            mask_block_cols = 20,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, threshold,
        mant_bits = 0,
        vec_size = 0, 
        fc1 = 0, 
        index = -1,
        samplepath = '',
        sample = 0
    ):
    
        hidden_states = self.dense(hidden_states, threshold=threshold, filepath = samplepath + 'weights/fc1-' + str(index), mant_bits = mant_bits, vec_size = vec_size, sample = sample, toggle_bfp=fc1)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        threshold=None,
        
        mant_bits = 0,
        vec_size = 0, 
        Wqkv = 0, 
        qk = 0,
        av = 0, 
        fc1 = 0, 
        index = -1,
        samplepath = '',
        sample = 0,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            threshold=threshold,
            
            mant_bits = mant_bits,
            vec_size = vec_size, 
            Wqkv = Wqkv, 
            qk = qk,
            av = av, 
            index = index,
            samplepath = samplepath,
            sample = sample
        )
        attention_output = self.output(self_outputs[0], hidden_states, threshold=threshold,
            mant_bits = mant_bits,
            vec_size = vec_size, 
            fc1 = fc1,
            index = index,
            samplepath = samplepath,
            sample = sample
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = MaskedLinear(
            config.hidden_size,
            config.intermediate_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
            mask_block_rows = 3,
            mask_block_cols = 20,
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, threshold,
        mant_bits = 0,
        vec_size = 0, 
        fc2 = 0, 
        index = -1,
        samplepath = '',
        sample = 0,
        attention_mask = None
    ):

        hidden_states = self.dense(hidden_states, threshold=threshold, filepath = samplepath + 'weights/fc2-' + str(index), mant_bits = mant_bits, vec_size = vec_size, sample=sample, toggle_bfp=fc2)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = MaskedLinear(
            config.intermediate_size,
            config.hidden_size,
            pruning_method=config.pruning_method,
            mask_init=config.mask_init,
            mask_scale=config.mask_scale,
            mask_block_rows = 3,
            mask_block_cols = 20,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, threshold,
        mant_bits = 0,
        vec_size = 0, 
        fc3 = 0,
        index = -1,
        samplepath = '',
        sample = 0,
        attention_mask = None
    ):

        hidden_states = self.dense(hidden_states, threshold=threshold, filepath = samplepath + 'weights/fc3-' + str(index), mant_bits = mant_bits, vec_size = vec_size, sample=sample, toggle_bfp=fc3)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        threshold=None,
        
        mant_bits = 0,
        vec_size = 0, 
        Wqkv = 0, 
        qk = 0,
        av = 0, 
        fc1 = 0, 
        fc2 = 0, 
        fc3 = 0,
        index = -1,
        samplepath = '',
        sample = 0,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, threshold=threshold,
            mant_bits = mant_bits,
            vec_size = vec_size, 
            Wqkv = Wqkv, 
            qk = qk,
            av = av, 
            fc1 = fc1,
            index = index,
            samplepath = samplepath,
            sample = sample
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output, threshold=threshold,
            mant_bits = mant_bits,
            vec_size = vec_size, 
            fc2 = fc2,
            index = index,
            samplepath = samplepath,
            sample = sample,
            attention_mask = attention_mask
        )
        layer_output = self.output(intermediate_output, attention_output, threshold=threshold,
            mant_bits = mant_bits,
            vec_size = vec_size, 
            fc3 = fc3,
            index = index,
            samplepath = samplepath,
            sample = sample,
            attention_mask = attention_mask
        )
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        threshold=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            #print(i)
            mant_bitsi = 3
            vec_sizei = 20
            Wqkvi = 1
            qki = 1
            avi = 1
            fc1i = 1
            fc2i = 1
            fc3i = 1
            sample = 0
        
            samplepath = 'share_qkv/sample/'
            


            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                threshold=threshold,
                
                mant_bits = mant_bitsi, 
                vec_size = vec_sizei,
                Wqkv = Wqkvi,
                qk = qki,
                av = avi,
                fc1 = fc1i,
                fc2 = fc2i,
                fc3 = fc3i,
                index = i,
                samplepath = samplepath,
                sample = sample
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MaskedBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = MaskedBertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


MASKED_BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~emmental.MaskedBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

MASKED_BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Masked Bert Model transformer outputting raw hidden-states without any specific head on top.",
    MASKED_BERT_START_DOCSTRING,
)
class MaskedBertModel(MaskedBertPreTrainedModel):
    """
    The `MaskedBertModel` class replicates the :class:`~transformers.BertModel` class
    and adds specific inputs to compute the adaptive mask on the fly.
    Note that we freeze the embeddings modules from their pre-trained values.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.embeddings.requires_grad_(requires_grad=False)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        threshold=None,
    ):
        r"""
        threshold (:obj:`float`):
            Threshold value (see :class:`~emmental.MaskedLinear`).

        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.

                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            threshold=threshold,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings(
    """Masked Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    MASKED_BERT_START_DOCSTRING,
)
class MaskedBertForSequenceClassification(MaskedBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MaskedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        threshold=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            threshold (:obj:`float`):
                Threshold value (see :class:`~emmental.MaskedLinear`).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification (or regression if config.num_labels==1) loss.
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            threshold=threshold,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Masked Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    MASKED_BERT_START_DOCSTRING,
)
class MaskedBertForMultipleChoice(MaskedBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = MaskedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        threshold=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the multiple choice classification loss.
                Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
                of the input tensors. (see `input_ids` above)
            threshold (:obj:`float`):
                Threshold value (see :class:`~emmental.MaskedLinear`).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
                Classification loss.
            classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
                `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

                Classification scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

        """
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            threshold=threshold,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Masked Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    MASKED_BERT_START_DOCSTRING,
)
class MaskedBertForTokenClassification(MaskedBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MaskedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        threshold=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Labels for computing the token classification loss.
                Indices should be in ``[0, ..., config.num_labels - 1]``.
            threshold (:obj:`float`):
                Threshold value (see :class:`~emmental.MaskedLinear`).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
                Classification loss.
            scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
                Classification scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            threshold=threshold,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings(
    """Masked Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    MASKED_BERT_START_DOCSTRING,
)
class MaskedBertForQuestionAnswering(MaskedBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MaskedBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(MASKED_BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        threshold=None,
    ):
        r"""
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`).
                Position outside of the sequence are not taken into account for computing the loss.
            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`).
                Position outside of the sequence are not taken into account for computing the loss.
            threshold (:obj:`float`):
                Threshold value (see :class:`~emmental.MaskedLinear`).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~emmental.MaskedBertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
                Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-start scores (before SoftMax).
            end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-end scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            threshold=threshold,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (
            start_logits,
            end_logits,
        ) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
