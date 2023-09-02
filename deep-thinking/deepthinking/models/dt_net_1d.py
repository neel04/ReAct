""" dt_net_1d.py
    DeepThinking 1D convolutional neural network.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""
from deepthinking.utils.corrupter import Adversarial_Perturbation

import web_pdb as pdb
import torch
import math
import torch.nn.functional as F

from torch import nn
from typing import List, Tuple, Any, Optional

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R091
class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def next_power(x: int):
    i = 1
    while i < x: i *= 2 # power of 2 less than x
    return i

class LiteAttention(nn.Module):
    '''
    Compute a data dependent vector of attention weights
    which is hadamard multiplied with the actual input to produce a weighted output. 
    Use Softmax to normalize the attention weights.
    '''
    def __init__(self, dim: int, s_dim: int = 1):
        super().__init__()
        self.attention_weights = nn.Linear(dim, dim, bias=True)
        self.gate = nn.Softmax(dim=s_dim)
    
    def forward(self, x):
        attention_weights = self.gate(self.attention_weights(x))
        return x * attention_weights

class AttentionBlock(nn.Module):
    """Basic MHSA residual block class for DeepThinking """
    
    def __init__(self, drop_rate:float, width: int):
        super().__init__()
        self.activation = NewGELU()
        self.input_dim = width

        self.attn_gate = LiteAttention(self.input_dim, s_dim=1)
        #self.attn_gate = torch.nn.MultiheadAttention(self.input_dim, next_power(self.input_dim // 64), bias=True, dropout=drop_rate) # width // 2 = bottleneck
        self.ln1 = nn.LayerNorm(self.input_dim)
        self.ln2 = nn.LayerNorm(self.input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            self.activation,
            nn.Linear(self.input_dim, self.input_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attn_gate(x)
        #x = x + self.attn_gate(x,x,x, need_weights=False)[0]
        x = self.ln2(x)
        x = x + self.mlp(x)

        return self.activation(x)

class RecurrentModule(nn.Module):
    """Main Module for holding recurrent modules"""

    def __init__(self, num_blocks: int, drop_rate: float, width: int, bottleneck: int, recall: bool = True):
        super(RecurrentModule, self).__init__()
        self.recall: bool = recall
        # Define the layers
        self.gelu = NewGELU()
        self.reshape_layer = nn.Linear(width, bottleneck) # downsampling layer

        self.attention_blocks = nn.Sequential(*[
            AttentionBlock(drop_rate, width)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor):
        # Add attention blocks
        #for attention_block in self.attention_blocks:
        x = self.attention_blocks(x)
        
        # Handling the recurrence by downsampling
        if self.recall:
            x = self.gelu(self.reshape_layer(x))

        return x

class OutputModule(nn.Module):
    def __init__(self, bottleneck: int, tgt_vocab_size: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(bottleneck, tgt_vocab_size))
    
    def forward(self, x: torch.Tensor):
        # x -> (batch_size, SEQLEN, bottleneck) | target -> (batch_size, SEQLEN, tgt_vocab_size)
        x = self.head(x) # x -> (batch_size, SEQLEN, tgt_vocab_size)
        return x

class DTNet1D(nn.Module):
    """DeepThinking 1D Network model class"""

    def __init__(self, num_blocks, width, recall, **kwargs):
        super().__init__()
        self.recall: bool = recall
        self.width = int(width) # width of the network layers
        self.bottleneck = width // 2 # bottleneck width

        self.vocab_size = 9
        self.tgt_vocab_size = 3 # usually equal to vocab_size
        self.SEQLEN = 32 # length of the input sequence
        self.drop_rate = 0.2 # dropout rate

        self.embed_layer = nn.Embedding(self.vocab_size, self.bottleneck, padding_idx=8) # embedding layer for the input sequence
        self.reshape_head = nn.Sequential(
            nn.Linear(self.bottleneck, self.bottleneck),
            NewGELU())

        self.recur_block = RecurrentModule(num_blocks, self.drop_rate, self.width, self.bottleneck) # Main recurrent block

        self.out_head = OutputModule(self.bottleneck, self.tgt_vocab_size) # Output module
        self.pos_enc = self.positional_encoding(self.bottleneck).to(torch.device('cuda:0'))
        
        self.perturber = Adversarial_Perturbation(out_head) # Perturber module
    
    @torch.no_grad()
    def positional_encoding(self, d_model):
        '''
        Generates the positional encoding for the input sequence
        of shape (batch_size, max_seq_len, d_model) which would be added
        to the sequence embeddings.
        '''
        position = torch.arange(self.SEQLEN, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(self.SEQLEN, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x, iters_to_do, interim_thought: Optional[torch.Tensor] = None, peturb_iters: list = [], **kwargs):
        # x is shape: (batch, SEQLEN)
        x = self.embed_layer(x) + self.pos_enc.to(x.device) # (batch, SEQLEN, bottleneck)
        initial_thought = self.reshape_head(x) # (batch, SEQLEN, bottleneck)

        if interim_thought is None:
            interim_thought = initial_thought

        # Only used during testing
        all_outputs = torch.zeros((x.size(0), iters_to_do, self.SEQLEN, self.tgt_vocab_size), device=x.device) if not self.training else None

        # Funneling all branches into a single, recursive set of blocks repeated `iters_to_do` times
        for i in range(iters_to_do):
            if self.recall:
                x = x.unsqueeze(-1) if x.dim() == 2 else x # (batch, 16) -> (batch, 16, 1) if needed
                # interim_thought is shape: (batch, bottleneck // 2) | x is shape: (batch, SEQLEN)
                interim_thought = torch.cat([interim_thought, x], 2) # (batch, SEQLEN, bottleneck // 2 + SEQLEN)

            interim_thought = self.recur_block(interim_thought) # the recursive block, bulk of the network | (batch, SEQLEN, bottleneck)

            if not self.training:
                # During testing, we need out for every iteration to append to all_outputs
                out = self.out_head(interim_thought) # (batch, SEQLEN, tgt_vocab_size)
                all_outputs[:, i] = out # storing intermediate computations for each iteration
            
            if i in perturb_iters:
                interim_thought, NUM_ERRORS = self.perturber.perturb(interim_thought)

        if self.training:
            # During training, we only need output when all the iterations are done, saving compute
            out = self.out_head(interim_thought) # (batch, SEQLEN, tgt_vocab_size)
            return out, interim_thought, NUM_ERRORS

        return all_outputs

def dt_net_1d(width, **kwargs):
    return DTNet1D(6, width, recall=False)


def dt_net_recall_1d(width, **kwargs):
    return DTNet1D(6, width, recall=True)


def dt_net_gn_1d(width, **kwargs):
    return DTNet1D(6, width, recall=False)


def dt_net_recall_gn_1d(width, **kwargs):
    return DTNet1D(6, width, recall=True)
