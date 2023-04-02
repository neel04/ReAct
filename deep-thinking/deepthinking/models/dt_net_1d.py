""" dt_net_1d.py
    DeepThinking 1D convolutional neural network.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""
import web_pdb as pdb
import torch
import math
import torch.nn.functional as F

from torch import nn
from .blocks import BasicBlock1D as BasicBlock
from .alibi import OptimizedALiBiMultiHeadAttention as ALiBiMHSA, VanillaALiBi
from .rope import RoPE_MHA
from .flash_mha import FlashMultiHeadAttention

# Enabling SDP backend
#torch.backends.cuda.enable_flash_sdp(enabled=True)
#print(f'\n{chr(0x26A1)*20}\nFlash Attention status: {torch.backends.cuda.flash_sdp_enabled()}\n{chr(0x26A1)*20}\n')

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class AttentionBlock1D(nn.Module):
    """Basic MHSA residual block class for DeepThinking """
    
    def __init__(self, drop_rate:int, width:int):
        super().__init__()
        self.width = width
        self.activation = NewGELU()

        self.attn_head = torch.nn.MultiheadAttention(self.width, self.width//32, bias=True, batch_first=True, dropout=0.05)
        self.linear1 = nn.Linear(self.width, self.width)

        self.ln1 = nn.LayerNorm(self.width)
        self.ln2 = nn.LayerNorm(self.width)

        self.mlp = nn.Sequential(
            nn.Linear(self.width, self.width),
            self.activation,
            nn.Linear(self.width, self.width),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attn_head(x, x, x, need_weights=False)[0]
        x = self.ln2(x)
        x = x + self.mlp(x)

        return self.activation(x)

class AttentionModule(nn.Module):
    def __init__(self, attn_head):
        super().__init__()
        self.attn_head = attn_head

    def forward(self, x):
        return self.attn_head(x,x,x, need_weights=False)[0]

class DTNet1D(nn.Module):
    """DeepThinking 1D Network model class"""

    def __init__(self, block, num_blocks, width, recall, group_norm=False, **kwargs):
        super().__init__()

        self.width = int(width) # width of the network layers
        self.bottleneck = self.width // 2 # bottleneck width
        self.recall = recall
        self.SEQLEN = 64 # length of the input sequence
        drop_rate = 0.1 # dropout rate

        self.reshape_layer = nn.Linear(self.width, self.bottleneck) # downsampling layer
        self.embed_layer = nn.Embedding(13, self.bottleneck, padding_idx=11) # embedding layer for the input sequence

        proj_linear = nn.Linear(self.bottleneck, self.bottleneck)
        head_linear = nn.Linear(self.bottleneck, 13)
        
        # Handling the recurrence 
        if self.recall:
            recur_layers = [self.reshape_layer, NewGELU()]
        else:
            recur_layers = []

        for i in range(num_blocks):
            recur_layers.insert(0, AttentionBlock1D(drop_rate, width)) # add attention blocks to the beginning of the list

        self.projection = nn.Sequential(proj_linear, NewGELU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_linear, NewGELU())

    def positional_encoding(self, max_seq_len, d_model):
        '''
        Generates the positional encoding for the input sequence
        of shape (batch_size, max_seq_len, d_model) which would be added
        to the sequence embeddings.
        '''
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        
        return pe

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        # x -> (batch, 16)
        x = self.embed_layer(x) + self.positional_encoding(self.SEQLEN, self.bottleneck).to(x.device, non_blocking=True)
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        # X -> (32, 1, 96), 32 is batch_size/#GPUs
        all_outputs = torch.zeros((x.size(0), iters_to_do, self.SEQLEN, 13)).to(x.device, non_blocking=True)

        for i in range(iters_to_do):
            if self.recall:
                x = x.unsqueeze(-1) if x.dim() == 2 else x # (batch, 16) -> (batch, 16, 1) if needed
                interim_thought = torch.cat([interim_thought, x], 2)

            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought

        return all_outputs


def dt_net_1d(width, **kwargs):
    return DTNet1D(BasicBlock, 6, width, recall=False)


def dt_net_recall_1d(width, **kwargs):
    return DTNet1D(BasicBlock, 6, width, recall=True)


def dt_net_gn_1d(width, **kwargs):
    return DTNet1D(BasicBlock, 6, width, recall=False, group_norm=True)


def dt_net_recall_gn_1d(width, **kwargs):
    return DTNet1D(BasicBlock, 6, width, recall=True, group_norm=True)
