import torch
import torch.nn.functional as F
import math
import einops

from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class ALiBiConfig:
    num_layers: int = 1
    d_model: int = 512
    num_heads: int = 16
    max_len: int = 16
    dropout: float = 0.0
    causal: bool = False

def get_relative_positions(seq_len: int, device: torch.device) -> torch.tensor:
    x = torch.arange(seq_len)[None, :].to(device, non_blocking=True)
    y = torch.arange(seq_len)[:, None].to(device, non_blocking=True)
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

class OptimizedALiBiMultiHeadAttention(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.causal = config.causal
        self.num_heads = config.num_heads
        self.scale = math.sqrt(config.d_model)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.register_buffer("m", get_alibi_slope(self.num_heads))
        self.kqv = torch.nn.Linear(config.d_model, 3 * config.d_model, bias=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.shape

        key, query, value = self.kqv(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)

        bias = (self.m * get_relative_positions(seq_len, self.m.device)).unsqueeze(0)
        # bias.shape == (1, num_heads, seq_len, seq_len)
        score = torch.einsum('b h s d, b h d n-> b h s n', query, key) / self.scale + bias
        # score.shape == (batch_size, num_heads, seq_len, seq_len)

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.dropout(out)

        return out