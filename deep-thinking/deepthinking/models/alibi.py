import torch
import torch.nn.functional as F
import math

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
    def __init__(self, d_model, num_heads, dropout=0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = math.sqrt(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("m", get_alibi_slope(self.num_heads))
        self.kqv = torch.nn.Linear(d_model, 3 * d_model, bias=False)

    def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor, need_weights=False) -> torch.tensor:
        assert q.shape == k.shape == v.shape, "q, k, v must have the same shape and should be the SAME tensor" # We accept 3 tensors to keep APi consistent with torch.nn.MultiHeadAttention
        assert need_weights == False, "We do not support returning attention weights"
        x = q # Use any, q, k, v

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

        return (out)