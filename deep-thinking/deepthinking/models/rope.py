import torch
from rotary_embedding_torch import RotaryEmbedding

class RoPE_MHA(torch.nn.Module):
    def __init__(self, d_model, num_heads, bias:bool = True, batch_first:bool = True, dropout:float = 0.0) -> None:
        super().__init__()
        assert batch_first == True, "We only support batch_first = True"

        inner_dim = d_model * num_heads
        self.heads = num_heads
        self.scale = d_model ** -0.5

        self.to_qkv = torch.nn.Linear(d_model, inner_dim * 3, bias=bias)
        self.to_out = torch.nn.Linear(inner_dim, d_model)

        self.dropout = torch.nn.Dropout(dropout)

        self.rotary_emb = RotaryEmbedding(dim = d_model)

    def forward(self, q, k, v, need_weights=True):
        assert q.shape == k.shape == v.shape, "q, k, v must have the same shape and should be the SAME tensor" # We accept 3 tensors to keep APi consistent with torch.nn.MultiHeadAttention
        assert need_weights == False, "We do not support returning attention weights"

        b, n, _, h = *q.shape, self.heads
        qkv = self.to_qkv(q).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)
        return (out,)