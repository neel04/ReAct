import torch.nn as nn
import torch.nn.functional as F
import torch

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, bias: bool = True, batch_first: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.dropout_p = dropout

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_linear = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, need_weights: bool = False):
        assert need_weights == False, "We do not support returning attention weights"

        if self.batch_first:
            q, k, v = [x.transpose(0, 1) for x in (q, k, v)]  # Convert to (seq_len, batch, d_model)

        batch_size = q.size(1)
        q = self.q_linear(q).view(-1, batch_size, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        k = self.k_linear(k).view(-1, batch_size, self.nhead, self.d_model // self.nhead).transpose(1, 2)
        v = self.v_linear(v).view(-1, batch_size, self.nhead, self.d_model // self.nhead).transpose(1, 2)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)

        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, batch_size, self.d_model)
        attn_output = self.out_linear(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # Convert back to (batch, seq_len, d_model)

        return (attn_output, )