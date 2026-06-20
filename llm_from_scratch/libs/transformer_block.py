import torch

from llm_from_scratch.libs.feedforward import FeedForward
from llm_from_scratch.libs.multihead_attention import MultiHeadAttention


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_bias=False):
        assert d_in == d_out, "d_in and d_out must match for residual connections"
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=dropout,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
        )
        self.mlp = FeedForward(emb_dim=d_in)
        self.ln_1 = torch.nn.LayerNorm(d_in)
        self.ln_2 = torch.nn.LayerNorm(d_in)
        self.drop_shortcut = torch.nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.ln_1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)

        x = x + residual

        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = self.drop_shortcut(x)
        x = x + residual

        return x