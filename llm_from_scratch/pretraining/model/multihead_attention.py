import torch
import math

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads=2, qkv_bias=False):

        super().__init__()

        self.n_heads = n_heads
        self.d_out = d_out

        assert (d_out % n_heads == 0), f"d_out {d_out} should be a multiple of n_heads {n_heads}"
        self.head_dim = d_out // n_heads

        self.c_attn = torch.nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.c_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, X):
        b, num_tokens, d_in = X.shape  # (b,num_tokens,d_in)
        
        Q, K, V = self.c_attn(X).split(self.d_out, dim=-1)  # each (b,num_tokens,d_out)

        # Split the weight matrices by n_heads dimension. (b,num_tokens,d_out) => (b,num_tokens,n_heads,head_dim)
        Q = Q.view(b, num_tokens, self.n_heads, self.head_dim)  # (b,num_tokens,n_heads,head_dim)
        K = K.view(b, num_tokens, self.n_heads, self.head_dim)
        V = V.view(b, num_tokens, self.n_heads, self.head_dim)

        # Transpose
        Q = Q.transpose(1,2)  # (b,n_heads,num_tokens,head_dim)
        K = K.transpose(1,2)  # (b,n_heads,num_tokens,head_dim)
        V = V.transpose(1,2)  # (b,n_heads,num_tokens,head_dim)


        S = (Q @ K.transpose(-2, -1)) /  math.sqrt(Q.shape[-1])  # (b,n_heads,num_tokens,head_dim) @ (b,n_heads,head_dim,num_tokens) => (b,n_heads,num_tokens,num_tokens)
        S = S.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        A = torch.softmax(S, dim=-1)
        A = self.dropout(A)
        C = A @ V  # (b,n_heads,num_tokens,num_tokens) @ (b,n_heads,num_tokens,head_dim) => (b,n_heads,num_tokens,head_dim)
        C = C.transpose(1,2)  # (b,num_tokens,n_heads,head_dim)
        C = C.contiguous().view(b, num_tokens, self.d_out)
        C = self.c_proj(C)

        return C
