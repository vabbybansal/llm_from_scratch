import torch

from llm_from_scratch.libs.transformer_block import TransformerBlock

class GPT(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers,
        d_in,
        context_length,
        dropout,
        n_heads,
        qkv_bias=False,
        tie_weights=True,
    ):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_in)
        self.pos_emb = torch.nn.Embedding(context_length, d_in)
        self.drop_emb = torch.nn.Dropout(dropout)

        self.trf_blocks = torch.nn.Sequential(
            *[
                TransformerBlock(
                    d_in, d_in, context_length, dropout, n_heads, qkv_bias=qkv_bias
                )
                for _ in range(n_layers)
            ]
        )

        self.ln_f = torch.nn.LayerNorm(d_in)
        self.lm_head = torch.nn.Linear(d_in, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # Pytorch saves (out_features, in_features)


    def forward(self, x):

        _, seq_len = x.shape
        token_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits