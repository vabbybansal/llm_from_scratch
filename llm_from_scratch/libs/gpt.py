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
        self.token_emb = torch.nn.Embedding(vocab_size, d_in)
        self.pos_emb = torch.nn.Embedding(context_length, d_in)
        self.drop_emb = torch.nn.Dropout(dropout)

        self.transformer_blocks = torch.nn.Sequential(
            *[
                TransformerBlock(
                    d_in, d_in, context_length, dropout, n_heads, qkv_bias=qkv_bias
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = torch.nn.LayerNorm(d_in)
        self.out_head = torch.nn.Linear(d_in, vocab_size, bias=False)
        if tie_weights:
            self.out_head.weight = self.token_emb.weight  # Pytorch saves (out_features, in_features)


    def forward(self, x):

        _, seq_len = x.shape
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits