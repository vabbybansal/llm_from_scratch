import torch

class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.c_fc = torch.nn.Linear(emb_dim, 4 * emb_dim)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))