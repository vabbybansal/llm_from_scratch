import torch

class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4*emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*emb_dim, emb_dim),
        )
    def forward(self, x):
        return self.layers(x)