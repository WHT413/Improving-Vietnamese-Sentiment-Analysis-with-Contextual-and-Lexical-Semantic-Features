"""Learnable gate that modulates lexical contributions per position."""

import torch
from torch import nn


class LearnableGate(nn.Module):
    def __init__(self, hidden_dim: int = 768, bias_init: float = -1.0):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.proj.bias, bias_init)

    def forward(self, context: torch.Tensor, lexical: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(torch.cat((context, lexical), dim=-1)))
