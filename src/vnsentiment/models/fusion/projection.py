"""Project 12-dimensional lexicon values to the PhoBERT hidden size."""

import torch
from torch import nn


class LexiconProjection(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 768):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mask_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.to(values.dtype)
        presence_f = presence.to(values.dtype)
        projected = (
            self.value_proj(values * mask_f)
            + self.mask_proj(mask_f)
            + self.bias
        )
        return projected * presence_f
