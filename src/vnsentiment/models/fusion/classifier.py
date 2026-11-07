"""Linear classifier over pooled sentence vectors."""

import torch
from torch import nn


class SentimentClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 768, num_labels: int = 3):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled)