from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LexicalBatch:
    values: torch.Tensor
    mask: torch.Tensor
    presence: torch.Tensor