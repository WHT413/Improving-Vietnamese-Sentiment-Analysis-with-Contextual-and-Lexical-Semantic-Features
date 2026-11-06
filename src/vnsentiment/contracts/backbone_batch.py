from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BackboneBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor
    word_ids: torch.Tensor
