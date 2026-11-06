from collections.abc import Sequence

import torch
from torch import nn

from vnsentiment.contracts.backbone_batch import BackboneBatch

from .batch import MAX_LENGTH, build_batch
from .encoder import PhoBertEncoder


class PhoBertBackbone(nn.Module):
    def __init__(self, tokenizer, encoder: PhoBertEncoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder

    def encode(
        self, texts: Sequence[str], max_length: int = MAX_LENGTH
    ) -> tuple[BackboneBatch, torch.Tensor]:
        batch = build_batch(self.tokenizer, texts, max_length=max_length)
        return batch, self.encoder(batch)
