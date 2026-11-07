"""Full sentiment model: PhoBERT branch plus lexicon fusion tail."""

from collections.abc import Sequence

import torch
from torch import nn

from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.pooling import pool_masked_mean
from vnsentiment.models.fusion.residual import fuse_residual
from vnsentiment.phobert_branch.branch import PhoBertBranch

_MODES = ("fused", "contextual")


class SentimentModel(nn.Module):
    def __init__(
        self,
        phobert: PhoBertBranch,
        lexicon: LexiconBranch,
        gate: LearnableGate,
        classifier: SentimentClassifier,
    ):
        super().__init__()
        self.phobert = phobert
        self.lexicon = lexicon
        self.gate = gate
        self.classifier = classifier

    def forward(self, texts: Sequence[str], mode: str = "fused") -> torch.Tensor:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        batch, hidden = self.phobert.encode(texts)
        if mode == "contextual":
            pooled = pool_masked_mean(hidden, batch.content_mask)
        else:
            lexical = self.lexicon(texts, batch)
            gates = self.gate(hidden, lexical)
            fused = fuse_residual(hidden, gates, lexical)
            pooled = pool_masked_mean(fused, batch.content_mask)
        return self.classifier(pooled)
