"""End-to-end lexicon branch: frozen lookup, subword alignment, projection."""

from collections.abc import Sequence

import torch
from torch import nn

from vnsentiment.contracts.backbone_batch import BackboneBatch
from vnsentiment.contracts.lexical_batch import LexicalBatch
from vnsentiment.lexicon.features import build_lexical_batch
from vnsentiment.lexicon.store import LexiconStore

from .projection import LexiconProjection


class LexiconBranch(nn.Module):
    def __init__(
        self,
        store: LexiconStore,
        projection: nn.Module | None = None,
    ):
        super().__init__()
        self.store = store
        self.projection = (
            projection if projection is not None else LexiconProjection()
        )

    def features(
        self, texts: Sequence[str], batch: BackboneBatch
    ) -> LexicalBatch:
        return build_lexical_batch(self.store, texts, batch)

    def forward(
        self, texts: Sequence[str], batch: BackboneBatch
    ) -> torch.Tensor:
        features = self.features(texts, batch)
        device, dtype = _projection_placement(self.projection, features)
        return self.projection(
            features.values.to(device=device, dtype=dtype),
            features.mask.to(device=device, dtype=dtype),
            features.presence.to(device=device, dtype=dtype),
        )


def _projection_placement(projection, features) -> tuple[torch.device, torch.dtype]:
    parameters = list(projection.parameters())
    buffers = list(projection.buffers())
    if parameters:
        device = parameters[0].device
    elif buffers:
        device = buffers[0].device
    else:
        device = features.values.device
    floating = next(
        (
            tensor.dtype
            for tensor in parameters + buffers
            if tensor.is_floating_point()
        ),
        None,
    )
    return device, floating or features.values.dtype