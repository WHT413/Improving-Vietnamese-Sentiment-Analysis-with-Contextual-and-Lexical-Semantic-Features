"""Aggregate token-level lexicon features into a sentence lexical profile."""

import torch

from vnsentiment.contracts.lexical_batch import LexicalBatch


def mean_profile(
    features: LexicalBatch,
    content_mask: torch.Tensor,
    word_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if content_mask.shape != features.values.shape[:2]:
        raise ValueError("content_mask must have shape [B, T]")
    mask = features.mask * content_mask.to(features.mask.dtype).unsqueeze(-1)
    if word_ids is not None:
        if word_ids.shape != content_mask.shape:
            raise ValueError("word_ids must have shape [B, T]")
        mask = mask * _first_subword_mask(word_ids).unsqueeze(-1)
    counts = mask.sum(dim=1)
    means = (features.values * mask).sum(dim=1) / counts.clamp(min=1.0)
    return means, counts


def _first_subword_mask(word_ids: torch.Tensor) -> torch.Tensor:
    keep = torch.zeros_like(word_ids, dtype=torch.bool)
    for row, indices in enumerate(word_ids.tolist()):
        seen: set[int] = set()
        for position, word_index in enumerate(indices):
            if word_index >= 0 and word_index not in seen:
                keep[row, position] = True
                seen.add(word_index)
    return keep