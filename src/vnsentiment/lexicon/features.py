"""Broadcast word-level lexicon lookups to PhoBERT subword positions."""

from collections.abc import Sequence

import torch

from vnsentiment.contracts.backbone_batch import BackboneBatch
from vnsentiment.contracts.lexical_batch import LexicalBatch

from .store import LexiconStore


def build_lexical_batch(
    store: LexiconStore, texts: Sequence[str], batch: BackboneBatch
) -> LexicalBatch:
    text_list = _text_list(texts)
    word_ids = _word_ids(batch, len(text_list))
    dimension = len(store.features)
    zero_values = [0.0] * dimension
    zero_mask = [0] * dimension
    rows = [
        _encode_row(store, text, row_ids, zero_values, zero_mask)
        for text, row_ids in zip(text_list, word_ids.tolist())
    ]
    return _to_batch(rows, word_ids.device)


def _text_list(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")
    text_list = list(texts)
    if not text_list:
        raise ValueError("texts must not be empty")
    return text_list


def _word_ids(batch: BackboneBatch, batch_size: int) -> torch.Tensor:
    word_ids = batch.word_ids
    if word_ids.ndim != 2:
        raise ValueError("word_ids must have shape [B, T]")
    if word_ids.shape[0] != batch_size:
        raise ValueError("texts and word_ids must have the same batch size")
    return word_ids


def _encode_row(
    store: LexiconStore,
    text: str,
    row_ids: list[int],
    zero_values: list[float],
    zero_mask: list[int],
) -> tuple[list[list[float]], list[list[int]], list[list[float]]]:
    if not isinstance(text, str):
        raise TypeError("each text must be a string")
    words = text.split()
    if not words:
        raise ValueError("text must not be empty")

    values: list[list[float]] = []
    mask: list[list[int]] = []
    presence: list[list[float]] = []
    for word_index in row_ids:
        if word_index < 0:
            values.append(zero_values)
            mask.append(zero_mask)
            presence.append([0.0])
            continue
        if word_index >= len(words):
            raise ValueError(
                f"word_ids index {word_index} outside text with {len(words)} words"
            )
        hit = store.lookup(words[word_index])
        if hit is None:
            values.append(zero_values)
            mask.append(zero_mask)
            presence.append([0.0])
            continue
        word_values, word_mask = hit
        values.append(list(word_values))
        mask.append(list(word_mask))
        presence.append([1.0 if any(word_mask) else 0.0])
    return values, mask, presence


def _to_batch(rows, device: torch.device) -> LexicalBatch:
    return LexicalBatch(
        values=torch.tensor([row[0] for row in rows], dtype=torch.float32, device=device),
        mask=torch.tensor([row[1] for row in rows], dtype=torch.float32, device=device),
        presence=torch.tensor([row[2] for row in rows], dtype=torch.float32, device=device),
    )