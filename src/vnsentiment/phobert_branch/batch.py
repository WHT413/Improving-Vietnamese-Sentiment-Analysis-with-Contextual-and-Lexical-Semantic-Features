"""Build PhoBERT word/subword batches with masks and word ids."""

from collections.abc import Sequence

import torch

from vnsentiment.contracts.backbone_batch import BackboneBatch

MAX_LENGTH = 256
_MIN_LENGTH = 3


def _build_row(
    tokenizer, text: str, max_length: int, special_ids: set[int]
) -> tuple[list[int], list[int], list[int], list[int]]:
    if not isinstance(text, str):
        raise TypeError("each text must be a string")
    words = text.split()
    if not words:
        raise ValueError("text must not be empty")

    budget = max_length - 2
    content_ids: list[int] = []
    word_ids: list[int] = []
    for index, word in enumerate(words):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            raise ValueError(f"tokenizer returned no subwords for word {word!r}")
        if any(token_id in special_ids for token_id in ids):
            raise ValueError(f"literal special token in word {word!r}")
        if len(content_ids) + len(ids) > budget:
            break
        content_ids.extend(ids)
        word_ids.extend([index] * len(ids))

    if not content_ids:
        raise ValueError(f"no content subwords fit within max_length={max_length}")

    return (
        [tokenizer.bos_token_id] + content_ids + [tokenizer.eos_token_id],
        [1] * (len(content_ids) + 2),
        [0] + [1] * len(content_ids) + [0],
        [-1] + word_ids + [-1],
    )


def _pad_rows(rows, pad_token_id: int) -> BackboneBatch:
    width = max(len(row[0]) for row in rows)
    padded = []
    for input_ids, attention_mask, content_mask, word_ids in rows:
        pad = width - len(input_ids)
        padded.append(
            (
                input_ids + [pad_token_id] * pad,
                attention_mask + [0] * pad,
                content_mask + [0] * pad,
                word_ids + [-1] * pad,
            )
        )
    return BackboneBatch(
        input_ids=torch.tensor([row[0] for row in padded], dtype=torch.long),
        attention_mask=torch.tensor([row[1] for row in padded], dtype=torch.long),
        content_mask=torch.tensor([row[2] for row in padded], dtype=torch.long),
        word_ids=torch.tensor([row[3] for row in padded], dtype=torch.long),
    )


def build_batch(
    tokenizer, texts: Sequence[str], max_length: int = MAX_LENGTH
) -> BackboneBatch:
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise ValueError("max_length must be an integer")
    if not _MIN_LENGTH <= max_length <= MAX_LENGTH:
        raise ValueError(f"max_length must be between {_MIN_LENGTH} and {MAX_LENGTH}")

    text_list = list(texts)
    if not text_list:
        raise ValueError("texts must not be empty")

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if bos_token_id is None or eos_token_id is None or pad_token_id is None:
        raise ValueError("tokenizer must define bos, eos, and pad token ids")

    special_ids = {bos_token_id, eos_token_id, pad_token_id}
    rows = [
        _build_row(tokenizer, text, max_length, special_ids) for text in text_list
    ]
    return _pad_rows(rows, pad_token_id)
