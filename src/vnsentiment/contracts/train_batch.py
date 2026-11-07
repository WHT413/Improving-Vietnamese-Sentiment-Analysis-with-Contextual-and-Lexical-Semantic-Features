"""Batch contract for a single training step."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainBatch:
    texts: tuple[str, ...]
    labels: torch.Tensor

    @classmethod
    def from_samples(
        cls, samples: Iterable[Mapping[str, object]]
    ) -> "TrainBatch":
        rows = tuple(samples)
        if not rows:
            raise ValueError("samples must not be empty")
        return cls(
            texts=tuple(row["text"] for row in rows),
            labels=torch.tensor(
                [row["label"] for row in rows], dtype=torch.long
            ),
        )
