"""Residual fusion of contextual and gated lexical representations."""

import torch


def fuse_residual(
    context: torch.Tensor, gate: torch.Tensor, lexical: torch.Tensor
) -> torch.Tensor:
    return context + gate * lexical
