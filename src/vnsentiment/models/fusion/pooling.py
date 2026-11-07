"""Masked mean pooling over content subwords."""

import torch


def pool_masked_mean(fused: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
    if fused.ndim != 3:
        raise ValueError("fused must have shape [B, T, D]")
    if content_mask.shape != fused.shape[:2]:
        raise ValueError("content_mask must have shape [B, T]")
    accumulate_dtype = (
        torch.float32
        if fused.dtype in (torch.float16, torch.bfloat16)
        else fused.dtype
    )
    mask = content_mask.to(device=fused.device, dtype=accumulate_dtype)
    values = fused.to(accumulate_dtype)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    pooled = (values * mask.unsqueeze(-1)).sum(dim=1) / counts
    return pooled.to(fused.dtype)