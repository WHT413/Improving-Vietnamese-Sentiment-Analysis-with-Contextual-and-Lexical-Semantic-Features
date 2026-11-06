"""Unified raw -> normalized -> segmented preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .collapse import normalize_basic, normalize_text
from .variants import VariantMap, VariantMatcher

PREPROCESSING_VERSION = "1.0.0"
_ALLOWED_LABELS = (0, 1, 2)


class EmptyTextError(ValueError):
    """Raised when normalized input text has no content."""


class Segmenter(Protocol):
    def segment(self, text: str) -> str:
        """Return word-segmented text using underscores for compounds."""


class IdentitySegmenter:
    """Pass-through seam used until a real RDRSegmenter is wired in."""

    def segment(self, text: str) -> str:
        return text


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    text: str
    label: int | None = None


@dataclass(frozen=True)
class NormalizedSample:
    sample_id: str
    normalized_text: str  # Final pipeline output after segmentation; identity is a debug passthrough.
    label: int | None
    preprocessing_version: str
    variant_resource_version: str


def _validate_label(label: object) -> None:
    if label is None:
        return
    if isinstance(label, bool) or not isinstance(label, int) or label not in _ALLOWED_LABELS:
        raise ValueError(
            f"label must be one of {_ALLOWED_LABELS} or None, got {label!r}"
        )


class PreprocessingPipeline:
    def __init__(
        self,
        variant_map: VariantMap,
        segmenter: Segmenter | None = None,
        *,
        preprocessing_version: str = PREPROCESSING_VERSION,
        require_approved: bool = True,
    ):
        if require_approved and variant_map.review_status != "approved":
            raise ValueError(
                "variant map review_status must be 'approved' for runtime use, "
                f"got {variant_map.review_status!r}"
            )
        self._matcher = VariantMatcher(variant_map)
        self._segmenter = segmenter or IdentitySegmenter()
        self._preprocessing_version = preprocessing_version
        self._variant_resource_version = variant_map.version

    def process(self, sample: RawSample) -> NormalizedSample:
        if not isinstance(sample.text, str):
            raise TypeError("raw text must be a string")
        _validate_label(sample.label)
        normalized = normalize_text(sample.text)
        if not normalized:
            raise EmptyTextError(
                f"sample {sample.sample_id!r} is empty after normalization"
            )
        replaced = self._matcher.replace(normalized)
        segmented = self._segmenter.segment(replaced)
        if not isinstance(segmented, str):
            raise TypeError("segmenter must return a string")
        segmented = normalize_basic(segmented)
        if not segmented:
            raise EmptyTextError(
                f"sample {sample.sample_id!r} is empty after segmentation"
            )
        return NormalizedSample(
            sample_id=sample.sample_id,
            normalized_text=segmented,
            label=sample.label,
            preprocessing_version=self._preprocessing_version,
            variant_resource_version=self._variant_resource_version,
        )
