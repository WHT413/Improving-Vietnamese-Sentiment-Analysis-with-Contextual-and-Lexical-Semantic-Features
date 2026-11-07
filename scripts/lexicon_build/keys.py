"""Canonical key normalization shared by all lexicon sources."""

from vnsentiment.preprocessing.collapse import normalize_basic
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter


def canonicalize_key(raw: str, segmenter: VnCoreNLPSegmenter) -> str:
    text = str(raw).replace("_", " ")
    text = normalize_basic(text)
    if not text:
        return ""
    segmented = segmenter.segment(text).strip()
    tokens = [token.strip("_") for token in segmented.split() if token.strip("_")]
    return " ".join(tokens)