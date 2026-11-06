"""Baseline normalization and Report 3 regex elongation collapse."""

import re
import unicodedata

_ELONGATION_RE = re.compile(r"([^\W\d_])\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_basic(text: str) -> str:
    """Apply NFC, lowercase, and whitespace collapse without changing word forms."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    composed = unicodedata.normalize("NFC", text).lower()
    return _WHITESPACE_RE.sub(" ", composed).strip()


def collapse_elongations(text: str) -> str:
    """Collapse letter runs of three or more, keeping digits and punctuation."""
    return _ELONGATION_RE.sub(
        lambda match: match.group(1) if match.group(1).isalpha() else match.group(0),
        text,
    )


def normalize_text(text: str) -> str:
    """Run the required order: NFC/lowercase/whitespace, then elongation collapse."""
    return collapse_elongations(normalize_basic(text))
