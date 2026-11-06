"""Approved variant-map loading and boundary-safe single-pass replacement."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from .collapse import normalize_basic

_DEFAULT_METADATA = "unknown"


def _reject_duplicate_raw_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def _normalize_entries(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_key, raw_target in pairs:
        if not isinstance(raw_key, str) or not isinstance(raw_target, str):
            raise ValueError("variant keys and targets must be strings")
        key = normalize_basic(raw_key)
        target = normalize_basic(raw_target)
        if not key:
            raise ValueError("variant key is empty after normalization")
        if not target:
            raise ValueError(f"variant target is empty after normalization: {key!r}")
        if key in entries:
            raise ValueError(f"duplicate variant key after normalization: {key!r}")
        entries[key] = target
    return entries


def _metadata(document: Mapping[str, object], field: str) -> str:
    value = document.get(field, _DEFAULT_METADATA)
    if not isinstance(value, str):
        raise ValueError(f"variant map field {field!r} must be a string")
    return value


@dataclass(frozen=True)
class VariantMap:
    version: str
    source: str
    review_status: str
    entries: Mapping[str, str]

    @classmethod
    def from_entries(
        cls,
        entries: Mapping[str, str],
        *,
        version: str = _DEFAULT_METADATA,
        source: str = _DEFAULT_METADATA,
        review_status: str = _DEFAULT_METADATA,
    ) -> "VariantMap":
        normalized = _normalize_entries(entries.items())
        return cls(
            version=version,
            source=source,
            review_status=review_status,
            entries=MappingProxyType(normalized),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VariantMap":
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle, object_pairs_hook=_reject_duplicate_raw_keys)
        if not isinstance(document, dict):
            raise ValueError("variant map root must be a JSON object")

        entries_node = document.get("entries")
        if isinstance(entries_node, dict):
            pairs = entries_node.items()
        elif isinstance(entries_node, list):
            pairs = []
            for index, item in enumerate(entries_node):
                if not isinstance(item, dict) or "key" not in item or "target" not in item:
                    raise ValueError(f"entries[{index}] must contain 'key' and 'target'")
                pairs.append((item["key"], item["target"]))
        else:
            raise ValueError("variant map must contain an 'entries' object or list")

        entries = _normalize_entries(pairs)
        entry_count = document.get("entry_count")
        if entry_count is not None and entry_count != len(entries):
            raise ValueError(
                f"entry_count={entry_count!r} does not match {len(entries)} entries"
            )
        return cls(
            version=_metadata(document, "version"),
            source=_metadata(document, "source"),
            review_status=_metadata(document, "review_status"),
            entries=MappingProxyType(entries),
        )


class VariantMatcher:
    """Replace approved variants using Unicode word boundaries, longest match first."""

    def __init__(self, variant_map: VariantMap):
        self._entries = dict(variant_map.entries)
        if self._entries:
            keys = sorted(
                self._entries, key=lambda key: (-len(key.split()), -len(key), key)
            )
            alternation = "|".join(re.escape(key) for key in keys)
            self._pattern = re.compile(rf"(?<!\w)(?:{alternation})(?!\w)")
        else:
            self._pattern = None

    def replace(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda match: self._entries[match.group(0)], text)
