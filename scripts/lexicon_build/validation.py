"""Strict schema, range, mask, and spot-check validation."""

from __future__ import annotations

import unicodedata
from typing import Any

from .config import FEATURES


def validate_composite(payload: dict[str, Any]) -> None:
    assert isinstance(payload.get("source"), dict), "Missing 'source' metadata dict"
    assert tuple(payload.get("features", ())) == FEATURES, "Features tuple mismatch"
    entries = payload.get("entries")
    assert isinstance(entries, dict) and len(entries) > 0, "Empty or invalid 'entries'"

    for key, item in entries.items():
        _validate_key(key)
        _validate_entry(key, item)

    _validate_spot_checks(entries)


def _validate_key(key: str) -> None:
    assert isinstance(key, str) and key, "Empty key found"
    assert " " not in key, f"Multi-token key with space found: {key!r}"
    assert not key.startswith("_") and not key.endswith("_"), (
        f"Key has leading/trailing underscore: {key!r}"
    )
    assert unicodedata.is_normalized("NFC", key), f"Key not NFC normalized: {key!r}"
    assert key == key.lower(), f"Key not lowercase: {key!r}"


def _validate_entry(key: str, item: dict[str, list[Any]]) -> None:
    values = item["values"]
    mask = item["mask"]
    assert len(values) == 12 and len(mask) == 12, f"Invalid length in {key!r}"
    assert any(flag == 1 for flag in mask), f"All-zero mask entry found: {key!r}"

    for index in range(12):
        flag = mask[index]
        value = values[index]
        assert flag in (0, 1), f"Invalid mask {flag} at dim {index} for {key!r}"
        if flag == 0:
            assert value == 0.0, (
                f"Masked value must be 0.0 at dim {index} for {key!r}, got {value}"
            )
        elif index == 0:
            assert -1.0 <= value <= 1.0, f"Polarity out of [-1, 1] for {key!r}: {value}"
        elif 1 <= index <= 3:
            assert 0.0 <= value <= 1.0, f"VAD out of [0, 1] at dim {index} for {key!r}: {value}"
        else:
            assert value in (0.0, 1.0), (
                f"Emotion not in {{0, 1}} at dim {index} for {key!r}: {value}"
            )


def _validate_spot_checks(entries: dict[str, dict[str, list[Any]]]) -> None:
    assert "an_ủi" in entries, "Spot-check failed: 'an_ủi' missing"
    assert entries["an_ủi"]["values"][0] == 0.625 and entries["an_ủi"]["mask"][0] == 1, (
        f"Spot-check failed for 'an_ủi': {entries['an_ủi']}"
    )

    assert "tốt" in entries, "Spot-check failed: 'tốt' missing"
    assert entries["tốt"]["values"][0] == 0.875 and entries["tốt"]["mask"][0] == 1, (
        f"Spot-check failed for 'tốt' polarity: {entries['tốt']}"
    )
    assert all(flag == 1 for flag in entries["tốt"]["mask"][4:]), (
        f"Spot-check failed for 'tốt' emotions: {entries['tốt']}"
    )

    assert "này" in entries, "Spot-check failed: 'này' missing"
    assert entries["này"]["values"][4:] == [0.0] * 8 and entries["này"]["mask"][4:] == [1] * 8, (
        f"Spot-check failed for 'này': {entries['này']}"
    )

    assert "xyz_oov_test" not in entries, "Spot-check failed: OOV token present in entries"