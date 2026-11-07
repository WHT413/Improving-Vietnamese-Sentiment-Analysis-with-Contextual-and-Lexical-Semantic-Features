"""Load the frozen composite lexicon and rescale polarity at load time."""

import json
from pathlib import Path

_FEATURES = (
    "polarity",
    "valence",
    "arousal",
    "dominance",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "trust",
    "anticipation",
)
_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "lexicons"
    / "composite_12d"
    / "v0.2.0"
    / "composite_lexicon.json"
)


class LexiconStore:
    def __init__(self, payload: dict):
        features = payload.get("features")
        if tuple(features or ()) != _FEATURES:
            raise ValueError("composite features do not match the 12-dim contract")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            raise ValueError("composite entries must be an object")
        self._entries = {
            key: self._prepare_entry(key, entry)
            for key, entry in raw_entries.items()
        }

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LexiconStore":
        source = Path(path) if path is not None else _DEFAULT_PATH
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(payload)

    @property
    def features(self) -> tuple[str, ...]:
        return _FEATURES

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, word: str) -> tuple[tuple[float, ...], tuple[int, ...]] | None:
        return self._entries.get(word)

    @staticmethod
    def _prepare_entry(
        key: str, entry: dict
    ) -> tuple[tuple[float, ...], tuple[int, ...]]:
        if not isinstance(key, str) or not key or " " in key:
            raise ValueError(f"invalid lexicon key: {key!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"entry {key!r} must be an object")
        values = entry.get("values")
        mask = entry.get("mask")
        if not isinstance(values, (list, tuple)) or len(values) != len(_FEATURES):
            raise ValueError(f"entry {key!r} must have 12 values")
        if not isinstance(mask, (list, tuple)) or len(mask) != len(_FEATURES):
            raise ValueError(f"entry {key!r} must have 12 mask flags")

        rescaled: list[float] = []
        for index, (value, present) in enumerate(zip(values, mask)):
            rescaled.append(
                LexiconStore._prepare_value(key, index, value, present)
            )
        return tuple(rescaled), tuple(int(flag) for flag in mask)

    @staticmethod
    def _prepare_value(key: str, index: int, value, present) -> float:
        if present not in (0, 1):
            raise ValueError(f"entry {key!r} dim {index} has invalid mask {present!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"entry {key!r} dim {index} is not numeric")
        if present == 0:
            if value != 0:
                raise ValueError(
                    f"entry {key!r} dim {index} is masked but nonzero: {value!r}"
                )
            return 0.0
        if index == 0:
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"entry {key!r} polarity outside [-1, 1]: {value!r}"
                )
            return (float(value) + 1.0) / 2.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"entry {key!r} dim {index} outside [0, 1]: {value!r}"
            )
        if index >= 4 and value not in (0.0, 1.0):
            raise ValueError(
                f"entry {key!r} dim {index} is not binary: {value!r}"
            )
        return float(value)