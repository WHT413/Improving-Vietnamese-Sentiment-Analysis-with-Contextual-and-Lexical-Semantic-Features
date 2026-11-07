"""VnCoreNLP / RDRSegmenter adapter for the preprocessing pipeline."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable, Protocol

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "tools" / "vncorenlp"
REQUIRED_MODEL_FILES = (
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
)
_WHITESPACE_RE = re.compile(r"\s")


class _Annotator(Protocol):
    def word_segment(self, text: str) -> list[str]:
        ...


AnnotatorFactory = Callable[[Path, str], _Annotator]


def _default_model_dir() -> Path:
    configured = os.environ.get("VNCORENLP_DIR")
    return Path(configured) if configured else DEFAULT_MODEL_DIR


def _has_whitespace(path: Path) -> bool:
    return _WHITESPACE_RE.search(str(path)) is not None


def _cache_dir() -> Path:
    configured = os.environ.get("VNCORENLP_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "vnsentiment" / "vncorenlp-1.2"
    return Path.home() / ".cache" / "vnsentiment" / "vncorenlp-1.2"


def _validate_model_files(model_dir: Path) -> None:
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"VnCoreNLP model files missing under {model_dir}: {', '.join(missing)}"
        )


def _sync_model_files(source_dir: Path, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_MODEL_FILES:
        source = source_dir / relative
        destination = cache_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            shutil.copy2(source, destination)


def resolve_runtime_model_dir(model_dir: str | Path) -> Path:
    """Return a model directory Java can load without URL-encoding spaces."""
    resolved = Path(model_dir).expanduser().resolve()
    _validate_model_files(resolved)
    if not _has_whitespace(resolved):
        return resolved

    cache_dir = _cache_dir()
    if _has_whitespace(cache_dir):
        raise RuntimeError(
            "VnCoreNLP cache path contains whitespace; set VNCORENLP_CACHE_DIR "
            f"to a path without spaces (current: {cache_dir})"
        )
    _sync_model_files(resolved, cache_dir)
    return cache_dir


def _create_annotator(model_dir: Path, max_heap_size: str) -> _Annotator:
    import py_vncorenlp

    return py_vncorenlp.VnCoreNLP(
        annotators=["wseg"], save_dir=str(model_dir), max_heap_size=max_heap_size
    )


class VnCoreNLPSegmenter:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        annotator_factory: AnnotatorFactory | None = None,
        max_heap_size: str = "-Xmx2g",
    ):
        self._model_dir = Path(model_dir) if model_dir is not None else _default_model_dir()
        self._annotator_factory = annotator_factory or _create_annotator
        self._max_heap_size = max_heap_size
        self._annotator: _Annotator | None = None

    def segment(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self._annotator is None:
            runtime_dir = resolve_runtime_model_dir(self._model_dir)
            previous = os.getcwd()
            try:
                self._annotator = self._annotator_factory(
                    runtime_dir, self._max_heap_size
                )
            finally:
                os.chdir(previous)
        sentences = self._annotator.word_segment(text)
        return " ".join(sentences).strip()
