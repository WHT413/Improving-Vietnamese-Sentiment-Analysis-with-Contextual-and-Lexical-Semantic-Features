import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vnsentiment.preprocessing.segmenter import (
    DEFAULT_MODEL_DIR,
    VnCoreNLPSegmenter,
    resolve_runtime_model_dir,
)

REQUIRED_FILES = (
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
)


def create_model_dir(root: Path, name: str = "vncorenlp") -> Path:
    model_dir = root / name
    for relative in REQUIRED_FILES:
        path = model_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dummy model data")
    return model_dir


class FakeAnnotator:
    def __init__(self, sentences):
        self._sentences = sentences
        self.calls = []

    def word_segment(self, text):
        self.calls.append(text)
        return list(self._sentences)


class VnCoreNLPSegmenterTests(unittest.TestCase):
    def test_segment_joins_sentences_with_space(self):
        annotator = FakeAnnotator(["sản_phẩm này", "không tệ"])
        segmenter = VnCoreNLPSegmenter(annotator_factory=lambda _dir, _heap: annotator)
        self.assertEqual(segmenter.segment("sản phẩm này không tệ"), "sản_phẩm này không tệ")
        self.assertEqual(annotator.calls, ["sản phẩm này không tệ"])

    def test_annotator_is_created_lazily_once(self):
        calls = []

        def factory(_dir, _heap):
            calls.append("init")
            return FakeAnnotator(["ok"])

        segmenter = VnCoreNLPSegmenter(annotator_factory=factory)
        self.assertEqual(calls, [])
        segmenter.segment("a")
        segmenter.segment("b")
        self.assertEqual(calls, ["init"])

    def test_segment_rejects_non_string(self):
        segmenter = VnCoreNLPSegmenter(annotator_factory=lambda _dir, _heap: FakeAnnotator([]))
        with self.assertRaises(TypeError):
            segmenter.segment(None)  # type: ignore[arg-type]

    def test_empty_sentence_list_returns_empty(self):
        segmenter = VnCoreNLPSegmenter(annotator_factory=lambda _dir, _heap: FakeAnnotator([]))
        self.assertEqual(segmenter.segment("a"), "")

    def test_default_model_dir_points_to_artifacts(self):
        self.assertEqual(DEFAULT_MODEL_DIR.name, "vncorenlp")
        self.assertEqual(DEFAULT_MODEL_DIR.parent.name, "tools")
        self.assertEqual(DEFAULT_MODEL_DIR.parent.parent.name, "artifacts")


class ResolveRuntimeModelDirTests(unittest.TestCase):
    def test_directory_without_spaces_is_used_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = create_model_dir(Path(directory))
            self.assertEqual(resolve_runtime_model_dir(model_dir), model_dir.resolve())

    def test_directory_with_spaces_is_copied_to_no_space_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = create_model_dir(root, "model dir with spaces")
            cache_dir = root / "cache"
            with patch.dict(os.environ, {"VNCORENLP_CACHE_DIR": str(cache_dir)}):
                resolved = resolve_runtime_model_dir(model_dir)
            self.assertEqual(resolved, cache_dir.resolve())
            for relative in REQUIRED_FILES:
                self.assertTrue((resolved / relative).is_file(), relative)
                self.assertEqual(
                    (resolved / relative).read_bytes(),
                    (model_dir / relative).read_bytes(),
                )

    def test_missing_model_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_runtime_model_dir(Path(directory))

    def test_cache_path_with_spaces_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = create_model_dir(root, "model dir with spaces")
            cache_dir = root / "cache dir with spaces"
            with patch.dict(os.environ, {"VNCORENLP_CACHE_DIR": str(cache_dir)}):
                with self.assertRaises(RuntimeError):
                    resolve_runtime_model_dir(model_dir)

    def test_segmenter_passes_resolved_cache_dir_to_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = create_model_dir(root, "model dir with spaces")
            cache_dir = root / "cache"
            received = []

            def factory(resolved_dir, _heap):
                received.append(Path(resolved_dir))
                return FakeAnnotator(["ok"])

            with patch.dict(os.environ, {"VNCORENLP_CACHE_DIR": str(cache_dir)}):
                segmenter = VnCoreNLPSegmenter(model_dir=model_dir, annotator_factory=factory)
                segmenter.segment("a")
            self.assertEqual(received, [cache_dir.resolve()])


if __name__ == "__main__":
    unittest.main()
