import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vnsentiment.training import data


def _write_split(directory, split, rows):
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path = Path(directory) / f"{split}.jsonl"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(directory, split, sha256):
    manifest = {"splits": {split: {"output_sha256": sha256}}}
    (Path(directory) / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class LoadSplitTests(unittest.TestCase):
    def test_real_splits_have_expected_rows(self):
        self.assertEqual(len(data.load_split("train")), 11426)
        self.assertEqual(len(data.load_split("dev")), 1583)
        self.assertEqual(len(data.load_split("test", allow_test=True)), 3166)

    def test_test_split_is_locked_by_default(self):
        with self.assertRaises(ValueError):
            data.load_split("test")

    def test_unknown_split_raises(self):
        with self.assertRaises(ValueError):
            data.load_split("validation")

    def test_sha_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_split(directory, "train", [{"text": "tốt", "label": 2}])
            _write_manifest(directory, "train", "0" * 64)
            with mock.patch.object(data, "_DATA_DIR", Path(directory)):
                with self.assertRaises(ValueError):
                    data.load_split("train")

    def test_label_outside_range_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            sha = _write_split(directory, "train", [{"text": "tốt", "label": 3}])
            _write_manifest(directory, "train", sha)
            with mock.patch.object(data, "_DATA_DIR", Path(directory)):
                with self.assertRaises(ValueError):
                    data.load_split("train")

    def test_empty_text_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            sha = _write_split(directory, "train", [{"text": "  ", "label": 2}])
            _write_manifest(directory, "train", sha)
            with mock.patch.object(data, "_DATA_DIR", Path(directory)):
                with self.assertRaises(ValueError):
                    data.load_split("train")


if __name__ == "__main__":
    unittest.main()
