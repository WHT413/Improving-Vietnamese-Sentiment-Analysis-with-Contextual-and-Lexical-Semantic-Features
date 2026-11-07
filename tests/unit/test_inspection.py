import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

from vnsentiment.phobert_branch.batch import build_batch
from vnsentiment.inspection import show_sample
from tests.helpers.fake_tokenizer import FakeTokenizer

PIECES = {
    "sản_phẩm": [10, 11],
    "này": [12],
    "không": [13, 14],
    "tệ": [15],
}


class FakeMatcher:
    def replace(self, text):
        return text


class FakePipeline:
    def process(self, sample):
        return SimpleNamespace(normalized_text="sản_phẩm này không tệ")


class FakePhoBertBranch:
    def __init__(self):
        self.tokenizer = FakeTokenizer(PIECES)

    def encode(self, texts):
        batch = build_batch(self.tokenizer, texts)
        return batch, torch.ones(1, batch.input_ids.shape[1], 4)


class FakeLexicon:
    def __init__(self):
        self.texts = []

    def print_report(self, text):
        self.texts.append(text)
        print(f"LEXICON {text}")


class ShowSampleTests(unittest.TestCase):
    def test_prints_preprocessing_lexicon_and_hidden(self):
        lexicon = FakeLexicon()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            show_sample(
                "sản phẩm này không tệ",
                FakeMatcher(),
                FakePipeline(),
                FakePhoBertBranch(),
                lexicon,
            )

        output = buffer.getvalue()
        self.assertIn("Raw       : sản phẩm này không tệ", output)
        self.assertIn("Segmented : sản_phẩm này không tệ", output)
        self.assertIn("LEXICON sản_phẩm này không tệ", output)
        self.assertEqual(lexicon.texts, ["sản_phẩm này không tệ"])
        self.assertIn("H          : shape=(1, 8, 4)", output)
        self.assertIn("pooled     :", output)

    def test_without_optional_branches_still_prints_preprocessing(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            show_sample("sản phẩm", FakeMatcher(), FakePipeline())

        output = buffer.getvalue()
        self.assertIn("Raw       : sản phẩm", output)
        self.assertNotIn("H          :", output)
        self.assertNotIn("LEXICON", output)


if __name__ == "__main__":
    unittest.main()