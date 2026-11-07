import json
import unittest
from pathlib import Path

import torch

from vnsentiment.phobert_branch.batch import build_batch
from vnsentiment.contracts.backbone_batch import BackboneBatch
from vnsentiment.lexicon.features import build_lexical_batch
from vnsentiment.lexicon.store import LexiconStore
from tests.helpers.fake_tokenizer import FakeTokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_fixture.json"
PIECES = {
    "tốt": [10, 11],
    "này": [12],
    "không": [13, 14],
    "tệ": [15],
    "sản_phẩm": [16, 17],
}


class LexicalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LexiconStore(json.loads(FIXTURE.read_text(encoding="utf-8")))

    def setUp(self):
        self.tokenizer = FakeTokenizer(PIECES)

    def test_broadcasts_word_features_to_every_subword(self):
        text = "tốt này không tệ"
        batch = build_batch(self.tokenizer, [text])
        features = build_lexical_batch(self.store, [text], batch)

        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, 0, 1, 2, 2, 3, -1]])
        self.assertEqual(features.values.shape, (1, 8, 12))
        self.assertEqual(features.mask.shape, (1, 8, 12))
        self.assertEqual(features.presence.shape, (1, 8, 1))

        expected_values, expected_mask = self.store.lookup("tốt")
        expected = torch.tensor(expected_values, dtype=torch.float32)
        self.assertTrue(torch.equal(features.values[0, 1], expected))
        self.assertTrue(torch.equal(features.values[0, 2], expected))
        self.assertEqual(tuple(features.mask[0, 1].tolist()), expected_mask)

    def test_special_and_padded_positions_are_zero(self):
        text = "tốt"
        batch = build_batch(self.tokenizer, [text])
        features = build_lexical_batch(self.store, [text], batch)

        for position in (0, 3):
            self.assertTrue(torch.equal(features.values[0, position], torch.zeros(12)))
            self.assertTrue(torch.equal(features.mask[0, position], torch.zeros(12)))
            self.assertEqual(features.presence[0, position].item(), 0.0)

    def test_batch_padding_stays_zero(self):
        texts = ["tốt này", "tệ"]
        batch = build_batch(self.tokenizer, texts)
        features = build_lexical_batch(self.store, texts, batch)

        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, 0, 1, -1], [-1, 0, -1, -1, -1]])
        self.assertEqual(features.values.shape, (2, 5, 12))
        self.assertTrue(torch.equal(features.values[1, 2:], torch.zeros(3, 12)))
        self.assertTrue(torch.equal(features.presence[1, 2:], torch.zeros(3, 1)))

    def test_oov_is_all_zero_without_presence(self):
        text = "xyz_oov_test"
        tokenizer = FakeTokenizer({**PIECES, text: [99]})
        batch = build_batch(tokenizer, [text])
        features = build_lexical_batch(self.store, [text], batch)

        self.assertTrue(torch.equal(features.values, torch.zeros(1, 3, 12)))
        self.assertTrue(torch.equal(features.mask, torch.zeros(1, 3, 12)))
        self.assertTrue(torch.equal(features.presence, torch.zeros(1, 3, 1)))

    def test_true_zero_has_presence_and_mask(self):
        text = "này"
        batch = build_batch(self.tokenizer, [text])
        features = build_lexical_batch(self.store, [text], batch)

        self.assertEqual(features.presence[0, 1].item(), 1.0)
        self.assertTrue(torch.equal(features.values[0, 1], torch.zeros(12)))
        self.assertEqual(tuple(features.mask[0, 1, 4:].tolist()), (1,) * 8)

    def test_truncated_words_are_not_looked_up(self):
        tokenizer = FakeTokenizer({"aa": [10, 11], "bbb": [12, 13, 14], "c": [15]})
        text = "aa bbb c"
        batch = build_batch(tokenizer, [text], max_length=6)
        features = build_lexical_batch(self.store, [text], batch)

        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, 0, -1]])
        self.assertEqual(features.values.shape, (1, 4, 12))
        self.assertTrue(torch.equal(features.values[0, 3], torch.zeros(12)))

    def test_returns_float_tensors(self):
        text = "tốt"
        batch = build_batch(self.tokenizer, [text])
        features = build_lexical_batch(self.store, [text], batch)

        self.assertEqual(features.values.dtype, torch.float32)
        self.assertEqual(features.mask.dtype, torch.float32)
        self.assertEqual(features.presence.dtype, torch.float32)

    def test_rejects_single_string(self):
        batch = build_batch(self.tokenizer, ["tốt"])
        with self.assertRaises(TypeError):
            build_lexical_batch(self.store, "tốt", batch)

    def test_rejects_batch_size_mismatch(self):
        batch = build_batch(self.tokenizer, ["tốt này"])
        with self.assertRaises(ValueError):
            build_lexical_batch(self.store, ["tốt"], batch)

    def test_rejects_word_index_outside_text(self):
        batch = BackboneBatch(
            input_ids=torch.tensor([[0, 10, 2]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            content_mask=torch.tensor([[0, 1, 0]]),
            word_ids=torch.tensor([[-1, 3, -1]]),
        )
        with self.assertRaises(ValueError):
            build_lexical_batch(self.store, ["tốt"], batch)


if __name__ == "__main__":
    unittest.main()