import json
import unittest
from pathlib import Path
from unittest import mock

import torch

from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.projection import LexiconProjection
from vnsentiment.models.sentiment import SentimentModel
from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from tests.helpers.fake_encoder import FakeEncoder
from tests.helpers.fake_tokenizer import FakeTokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_fixture.json"
PIECES = {
    "tốt": [10, 11],
    "này": [12],
    "không": [13, 14],
    "tệ": [15],
}


def _empty_store():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["entries"] = {}
    return LexiconStore(payload)


class SentimentModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        tokenizer = FakeTokenizer(PIECES)
        phobert = PhoBertBranch(
            tokenizer,
            PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu"),
        )
        self.model = SentimentModel(
            phobert=phobert,
            lexicon=LexiconBranch(
                LexiconStore(json.loads(FIXTURE.read_text(encoding="utf-8"))),
                LexiconProjection(hidden_dim=4),
            ),
            gate=LearnableGate(hidden_dim=4, bias_init=0.0),
            classifier=SentimentClassifier(hidden_dim=4),
        )

    def test_both_modes_return_logits(self):
        texts = ["tốt này không tệ", "tệ"]
        for mode in ("fused", "contextual"):
            with self.subTest(mode=mode):
                logits = self.model(texts, mode=mode)
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(logits).all())

    def test_contextual_mode_ignores_lexicon(self):
        texts = ["tốt này không tệ"]
        before = self.model(texts, mode="contextual")
        self.model.lexicon.store = _empty_store()
        after = self.model(texts, mode="contextual")
        self.assertTrue(torch.allclose(before, after))

    def test_gradients_reach_classifier_in_both_modes(self):
        for mode in ("fused", "contextual"):
            with self.subTest(mode=mode):
                self.model.zero_grad(set_to_none=True)
                self.model.classifier.fc.weight.grad = None
                logits = self.model(["tốt"], mode=mode)
                logits.sum().backward()
                grad = self.model.classifier.fc.weight.grad
                self.assertIsNotNone(grad)
                self.assertGreater(grad.abs().sum().item(), 0)

    def test_forward_calls_encode_once_per_mode(self):
        for mode in ("fused", "contextual"):
            with self.subTest(mode=mode):
                with mock.patch.object(
                    self.model.phobert,
                    "encode",
                    wraps=self.model.phobert.encode,
                ) as encode:
                    self.model(["tốt này không tệ"], mode=mode)
                self.assertEqual(encode.call_count, 1)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self.model(["tốt"], mode="concat")


if __name__ == "__main__":
    unittest.main()
