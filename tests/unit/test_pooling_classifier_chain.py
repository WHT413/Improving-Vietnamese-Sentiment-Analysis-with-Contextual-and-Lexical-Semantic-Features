import json
import unittest
from pathlib import Path

import torch

from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.pooling import pool_masked_mean
from vnsentiment.models.fusion.projection import LexiconProjection
from vnsentiment.models.fusion.residual import fuse_residual
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


class PoolingClassifierChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LexiconStore(json.loads(FIXTURE.read_text(encoding="utf-8")))

    def setUp(self):
        torch.manual_seed(0)
        self.tokenizer = FakeTokenizer(PIECES)
        self.phobert = PhoBertBranch(
            self.tokenizer,
            PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu"),
        )
        self.lexicon = LexiconBranch(
            self.store, LexiconProjection(hidden_dim=4)
        )
        self.gate = LearnableGate(hidden_dim=4, bias_init=0.0)
        self.classifier = SentimentClassifier(hidden_dim=4)

    def _chain(self, text):
        batch, hidden = self.phobert.encode([text])
        lexical = self.lexicon([text], batch)
        gates = self.gate(hidden, lexical)
        fused = fuse_residual(hidden, gates, lexical)
        pooled = pool_masked_mean(fused, batch.content_mask)
        logits = self.classifier(pooled)
        return batch, hidden, fused, pooled, logits

    def test_chain_reaches_logits(self):
        batch, hidden, fused, pooled, logits = self._chain("tốt này không tệ")

        self.assertEqual(tuple(hidden.shape), (1, 8, 4))
        self.assertEqual(tuple(fused.shape), (1, 8, 4))
        self.assertEqual(tuple(pooled.shape), (1, 4))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_oov_chain_still_reaches_finite_logits(self):
        tokenizer = FakeTokenizer({"xyz_oov_test": [99]})
        phobert = PhoBertBranch(
            tokenizer,
            PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu"),
        )
        batch, hidden = phobert.encode(["xyz_oov_test"])
        lexical = self.lexicon(["xyz_oov_test"], batch)
        gates = self.gate(hidden, lexical)
        fused = fuse_residual(hidden, gates, lexical)
        pooled = pool_masked_mean(fused, batch.content_mask)
        logits = self.classifier(pooled)

        self.assertTrue(torch.equal(fused, hidden))
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.isfinite(logits).all())

    def test_gradients_reach_classifier(self):
        with torch.no_grad():
            self.classifier.fc.weight.zero_()
            self.classifier.fc.bias.zero_()
        batch, hidden, fused, pooled, logits = self._chain("tốt")
        logits.sum().backward()

        self.assertIsNotNone(self.classifier.fc.weight.grad)
        self.assertGreater(
            self.classifier.fc.weight.grad.abs().sum().item(), 0
        )


if __name__ == "__main__":
    unittest.main()