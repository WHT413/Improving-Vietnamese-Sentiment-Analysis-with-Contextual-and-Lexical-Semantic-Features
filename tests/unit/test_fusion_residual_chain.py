import json
import unittest
from pathlib import Path

import torch

from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
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


class FusionResidualChainTests(unittest.TestCase):
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

    def _chain(self, text):
        batch, hidden = self.phobert.encode([text])
        with torch.no_grad():
            lexical = self.lexicon([text], batch)
            gates = self.gate(hidden, lexical)
            fused = fuse_residual(hidden, gates, lexical)
        return batch, hidden, lexical, gates, fused

    def test_chain_through_residual_shapes_and_formula(self):
        batch, hidden, lexical, gates, fused = self._chain("tốt này không tệ")

        self.assertEqual(tuple(hidden.shape), (1, 8, 4))
        self.assertEqual(tuple(lexical.shape), (1, 8, 4))
        self.assertEqual(tuple(gates.shape), (1, 8, 4))
        self.assertEqual(tuple(fused.shape), (1, 8, 4))
        self.assertTrue(torch.equal(fused, hidden + gates * lexical))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertFalse(torch.equal(fused[0, 1], hidden[0, 1]))

    def test_gate_values_are_bounded(self):
        _, _, _, gates, _ = self._chain("tốt này không tệ")
        self.assertTrue(bool((gates > 0).all()))
        self.assertTrue(bool((gates < 1).all()))

    def test_special_positions_keep_phobert_exactly(self):
        _, hidden, _, _, fused = self._chain("tốt")
        self.assertTrue(torch.equal(fused[0, 0], hidden[0, 0]))
        self.assertTrue(torch.equal(fused[0, 3], hidden[0, 3]))

    def test_oov_falls_back_to_phobert_exactly(self):
        tokenizer = FakeTokenizer({"xyz_oov_test": [99]})
        phobert = PhoBertBranch(
            tokenizer,
            PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu"),
        )
        batch, hidden = phobert.encode(["xyz_oov_test"])
        with torch.no_grad():
            lexical = self.lexicon(["xyz_oov_test"], batch)
            gates = self.gate(hidden, lexical)
            fused = fuse_residual(hidden, gates, lexical)

        self.assertEqual(tuple(lexical.shape), (1, 3, 4))
        self.assertTrue(torch.equal(lexical, torch.zeros(1, 3, 4)))
        self.assertTrue(torch.equal(fused, hidden))


if __name__ == "__main__":
    unittest.main()