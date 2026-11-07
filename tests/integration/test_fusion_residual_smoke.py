import os
import unittest

import torch

from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.residual import fuse_residual
from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from vnsentiment.phobert_branch.tokenizer import load_phobert_tokenizer


@unittest.skipUnless(
    os.environ.get("RUN_PHOBERT_SMOKE") == "1",
    "set RUN_PHOBERT_SMOKE=1 to run the real fusion residual smoke test",
)
class FusionResidualSmokeTests(unittest.TestCase):
    def test_real_branches_through_gate_and_residual(self):
        tokenizer = load_phobert_tokenizer()
        encoder = PhoBertEncoder(device="cpu")
        encoder.model.eval()
        phobert = PhoBertBranch(tokenizer, encoder)
        lexicon = LexiconBranch(LexiconStore.load())
        gate = LearnableGate()
        texts = ["sản_phẩm này không tệ", "xyz_oov_test"]

        with torch.no_grad():
            batch, hidden = phobert.encode(texts)
            lexical = lexicon(texts, batch)
            gates = gate(hidden, lexical)
            fused = fuse_residual(hidden, gates, lexical)

        width = batch.input_ids.shape[1]
        expected = (2, width, 768)
        self.assertEqual(tuple(hidden.shape), expected)
        self.assertEqual(tuple(lexical.shape), expected)
        self.assertEqual(tuple(gates.shape), expected)
        self.assertEqual(tuple(fused.shape), expected)
        self.assertTrue(torch.equal(fused, hidden + gates * lexical))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(bool((gates > 0).all()))
        self.assertTrue(bool((gates < 1).all()))
        self.assertFalse(torch.equal(fused[0], hidden[0]))
        self.assertTrue(torch.equal(fused[1], hidden[1]))


if __name__ == "__main__":
    unittest.main()