import os
import unittest

import torch

from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.pooling import pool_masked_mean
from vnsentiment.models.fusion.residual import fuse_residual
from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from vnsentiment.phobert_branch.tokenizer import load_phobert_tokenizer


@unittest.skipUnless(
    os.environ.get("RUN_PHOBERT_SMOKE") == "1",
    "set RUN_PHOBERT_SMOKE=1 to run the real pooling/classifier smoke test",
)
class PoolingClassifierSmokeTests(unittest.TestCase):
    def test_real_chain_reaches_logits(self):
        tokenizer = load_phobert_tokenizer()
        encoder = PhoBertEncoder(device="cpu")
        encoder.model.eval()
        phobert = PhoBertBranch(tokenizer, encoder)
        lexicon = LexiconBranch(LexiconStore.load())
        gate = LearnableGate()
        classifier = SentimentClassifier()
        texts = ["sản_phẩm này không tệ", "xyz_oov_test"]

        with torch.no_grad():
            batch, hidden = phobert.encode(texts)
            lexical = lexicon(texts, batch)
            gates = gate(hidden, lexical)
            fused = fuse_residual(hidden, gates, lexical)
            pooled = pool_masked_mean(fused, batch.content_mask)
            logits = classifier(pooled)

        width = batch.input_ids.shape[1]
        self.assertEqual(tuple(hidden.shape), (2, width, 768))
        self.assertEqual(tuple(fused.shape), (2, width, 768))
        self.assertEqual(tuple(pooled.shape), (2, 768))
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(fused[1], hidden[1]))


if __name__ == "__main__":
    unittest.main()