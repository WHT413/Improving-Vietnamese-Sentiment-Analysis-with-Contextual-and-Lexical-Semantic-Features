import os
import unittest

import torch

from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from vnsentiment.phobert_branch.tokenizer import load_phobert_tokenizer
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.lexicon.store import LexiconStore


@unittest.skipUnless(
    os.environ.get("RUN_PHOBERT_SMOKE") == "1",
    "set RUN_PHOBERT_SMOKE=1 to run the real lexicon projection smoke test",
)
class LexiconProjectionSmokeTests(unittest.TestCase):
    def test_real_lexicon_projects_to_phobert_base_hidden_size(self):
        tokenizer = load_phobert_tokenizer()
        self.assertFalse(tokenizer.is_fast)
        encoder = PhoBertEncoder(device="cpu")
        encoder.model.eval()
        phobert_branch = PhoBertBranch(tokenizer, encoder)

        store = LexiconStore.load()
        branch = LexiconBranch(store)
        texts = ["sản_phẩm này không tệ", "xyz_oov_test"]

        with torch.no_grad():
            batch, hidden = phobert_branch.encode(texts)
            features = branch.features(texts, batch)
            projected = branch(texts, batch)

        self.assertEqual(tuple(hidden.shape), (2, batch.input_ids.shape[1], 768))
        self.assertEqual(tuple(projected.shape), tuple(hidden.shape))
        self.assertEqual(features.values.shape, (2, batch.input_ids.shape[1], 12))
        self.assertEqual(features.mask.shape, features.values.shape)
        self.assertEqual(features.presence.shape, (2, batch.input_ids.shape[1], 1))

        self.assertEqual(batch.word_ids[0].tolist()[:6], [-1, 0, 1, 2, 3, -1])
        self.assertAlmostEqual(features.values[0, 1, 1].item(), 0.646, places=5)
        self.assertAlmostEqual(features.values[0, 4, 0].item(), 0.1875, places=6)
        self.assertEqual(features.presence[0, 4].item(), 1.0)
        self.assertTrue(torch.equal(features.values[0, 2, 4:], torch.zeros(8)))
        self.assertEqual(tuple(features.mask[0, 2, 4:].tolist()), (1,) * 8)

        self.assertTrue(torch.equal(projected[1], torch.zeros_like(projected[1])))
        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue(torch.isfinite(hidden + projected).all())


if __name__ == "__main__":
    unittest.main()