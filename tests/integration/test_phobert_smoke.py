import os
import unittest

import torch

from vnsentiment.backbones.backbone import PhoBertBackbone
from vnsentiment.backbones.encoder import PhoBertEncoder
from vnsentiment.backbones.tokenizer import load_phobert_tokenizer


@unittest.skipUnless(
    os.environ.get("RUN_PHOBERT_SMOKE") == "1",
    "set RUN_PHOBERT_SMOKE=1 to run the real PhoBERT smoke test",
)
class PhoBertSmokeTests(unittest.TestCase):
    def test_real_phobert_forward(self):
        tokenizer = load_phobert_tokenizer()
        self.assertFalse(tokenizer.is_fast)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = PhoBertEncoder(device=device)
        encoder.model.eval()
        backbone = PhoBertBackbone(tokenizer, encoder)

        with torch.no_grad():
            batch, hidden = backbone.encode(["sản_phẩm này không tệ"])

        self.assertEqual(batch.input_ids.tolist(), [[0, 265, 23, 17, 4152, 2]])
        self.assertEqual(batch.attention_mask.tolist(), [[1, 1, 1, 1, 1, 1]])
        self.assertEqual(batch.content_mask.tolist(), [[0, 1, 1, 1, 1, 0]])
        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, 1, 2, 3, -1]])
        self.assertEqual(tuple(hidden.shape), (1, 6, 768))
        self.assertEqual(hidden.device.type, device)


if __name__ == "__main__":
    unittest.main()
