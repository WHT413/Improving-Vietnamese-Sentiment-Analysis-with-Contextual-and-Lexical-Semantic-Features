import unittest

import torch

from vnsentiment.contracts.train_batch import TrainBatch


class TrainBatchTests(unittest.TestCase):
    def test_from_samples_preserves_texts_and_labels(self):
        batch = TrainBatch.from_samples(
            [
                {"text": "tốt này không tệ", "label": 2},
                {"text": "sản_phẩm tệ", "label": 0},
            ]
        )

        self.assertEqual(batch.texts, ("tốt này không tệ", "sản_phẩm tệ"))
        self.assertEqual(batch.labels.dtype, torch.long)
        self.assertEqual(batch.labels.tolist(), [2, 0])

    def test_from_samples_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            TrainBatch.from_samples([])


if __name__ == "__main__":
    unittest.main()
