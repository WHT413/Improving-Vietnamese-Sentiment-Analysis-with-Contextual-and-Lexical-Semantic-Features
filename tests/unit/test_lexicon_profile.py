import unittest

import torch

from vnsentiment.contracts.lexical_batch import LexicalBatch
from vnsentiment.lexicon.profile import mean_profile


def _features(values, mask):
    values = torch.tensor(values, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)
    return LexicalBatch(
        values=values,
        mask=mask,
        presence=torch.ones(values.shape[0], values.shape[1], 1),
    )


class MeanProfileTests(unittest.TestCase):
    def test_averages_only_present_content_positions(self):
        values = [[[9.0] * 12, [0.5, 0.2] + [0.0] * 10,
                   [0.0] * 12, [9.0] * 12]]
        mask = [[[1] * 12, [1, 1] + [0] * 10,
                 [1] + [0] * 11, [1] * 12]]
        features = _features(values, mask)
        content_mask = torch.tensor([[0, 1, 1, 0]])

        means, counts = mean_profile(features, content_mask)

        self.assertEqual(tuple(means.shape), (1, 12))
        self.assertEqual(tuple(counts.shape), (1, 12))
        self.assertAlmostEqual(means[0, 0].item(), 0.25)
        self.assertAlmostEqual(means[0, 1].item(), 0.2)
        self.assertEqual(counts[0, 0].item(), 2)
        self.assertEqual(counts[0, 1].item(), 1)
        self.assertEqual(counts[0, 2].item(), 0)
        self.assertEqual(means[0, 2].item(), 0.0)

    def test_ignores_special_and_padded_positions(self):
        values = [[[9.0] * 12, [0.4] + [0.0] * 11,
                   [9.0] * 12, [9.0] * 12]]
        mask = [[[1] * 12, [1] + [0] * 11, [1] * 12, [1] * 12]]
        features = _features(values, mask)
        content_mask = torch.tensor([[0, 1, 0, 0]])

        means, counts = mean_profile(features, content_mask)

        self.assertAlmostEqual(means[0, 0].item(), 0.4)
        self.assertEqual(counts[0, 0].item(), 1)

    def test_missing_dimension_has_zero_mean_and_count(self):
        features = _features([[[0.0] * 12]], [[[0] * 12]])
        means, counts = mean_profile(
            features, torch.tensor([[1]])
        )
        self.assertTrue(torch.equal(means, torch.zeros(1, 12)))
        self.assertTrue(torch.equal(counts, torch.zeros(1, 12)))

    def test_supports_multiple_rows(self):
        values = [[[0.4] + [0.0] * 11], [[0.8] + [0.0] * 11]]
        mask = [[[1] + [0] * 11], [[1] + [0] * 11]]
        features = _features(values, mask)
        content_mask = torch.tensor([[1], [1]])

        means, counts = mean_profile(features, content_mask)

        self.assertAlmostEqual(means[0, 0].item(), 0.4)
        self.assertAlmostEqual(means[1, 0].item(), 0.8)
        self.assertEqual(counts[:, 0].tolist(), [1.0, 1.0])

    def test_can_deduplicate_subwords_per_word(self):
        values = [
            [[0.4] + [0.0] * 11, [0.8] + [0.0] * 11, [0.0] * 12]
        ]
        mask = [
            [[1] + [0] * 11, [1] + [0] * 11, [1] + [0] * 11]
        ]
        features = _features(values, mask)

        means, counts = mean_profile(
            features,
            torch.tensor([[1, 1, 1]]),
            torch.tensor([[0, 0, 1]]),
        )

        self.assertAlmostEqual(means[0, 0].item(), 0.2)
        self.assertEqual(counts[0, 0].item(), 2)

    def test_rejects_content_mask_with_wrong_shape(self):
        features = _features([[[0.0] * 12]], [[[0] * 12]])
        with self.assertRaises(ValueError):
            mean_profile(features, torch.tensor([[1, 0]]))


if __name__ == "__main__":
    unittest.main()