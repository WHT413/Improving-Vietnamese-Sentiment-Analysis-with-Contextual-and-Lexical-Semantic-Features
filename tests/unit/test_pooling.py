import unittest

import torch

from vnsentiment.models.fusion.pooling import pool_masked_mean


class PoolMaskedMeanTests(unittest.TestCase):
    def test_matches_manual_masked_mean(self):
        fused = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        mask = torch.tensor([[1, 0, 1, 0]])

        pooled = pool_masked_mean(fused, mask)

        self.assertEqual(pooled.shape, (1, 2))
        self.assertTrue(torch.equal(pooled, torch.tensor([[3.0, 4.0]])))

    def test_masked_positions_do_not_affect_output(self):
        mask = torch.tensor([[1, 0, 1, 0]])
        fused_a = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])
        fused_b = fused_a.clone()
        fused_b[0, 1] = 999.0
        fused_b[0, 3] = -999.0

        pooled_a = pool_masked_mean(fused_a, mask)
        pooled_b = pool_masked_mean(fused_b, mask)

        self.assertTrue(torch.equal(pooled_a, pooled_b))

    def test_all_zero_mask_row_returns_zero_without_nan(self):
        fused = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.zeros(1, 3, dtype=torch.long)

        pooled = pool_masked_mean(fused, mask)

        self.assertTrue(torch.equal(pooled, torch.zeros(1, 2)))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_gradient_only_reaches_unmasked_positions(self):
        fused = torch.ones(1, 3, 2, requires_grad=True)
        mask = torch.tensor([[1, 0, 1]])

        pool_masked_mean(fused, mask).sum().backward()

        self.assertTrue(torch.equal(fused.grad[0, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(fused.grad[0, 0], torch.full((2,), 0.5)))
        self.assertTrue(torch.equal(fused.grad[0, 2], torch.full((2,), 0.5)))

    def test_float16_input_keeps_dtype_and_expected_mean(self):
        fused = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float16)
        mask = torch.tensor([[1, 0, 1]])

        pooled = pool_masked_mean(fused, mask)

        self.assertEqual(pooled.dtype, torch.float16)
        self.assertAlmostEqual(pooled.item(), 2.0)

    def test_float16_accumulation_does_not_overflow(self):
        fused = torch.tensor([[[40000.0], [40000.0]]], dtype=torch.float16)
        mask = torch.tensor([[1, 1]])

        pooled = pool_masked_mean(fused, mask)

        self.assertEqual(pooled.dtype, torch.float16)
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertAlmostEqual(pooled.item(), 40000.0, delta=32.0)

    def test_boolean_mask_matches_integer_mask(self):
        fused = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        integer_mask = torch.tensor([[1, 0, 1]])
        boolean_mask = integer_mask.bool()

        pooled_int = pool_masked_mean(fused, integer_mask)
        pooled_bool = pool_masked_mean(fused, boolean_mask)

        self.assertTrue(torch.equal(pooled_int, pooled_bool))

    def test_all_zero_mask_backward_is_zero(self):
        fused = torch.ones(1, 3, 2, requires_grad=True)
        mask = torch.zeros(1, 3, dtype=torch.long)

        pool_masked_mean(fused, mask).sum().backward()

        self.assertTrue(torch.equal(fused.grad, torch.zeros(1, 3, 2)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_moves_mask_to_fused_device(self):
        fused = torch.ones(1, 3, 2, device="cuda")
        mask = torch.tensor([[1, 1, 0]])

        pooled = pool_masked_mean(fused, mask)

        self.assertEqual(pooled.device.type, "cuda")
        self.assertTrue(torch.equal(pooled, torch.ones(1, 2, device="cuda")))

    def test_rejects_non_three_dimensional_fused(self):
        with self.assertRaises(ValueError):
            pool_masked_mean(torch.ones(4, 2), torch.ones(4))

    def test_rejects_mask_with_wrong_shape(self):
        with self.assertRaises(ValueError):
            pool_masked_mean(torch.ones(1, 4, 2), torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()