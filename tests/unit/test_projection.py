import unittest

import torch

from vnsentiment.models.fusion.projection import LexiconProjection


class LexiconProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.values = torch.rand(2, 3, 12)
        self.mask = torch.ones(2, 3, 12)
        self.presence = torch.ones(2, 3, 1)

    def test_default_projects_12_to_768(self):
        module = LexiconProjection()
        output = module(self.values, self.mask, self.presence)
        self.assertEqual(output.shape, (2, 3, 768))

    def test_oov_positions_are_exactly_zero_even_with_nonzero_bias(self):
        module = LexiconProjection(hidden_dim=16)
        with torch.no_grad():
            module.bias.fill_(1.0)
        presence = torch.zeros(2, 3, 1)
        presence[0, 0, 0] = 1
        output = module(self.values, self.mask, presence)
        self.assertFalse(torch.equal(output[0, 0], torch.zeros(16)))
        self.assertTrue(torch.equal(output[0, 1], torch.zeros(16)))
        self.assertTrue(torch.equal(output[1], torch.zeros(3, 16)))

    def test_masked_dimensions_do_not_affect_output(self):
        module = LexiconProjection(hidden_dim=16)
        mask = torch.zeros(1, 1, 12)
        mask[..., 0] = 1
        values_a = torch.zeros(1, 1, 12)
        values_b = torch.zeros(1, 1, 12)
        values_b[..., 5] = 99.0
        presence = torch.ones(1, 1, 1)
        out_a = module(values_a, mask, presence)
        out_b = module(values_b, mask, presence)
        self.assertTrue(torch.equal(out_a, out_b))

    def test_true_zero_differs_from_missing(self):
        module = LexiconProjection(hidden_dim=16)
        with torch.no_grad():
            module.bias.fill_(0.5)
        values = torch.zeros(1, 1, 12)
        present_mask = torch.zeros(1, 1, 12)
        present_mask[..., 0] = 1
        present = torch.ones(1, 1, 1)
        missing_mask = torch.zeros(1, 1, 12)
        missing = torch.zeros(1, 1, 1)
        out_present = module(values, present_mask, present)
        out_missing = module(values, missing_mask, missing)
        self.assertFalse(torch.equal(out_present, torch.zeros(1, 1, 16)))
        self.assertTrue(torch.equal(out_missing, torch.zeros(1, 1, 16)))

    def test_boolean_masks_match_float_masks(self):
        module = LexiconProjection(hidden_dim=16)
        out_bool = module(self.values, self.mask.bool(), self.presence.bool())
        out_float = module(self.values, self.mask, self.presence)
        self.assertTrue(torch.equal(out_bool, out_float))

    def test_gradients_flow_for_present_tokens(self):
        module = LexiconProjection(hidden_dim=16)
        values = self.values.clone().requires_grad_(True)
        output = module(values, self.mask, self.presence)
        output.sum().backward()
        self.assertIsNotNone(module.value_proj.weight.grad)
        self.assertIsNotNone(module.mask_proj.weight.grad)
        self.assertIsNotNone(module.bias.grad)
        self.assertGreater(module.value_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.mask_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(module.bias.grad.abs().sum().item(), 0)

    def test_oov_does_not_update_bias(self):
        module = LexiconProjection(hidden_dim=16)
        presence = torch.zeros(2, 3, 1)
        output = module(self.values, self.mask, presence)
        output.sum().backward()
        self.assertIsNotNone(module.bias.grad)
        self.assertTrue(torch.equal(module.bias.grad, torch.zeros(16)))


if __name__ == "__main__":
    unittest.main()
