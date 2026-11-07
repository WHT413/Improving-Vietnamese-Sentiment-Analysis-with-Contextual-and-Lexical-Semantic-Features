import unittest

import torch

from vnsentiment.models.fusion.residual import fuse_residual


class ResidualFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.context = torch.rand(2, 4, 16)
        self.gate = torch.rand(2, 4, 16)
        self.lexical = torch.rand(2, 4, 16)

    def test_output_shape_matches_inputs(self):
        output = fuse_residual(self.context, self.gate, self.lexical)
        self.assertEqual(output.shape, (2, 4, 16))

    def test_adds_gated_lexical_to_context(self):
        output = fuse_residual(self.context, self.gate, self.lexical)
        expected = self.context + self.gate * self.lexical
        self.assertTrue(torch.allclose(output, expected))

    def test_zero_lexical_returns_context_exactly(self):
        lexical = torch.zeros_like(self.lexical)
        output = fuse_residual(self.context, self.gate, lexical)
        self.assertTrue(torch.equal(output, self.context))

    def test_zero_gate_returns_context_exactly(self):
        gate = torch.zeros_like(self.gate)
        output = fuse_residual(self.context, gate, self.lexical)
        self.assertTrue(torch.equal(output, self.context))

    def test_gradients_flow_to_all_inputs(self):
        context = self.context.clone().requires_grad_(True)
        gate = self.gate.clone().requires_grad_(True)
        lexical = self.lexical.clone().requires_grad_(True)
        fuse_residual(context, gate, lexical).sum().backward()
        self.assertGreater(context.grad.abs().sum().item(), 0)
        self.assertGreater(gate.grad.abs().sum().item(), 0)
        self.assertGreater(lexical.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
