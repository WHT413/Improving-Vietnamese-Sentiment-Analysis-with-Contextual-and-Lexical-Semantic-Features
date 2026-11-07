import unittest

import torch

from vnsentiment.models.fusion.gate import LearnableGate


class LearnableGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.context = torch.rand(2, 4, 16)
        self.lexical = torch.rand(2, 4, 16)

    def test_default_gate_shape_is_768(self):
        gate = LearnableGate()
        context = torch.rand(1, 2, 768)
        lexical = torch.rand(1, 2, 768)
        self.assertEqual(gate(context, lexical).shape, (1, 2, 768))

    def test_gate_values_are_in_unit_interval(self):
        gate = LearnableGate(hidden_dim=16)
        output = gate(self.context, self.lexical)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_gate_depends_on_context(self):
        gate = LearnableGate(hidden_dim=16)
        output_a = gate(self.context, self.lexical)
        output_b = gate(self.context + 1.0, self.lexical)
        self.assertFalse(torch.allclose(output_a, output_b))

    def test_gate_depends_on_lexical(self):
        gate = LearnableGate(hidden_dim=16)
        output_a = gate(self.context, self.lexical)
        output_b = gate(self.context, self.lexical + 1.0)
        self.assertFalse(torch.allclose(output_a, output_b))

    def test_bias_starts_negative(self):
        gate = LearnableGate(hidden_dim=16)
        self.assertTrue(torch.all(gate.proj.bias < 0))

    def test_initial_mean_gate_is_below_half(self):
        gate = LearnableGate(hidden_dim=16)
        output = gate(self.context, self.lexical)
        self.assertLess(output.mean().item(), 0.5)

    def test_bias_only_mean_gate_is_below_half(self):
        gate = LearnableGate(hidden_dim=16)
        with torch.no_grad():
            gate.proj.weight.zero_()
        output = gate(self.context, self.lexical)
        self.assertLess(output.mean().item(), 0.5)

    def test_gradients_flow_to_inputs_and_parameters(self):
        gate = LearnableGate(hidden_dim=16)
        context = self.context.clone().requires_grad_(True)
        lexical = self.lexical.clone().requires_grad_(True)
        gate(context, lexical).sum().backward()
        self.assertIsNotNone(gate.proj.weight.grad)
        self.assertGreater(gate.proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(context.grad.abs().sum().item(), 0)
        self.assertGreater(lexical.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
