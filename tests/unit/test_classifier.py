import unittest

import torch

from vnsentiment.models.fusion.classifier import SentimentClassifier


class SentimentClassifierTests(unittest.TestCase):
    def test_defaults_to_768_to_3(self):
        classifier = SentimentClassifier()
        self.assertEqual(classifier.fc.in_features, 768)
        self.assertEqual(classifier.fc.out_features, 3)

    def test_forward_shape(self):
        classifier = SentimentClassifier(hidden_dim=4)
        logits = classifier(torch.randn(2, 4))
        self.assertEqual(logits.shape, (2, 3))

    def test_logits_are_finite_and_depend_on_input(self):
        classifier = SentimentClassifier(hidden_dim=4)
        z = torch.randn(2, 4)
        logits_a = classifier(z)
        logits_b = classifier(z + 1.0)
        self.assertTrue(torch.isfinite(logits_a).all())
        self.assertFalse(torch.allclose(logits_a, logits_b))

    def test_returns_raw_logits_without_softmax(self):
        classifier = SentimentClassifier(hidden_dim=2, num_labels=3)
        with torch.no_grad():
            classifier.fc.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            )
            classifier.fc.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))

        logits = classifier(torch.tensor([[2.0, 3.0]]))

        self.assertTrue(
            torch.equal(logits, torch.tensor([[2.0, 4.0, 7.0]]))
        )

    def test_supports_custom_num_labels(self):
        classifier = SentimentClassifier(hidden_dim=4, num_labels=5)
        logits = classifier(torch.randn(2, 4))
        self.assertEqual(logits.shape, (2, 5))

    def test_gradients_reach_classifier_parameters(self):
        classifier = SentimentClassifier(hidden_dim=4)
        classifier.zero_grad()
        z = torch.randn(2, 4, requires_grad=True)

        classifier(z).sum().backward()

        self.assertIsNotNone(classifier.fc.weight.grad)
        self.assertIsNotNone(classifier.fc.bias.grad)
        self.assertGreater(classifier.fc.weight.grad.abs().sum().item(), 0)
        self.assertGreater(classifier.fc.bias.grad.abs().sum().item(), 0)
        self.assertIsNotNone(z.grad)
        self.assertGreater(z.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()