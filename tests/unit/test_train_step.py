import json
import unittest
from pathlib import Path

import torch

from vnsentiment.contracts.train_batch import TrainBatch
from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.projection import LexiconProjection
from vnsentiment.models.sentiment import SentimentModel
from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from vnsentiment.training.step import train_step
from tests.helpers.fake_encoder import FakeEncoder
from tests.helpers.fake_tokenizer import FakeTokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_fixture.json"
PIECES = {
    "tốt": [10, 11],
    "này": [12],
    "không": [13, 14],
    "tệ": [15],
}
SAMPLES = [
    {"text": "tốt", "label": 2},
    {"text": "tệ", "label": 0},
    {"text": "tốt này không tệ", "label": 2},
    {"text": "không tốt", "label": 0},
]


def _build_model(store):
    return SentimentModel(
        phobert=PhoBertBranch(
            FakeTokenizer(PIECES),
            PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu"),
        ),
        lexicon=LexiconBranch(store, LexiconProjection(hidden_dim=4)),
        gate=LearnableGate(hidden_dim=4, bias_init=0.0),
        classifier=SentimentClassifier(hidden_dim=4),
    )


class TrainStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LexiconStore(
            json.loads(FIXTURE.read_text(encoding="utf-8"))
        )

    def setUp(self):
        torch.manual_seed(0)
        self.model = _build_model(self.store)
        self.batch = TrainBatch.from_samples(SAMPLES[:2])

    def test_backward_populates_classifier_gradient_without_stepping(self):
        before = [p.detach().clone() for p in self.model.parameters()]
        loss = train_step(self.model, self.batch)
        after = list(self.model.parameters())

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        grad = self.model.classifier.fc.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)
        for original, current in zip(before, after):
            self.assertTrue(torch.equal(original, current))

    def test_padded_batch_loss_matches_single_sample_losses(self):
        samples = SAMPLES[0:2]
        padded = TrainBatch.from_samples(samples)
        batch_loss = train_step(self.model, padded)
        self.model.zero_grad(set_to_none=True)

        single_losses = [
            train_step(self.model, TrainBatch.from_samples([sample]))
            for sample in samples
        ]
        self.assertTrue(
            torch.allclose(
                torch.tensor(batch_loss),
                torch.tensor(sum(single_losses) / len(single_losses)),
            )
        )

    def test_accumulated_grads_match_a_single_combined_batch(self):
        torch.manual_seed(7)
        accumulated = _build_model(self.store)
        combined = _build_model(self.store)
        combined.load_state_dict(accumulated.state_dict())

        accumulated.zero_grad(set_to_none=True)
        train_step(
            accumulated, TrainBatch.from_samples(SAMPLES[0:2]), scale=0.5
        )
        train_step(
            accumulated, TrainBatch.from_samples(SAMPLES[2:4]), scale=0.5
        )

        combined.zero_grad(set_to_none=True)
        train_step(combined, TrainBatch.from_samples(SAMPLES), scale=1.0)

        for (name, left), (_, right) in zip(
            accumulated.named_parameters(), combined.named_parameters()
        ):
            if left.grad is None or right.grad is None:
                self.assertIs(left.grad, right.grad, name)
                continue
            self.assertTrue(
                torch.allclose(left.grad, right.grad, atol=1e-6),
                name,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_model_accepts_cpu_labels(self):
        model = _build_model(self.store).to("cuda")
        loss = train_step(model, self.batch)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_three_epochs_on_eight_sentences_stay_finite(self):
        samples = (SAMPLES * 2)[:8]
        batch = TrainBatch.from_samples(samples)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        for _ in range(3):
            optimizer.zero_grad()
            loss = train_step(self.model, batch)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
