import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.projection import LexiconProjection
from vnsentiment.models.sentiment import SentimentModel
from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from vnsentiment.training.loop import Trainer
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
    {"text": "sản_phẩm này tốt", "label": 2},
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


def _dev_metrics(macro_f1):
    return {
        "accuracy": macro_f1,
        "macro_f1": macro_f1,
        "f1_per_class": (macro_f1, macro_f1, macro_f1),
    }


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LexiconStore(
            json.loads(FIXTURE.read_text(encoding="utf-8"))
        )

    def setUp(self):
        torch.manual_seed(0)
        self.model = _build_model(self.store)

    def test_optimizer_has_two_parameter_groups(self):
        trainer = Trainer(self.model)

        encoder_group, fusion_group = trainer.optimizer.param_groups
        self.assertAlmostEqual(encoder_group["lr"], 2e-5)
        self.assertAlmostEqual(fusion_group["lr"], 1e-4)
        self.assertAlmostEqual(encoder_group["weight_decay"], 0.01)
        self.assertAlmostEqual(fusion_group["weight_decay"], 0.01)

    def test_five_micro_batches_with_accum_four_step_twice(self):
        trainer = Trainer(
            self.model, batch_size=1, accum_steps=4, checkpoint_path=None
        )

        with mock.patch(
            "vnsentiment.training.loop.clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as clip, mock.patch.object(
            trainer.optimizer, "step", wraps=trainer.optimizer.step
        ) as step:
            loss = trainer.run_epoch(SAMPLES, epoch=1)

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(step.call_count, 2)
        self.assertEqual(clip.call_count, 2)

    def test_evaluate_returns_metrics_without_updating_weights(self):
        trainer = Trainer(self.model, batch_size=2)
        before = [p.detach().clone() for p in self.model.parameters()]

        metrics = trainer.evaluate(SAMPLES)

        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(len(metrics["f1_per_class"]), 3)
        self.assertFalse(trainer.model.training)
        for original, current in zip(before, self.model.parameters()):
            self.assertTrue(torch.equal(original, current))

    def test_fit_keeps_best_checkpoint_and_early_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            trainer = Trainer(
                self.model,
                max_epochs=10,
                patience=2,
                checkpoint_path=checkpoint,
            )
            trainer.run_epoch = mock.Mock(return_value=1.0)
            trainer.evaluate = mock.Mock(
                side_effect=[
                    _dev_metrics(0.5),
                    _dev_metrics(0.6),
                    _dev_metrics(0.55),
                    _dev_metrics(0.55),
                ]
            )

            summary = trainer.fit(SAMPLES, SAMPLES)

            self.assertEqual(trainer.run_epoch.call_count, 4)
            self.assertEqual(summary["best_epoch"], 2)
            self.assertAlmostEqual(summary["best_metrics"]["macro_f1"], 0.6)
            self.assertEqual(len(summary["history"]), 4)
            self.assertTrue(checkpoint.is_file())

            payload = torch.load(checkpoint, weights_only=True)
            self.assertEqual(payload["epoch"], 2)
            self.assertAlmostEqual(payload["dev_metrics"]["macro_f1"], 0.6)
            self.assertEqual(payload["seed"], 42)


if __name__ == "__main__":
    unittest.main()
