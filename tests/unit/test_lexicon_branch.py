import json
import unittest
from pathlib import Path

import torch

from vnsentiment.phobert_branch.batch import build_batch
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.projection import LexiconProjection
from tests.helpers.fake_tokenizer import FakeTokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_fixture.json"
PIECES = {
    "tốt": [10, 11],
    "này": [12],
    "tệ": [15],
}


class _EchoProjection(torch.nn.Module):
    def forward(self, values, mask, presence):
        return values


class _IntegerBufferProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

    def forward(self, values, mask, presence):
        self.counter += 1
        return values


class LexiconBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LexiconStore(json.loads(FIXTURE.read_text(encoding="utf-8")))

    def setUp(self):
        self.tokenizer = FakeTokenizer(PIECES)

    def test_default_projection_matches_phobert_base_hidden_size(self):
        text = "tốt này"
        batch = build_batch(self.tokenizer, [text])
        branch = LexiconBranch(self.store)
        with torch.no_grad():
            projected = branch([text], batch)
        self.assertEqual(projected.shape, (1, 5, 768))

    def test_injected_projection_is_used(self):
        text = "tốt này"
        batch = build_batch(self.tokenizer, [text])
        projection = LexiconProjection(hidden_dim=16)
        branch = LexiconBranch(self.store, projection)
        with torch.no_grad():
            features = branch.features([text], batch)
            projected = branch([text], batch)
            expected = projection(
                features.values, features.mask, features.presence
            )
        self.assertIs(branch.projection, projection)
        self.assertTrue(torch.equal(projected, expected))

    def test_oov_positions_are_exactly_zero_with_nonzero_bias(self):
        text = "xyz_oov_test"
        tokenizer = FakeTokenizer({text: [99]})
        batch = build_batch(tokenizer, [text])
        projection = LexiconProjection(hidden_dim=16)
        with torch.no_grad():
            projection.bias.fill_(1.0)
        branch = LexiconBranch(self.store, projection)
        with torch.no_grad():
            projected = branch([text], batch)
        self.assertTrue(torch.equal(projected, torch.zeros(1, 3, 16)))

    def test_present_tokens_are_projected(self):
        text = "tốt"
        batch = build_batch(self.tokenizer, [text])
        projection = LexiconProjection(hidden_dim=16)
        with torch.no_grad():
            projection.bias.fill_(1.0)
        branch = LexiconBranch(self.store, projection)
        with torch.no_grad():
            projected = branch([text], batch)
        self.assertFalse(torch.equal(projected[0, 1], torch.zeros(16)))
        self.assertTrue(torch.equal(projected[0, 0], torch.zeros(16)))

    def test_features_exposes_aligned_twelve_dimensions(self):
        text = "tốt này"
        batch = build_batch(self.tokenizer, [text])
        branch = LexiconBranch(self.store)
        features = branch.features([text], batch)
        self.assertEqual(features.values.shape, (1, 5, 12))
        self.assertEqual(features.mask.shape, (1, 5, 12))
        self.assertEqual(features.presence.shape, (1, 5, 1))

    def test_projection_parameters_are_registered(self):
        branch = LexiconBranch(self.store)
        self.assertIn("projection.value_proj.weight", branch.state_dict())
        self.assertTrue(list(branch.parameters()))

    def test_gradients_reach_projection(self):
        text = "tốt"
        batch = build_batch(self.tokenizer, [text])
        branch = LexiconBranch(self.store, LexiconProjection(hidden_dim=16))
        branch.zero_grad()
        branch([text], batch).sum().backward()
        self.assertIsNotNone(branch.projection.value_proj.weight.grad)
        self.assertGreater(
            branch.projection.value_proj.weight.grad.abs().sum().item(), 0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_forward_uses_projection_device(self):
        text = "tốt"
        batch = build_batch(self.tokenizer, [text])
        branch = LexiconBranch(self.store, LexiconProjection(hidden_dim=16)).cuda()
        with torch.no_grad():
            projected = branch([text], batch)
        self.assertEqual(projected.device.type, "cuda")

    def test_forward_casts_features_to_projection_dtype(self):
        text = "tốt"
        batch = build_batch(self.tokenizer, [text])
        branch = LexiconBranch(
            self.store, LexiconProjection(hidden_dim=16)
        ).to(dtype=torch.bfloat16)
        with torch.no_grad():
            projected = branch([text], batch)
        self.assertEqual(projected.dtype, torch.bfloat16)

    def test_parameterless_projection_falls_back_to_feature_placement(self):
        text = "tốt"
        batch = build_batch(self.tokenizer, [text])
        branch = LexiconBranch(self.store, _EchoProjection())
        with torch.no_grad():
            projected = branch([text], batch)
        self.assertEqual(projected.shape, (1, 4, 12))
        self.assertEqual(projected.dtype, torch.float32)

    def test_integer_buffer_does_not_truncate_float_features(self):
        text = "tốt"
        batch = build_batch(self.tokenizer, [text])
        branch = LexiconBranch(self.store, _IntegerBufferProjection())
        with torch.no_grad():
            projected = branch([text], batch)
            features = branch.features([text], batch)

        self.assertEqual(projected.dtype, torch.float32)
        self.assertTrue(torch.equal(projected, features.values))
        self.assertAlmostEqual(projected[0, 1, 0].item(), 0.9375)

    def test_rejects_text_batch_mismatch(self):
        batch = build_batch(self.tokenizer, ["tốt này", "tệ"])
        branch = LexiconBranch(self.store)
        with self.assertRaises(ValueError):
            branch(["tốt"], batch)


if __name__ == "__main__":
    unittest.main()