import unittest

import torch

from vnsentiment.backbones.backbone import PhoBertBackbone
from vnsentiment.backbones.batch import MAX_LENGTH, build_batch
from vnsentiment.backbones.encoder import PhoBertEncoder
from tests.helpers.fake_encoder import FakeEncoder
from tests.helpers.fake_tokenizer import FakeTokenizer

SIMPLE_PIECES = {
    "sản_phẩm": [10, 11],
    "này": [12],
    "không": [13, 14],
    "tệ": [15],
}


class BuildBatchTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer(SIMPLE_PIECES)

    def test_adds_special_tokens_and_masks(self):
        batch = build_batch(self.tokenizer, ["sản_phẩm này"])
        self.assertEqual(batch.input_ids.tolist(), [[0, 10, 11, 12, 2]])
        self.assertEqual(batch.attention_mask.tolist(), [[1, 1, 1, 1, 1]])
        self.assertEqual(batch.content_mask.tolist(), [[0, 1, 1, 1, 0]])
        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, 0, 1, -1]])

    def test_pads_shorter_sequence(self):
        batch = build_batch(self.tokenizer, ["sản_phẩm", "này"])
        self.assertEqual(batch.input_ids.tolist(), [[0, 10, 11, 2], [0, 12, 2, 1]])
        self.assertEqual(batch.attention_mask.tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(batch.content_mask.tolist(), [[0, 1, 1, 0], [0, 1, 0, 0]])
        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, 0, -1], [-1, 0, -1, -1]])

    def test_word_ids_share_subwords(self):
        batch = build_batch(self.tokenizer, ["không"])
        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, 0, -1]])
        self.assertEqual(batch.content_mask.tolist(), [[0, 1, 1, 0]])

    def test_punctuation_is_content(self):
        tokenizer = FakeTokenizer({".": [16]})
        batch = build_batch(tokenizer, ["."])
        self.assertEqual(batch.content_mask.tolist(), [[0, 1, 0]])
        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, -1]])

    def test_minimum_max_length_keeps_one_subword(self):
        batch = build_batch(self.tokenizer, ["này"], max_length=3)
        self.assertEqual(batch.input_ids.tolist(), [[0, 12, 2]])
        self.assertEqual(batch.content_mask.tolist(), [[0, 1, 0]])

    def test_uses_tokenizer_special_ids(self):
        tokenizer = FakeTokenizer({"a": [10]})
        tokenizer.bos_token_id = 100
        tokenizer.eos_token_id = 200
        tokenizer.pad_token_id = 300
        batch = build_batch(tokenizer, ["a"])
        self.assertEqual(batch.input_ids.tolist(), [[100, 10, 200]])

    def test_truncates_whole_word_and_stops(self):
        tokenizer = FakeTokenizer(
            {"aa": [10, 11], "bbb": [12, 13, 14], "c": [15]}
        )
        batch = build_batch(tokenizer, ["aa bbb c"], max_length=6)
        self.assertEqual(batch.input_ids.tolist(), [[0, 10, 11, 2]])
        self.assertEqual(batch.content_mask.tolist(), [[0, 1, 1, 0]])
        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, 0, -1]])

    def test_raises_when_no_content_fits(self):
        tokenizer = FakeTokenizer({"long": list(range(100, 400))})
        with self.assertRaises(ValueError):
            build_batch(tokenizer, ["long"])

    def test_rejects_empty_texts(self):
        for text in ("", "   \n"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    build_batch(self.tokenizer, [text])

    def test_rejects_single_string_input(self):
        with self.assertRaises(TypeError):
            build_batch(self.tokenizer, "sản_phẩm")

    def test_rejects_non_string_text(self):
        with self.assertRaises(TypeError):
            build_batch(self.tokenizer, [None])  # type: ignore[list-item]

    def test_rejects_max_length_out_of_bounds(self):
        for max_length in (2, MAX_LENGTH + 1):
            with self.subTest(max_length=max_length):
                with self.assertRaises(ValueError):
                    build_batch(self.tokenizer, ["sản_phẩm"], max_length=max_length)

    def test_rejects_literal_special_tokens(self):
        for word, token_id in (("<s>", 0), ("</s>", 2), ("<pad>", 1)):
            with self.subTest(word=word):
                tokenizer = FakeTokenizer({word: [token_id]})
                with self.assertRaises(ValueError):
                    build_batch(tokenizer, [word])

    def test_rejects_special_token_mid_sentence(self):
        tokenizer = FakeTokenizer({"a": [10], "<s>": [0], "b": [11]})
        with self.assertRaises(ValueError):
            build_batch(tokenizer, ["a <s> b"])

    def test_allows_unk_as_content(self):
        tokenizer = FakeTokenizer({"<unk>": [3]})
        batch = build_batch(tokenizer, ["<unk>"])
        self.assertEqual(batch.content_mask.tolist(), [[0, 1, 0]])
        self.assertEqual(batch.word_ids.tolist(), [[-1, 0, -1]])

    def test_returns_long_tensors(self):
        batch = build_batch(self.tokenizer, ["sản_phẩm"])
        self.assertEqual(batch.input_ids.dtype, torch.long)
        self.assertEqual(batch.attention_mask.dtype, torch.long)
        self.assertEqual(batch.content_mask.dtype, torch.long)
        self.assertEqual(batch.word_ids.dtype, torch.long)

    def test_default_max_length_is_256(self):
        self.assertEqual(MAX_LENGTH, 256)


class BackboneTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer(SIMPLE_PIECES)
        self.encoder = PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu")
        self.backbone = PhoBertBackbone(self.tokenizer, self.encoder)

    def test_encode_returns_batch_and_hidden(self):
        batch, hidden = self.backbone.encode(["sản_phẩm này"])
        self.assertEqual(batch.input_ids.shape, (1, 5))
        self.assertEqual(tuple(hidden.shape), (1, 5, 4))

    def test_encode_handles_multiple_texts(self):
        batch, hidden = self.backbone.encode(["sản_phẩm này", "này"])
        self.assertEqual(batch.input_ids.shape, (2, 5))
        self.assertEqual(tuple(hidden.shape), (2, 5, 4))

    def test_encode_runs_encoder_forward_hooks(self):
        calls = []
        handle = self.encoder.register_forward_hook(
            lambda _module, inputs, output: calls.append((inputs[0], output))
        )
        try:
            batch, hidden = self.backbone.encode(["sản_phẩm này"])
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], batch)
        self.assertIs(calls[0][1], hidden)

    def test_encoder_wrapper_uses_hidden_size(self):
        encoder = PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu")
        batch = build_batch(self.tokenizer, ["sản_phẩm"])
        hidden = encoder.forward(batch)
        self.assertEqual(encoder.hidden_size, 4)
        self.assertEqual(tuple(hidden.shape), (1, 4, 4))

    def test_backbone_registers_encoder_parameters(self):
        backbone = PhoBertBackbone(
            self.tokenizer, PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu")
        )
        self.assertIsInstance(backbone, torch.nn.Module)
        self.assertTrue(list(backbone.parameters()))
        self.assertIn("encoder.model.scale", backbone.state_dict())

    def test_train_eval_propagate_to_encoder(self):
        backbone = PhoBertBackbone(
            self.tokenizer, PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu")
        )
        backbone.train()
        self.assertTrue(backbone.encoder.training)
        self.assertTrue(backbone.encoder.model.training)
        backbone.eval()
        self.assertFalse(backbone.encoder.training)
        self.assertFalse(backbone.encoder.model.training)

    def test_backward_updates_encoder_parameter(self):
        backbone = PhoBertBackbone(
            self.tokenizer, PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu")
        )
        backbone.zero_grad()
        _, hidden = backbone.encode(["sản_phẩm này"])
        hidden.sum().backward()
        scale = backbone.encoder.model.scale
        self.assertIsNotNone(scale.grad)
        before = scale.detach().clone()
        optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)
        optimizer.step()
        self.assertFalse(torch.equal(before, scale.detach()))

    def test_encoder_derives_device_from_model_after_to(self):
        encoder = PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu")
        encoder.device = torch.device("meta")
        batch = build_batch(self.tokenizer, ["sản_phẩm"])
        hidden = encoder.forward(batch)
        self.assertEqual(hidden.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
