# P-JiT 21K support for Forge Neo
# The text engine: upstream's chat template with thinking off, the 256-token cut, the zero padding and the (256, D) layout.

import importlib
import unittest

import torch

from tests import forge_stubs


class TestTextEngine(unittest.TestCase):
    def setUp(self):
        self.ctx = forge_stubs.installed(forge_stubs.text_processing())
        self.ctx.__enter__()
        self.te = importlib.import_module("forge_pjit.native.text_engine")
        model = torch.nn.Module()
        model.num_layers = 32
        self.engine = self.te.PJiTTextEngine(model, forge_stubs.FakeHFTokenizer())

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_encoder_settings(self):
        kwargs = self.engine.text_encoder.kwargs
        self.assertEqual(kwargs["layer"], [7, 15, 27])
        self.assertEqual(kwargs["special_tokens"], {"pad": 248044})

    def test_template_with_thinking_off(self):
        # chat_template.jinja of Qwen/Qwen3.5-4B with add_generation_prompt and enable_thinking=False
        self.engine(["a cat"])
        text = "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.assertEqual(self.engine.text_encoder.calls[-1], [1000 + ord(c) for c in text])

    def test_whitespace_is_the_empty_prompt(self):
        self.assertEqual(self.engine.token_ids("  "), self.engine.token_ids(""))
        self.assertEqual(self.engine.token_ids(" a cat  "), self.engine.token_ids("a cat"))

    def test_cut_at_256_tokens_template_included(self):
        ids = self.engine.token_ids("x" * 500)
        self.assertEqual(len(ids), 256)
        self.assertEqual(ids[-1], 1000 + ord("x"))

    def test_caption_is_zero_padded_to_256(self):
        cond = self.engine(["a cat"])[0]
        length = len(self.engine.token_ids("a cat"))
        self.assertEqual(cond.shape, (256, forge_stubs.FakeSDClipModel.dim))
        self.assertTrue(cond[length:].eq(0).all())
        self.assertEqual(cond[1:length].ne(0).any(dim=-1).sum().item(), length - 1)

    def test_every_caption_has_the_same_shape(self):
        cond = self.engine(["x", "x", "a longer caption"])
        self.assertEqual({c.shape for c in cond}, {torch.Size((256, forge_stubs.FakeSDClipModel.dim))})
        self.assertEqual(len(self.engine.text_encoder.calls), 2)  # the repeated caption is encoded once

    def test_token_counts(self):
        lengths = [len(self.te.TEMPLATE.format(text)) for text in ("", "ab")]
        self.assertEqual(self.engine.tokenize(["", "ab"]), lengths)
        self.assertEqual(self.engine.tokenize("ab"), lengths[1])


if __name__ == "__main__":
    unittest.main()
