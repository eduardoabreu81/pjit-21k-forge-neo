# P-JiT 21K support for Forge Neo
# The Qwen3.5 text encoder port against transformers' text decoder (tests/reference/qwen35_hf.py), its configuration
# against the released one, and its parameter names and shapes against the real Comfy-Org text encoder file (header in
# tests/fixtures).

import json
import unittest
from pathlib import Path

import torch

from forge_pjit.native import checkpoints, qwen35
from tests import forge_stubs
from tests.reference import qwen35_hf

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
EMBEDDED_CONFIG = ROOT / "forge_pjit" / "native" / "huggingface" / "pjit-21k" / "text_encoder" / "config.json"
# Qwen/Qwen3.5-4B config.json, the parts that shape the language tower
RELEASED_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
    "text_config": {
        "attn_output_gate": True, "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
        "hidden_size": 2560, "intermediate_size": 9216, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4, "rms_norm_eps": 1e-06,
        "vocab_size": 248320,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000, "rope_type": "default"},
    },
}

# small enough for the CPU, large enough for every path: grouped heads in both attentions, a full-attention layer,
# partial rotary, several delta-rule chunks
TINY = {
    "vocab_size": 300, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 8,
    "full_attention_interval": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "partial_rotary_factor": 0.25, "rope_theta": 10000.0, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
    "linear_key_head_dim": 16, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6,
}


def reference_config(config):
    layer_types = ["full_attention" if (i + 1) % config["full_attention_interval"] == 0 else "linear_attention"
                   for i in range(config["num_hidden_layers"])]
    return {**config, "layer_types": layer_types, "mrope_section": [2, 1, 1]}


def randomized(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("A_log"):
                param.copy_(torch.empty(param.shape).uniform_(0.01, 16, generator=generator).log())
            elif "norm" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.2)
            else:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.15)
    return model


def built(config):
    with forge_stubs.forge_operations():
        return qwen35.Qwen35TextModel(config).eval()


class TestQwen35(unittest.TestCase):
    def setUp(self):
        self.reference = randomized(qwen35_hf.Qwen3_5TextModel(reference_config(TINY)), seed=0).eval()
        self.model = built(TINY)
        missing, unexpected = self.model.load_state_dict(
            {f"model.{k}": v for k, v in self.reference.state_dict().items()}, strict=False)
        self.assertEqual((missing, unexpected), ([], []))
        self.ids = torch.randint(0, TINY["vocab_size"], (1, 150), generator=torch.Generator().manual_seed(1))

    def test_last_state_matches_transformers(self):
        with torch.no_grad():
            expected = self.reference(self.ids)
            got, tapped = self.model(None, embeds=self.model.get_input_embeddings()(self.ids).float(),
                                     dtype=torch.float32)
        self.assertIsNone(tapped)
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)

    def test_tapped_layers_match_transformers(self):
        # upstream reads hidden_states[i + 1], the output of block i; layers 3 and 7 are full attention here, as 7, 15
        # and 27 are in the 4B
        captured = {}
        for index in (3, 7):
            self.reference.layers[index].register_forward_hook(
                lambda module, args, output, index=index: captured.__setitem__(
                    index, output[0] if isinstance(output, tuple) else output))
        with torch.no_grad():
            self.reference(self.ids)
            _, tapped = self.model(None, embeds=self.model.get_input_embeddings()(self.ids).float(),
                                   intermediate_output=[3, 7], dtype=torch.float32)
        self.assertEqual(tapped.shape, (1, 150, 2 * TINY["hidden_size"]))
        torch.testing.assert_close(tapped, torch.cat([captured[3], captured[7]], dim=-1), rtol=1e-4, atol=1e-4)

    def test_configuration(self):
        released = qwen35.Qwen35Config.from_dict(RELEASED_CONFIG)
        released_rope = RELEASED_CONFIG["text_config"]["rope_parameters"]
        self.assertEqual(released_rope["partial_rotary_factor"], released.partial_rotary_factor)
        embedded = qwen35.Qwen35Config.from_dict(json.loads(EMBEDDED_CONFIG.read_text()))
        self.assertEqual(released, qwen35.Qwen35Config())
        self.assertEqual(embedded, qwen35.Qwen35Config())
        # Forge's Qwen3 loader branch picks its 4B class (the one swapped for the port) on this hidden size
        self.assertEqual(json.loads(EMBEDDED_CONFIG.read_text())["hidden_size"], 2560)

    def test_the_layers_upstream_reads_are_full_attention(self):
        config = qwen35.Qwen35Config()
        for layer in (7, 15, 27):
            self.assertTrue(config.is_full_attention(layer))


class TestRealCheckpoint(unittest.TestCase):
    def setUp(self):
        self.header = json.loads((FIXTURES / "qwen3.5_4b_bf16.shapes.json").read_text())

    def test_weights_match_the_model(self):
        prepared = checkpoints.prepare_text_encoder(dict(self.header))
        with torch.device("meta"):
            model = built({})
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        self.assertEqual({k: tuple(v["shape"]) for k, v in prepared.items()}, expected)

    def test_the_int8_build_fits(self):
        header = json.loads((FIXTURES / "qwen3.5_4b_int8_convrot.shapes.json").read_text())
        prepared = checkpoints.prepare_text_encoder(header)
        with torch.device("meta"):
            model = built({})
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        weights = {k: tuple(v["shape"]) for k, v in prepared.items() if not k.endswith((".comfy_quant", ".weight_scale"))}
        self.assertEqual(weights, expected)

    def test_other_parts_are_left_out(self):
        state_dict = dict(self.header)
        prepared = checkpoints.prepare_text_encoder(state_dict)
        self.assertTrue(all(k.startswith("model.") and ".visual." not in k for k in prepared))
        self.assertFalse(any(k.startswith("model.language_model.") for k in state_dict))
        self.assertTrue(any(k.startswith("model.visual.") for k in state_dict))
        self.assertTrue(any(k.startswith("mtp.") for k in state_dict))

    def test_another_size_is_refused_by_name(self):
        wrong = dict(self.header)
        wrong["model.language_model.norm.weight"] = {"shape": [4096]}
        with self.assertRaisesRegex(ValueError, "another size"):
            checkpoints.text_encoder_prefix(wrong)
        with self.assertRaisesRegex(ValueError, "qwen3.5_4b_bf16"):
            checkpoints.text_encoder_prefix({})


if __name__ == "__main__":
    unittest.main()
