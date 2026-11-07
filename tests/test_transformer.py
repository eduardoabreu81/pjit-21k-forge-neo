# P-JiT 21K support for Forge Neo
# The DiT, the guidance and the whole sampling loop against upstream's code (tests/reference/pjit_21k, commit in
# native/__init__.py), on tiny random models on the CPU. Under fp32, where upstream's autocast has nothing to do.

import json
import sys
import unittest
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "reference"))

from pjit_21k import config as reference_config
from pjit_21k import model as reference_model
from pjit_21k import noise as reference_noise
from pjit_21k import sampler as reference_sampler

from forge_pjit.native import detect, guidance, pixel_space, scheduler
from forge_pjit.native.transformer import TEXT_LEN, PJiTDiT, rope_table

FIXTURES = Path(__file__).parent / "fixtures"
QUANT_SUFFIXES = (".comfy_quant", ".weight_scale")
TINY = reference_config.PyramidJiTConfig(dim=256, num_heads=2, num_layers=3, ffn_dim=480, refiner_blocks=1,
                                         refiner_ffn_dim=240, adaln_rank=32, text_dim=24, bottleneck_dim=64)


def pair(seed=0, config=TINY):
    torch.manual_seed(seed)
    reference = reference_model.PyramidJiT(config).eval()
    with torch.no_grad():
        for name, param in reference.named_parameters():
            if name.endswith("modulation"):
                param.zero_()
            elif param.ndim > 1:
                param.copy_(torch.randn(param.shape) * 0.05)
    state = reference.state_dict()
    model = PJiTDiT(**detect.detect(state)).eval()
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return reference, model


def caption(length, seed, config=TINY):
    text = torch.randn(length, config.text_dim, generator=torch.Generator().manual_seed(seed))
    return torch.cat([text, text.new_zeros(TEXT_LEN - length, config.text_dim)])[None]


class TestDiT(unittest.TestCase):
    def test_rope_table_matches(self):
        reference, _ = pair()
        expected = reference_model.build_rope_table(reference.freqs, 1, 3, 4, TEXT_LEN)
        cos, sin = rope_table(3, 4, TEXT_LEN, 128, "cpu")
        self.assertTrue(torch.equal(cos, expected.real.float()))
        self.assertTrue(torch.equal(sin, expected.imag.float()))

    def test_image_prediction_matches(self):
        reference, model = pair()
        for height, width in ((64, 96), (70, 100)):  # sides off the 32 grid are padded and cropped back
            x = torch.randn(2, 3, height, width, generator=torch.Generator().manual_seed(1)) * 4
            text = torch.cat([caption(10, 2), caption(40, 3)])
            lens = torch.tensor([10, 40], dtype=torch.int32)
            t = torch.tensor([980, 20])
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = reference(x_t=x.unsqueeze(2), text=text, text_lens=lens, t=t).squeeze(2)
                got = model.predict_image(x, t, text, lens, {})
            self.assertEqual(got.shape, (2, 3, height, width))
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

    def test_forward_is_the_velocity_of_the_scaled_state(self):
        _, model = pair()
        x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(4))
        sigma = torch.tensor([0.5])
        text = caption(12, 5)
        with torch.no_grad():
            velocity = model(x, sigma, context=text, transformer_options={})
            predicted = model.predict_image(x * 4, torch.tensor([500]), text, torch.tensor([12]), {})
        torch.testing.assert_close(x - sigma * velocity, predicted / 4)

    def test_caption_length_is_read_from_the_zeros(self):
        _, model = pair()
        x = torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(6))
        short = caption(12, 7)
        same_with_rows_after = short.clone()
        same_with_rows_after[:, 12:] = 1.0
        with torch.no_grad():
            a = model(x, torch.tensor([0.5]), context=short, transformer_options={})
            b = model(x, torch.tensor([0.5]), context=same_with_rows_after, transformer_options={})
        self.assertFalse(torch.equal(a, b))

    def test_the_released_checkpoint_fits(self):
        header = json.loads((FIXTURES / "pjit-21k.shapes.json").read_text())
        config = detect.detect(header)
        self.assertEqual(config, {
            "image_model": "pjit", "dim": 2944, "bottleneck_dim": 1024, "channels": 3, "num_heads": 23, "num_layers": 22,
            "refiner_blocks": 2, "ffn_hidden": 5376, "refiner_ffn_hidden": 2816, "adaln_rank": 256, "text_dim": 7680,
            "freq_dim": 256})
        with torch.device("meta"):
            model = PJiTDiT(**config)
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        self.assertEqual({k: tuple(v["shape"]) for k, v in header.items()}, expected)
        self.assertEqual(sum(v.numel() for v in model.parameters()), 2184091606)

    def test_the_int8_build_fits(self):
        header = json.loads((FIXTURES / "pjit_21k_int8_convrot.shapes.json").read_text())
        config = detect.detect(header)
        self.assertEqual(config, detect.detect(json.loads((FIXTURES / "pjit-21k.shapes.json").read_text())))
        with torch.device("meta"):
            model = PJiTDiT(**config)
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        weights = {k: tuple(v["shape"]) for k, v in header.items() if not k.endswith(QUANT_SUFFIXES)}
        self.assertEqual(weights, expected)
        # every INT8 layer is a Linear of the port, which Forge's mixed-precision operations replace
        quantized = {k.rsplit(".", 1)[0] for k in header if k.endswith(".comfy_quant")}
        linears = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
        self.assertTrue(quantized and quantized <= linears)

    def test_other_checkpoints_are_not_taken(self):
        self.assertIsNone(detect.detect({"double_blocks.0.img_attn.qkv.weight": {"shape": [1, 1]}}))


class TestGuidance(unittest.TestCase):
    def test_apg_matches_across_steps(self):
        generator = torch.Generator().manual_seed(8)
        momentum = torch.zeros(2, 3, 16, 16)
        for _ in range(4):
            v_cond = torch.randn(2, 3, 16, 16, generator=generator)
            v_uncond = torch.randn(2, 3, 16, 16, generator=generator)
            buffer = reference_sampler.MomentumBuffer(momentum=guidance.MOMENTUM)
            buffer.running_average = momentum.unsqueeze(2).to(torch.bfloat16).float()
            expected = reference_sampler.adaptive_projected_guidance(
                pred_cond=v_cond.unsqueeze(2), pred_other=v_uncond.unsqueeze(2), guidance_scale=15.0,
                momentum_buffer=buffer, eta=guidance.ETA, rescale=guidance.RESCALE).squeeze(2)
            got, momentum = guidance.apg(v_cond, v_uncond, momentum, 15.0)
            self.assertTrue(torch.equal(got, expected))
            self.assertTrue(torch.equal(momentum, buffer.running_average.squeeze(2)))

    def test_defaults_match(self):
        defaults = reference_config.SamplerConfig()
        self.assertEqual(guidance.MOMENTUM, defaults.apg_momentum)
        self.assertEqual(guidance.ETA, defaults.apg_eta)
        self.assertEqual(guidance.RESCALE, defaults.apg_rescale)
        self.assertEqual(pixel_space.NOISE_SCALE, defaults.noise_scale)

    def test_cfg_one_is_the_plain_prediction(self):
        x = torch.randn(1, 3, 8, 8)
        cond = torch.randn(1, 3, 8, 8)
        args = {"input": x, "cond": x - cond, "cond_scale": 1.0, "sigma": torch.tensor([0.5]), "cond_denoised": cond,
                "uncond_denoised": torch.zeros_like(x)}
        self.assertTrue(torch.equal(guidance.Guidance()(args), x - cond))

    def test_momentum_restarts_with_a_new_picture(self):
        guide = guidance.Guidance()
        x = torch.randn(1, 3, 8, 8)
        args = {"input": x, "cond": x * 0, "cond_scale": 15.0, "cond_denoised": x * 0.5, "uncond_denoised": x * 0.25}
        first = guide({**args, "sigma": torch.tensor([1.0])})
        guide({**args, "sigma": torch.tensor([0.5])})
        again = guide({**args, "sigma": torch.tensor([1.0])})
        self.assertTrue(torch.equal(first, again))


class TestSampling(unittest.TestCase):
    def sample_like_forge(self, model, noise, cond, uncond, steps, scale):
        """Forge's flow path: Euler over the scheduler's sigmas, the prediction and the guidance of the engine."""
        guide = guidance.Guidance(torch.bfloat16)
        sigmas = scheduler.sigmas(steps)
        x = noise / pixel_space.NOISE_SCALE
        batch = x.shape[0]
        context = torch.cat([cond, uncond])
        with torch.no_grad():
            for i in range(steps):
                sigma, sigma_next = sigmas[i], sigmas[i + 1]
                t = sigma.expand(batch)
                seen = guidance.round_state(x, torch.bfloat16)
                out = model(torch.cat([seen, seen]), t.repeat(2), context=context, transformer_options={})
                cond_denoised, uncond_denoised = (torch.cat([seen, seen]) - sigma * out).chunk(2)
                result = x - guide({"input": x, "cond": x - cond_denoised, "cond_scale": scale, "sigma": t,
                                    "cond_denoised": cond_denoised, "uncond_denoised": uncond_denoised})
                x = x + (x - result) / sigma * (sigma_next - sigma)
        return pixel_space.decode(x)

    def test_matches_upstream_generate(self):
        reference, model = pair(seed=9)
        cond_text = torch.randn(30, TINY.text_dim, generator=torch.Generator().manual_seed(10))
        uncond_text = torch.randn(9, TINY.text_dim, generator=torch.Generator().manual_seed(11))

        def encoder(prompts):
            return [cond_text if prompts[0] else uncond_text]

        sampler = reference_config.SamplerConfig(height=64, width=96, sampling_steps=12, noise_sm_count=None)
        seeds = [3, 4]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = torch.stack(reference_sampler.generate(reference, encoder, "a prompt", seeds, sampler,
                                                              device="cpu", quiet=True)).squeeze(2)
        noise = reference_noise.StackedRandomGenerator(device="cpu", seeds=seeds, sm_count=None).randn(
            (2, 3, 1, 64, 96), dtype=torch.float32, device="cpu").squeeze(2) * sampler.noise_scale

        def padded(text):
            return torch.cat([text, text.new_zeros(TEXT_LEN - text.shape[0], TINY.text_dim)])[None].repeat(2, 1, 1)

        got = self.sample_like_forge(model, noise, padded(cond_text), padded(uncond_text), 12, sampler.guidance_scale)
        got = ((got + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
        self.assertLessEqual((got.int() - expected.int()).abs().max().item(), 1)

    def test_the_schedule_is_upstream_grid(self):
        self.assertTrue(torch.equal(scheduler.schedule(50), torch.linspace(1.0, 0.0, 51)))


if __name__ == "__main__":
    unittest.main()
