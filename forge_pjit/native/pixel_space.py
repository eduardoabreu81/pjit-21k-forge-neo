# P-JiT 21K support for Forge Neo
# P-JiT has no autoencoder: it samples pixels. Forge's pipeline wants a latent and a VAE, so here the latent is the image
# itself over NOISE_SCALE.
#
# Upstream starts from 4 x randn (the noise level the model was trained on at 1024px), with the data in [-1, 1]. Forge's
# flow samplers start from 1 x randn, so the state they carry is upstream's over 4. That is the same trajectory under any
# of Forge's samplers, and 4 is a power of two, so the scale never rounds. The DiT scales back (transformer.py).

import types

import torch

NOISE_SCALE = 4.0


class PixelLatentFormat:
    """What Forge asks of a latent format: the scale, and the factors of its fast RGB preview."""

    latent_channels = 3
    scale_factor = 1.0 / NOISE_SCALE
    shift_factor = 0.0
    # the "latent" is the picture: the preview is the state back in [-1, 1]
    latent_rgb_factors = [[NOISE_SCALE, 0.0, 0.0], [0.0, NOISE_SCALE, 0.0], [0.0, 0.0, NOISE_SCALE]]
    latent_rgb_factors_bias = None
    taesd_decoder_name = None

    def __init__(self):
        self.scale_factor = 1.0 / NOISE_SCALE

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor


class PassthroughAutoencoder(torch.nn.Module):
    """Stands where Forge expects an autoencoder. Nothing goes through it in generation (the engine overrides
    encode_first_stage and decode_first_stage); it holds no weights and takes no memory."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(latent_channels=PixelLatentFormat.latent_channels)
        self.latent_format = PixelLatentFormat()

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def process_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_out(self, latent):
        return self.latent_format.process_out(latent)


def encode(images: torch.Tensor) -> torch.Tensor:
    """Pictures in [-1, 1] to the state."""
    return images / NOISE_SCALE


def decode(state: torch.Tensor) -> torch.Tensor:
    """The state to pictures in [-1, 1]; upstream clamps them just the same when it writes the picture."""
    return (state * NOISE_SCALE).clamp(-1.0, 1.0)
