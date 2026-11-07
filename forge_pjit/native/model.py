# P-JiT 21K support for Forge Neo
# huggingface_guess registration, following the FLUX 3 extension's model.py.

import os

import torch
from huggingface_guess.model_list import BASE, ModelType

from . import detect as detection
from .pixel_space import PixelLatentFormat

# absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part. The
# folder name must hold none of the words Forge Neo keys its model handling on (klein, Wan, PiD, Anima, Krea-2...)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "pjit-21k")
TEXT_ENCODER = "qwen35_4b"


class PJiT(BASE):
    huggingface_repo = CONFIG_DIR

    unet_config = {"image_model": detection.IMAGE_MODEL}

    # rectified flow on a uniform grid, no shift; the DiT multiplies the time by 1000 itself, so the timestep is the sigma
    sampling_settings = {"shift": 1.0, "multiplier": 1.0}

    # Forge sizes a pass by the state's area, and here that is the picture's, 256 times a 16x latent's. FLUX 3's factor
    # (14.6) over 256, for the same work per token, times the 1,280 tokens at 1024px over the 4,096 + text of FLUX 3
    memory_usage_factor = 0.02
    # as Forge's own MOSS-Image lists them: a GPU without bfloat16 (RTX 20 and older) falls back to float32, and the Turing
    # FP16 extension, when installed and enabled, turns that fallback into float16. Listing float16 here would take that
    # choice away from its setting
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    unet_extra_config = {}
    latent_format = PixelLatentFormat

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        # filled by patches._hook_text_encoder_file from the file selected under VAE / Text Encoder
        return {f"{TEXT_ENCODER}.transformer": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLOW


def detect(state_dict: dict, key_prefix: str) -> dict | None:
    return detection.detect(state_dict, key_prefix)
