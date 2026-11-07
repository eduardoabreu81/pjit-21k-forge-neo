# P-JiT 21K support for Forge Neo
# The P-JiT diffusion engine: text-to-image and image-to-image in pixel space, under adaptive projected guidance.

import torch
from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.modules.k_prediction import PredictionDiscreteFlow
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from . import pixel_space
from .guidance import Guidance, round_state
from .model import TEXT_ENCODER, PJiT
from .text_engine import MAX_LENGTH, PJiTTextEngine


class PixelPrediction(PredictionDiscreteFlow):
    """Forge's flow prediction, from the state the model saw: upstream carries the state in fp32, shows it in bf16 and
    takes the step from what it showed. Forge would subtract the velocity from the fp32 state."""

    state_dtype = None

    def calculate_denoised(self, sigma, model_output, model_input):
        return super().calculate_denoised(sigma, model_output, round_state(model_input, self.state_dtype))


class PJiTEngine(ForgeDiffusionEngine):
    matched_guesses = [PJiT]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={TEXT_ENCODER: huggingface_components["text_encoder"]},
                    tokenizer_dict={TEXT_ENCODER: huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"])
        # the state is the picture: Forge's processing.opt_f reads this when the model loads
        vae.upscale_ratio = 1
        vae.downscale_ratio = 1

        predictor = PixelPrediction(estimated_config)
        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=predictor, config=estimated_config)
        predictor.state_dtype = unet.model.computation_dtype
        unet.set_model_sampler_cfg_function(Guidance(unet.model.computation_dtype))

        self.text_engine_pjit = PJiTTextEngine(text_encoder=getattr(clip.cond_stage_model, TEXT_ENCODER),
                                               tokenizer=getattr(clip.tokenizer, TEXT_ENCODER))

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        dynamic_args.klein = False
        self.is_pjit = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_engine_pjit(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        return self.text_engine_pjit.tokenize(prompt), MAX_LENGTH

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        return pixel_space.encode(x)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        return pixel_space.decode(x)
