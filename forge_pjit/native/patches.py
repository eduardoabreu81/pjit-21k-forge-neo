# P-JiT 21K support for Forge Neo
# Runtime hooks into Forge Neo's loader and preview. Every hook hands anything that is not P-JiT to the original function
# untouched; Forge Neo itself stays unchanged. Follows the MOSS-Image extension's patches.py.

import functools
import logging

import torch
from backend import loader
from backend.nn import krea
from huggingface_guess import detection, model_list

from . import model, text_encoder
from .checkpoints import text_encoder_prefix
from .engine import PJiTEngine
from .pixel_space import PassthroughAutoencoder
from .transformer import PJiTDiT

logger = logging.getLogger("forge_pjit")

_applied = False


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = model.detect(state_dict, key_prefix)
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_components() -> None:
    original = loader.load_huggingface_component

    # wraps leaves __wrapped__ behind, which inspect.getsource follows: the compat checks of the other extensions that
    # read Forge's loader source still reach it through this hook
    @functools.wraps(original)
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.PJiT):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "PJiTTransformer":
            # the Krea 2 branch is Forge's generic DiT load: dtype, quantization and device handling; it imports
            # backend.nn.krea.SingleStreamDiT inside the function, so the swap routes the build to us
            original_krea = krea.SingleStreamDiT
            krea.SingleStreamDiT = PJiTDiT
            try:
                return original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            finally:
                krea.SingleStreamDiT = original_krea
        if cls_name == text_encoder.LOADER_CLASS:
            return text_encoder.load(original, guess, component_name, lib_name, repo_path, state_dict)
        if cls_name == "PJiTPixelSpace":
            return PassthroughAutoencoder()

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _hook_text_encoder_file() -> None:
    # Forge sorts the files selected under VAE / Text Encoder by their keys and knows no Qwen3.5-4B: the file goes where
    # the P-JiT guess looks for its text encoder (model.clip_target)
    original = loader.replace_state_dict

    def replace_state_dict(sd, asd, guess, path):
        if isinstance(guess, model.PJiT):
            try:
                text_encoder_prefix(asd)
            except ValueError:
                pass
            else:
                prefix = f"{guess.text_encoder_key_prefix[0]}{model.TEXT_ENCODER}.transformer."
                for key in [k for k in sd if k.startswith(prefix)]:
                    del sd[key]
                sd.update({f"{prefix}{k}": v for k, v in asd.items()})
                return sd
        return original(sd, asd, guess, path)

    loader.replace_state_dict = replace_state_dict


def _hook_preview() -> None:
    # Approx NN and TAESD decode a latent, and there is none: the state is the picture, under every preview setting
    from modules import sd_samplers_common, shared

    original = sd_samplers_common.samples_to_images_tensor

    def samples_to_images_tensor(sample, approximation=None, model=None):
        sd_model = model or shared.sd_model
        if getattr(sd_model, "is_pjit", False):
            return sd_model.decode_first_stage(sample)
        return original(sample, approximation, model)

    sd_samplers_common.samples_to_images_tensor = samples_to_images_tensor


def _register_scheduler() -> None:
    from modules import sd_schedulers, shared

    from . import scheduler

    if scheduler.LABEL in sd_schedulers.schedulers_map:
        return
    entry = sd_schedulers.Scheduler(scheduler.NAME, scheduler.LABEL, scheduler.sigmas, need_inner_model=True,
                                    aliases=list(scheduler.ALIASES))
    sd_schedulers.all_schedulers.append(entry)
    if entry.label not in shared.opts.hide_schedulers:
        sd_schedulers.schedulers.append(entry)
    sd_schedulers.schedulers_map.update({entry.name: entry, entry.label: entry,
                                         **{alias: entry for alias in (entry.aliases or [])}})


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.PJiT not in model_list.models:
        model_list.models.append(model.PJiT)
    if PJiTEngine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, PJiTEngine)
    _hook_components()
    _hook_text_encoder_file()
    _hook_preview()
    try:
        _register_scheduler()
    except Exception as e:
        # the model still works: Simple gives the same grid at 50 steps
        logger.warning(f"[P-JiT] could not add the P-JiT scheduler: {e}")
    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works: Euler, P-JiT, 50 steps, CFG 15 by hand
        logger.warning(f"[P-JiT] could not add the pjit UI preset: {e}")

    _applied = True
    print(f"[P-JiT] native backend enabled (torch {torch.__version__})")
