# P-JiT 21K support for Forge Neo
# Checks that the installed Forge Neo has every piece the P-JiT backend builds on, before anything is patched. Same
# mechanism as the MOSS-Image extension's compat.py.

import importlib
import inspect

# what the backend imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.args": ("dynamic_args",),
    "backend.attention": ("attention_function",),
    "backend.memory_management": ("load_model_gpu", "cast_to"),
    "backend.loader": ("load_huggingface_component", "possible_models", "replace_state_dict"),
    "backend.modules.k_prediction": ("PredictionDiscreteFlow",),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.nn.llm.llama": ("Qwen3_4B",),
    "backend.operations": ("ForgeOperations",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine", "ForgeObjects"),
    "backend.patcher.clip": ("CLIP",),
    "backend.patcher.unet": ("UnetPatcher",),
    "backend.patcher.vae": ("VAE",),
    "backend.text_processing._comfy": ("SDClipModel", "SDTokenizer", "INF"),
    "backend.text_processing.emphasis": ("EmphasisNone", "uses_emphasis"),
    "modules.rng": ("ImageRNG",),
    "modules.sd_samplers_common": ("samples_to_images_tensor",),
    "modules.sd_schedulers": ("Scheduler", "all_schedulers", "schedulers", "schedulers_map"),
    "modules_forge.presets": ("PresetArch", "SAMPLERS", "SCHEDULERS", "STEPS", "CFG", "register"),
    "huggingface_guess.detection": ("detect_unet_config",),
    "huggingface_guess.model_list": ("models", "BASE", "ModelType"),
}

# the transformer goes through the Krea 2 branch of the loader, the text encoder through the Qwen3 one (which must still
# pick its Qwen3-4B class by name, inside the call)
LOADER_BRANCHES = ("Krea2Transformer2DModel", "Qwen3Model")
LOADER_SOURCE = ("from backend.nn.llm.llama import Qwen3_4B", "from backend.nn.krea import SingleStreamDiT")


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        from backend import loader, operations

        problems += loader_problems(loader, operations)

    return problems


def loader_problems(loader, operations) -> list[str]:
    problems: list[str] = []

    # the whole module, not load_huggingface_component: another extension that loaded first has already put its own
    # wrapper there, and a wrapper's source holds none of Forge's branches
    source = inspect.getsource(loader)
    missing = [b for b in LOADER_BRANCHES if f'"{b}"' not in source]
    if missing:
        problems.append(f"the model loader has no {', '.join(missing)} branch")
    missing = [s for s in LOADER_SOURCE if s not in source]
    if missing:
        problems.append(f"the model loader no longer does {'; '.join(missing)}")
    # the Qwen3.5 norms are Forge's RMSNorm(add=True), the 1 + weight scale its Qwen3-VL uses
    if "add" not in inspect.signature(operations.ForgeOperations.RMSNorm.__init__).parameters:
        problems.append("backend.operations.RMSNorm has no add argument")

    return problems


def report(problems: list[str]) -> None:
    print("[P-JiT] this Forge Neo is too old or too new for the extension, so it stays disabled "
          "(Forge Neo itself keeps working as usual):")
    for problem in problems:
        print(f"          {problem}")
