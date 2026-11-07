# P-JiT 21K support for Forge Neo
# What the loader hook checks and cleans in the selected files before Forge builds the components. No Forge imports: the
# CPU tests run it on the real checkpoints' headers.

# the Qwen3.5-4B language tower: 32 layers, hidden 2560, a Gated DeltaNet in layer 0 and full attention in layer 3.
# Comfy-Org's qwen3.5_4b_bf16 and PedroMarinhoDev's qwen3.5_4b_int8_convrot store it as model.language_model.* (with the
# vision tower and the multi-token prediction head beside it), transformers' Qwen3_5Model as language_model.*, a
# text-only export as model.*
TE_PREFIXES = ("model.language_model.", "language_model.", "model.")
TE_KEYS = ("layers.0.linear_attn.A_log", "layers.3.self_attn.q_norm.weight", "norm.weight")
TE_NORM = "norm.weight"
TE_HIDDEN = 2560
MISSING_TE = ("P-JiT 21K needs its Qwen3.5 4B text encoder: select qwen3.5_4b_int8_convrot.safetensors or "
              "qwen3.5_4b_bf16.safetensors under VAE / Text Encoder")
WRONG_TE = ("P-JiT 21K needs the Qwen3.5 4B text encoder, and the Qwen3.5 selected under VAE / Text Encoder is another "
            "size (the 9B of MOSS-Image will not do): select qwen3.5_4b_int8_convrot.safetensors or "
            "qwen3.5_4b_bf16.safetensors")


def shape_of(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    if isinstance(value, dict):
        return tuple(value["shape"])
    return tuple(value.shape)


def text_encoder_prefix(state_dict) -> str:
    """The prefix of the Qwen3.5 language tower's keys, or the user-facing error when the file is not one."""
    if not isinstance(state_dict, dict) or len(state_dict) <= 16:
        raise ValueError(MISSING_TE)
    prefix = next((p for p in TE_PREFIXES if all(f"{p}{k}" in state_dict for k in TE_KEYS)), None)
    if prefix is None:
        raise ValueError(MISSING_TE)
    if shape_of(state_dict[f"{prefix}{TE_NORM}"])[0] != TE_HIDDEN:
        raise ValueError(WRONG_TE)
    return prefix


def prepare_text_encoder(state_dict: dict) -> dict:
    """The language tower under model.* (Qwen35TextModel's names), every other part of the file left out. Keys move
    out of the given dict, so the vision tower and the heads are not kept alive by it."""
    prefix = text_encoder_prefix(state_dict)
    keys = [k for k in state_dict if k.startswith(prefix) and not k.startswith(f"{prefix}visual.")]
    return {f"model.{k[len(prefix):]}": state_dict.pop(k) for k in keys}
