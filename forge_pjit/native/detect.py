# P-JiT 21K support for Forge Neo
# Recognize a P-JiT DiT and read its configuration from tensor shapes.
#
# Works on a loaded state dict or on a safetensors header: shapes are read through shape_of, so no tensor has to be
# materialized.

import re

IMAGE_MODEL = "pjit"
IDENTITY_KEYS = ("patch_embed.proj1.weight", "image_refiner.0.adaln_up.weight", "text_refiner.0.adaln_up.weight",
                 "head.modulation")


def shape_of(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    if isinstance(value, dict):
        return tuple(value["shape"])
    return tuple(value.shape)


def count_blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    found = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    return max(found) + 1 if found else 0


def is_pjit(state_dict, key_prefix=""):
    return all(f"{key_prefix}{k}" in state_dict for k in IDENTITY_KEYS)


def detect(state_dict, key_prefix=""):
    """The PJiTDiT keyword arguments for this checkpoint, or None when it is not a P-JiT DiT."""
    if not is_pjit(state_dict, key_prefix):
        return None

    def shape(key):
        return shape_of(state_dict[f"{key_prefix}{key}"])

    keys = list(state_dict.keys())
    return {
        "image_model": IMAGE_MODEL,
        "dim": shape("patch_embed.proj2.weight")[0],
        "bottleneck_dim": shape("patch_embed.proj1.weight")[0],
        "channels": shape("patch_embed.proj1.weight")[1],
        "num_heads": shape("blocks.0.attn.gate_proj.weight")[0],
        "num_layers": count_blocks(keys, f"{key_prefix}blocks."),
        "refiner_blocks": count_blocks(keys, f"{key_prefix}image_refiner."),
        # w13 carries the SwiGLU value and gate halves; read off its rows, which quantized builds never pack
        "ffn_hidden": shape("blocks.0.ffn.w13.weight")[0] // 2,
        "refiner_ffn_hidden": shape("image_refiner.0.ffn.w13.weight")[0] // 2,
        "adaln_rank": shape("image_adaln.1.weight")[0],
        "text_dim": shape("text_proj.0.weight")[1],
        "freq_dim": shape("time_embed.0.weight")[1],
    }
