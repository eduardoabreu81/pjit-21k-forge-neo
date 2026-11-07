# P-JiT 21K support for Forge Neo
# The few Forge Neo runtime pieces the DiT calls, with plain torch fallbacks outside Forge (CPU tests).
# Same module as the FLUX 3 extension's kernels.py, with the attention mask P-JiT's padded caption needs.

import torch

try:
    from backend import attention as _forge_attention
    from backend.memory_management import cast_to as _forge_cast_to
except ImportError:  # outside Forge Neo: eager backend on CPU
    _forge_attention = None
    _forge_cast_to = None

__all__ = ["attention", "cast_to"]


def cast_to(weight, dtype=None, device=None):
    if _forge_cast_to is not None:
        return _forge_cast_to(weight, dtype=dtype, device=device)
    return weight.to(dtype=dtype, device=device)


def attention(q, k, v, mask=None, transformer_options=None):
    """q, k, v: [B, H, S, D] -> [B, S, H * D]; mask: boolean [B, 1, S, S], True where a row may attend, or None.

    transformer_options reach Forge's attention wrapper, which applies attention overrides such as the Sparse Attention
    Integrated extension.
    """
    if _forge_attention is not None:
        # looked up per call: Forge picks its attention function at import time and other extensions may wrap it
        return _forge_attention.attention_function(q, k, v, q.shape[1], mask=mask, skip_reshape=True,
                                                   transformer_options=transformer_options or {})
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
