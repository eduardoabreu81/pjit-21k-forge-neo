# P-JiT 21K support for Forge Neo
# Adaptive projected guidance (Sadat et al., ICLR 2025, arXiv:2410.02416), as upstream's pjit_21k/sampler.py applies it:
# in velocity space, with a momentum buffer across the steps, the difference clipped to a norm of 10, and only its part
# orthogonal to the conditional prediction scaled up. It runs as Forge's sampler_cfg_function, and the CFG Scale slider
# is its guidance scale, 15 by default. The values are upstream's; the model was tuned on them.
#
# No Forge imports: the CPU tests run this against upstream's own code.

import torch

from .pixel_space import NOISE_SCALE

MOMENTUM = -0.75
ETA = 0.0
RESCALE = 10.0


def round_state(x: torch.Tensor, dtype) -> torch.Tensor:
    """The state as the model saw it. Upstream carries it in fp32 and shows it to the model in bf16, and every step works
    from what the model saw; under another compute dtype Forge's cast does not round to bf16 and neither does this."""
    return x.to(torch.bfloat16).float() if dtype == torch.bfloat16 else x


def project(v0: torch.Tensor, v1: torch.Tensor):
    """v0's parts parallel and orthogonal to v1, per sample, in float64."""
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    dims = list(range(1, v1.ndim))
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    return parallel.to(dtype), (v0 - parallel).to(dtype)


def apg(v_cond: torch.Tensor, v_uncond: torch.Tensor, momentum: torch.Tensor, scale: float):
    """The guided velocity and the new momentum buffer, for velocities in pixel units (B, C, H, W)."""
    difference = v_cond - v_uncond + MOMENTUM * momentum.to(torch.bfloat16).float()
    momentum = difference
    norm = difference.norm(p=2, dim=[1, 2, 3], keepdim=True)
    clipped = difference * torch.minimum(torch.ones_like(difference), RESCALE / norm)
    parallel, orthogonal = project(clipped, v_cond)
    return v_cond + scale * (orthogonal + ETA * parallel), momentum


class Guidance:
    """Holds the momentum between steps. A step whose sigma does not fall (a new picture, the hires pass) starts over."""

    def __init__(self, dtype=None):
        self.dtype = dtype
        self.momentum = None
        self.sigma = None

    def __call__(self, args: dict) -> torch.Tensor:
        # Forge's contract: cfg_result = x - returned, on the denoised predictions
        x = args["input"]
        scale = args["cond_scale"]
        if abs(scale - 1.0) < 1e-6:
            # Forge skips the negative pass at CFG 1, so there is nothing to guide with
            return args["cond"]

        sigma = args["sigma"]
        t = sigma.float().view(-1, *([1] * (x.ndim - 1)))
        current = float(sigma.max())
        if self.momentum is None or self.momentum.shape != x.shape or self.sigma is None or current >= self.sigma:
            self.momentum = torch.zeros_like(x, dtype=torch.float32)
        self.sigma = current

        x_seen = round_state(x, self.dtype)
        v_cond = (x_seen - args["cond_denoised"]).float() * NOISE_SCALE / t
        v_uncond = (x_seen - args["uncond_denoised"]).float() * NOISE_SCALE / t
        guided, self.momentum = apg(v_cond, v_uncond, self.momentum.to(x), scale)
        return x - (x_seen - t * guided / NOISE_SCALE)
