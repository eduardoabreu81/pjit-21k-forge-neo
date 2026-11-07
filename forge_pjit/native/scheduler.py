# P-JiT 21K support for Forge Neo
# The "P-JiT" scheduler: upstream's uniform schedule, linspace(1, 0, steps + 1), so the sigmas are exactly the ones
# upstream samples at. With 50 steps Forge's Simple lands on the same grid, up to the last bit of each sigma, and the DiT
# truncates sigma * 1000, so that last bit can move a step to the timestep next to it.

import torch

NAME = "pjit"
LABEL = "P-JiT"
ALIASES = ("PJiT", "pjit-21k")


def schedule(n: int) -> torch.Tensor:
    return torch.linspace(1.0, 0.0, n + 1)


def sigmas(n, sigma_min=None, sigma_max=None, inner_model=None, device="cpu", **kwargs):
    return schedule(n).to(device)
