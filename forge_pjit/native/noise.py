# P-JiT 21K support for Forge Neo
# The initial noise as an H100 SXM draws it, the GPU the model was trained on, so a seed gives the picture Linum published
# for it. Ported from upstream's pjit_21k/noise.py, which reproduces torch.randn's CUDA kernel in Triton: here it is numpy
# on the CPU, so it needs no Triton and no particular GPU.
#
# torch's kernel runs one Philox4x32-10 stream per thread and writes each stream's normals to every T-th element, T being the
# thread count, which depends on the GPU's SM count: that is why a seed draws different noise on different GPU models.
# The sines and logarithms are exact here and hardware approximations there, a difference far below what a picture shows.
#
# No Forge imports: the CPU tests run it as is.

import math

import numpy as np
import torch

SM_COUNT = 132  # H100 SXM
THREADS_PER_SM = 2048
BLOCK = 256
UNROLL = 4
ROUND_A = 0xD2511F53
ROUND_B = 0xCD9E8D57
KEY_A = 0x9E3779B9
KEY_B = 0xBB67AE85
MASK = 0xFFFFFFFF
# curand_globals.h constants, folded in fp32 as nvcc folds them
INV_2POW32 = float(np.float32(2.3283064e-10))
INV_2POW32_2PI = float(np.float32(np.float32(2.3283064e-10) * np.float32(6.2831855)))
INV_2POW32_HALF = float(np.float32(INV_2POW32) / np.float32(2))
INV_2POW32_2PI_HALF = float(np.float32(INV_2POW32_2PI) / np.float32(2))


def philox(k0, k1, c0, c1, c2, c3, rounds=10):
    """Philox4x32: uint32 values in uint64 arrays in, the four output words out."""
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint64) for c in (c0, c1, c2, c3))
    round_a, round_b, shift, mask = (np.uint64(v) for v in (ROUND_A, ROUND_B, 32, MASK))
    for _ in range(rounds):
        product_a = c0 * round_a
        product_b = c2 * round_b
        c0, c1, c2, c3 = ((product_b >> shift) ^ c1 ^ np.uint64(k0), product_b & mask,
                          (product_a >> shift) ^ c3 ^ np.uint64(k1), product_a & mask)
        k0 = (k0 + KEY_A) & MASK
        k1 = (k1 + KEY_B) & MASK
    return c0, c1, c2, c3


def box_muller(x, y):
    """curand's Box-Muller, device branch: the 32-bit words to two normals."""
    u = (x.astype(np.float32).astype(np.float64) * INV_2POW32 + INV_2POW32_HALF).astype(np.float32)
    v = (y.astype(np.float32).astype(np.float64) * INV_2POW32_2PI + INV_2POW32_2PI_HALF).astype(np.float32)
    s = np.sqrt(-2.0 * np.log(u.astype(np.float64)).astype(np.float32))
    return np.sin(v.astype(np.float64)).astype(np.float32) * s, np.cos(v.astype(np.float64)).astype(np.float32) * s


def randn(seed: int, shape, sm_count: int = SM_COUNT) -> torch.Tensor:
    """torch.randn(shape) on a GPU with sm_count SMs and a generator freshly seeded with seed; float32 on the CPU."""
    seed = int(seed) % (1 << 32)
    numel = math.prod(shape)
    grid = min(math.ceil(numel / BLOCK), THREADS_PER_SM // BLOCK * sm_count)
    threads = BLOCK * grid
    iterations = (numel - 1) // (threads * UNROLL) + 1

    out = np.empty(numel, dtype=np.float32)
    streams = np.arange(threads, dtype=np.uint64)
    zero = np.zeros(threads, dtype=np.uint64)
    for iteration in range(iterations):
        r0, r1, r2, r3 = philox(seed, 0, zero + np.uint64(iteration), zero, streams, zero)
        normals = (*box_muller(r0, r1), *box_muller(r2, r3))
        base = streams.astype(np.int64) + threads * UNROLL * iteration
        for index, values in enumerate(normals):
            where = base + index * threads
            live = where < numel
            out[where[live]] = values[live]
    return torch.from_numpy(out).reshape(tuple(shape))
