# P-JiT 21K support for Forge Neo
# The P-JiT DiT, ported from Linum-AI/pjit-21k (pjit_21k/model.py, see PJIT_COMMIT in native/__init__.py): one
# single-stream DiT that reads the noisy image as 32x32-pixel patches, next to the caption, and predicts the clean image.
#
# Layers are plain torch modules: inside Forge's using_forge_operations they become Forge operations (dtype, offload,
# quantization), outside it they stay torch modules for the CPU tests. Upstream runs fp32 weights under bf16 autocast;
# Forge has no autocast, so what autocast does is written out: linear layers and attention take their inputs in the
# compute dtype, while the norms and the residual stream stay in fp32, as they end up under upstream's autocast.
#
# The module's interface is Forge's flow prediction, not upstream's: the state is the image over NOISE_SCALE, because
# Forge draws unit noise and upstream's starts at 4 (a power of two, so the scale never rounds), and the output is the
# velocity of that state. See pixel_space.py.

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kernels
from .pixel_space import NOISE_SCALE

PATCH = 32
TEXT_LEN = 256
HEAD_DIM = 128
EPS = 1e-6
ROPE_THETA = 10000.0
ROPE_SPATIAL_SCALE = 0.5
NUM_TIMESTEPS = 1000
# Under fp16 (GPUs without bf16) the caption's massive activations overflow the projection; the layers have no bias, so
# they run on a scaled input and the scale comes back in fp32. The FLUX 3 extension found the same on Qwen3-VL.
TEXT_FP16_SCALE = 2.0 ** -8


def sinusoidal_embedding(position, dim, dtype):
    half = dim // 2
    position = position.to(torch.float64)
    inv_freq = torch.pow(10000.0, -torch.arange(half).to(position) / half)
    angles = torch.outer(position, inv_freq)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).to(dtype)


def rope_table(h, w, text_len, head_dim, device):
    """cos and sin (L, head_dim / 2) in fp32 for [image h x w | text], computed in fp64. The time axis and the caption
    tokens sit at position 0, which is the identity rotation."""
    half = head_dim // 2
    spatial = head_dim // 6
    frequency = 1.0 / torch.pow(ROPE_THETA, torch.arange(0, 2 * spatial, 2, dtype=torch.float64) / (2 * spatial))
    rows = torch.outer(torch.arange(h, dtype=torch.float64) * ROPE_SPATIAL_SCALE, frequency)
    cols = torch.outer(torch.arange(w, dtype=torch.float64) * ROPE_SPATIAL_SCALE, frequency)
    angles = torch.zeros(h * w + text_len, half, dtype=torch.float64)
    temporal = half - 2 * spatial
    if h * w:
        angles[: h * w, temporal: temporal + spatial] = rows[:, None].expand(h, w, spatial).reshape(h * w, spatial)
        angles[: h * w, temporal + spatial:] = cols[None, :].expand(h, w, spatial).reshape(h * w, spatial)
    return torch.cos(angles).float().to(device), torch.sin(angles).float().to(device)


def apply_rope(x, table):
    """x (B, L, heads, head_dim) in fp32, rotated pair by pair."""
    cos, sin = table
    b, length, heads, head_dim = x.shape
    half = head_dim // 2
    cos = cos.view(1, length, 1, half)
    sin = sin.view(1, length, 1, half)
    pairs = x.reshape(b, length, heads, half, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(3)


def attention_mask(mask):
    """A row may attend to the valid rows, and a padded row to itself: its output is zeroed afterwards, and an empty
    row would be NaN. Boolean (B, 1, L, L), or None when every row is valid."""
    if bool(mask.all()):
        return None
    allowed = mask[:, :, None] & mask[:, None, :]
    allowed = allowed | torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    return allowed.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # torch.ones, not empty: the checkpoint overwrites it
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # normalized in fp32, rounded to the input's dtype, then the fp32 gain: the output is fp32, which is what
        # keeps upstream's residual stream in fp32 under bf16 autocast
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return normed.type_as(x) * kernels.cast_to(self.weight, torch.float32, x.device)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x3)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, num_heads)

    def forward(self, x, mask, sdpa_mask, table, dtype, transformer_options):
        b, length = x.shape[:2]
        heads, head_dim = self.num_heads, self.head_dim
        not_valid = ~mask.unsqueeze(-1)
        x_in = x.to(dtype)

        q = self.norm_q(self.q(x_in).masked_fill(not_valid, 0.0)).view(b, length, heads, head_dim)
        k = self.norm_k(self.k(x_in).masked_fill(not_valid, 0.0)).view(b, length, heads, head_dim)
        v = self.v(x_in).masked_fill(not_valid, 0.0).view(b, length, heads, head_dim)
        q = apply_rope(q, table).to(dtype)
        k = apply_rope(k, table).to(dtype)

        out = kernels.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), sdpa_mask, transformer_options)
        out = out.view(b, length, heads, head_dim) * mask[:, :, None, None]

        gate = torch.sigmoid(self.gate_proj(x_in).masked_fill(not_valid, -1e4)).unsqueeze(-1)
        out = (out * gate).flatten(2)
        return self.o(out.to(dtype)).masked_fill(not_valid, 0.0)


class Block(nn.Module):
    def __init__(self, dim, ffn_hidden, num_heads, adaln_rank):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.post_norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden)
        self.post_norm2 = RMSNorm(dim)
        self.adaln_up = nn.Linear(adaln_rank, 4 * dim)

    def forward(self, x, c, mask, sdpa_mask, table, dtype, transformer_options):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln_up(c).unsqueeze(1).chunk(4, dim=2)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
        valid = mask.unsqueeze(-1)

        y = self.attn(self.norm1(x) * scale_msa, mask, sdpa_mask, table, dtype, transformer_options)
        x = x + gate_msa * (self.post_norm1(y) * valid)

        y = self.ffn((self.norm2(x) * scale_mlp).to(dtype))
        return x + gate_mlp * (self.post_norm2(y) * valid)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, dim, bottleneck_dim):
        super().__init__()
        patch = (1, PATCH, PATCH)
        self.proj1 = nn.Conv3d(in_channels, bottleneck_dim, kernel_size=patch, stride=patch, bias=False)
        self.proj2 = nn.Conv3d(bottleneck_dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        return self.proj2(self.proj1(x))


class OutputHead(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, PATCH * PATCH * out_channels)
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

    def forward(self, x, e, dtype):
        # fp32 under upstream's autocast too: layer_norm keeps its input's dtype, and its weights are fp32 masters
        modulation = kernels.cast_to(self.modulation, torch.float32, x.device)
        shift, scale = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        norm = self.norm
        normed = F.layer_norm(x, (x.shape[-1],), kernels.cast_to(norm.weight, torch.float32, x.device),
                              kernels.cast_to(norm.bias, torch.float32, x.device), norm.eps)
        return self.proj((normed * (1 + scale) + shift).to(dtype))


class PJiTDiT(nn.Module):
    def __init__(self, image_model=None, dim=2944, num_heads=23, num_layers=22, refiner_blocks=2, ffn_hidden=5376,
                 refiner_ffn_hidden=2816, adaln_rank=256, text_dim=7680, bottleneck_dim=1024, freq_dim=256, channels=3,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.channels = channels
        self.head_dim = dim // num_heads

        self.patch_embed = PatchEmbed(channels, dim, bottleneck_dim)
        self.time_embed = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.image_adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, adaln_rank))
        self.text_adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, adaln_rank))
        # no bias: zero-padded caption rows stay exactly zero
        self.text_proj = nn.Sequential(nn.Linear(text_dim, dim, bias=False), nn.GELU(approximate="tanh"),
                                       nn.Linear(dim, dim, bias=False))
        self.image_refiner = nn.ModuleList(
            [Block(dim, refiner_ffn_hidden, num_heads, adaln_rank) for _ in range(refiner_blocks)])
        self.text_refiner = nn.ModuleList(
            [Block(dim, refiner_ffn_hidden, num_heads, adaln_rank) for _ in range(refiner_blocks)])
        self.blocks = nn.ModuleList([Block(dim, ffn_hidden, num_heads, adaln_rank) for _ in range(num_layers)])
        self.head = OutputHead(dim, channels)

    def project_text(self, text, dtype):
        first, activation, second = self.text_proj
        if dtype != torch.float16:
            return second(activation(first(text.to(dtype))))
        hidden = activation(first((text.float() * TEXT_FP16_SCALE).to(dtype)).float() / TEXT_FP16_SCALE)
        return second((hidden * TEXT_FP16_SCALE).to(dtype)).float() / TEXT_FP16_SCALE

    def predict_image(self, x_t, t, text, text_lens, transformer_options):
        """Upstream's PyramidJiT.forward: x_t (B, 3, H, W) in pixels, integer timesteps, caption zero-padded to
        TEXT_LEN; the clean image (B, 3, H, W) in the compute dtype."""
        dtype = x_t.dtype
        device = x_t.device
        h_in, w_in = x_t.shape[-2:]
        pad_h, pad_w = -h_in % PATCH, -w_in % PATCH
        if pad_h or pad_w:
            x_t = F.pad(x_t, (0, pad_w, 0, pad_h), mode="constant", value=0)

        embedded = self.patch_embed(x_t.unsqueeze(2))
        h_p, w_p = embedded.shape[-2:]
        x_img = embedded.flatten(2).transpose(1, 2)
        b, l_img = x_img.shape[:2]

        e = self.time_embed(sinusoidal_embedding(t, self.freq_dim, dtype))
        c_image = self.image_adaln(e)
        c_text = self.text_adaln(e)
        text_emb = self.project_text(text, dtype)

        img_mask = torch.ones(b, l_img, dtype=torch.bool, device=device)
        text_mask = torch.arange(TEXT_LEN, device=device)[None, :] < text_lens[:, None]

        table = rope_table(h_p, w_p, 0, self.head_dim, device)
        for block in self.image_refiner:
            x_img = block(x_img, c_image, img_mask, None, table, dtype, transformer_options)

        table = rope_table(0, 0, TEXT_LEN, self.head_dim, device)
        sdpa_mask = attention_mask(text_mask)
        for block in self.text_refiner:
            text_emb = block(text_emb, c_text, text_mask, sdpa_mask, table, dtype, transformer_options)

        mask = torch.cat([img_mask, text_mask], dim=1)
        seq = torch.cat([x_img, text_emb], dim=1).masked_fill(~mask.unsqueeze(-1), 0.0)
        table = rope_table(h_p, w_p, TEXT_LEN, self.head_dim, device)
        sdpa_mask = attention_mask(mask)
        for block in self.blocks:
            seq = block(seq, c_image, mask, sdpa_mask, table, dtype, transformer_options)

        out = self.head(seq[:, :l_img], e, dtype)
        out = out.view(b, h_p, w_p, PATCH, PATCH, self.channels)
        out = torch.einsum("bhwqrc->bchqwr", out).reshape(b, self.channels, h_p * PATCH, w_p * PATCH)
        return out[:, :, :h_in, :w_in]

    def forward(self, x, timestep, context=None, control=None, transformer_options=None, **kwargs):
        sigma = timestep.float()
        # upstream truncates, it does not round: its timesteps are (t * 1000).long()
        t = (sigma * NUM_TIMESTEPS).long()
        # the caption's padding is what the text engine zeroed; every caption has its chat template, so never empty
        text_lens = context.ne(0).any(dim=-1).sum(dim=1)
        x_t = x * NOISE_SCALE
        predicted = self.predict_image(x_t, t, context, text_lens, transformer_options or {})
        # x_t = (1 - t) x_0 + t noise, so the velocity is (x_t - x_0) / t; the sigma is above 0 wherever Euler evaluates
        sigma = sigma.clamp(min=1.0 / NUM_TIMESTEPS).view(-1, 1, 1, 1)
        velocity = (x_t.float() - predicted.float()) / sigma
        return (velocity / NOISE_SCALE).to(x.dtype)

