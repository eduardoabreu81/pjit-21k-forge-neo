# P-JiT 21K support for Forge Neo
# The Qwen3.5 language tower P-JiT reads its captions with (Qwen3.5-4B). Forge Neo has no Qwen3.5-4B, so the text layers
# live here: three Gated DeltaNet (linear attention) layers for every gated full-attention layer, each followed by a
# SwiGLU MLP. The same port as the MOSS-Image extension's, which encodes with the 9B and reads only the last layer; this
# one hands out the outputs of the layers P-JiT reads.
#
# Ported from ComfyUI's comfy/text_encoders/qwen35.py (COMFYUI_COMMIT in native/__init__.py), with the numerics of
# transformers' modeling_qwen3_5.py (TRANSFORMERS_COMMIT): queries and keys L2-normalized in float32, the delta rule
# and the decay in float32, the gated norm before its gate. Prompt encoding only: one causal pass over the whole
# prompt, so no KV cache, no recurrent decode, no generation and no vision.
#
# The layers come from torch.nn classes looked up when the model is built, so Forge's using_forge_operations swaps in
# its own (offloading, manual cast, INT8 ConvRot, FP8 and GGUF weights). The norms ask for Forge's RMSNorm(add=True),
# Qwen3.5's zero-centred norm (scale 1 + weight), as Forge Neo's own Qwen3-VL and ComfyUI do.

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

L2NORM_EPS = 1e-6
CHUNK_SIZE = 64


@dataclass
class Qwen35Config:
    """Qwen3.5-4B's text configuration (the text_config of its transformers config)."""

    vocab_size: int = 248320
    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    full_attention_interval: int = 4
    # gated full attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000000.0
    # Gated DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, config: dict) -> "Qwen35Config":
        # a full transformers config nests the text part; rope_theta sits in rope_parameters there
        config = {**config, **config.get("text_config", {})}
        config.update({k: v for k, v in config.get("rope_parameters", {}).items() if k in ("rope_theta",)})
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in names})

    def is_full_attention(self, index: int) -> bool:
        return (index + 1) % self.full_attention_interval == 0


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + L2NORM_EPS)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=CHUNK_SIZE):
    """The gated delta rule over a whole sequence from an empty state, chunk by chunk (transformers'
    torch_chunk_gated_delta_rule with use_qk_l2norm_in_kernel). Inputs (B, T, H, D) and (B, T, H) for g and beta,
    output (B, T, H, Dv) in float32."""
    batch, length = key.shape[:2]
    heads, v_dim = value.shape[-2:]
    k_dim = key.shape[-1]
    query, key, value, beta, decay = (t.transpose(1, 2).float().contiguous() for t in (query, key, value, beta, g))
    query = l2norm(query) * k_dim**-0.5
    key = l2norm(key)

    pad = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = (F.pad(t, (0, 0, 0, pad)) for t in (query, key, value))
    beta, decay = (F.pad(t, (0, pad)) for t in (beta, decay))
    chunks = (length + pad) // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = (t.reshape(batch, heads, chunks, chunk_size, -1) for t in (query, key, k_beta, v_beta))
    cum_decay = decay.reshape(batch, heads, chunks, chunk_size).cumsum(dim=3)

    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, k_beta * cum_decay.exp().unsqueeze(-1), upper=False,
                                               unitriangular=True)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    state = query.new_zeros(batch, heads, k_dim, v_dim)
    out = torch.empty_like(new_values)
    for i in range(chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    return out.reshape(batch, heads, -1, v_dim)[:, :, :length].transpose(1, 2)


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        hidden = config.hidden_size

        self.in_proj_qkv = nn.Linear(hidden, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)
        # depthwise causal convolution over the q, k, v channels
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=config.linear_conv_kernel_dim, groups=conv_dim,
                                padding=config.linear_conv_kernel_dim - 1, bias=False)
        # per-head decay, kept in float32 (the checkpoint stores A_log in float32)
        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads, dtype=torch.float32), requires_grad=False)
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads, dtype=torch.float32), requires_grad=False)
        # gated norm of the delta rule's output: a plain RMS weight, then the SiLU gate (transformers'
        # Qwen3_5RMSNormGated)
        self.norm = nn.RMSNorm(self.value_head_dim, eps=config.rms_norm_eps)

    def forward(self, x):
        batch, length, _ = x.shape
        z = self.in_proj_z(x)
        beta = self.in_proj_b(x).float().sigmoid()
        a = self.in_proj_a(x).float()
        qkv = F.silu(self.conv1d(self.in_proj_qkv(x).transpose(1, 2))[:, :, :length]).transpose(1, 2).to(x.dtype)
        query, key, value = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, length, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch, length, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch, length, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            repeat = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        a_log = self.A_log.to(device=x.device, dtype=torch.float32)
        dt_bias = self.dt_bias.to(device=x.device, dtype=torch.float32)
        g = -a_log.exp() * F.softplus(a + dt_bias)

        out = chunk_gated_delta_rule(query, key, value, g, beta).to(x.dtype).reshape(-1, self.value_head_dim)
        out = (self.norm(out) * F.silu(z.reshape(-1, self.value_head_dim).float())).to(x.dtype)
        return self.out_proj(out.reshape(batch, length, self.value_dim))


def rotary(position_ids: torch.Tensor, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin (T, dim) for text positions; MRoPE's three axes coincide on text, so this is plain RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=position_ids.device, dtype=torch.float32) / dim))
    freqs = position_ids.float()[:, None] * inv_freq[None]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first cos.shape[-1] channels of (B, H, T, D) by halves, in float32."""
    dim = cos.shape[-1]
    rot, rest = x[..., :dim].float(), x[..., dim:]
    half = dim // 2
    rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
    return torch.cat(((rot * cos + rotated * sin).to(x.dtype), rest), dim=-1)


class GatedAttention(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        inner = self.num_heads * self.head_dim
        # q_proj gives each head its query and then its output gate
        self.q_proj = nn.Linear(config.hidden_size, 2 * inner, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=True)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=True)

    def forward(self, x, cos, sin):
        batch, length, _ = x.shape
        query, gate = self.q_proj(x).view(batch, length, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        gate = gate.reshape(batch, length, -1)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query = apply_partial_rope(query, cos, sin)
        key = apply_partial_rope(key, cos, sin)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(query, key.to(query.dtype), value.to(query.dtype), is_causal=True)
        out = out.transpose(1, 2).reshape(batch, length, -1) * gate.sigmoid()
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, index: int):
        super().__init__()
        self.full_attention = config.is_full_attention(index)
        if self.full_attention:
            self.self_attn = GatedAttention(config)
        else:
            self.linear_attn = GatedDeltaNet(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True)

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        x = x + (self.self_attn(h, cos, sin) if self.full_attention else self.linear_attn(h))
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen35Transformer(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=True)

    def forward(self, x: torch.Tensor, tap_layers=None):
        """The final state after the norm, and the outputs of tap_layers (0-indexed, before any norm), side by side:
        transformers' hidden_states[i + 1], except the last layer's, which transformers takes after the norm."""
        positions = torch.arange(x.shape[1], device=x.device)
        cos, sin = rotary(positions, int(self.config.head_dim * self.config.partial_rotary_factor),
                          self.config.rope_theta)
        tapped = []
        for index, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if tap_layers is not None and index in tap_layers:
                tapped.append(x)
        return self.norm(x), torch.cat(tapped, dim=-1) if tapped else None


class Qwen35TextModel(nn.Module):
    """What Forge Neo's Qwen3 loader branch builds for P-JiT, in place of its Qwen3-4B (same call signature):
    checkpoint keys model.embed_tokens.*, model.layers.*, model.norm.*."""

    def __init__(self, config_dict: dict):
        super().__init__()
        config = Qwen35Config.from_dict(config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = Qwen35Transformer(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=True, dtype=None, **kwargs):
        # Forge's SDClipModel passes the token embeddings in float32, and with a list of layers reads the second output.
        # Captions are encoded one at a time and unpadded, as upstream does
        if embeds is None:
            embeds = self.model.embed_tokens(x)
        return self.model(embeds.to(dtype or embeds.dtype), intermediate_output)
