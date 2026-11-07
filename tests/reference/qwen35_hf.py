# Reference for the tests only: the Qwen3.5 text decoder of transformers' modeling_qwen3_5.py
# (huggingface/transformers commit 09281b8d56b08199ce9f397c694f38853cc45278, Apache License 2.0), transcribed
# without the transformers framework (no cache, no kernels hub, eager attention). The token mixer uses
# torch_recurrent_gated_delta_rule, the token-by-token form, where the extension uses the chunked one: the two
# algorithms must agree. Parameter names are transformers' Qwen3_5TextModel's.

import torch
import torch.nn.functional as F
from torch import nn


def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    decay = g
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)
    last_recurrent_state = torch.zeros((batch_size, num_v_heads, k_head_dim, v_head_dim), dtype=value.dtype)
    core_attn_out = torch.zeros_like(value)
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        decay_t = decay[:, :, i].exp()[..., None, None]
        last_recurrent_state = last_recurrent_state * decay_t
        beta_t = beta[:, :, i].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.num_v_heads = c["linear_num_value_heads"]
        self.num_k_heads = c["linear_num_key_heads"]
        self.head_k_dim = c["linear_key_head_dim"]
        self.head_v_dim = c["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        kernel = c["linear_conv_kernel_dim"]
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=kernel, groups=self.conv_dim,
                                padding=kernel - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=c["rms_norm_eps"])
        self.out_proj = nn.Linear(self.value_dim, c["hidden_size"], bias=False)
        self.in_proj_qkv = nn.Linear(c["hidden_size"], self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(c["hidden_size"], self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(c["hidden_size"], self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(c["hidden_size"], self.num_v_heads, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        # causal_conv1d_fn
        weight = self.conv1d.weight.squeeze(1)
        mixed_qkv = F.conv1d(mixed_qkv.to(weight.dtype), weight=weight.unsqueeze(1), padding=weight.shape[-1] - 1,
                             groups=mixed_qkv.shape[1])[:, :, :seq_len]
        mixed_qkv = F.silu(mixed_qkv).to(hidden_states.dtype).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        core_attn_out = torch_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta,
                                                         use_qk_l2norm_in_kernel=True)
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        dim = int(c["head_dim"] * c["partial_rotary_factor"])
        self.inv_freq = 1.0 / (c["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.mrope_section = c["mrope_section"]

    def forward(self, x, position_ids):
        position_ids = position_ids.expand(3, -1, -1)
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos = self.recomposition_frequencies(freqs.cos())
        sin = self.recomposition_frequencies(freqs.sin())
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq):
        freqs_thw = freq[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


class Qwen3_5Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.head_dim = c["head_dim"]
        self.num_key_value_groups = c["num_attention_heads"] // c["num_key_value_heads"]
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(c["hidden_size"], c["num_attention_heads"] * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(c["hidden_size"], c["num_key_value_heads"] * self.head_dim, bias=False)
        self.v_proj = nn.Linear(c["hidden_size"], c["num_key_value_heads"] * self.head_dim, bias=False)
        self.o_proj = nn.Linear(c["num_attention_heads"] * self.head_dim, c["hidden_size"], bias=False)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=c["rms_norm_eps"])
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=c["rms_norm_eps"])

    def forward(self, hidden_states, position_embeddings):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # eager_attention_forward with the causal mask
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        length = attn_weights.shape[-1]
        causal = torch.full((length, length), torch.finfo(attn_weights.dtype).min).triu(1)
        attn_weights = F.softmax(attn_weights + causal, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen3_5MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, c, layer_idx):
        super().__init__()
        self.block_type = c["layer_types"][layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(c)
        else:
            self.self_attn = Qwen3_5Attention(c)
        self.mlp = Qwen3_5MLP(c)
        self.input_layernorm = Qwen3_5RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.post_attention_layernorm = Qwen3_5RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])

    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class Qwen3_5TextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(c, i) for i in range(c["num_hidden_layers"])])
        self.norm = Qwen3_5RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(c)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1]).view(1, 1, -1).expand(3, hidden_states.shape[0], -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        return self.norm(hidden_states)
