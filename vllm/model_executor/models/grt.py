# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Ported from the transformers-grt fork:
# https://github.com/MatterWaveAI/transformers (src/transformers/models/grt/modeling_grt.py)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GRT model (Geometric Rotational Transformer).

GRT replaces every dense weight matrix with chains of Givens rotations, so
there are no `nn.Linear` layers and no tensor-parallel sharding of weights;
tensor parallelism must therefore be 1. The LM head is a cosine similarity
against the (tied) token embedding scaled by a learned temperature, computed
directly in `compute_logits` rather than via `ParallelLMHead`.

Positional information enters through a per-token rotation of the embedding
(`pos_encoding`), so the Q/K/V projection chains are per-position independent
and the paged KV cache is exact.

Known checkpoint quirk: released weights carry twelve `decay_raw` tensors per
model belonging to a FlashGivens kernel that neither the transformers fork nor
this file implements; `load_weights` ignores them explicitly.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GRTConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


def _softplus_scale(raw_scale: torch.Tensor) -> torch.Tensor:
    """Map a raw scale parameter to a positive scale."""
    return F.softplus(raw_scale)


def _build_butterfly_perms(n_stages: int, dim: int) -> list[torch.Tensor]:
    """Butterfly permutation for each stage; stage `k` pairs dims at stride `2 ** (k % n_levels)`."""
    half = dim // 2
    n_levels = max(1, int(math.log2(half))) if half > 1 else 1
    perms = []
    for k in range(n_stages):
        stride = 1 << (k % n_levels)
        order = []
        for block_start in range(0, dim, 2 * stride):
            for i in range(stride):
                order.append(block_start + i)
                order.append(block_start + i + stride)
        perm = torch.tensor(order, dtype=torch.long)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(dim)
        perms.append((perm, inv_perm))
    return perms


def _clifford_features(
    x0: torch.Tensor, x1: torch.Tensor, n_features: int, mixer_bivector: bool = False
) -> torch.Tensor:
    """Content features that modulate rotation angles (Cl(2,0), Stokes, or energy)."""
    if n_features == 1:
        return x0 * x0 + x1 * x1
    if n_features == 3:
        return torch.cat([x0 * x0 + x1 * x1, x0 * x0 - x1 * x1, 2.0 * x0 * x1], dim=-1)
    x0_next = torch.cat([x0[..., 1:], x0[..., :1]], dim=-1)
    x1_next = torch.cat([x1[..., 1:], x1[..., :1]], dim=-1)
    bivector = x0 * x0_next - x1 * x1_next if mixer_bivector else x0 * x1_next - x1 * x0_next
    return torch.cat(
        [
            x0 * x0 + x1 * x1,
            x0,
            x1,
            bivector,
        ],
        dim=-1,
    )


def _coupled_rotate(
    hidden_states: torch.Tensor,
    angles: torch.Tensor,
    coupling: torch.Tensor,
    scales: torch.Tensor,
    perms: torch.Tensor | None,
    inv_perms: torch.Tensor | None,
    n_features: int,
) -> torch.Tensor:
    """Apply `K` content-modulated rotation + scale stages.

    Args:
        hidden_states: `(num_tokens, num_heads, head_dim)`.
        angles: `(K, num_heads, head_dim // 2)` base rotation angles.
        coupling: `(K, num_heads, n_features * head_dim // 2, head_dim // 2)`.
        scales: `(K, num_heads, head_dim)`, already positive.
        perms / inv_perms: `(K, head_dim)` butterfly permutations, or `None`.
    """
    num_stages = angles.shape[0]
    out = hidden_states

    for k in range(num_stages):
        if perms is not None:
            out = out[..., perms[k]]

        pairs = out.view(*out.shape[:-1], -1, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]

        features = _clifford_features(x0, x1, n_features)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-6)

        modulation = torch.einsum("thj,hji->thi", features, coupling[k])
        theta = angles[k] + torch.tanh(modulation)

        cos, sin = torch.cos(theta), torch.sin(theta)
        rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        out = rotated.view(*out.shape) * scales[k]

        if inv_perms is not None:
            out = out[..., inv_perms[k]]

    return out


def _plain_rotate(
    hidden_states: torch.Tensor,
    angles: torch.Tensor,
    raw_scales: torch.Tensor,
    perms: torch.Tensor | None,
    inv_perms: torch.Tensor | None,
) -> torch.Tensor:
    """Apply `K` fixed rotation + scale stages (no content modulation)."""
    scales = _softplus_scale(raw_scales)
    orig_shape = hidden_states.shape
    out = hidden_states.reshape(-1, orig_shape[-1])
    num_tokens, dim = out.shape
    num_stages, n_pairs = angles.shape

    for k in range(num_stages):
        if perms is not None:
            out = out[:, perms[k]]

        pairs = out.view(num_tokens, n_pairs, 2)
        x0, x1 = pairs[:, :, 0], pairs[:, :, 1]

        cos, sin = torch.cos(angles[k]), torch.sin(angles[k])
        rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        out = rotated.view(num_tokens, dim) * scales[k]

        if inv_perms is not None:
            out = out[:, inv_perms[k]]

    return out.reshape(orig_shape)


class GRTPearlNorm(nn.Module):
    """Pair-aware RMSNorm: divides by the RMS of per-pair radii, phases untouched.

    Accepts the same `(x, residual)` optional-residual contract as vLLM's
    `RMSNorm`, returning `(normalized, x + residual)` when a residual is given.
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.d_model = d_model

    def _normalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pairs = hidden_states.view(*hidden_states.shape[:-1], self.d_model // 2, 2)
        radius_sq = (pairs**2).sum(dim=-1)
        rms = torch.sqrt(radius_sq.mean(dim=-1, keepdim=True) + self.eps)
        return hidden_states / rms * self.scale

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._normalize(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        return self._normalize(hidden_states), residual


class GRTRadialGELU(nn.Module):
    """GELU applied to pair magnitudes, preserving phase."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pairs = hidden_states.view(*hidden_states.shape[:-1], self.dim // 2, 2)
        radius = torch.norm(pairs, dim=-1, keepdim=True)
        gate = F.gelu(radius) / (radius + self.eps)
        return (pairs * gate).view(*hidden_states.shape)


class GRTEmbedding(nn.Module):
    """Phase-angle token embedding: `d_model // 2` angles plus a log magnitude."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.phase_angles = nn.Parameter(torch.empty(vocab_size, d_model // 2))
        self.log_magnitude = nn.Parameter(torch.empty(vocab_size))

    def token_vectors(self) -> torch.Tensor:
        """Full embedding vectors for every token, shape `(vocab_size, d_model)`."""
        angles = self.phase_angles
        pairs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        embeddings = pairs.reshape(self.vocab_size, self.d_model)
        magnitude = torch.exp(self.log_magnitude.clamp(-5.0, 5.0)).unsqueeze(-1)
        return embeddings * magnitude

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        angles = self.phase_angles[token_ids]
        pairs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        embeddings = pairs.reshape(*token_ids.shape, self.d_model)
        magnitude = torch.exp(self.log_magnitude[token_ids].clamp(-5.0, 5.0)).unsqueeze(-1)
        return embeddings * magnitude


class GRTRotaryPositionEmbedding(nn.Module):
    """Rotational positional encoding: rotates each pair by `position * freq_i`."""

    def __init__(self, config: GRTConfig):
        super().__init__()
        n_pairs = config.d_model // 2
        log_freqs = -torch.arange(0, n_pairs).float() / n_pairs * math.log(config.pos_freq_base)
        if config.learnable_rope:
            self.log_freqs = nn.Parameter(log_freqs)
        else:
            self.register_buffer("log_freqs", log_freqs, persistent=False)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        num_tokens, d_model = hidden_states.shape
        pairs = hidden_states.view(num_tokens, d_model // 2, 2)

        freqs = torch.exp(self.log_freqs.to(hidden_states.dtype))
        angles = positions.to(hidden_states.dtype).unsqueeze(-1) * freqs
        cos, sin = torch.cos(angles), torch.sin(angles)

        x0, x1 = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return rotated.view(num_tokens, d_model)


class GRTAttention(nn.Module):
    """Multi-head attention whose Q/K/V/O projections are rotation chains."""

    def __init__(
        self,
        config: GRTConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError(
                "GRT has no dense weight matrices to shard; tensor parallelism "
                "must be 1."
            )
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_stages = config.n_qkv_rotations
        self.n_features = {"clifford": 4, "stokes": 4, "energy": 1}[config.coupling_features]
        self.scaling = self.head_dim**-0.5

        n_pairs = self.head_dim // 2
        angle_shape = (self.n_stages, self.n_heads, n_pairs)
        scale_shape = (self.n_stages, self.n_heads, self.head_dim)
        coupling_shape = (self.n_stages, self.n_heads, self.n_features * n_pairs, n_pairs)

        for name in ("q", "k", "v", "o"):
            self.register_parameter(f"{name}_angles", nn.Parameter(torch.empty(angle_shape)))
            self.register_parameter(f"{name}_scales", nn.Parameter(torch.empty(scale_shape)))
            self.register_parameter(f"{name}_coupling", nn.Parameter(torch.empty(coupling_shape)))

        if self.n_stages > 1:
            perms = _build_butterfly_perms(self.n_stages, self.head_dim)
            self.register_buffer("_attn_perms", torch.stack([p for p, _ in perms]), persistent=True)
            self.register_buffer("_attn_inv_perms", torch.stack([ip for _, ip in perms]), persistent=True)

        self.attn = Attention(
            self.n_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.n_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def _project(self, hidden_states, angles, raw_scales, coupling):
        return _coupled_rotate(
            hidden_states,
            angles,
            coupling,
            _softplus_scale(raw_scales),
            getattr(self, "_attn_perms", None),
            getattr(self, "_attn_inv_perms", None),
            self.n_features,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`hidden_states`: `(num_tokens, d_model)` -> `(num_tokens, d_model)`."""
        num_tokens = hidden_states.shape[0]
        flat = hidden_states.view(num_tokens, self.n_heads, self.head_dim)
        query = self._project(flat, self.q_angles, self.q_scales, self.q_coupling)
        key = self._project(flat, self.k_angles, self.k_scales, self.k_coupling)
        value = self._project(flat, self.v_angles, self.v_scales, self.v_coupling)

        attn_output = self.attn(
            query.reshape(num_tokens, -1),
            key.reshape(num_tokens, -1),
            value.reshape(num_tokens, -1),
        )

        out_flat = attn_output.view(num_tokens, self.n_heads, self.head_dim)
        out = self._project(out_flat, self.o_angles, self.o_scales, self.o_coupling)
        return out.reshape(num_tokens, self.d_model)


class GRTMixer(nn.Module):
    """Cross-head mixing at full `d_model` width with block-diagonal coupling."""

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_rotations = config.n_mixer_rotations
        self.n_pairs = config.d_model // 2
        self.n_features = {"clifford": 4, "stokes": 4, "energy": 1}[config.coupling_features]
        self.group_size = next((g for g in (16, 8, 4, 2) if self.n_pairs % g == 0), 1)
        n_groups = self.n_pairs // self.group_size

        self.angles = nn.Parameter(torch.empty(self.n_rotations, self.n_pairs))
        self.scales = nn.Parameter(torch.empty(self.n_rotations, config.d_model))
        self.coupling = nn.Parameter(
            torch.empty(self.n_rotations, n_groups, self.n_features * self.group_size, self.group_size)
        )

        perms = _build_butterfly_perms(self.n_rotations, config.d_model)
        self.register_buffer("_perms_stack", torch.stack([p for p, _ in perms]), persistent=True)
        self.register_buffer("_inv_perms_stack", torch.stack([ip for _, ip in perms]), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        group_size = self.group_size
        n_groups = self.n_pairs // group_size
        num_tokens = hidden_states.shape[0]

        for k in range(self.n_rotations):
            hidden_states = hidden_states[..., self._perms_stack[k]]

            flat = hidden_states.reshape(num_tokens, self.d_model)
            grouped = flat.view(num_tokens, n_groups, 2 * group_size)
            pairs = grouped.view(num_tokens, n_groups, group_size, 2)
            x0, x1 = pairs[..., 0], pairs[..., 1]

            features = _clifford_features(x0, x1, self.n_features, mixer_bivector=True)
            # eps=1e-2 (not 1e-6) to match the released checkpoints' mixer.
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-2)

            modulation = torch.einsum("tgj,gji->tgi", features, self.coupling[k])
            theta = self.angles[k].view(n_groups, group_size) + torch.tanh(modulation)

            cos, sin = torch.cos(theta), torch.sin(theta)
            rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
            rotated = rotated.view(num_tokens, n_groups, 2 * group_size)
            rotated = rotated * _softplus_scale(self.scales[k]).view(n_groups, 2 * group_size)

            hidden_states = rotated.view(num_tokens, self.d_model)
            hidden_states = hidden_states[..., self._inv_perms_stack[k]]

        return hidden_states


class GRTFeedForward(nn.Module):
    """Rotation-chain FFN: pad to `d_ff`, rotate, RadialGELU, rotate, truncate."""

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = int(config.d_ff) if config.d_ff is not None else config.ffn_mult * config.d_model
        self.n_rotations = config.ffn_n_rotations
        n_pairs = self.d_ff // 2
        self.activation = GRTRadialGELU(self.d_ff)

        self.expand_angles = nn.Parameter(torch.empty(self.n_rotations, n_pairs))
        self.expand_scales = nn.Parameter(torch.empty(self.n_rotations, self.d_ff))
        self.contract_angles = nn.Parameter(torch.empty(self.n_rotations, n_pairs))
        self.contract_scales = nn.Parameter(torch.empty(self.n_rotations, self.d_ff))

        if self.n_rotations > 1:
            perms = _build_butterfly_perms(self.n_rotations, self.d_ff)
            self.register_buffer("_ffn_perms", torch.stack([p for p, _ in perms]), persistent=True)
            self.register_buffer("_ffn_inv_perms", torch.stack([ip for _, ip in perms]), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        perms = getattr(self, "_ffn_perms", None)
        inv_perms = getattr(self, "_ffn_inv_perms", None)

        hidden_states = F.pad(hidden_states, (0, self.d_ff - self.d_model))
        hidden_states = _plain_rotate(hidden_states, self.expand_angles, self.expand_scales, perms, inv_perms)
        hidden_states = self.activation(hidden_states)
        hidden_states = _plain_rotate(hidden_states, self.contract_angles, self.contract_scales, perms, inv_perms)
        return hidden_states[..., : self.d_model]


class GRTDecoderLayer(nn.Module):
    """One GRT block: norm -> attention -> norm -> mixer -> norm -> feed-forward.

    Each sublayer writes into the residual stream through a learnable gate.
    """

    def __init__(
        self,
        config: GRTConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_mode = config.residual_gate_mode if config.learnable_residual_gates else "none"

        if self.gate_mode == "free":
            gate_init = 1.0
        elif self.gate_mode == "bounded":
            gate_init = 0.0
        elif self.gate_mode == "bounded_depth":
            target = 1.0 / max(1, config.n_blocks) ** 0.5
            t = min(0.9999, max(-0.9999, (target - 0.1) / 0.45 - 1.0))
            gate_init = 0.5 * math.log((1 + t) / (1 - t))
        else:
            gate_init = None

        if gate_init is None:
            self.res_gate_attn = None
            self.res_gate_ffn = None
            self.res_gate_mixer = None
        else:
            self.res_gate_attn = nn.Parameter(torch.empty(()))
            self.res_gate_ffn = nn.Parameter(torch.empty(()))
            self.res_gate_mixer = nn.Parameter(torch.empty(()))
        self.gate_init = gate_init

        self.norm1 = GRTPearlNorm(config.d_model)
        self.attention = GRTAttention(
            config, cache_config=cache_config, quant_config=quant_config, prefix=f"{prefix}.attention"
        )

        self.mixer = None
        if config.n_mixer_rotations > 0:
            self.norm_mixer = GRTPearlNorm(config.d_model)
            self.mixer = GRTMixer(config)

        self.norm_ffn = GRTPearlNorm(config.d_model)
        self.ffn = GRTFeedForward(config)

        self.post_norm_attn = GRTPearlNorm(config.d_model) if config.peri_ln else None
        self.post_norm_ffn = GRTPearlNorm(config.d_model) if config.peri_ln else None
        if config.residual_norm_mode == "pearlnorm":
            self.res_norm_block = GRTPearlNorm(config.d_model)
        elif config.residual_norm_mode != "none":
            raise ValueError(f"Unknown residual_norm_mode: {config.residual_norm_mode!r}")
        else:
            self.res_norm_block = None

    def _gate(self, raw):
        if raw is None:
            return 1.0
        if self.gate_mode == "bounded":
            return 1.0 + torch.tanh(raw) * 0.5
        if self.gate_mode == "bounded_depth":
            return 0.55 + torch.tanh(raw) * 0.45
        return raw

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
        else:
            hidden_states, residual = self.norm1(hidden_states, residual)

        attn_out = self.attention(hidden_states)
        if self.post_norm_attn is not None:
            attn_out = self.post_norm_attn(attn_out)
        attn_out = self._gate(self.res_gate_attn) * attn_out

        if self.mixer is not None:
            hidden_states, residual = self.norm_mixer(attn_out, residual)
            mixer_out = self._gate(self.res_gate_mixer) * self.mixer(hidden_states)
        else:
            mixer_out = attn_out
        hidden_states, residual = self.norm_ffn(mixer_out, residual)

        ffn_out = self.ffn(hidden_states)
        if self.post_norm_ffn is not None:
            ffn_out = self.post_norm_ffn(ffn_out)
        ffn_out = self._gate(self.res_gate_ffn) * ffn_out

        if self.res_norm_block is not None:
            # The normalized stream becomes the whole residual stream; the next
            # layer starts from a fresh stream.
            hidden_states = self.res_norm_block(ffn_out + residual)
            return hidden_states, None
        return ffn_out, residual


class GRTModel(nn.Module):
    """GRT encoder-decoder-less transformer: embedding, block stack, final norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        if get_pp_group().is_first_rank:
            self.embedding = GRTEmbedding(config.vocab_size, config.d_model)
            self.pos_encoding = GRTRotaryPositionEmbedding(config)
        else:
            self.embedding = PPMissingLayer()
            self.pos_encoding = PPMissingLayer()

        self.start_layer, self.end_layer, self.blocks = make_layers(
            config.n_blocks,
            lambda prefix: GRTDecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.blocks",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.d_model
        )
        if get_pp_group().is_last_rank:
            self.final_norm = GRTPearlNorm(config.d_model)
            self.output_scale = nn.Parameter(torch.empty(()))
        else:
            self.final_norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Cosine-similarity head against the full (tied) embedding table."""
        token_embeddings = self.embedding.token_vectors()
        hidden_states = hidden_states.to(torch.float32)
        token_embeddings = token_embeddings.to(torch.float32)
        hidden_norm = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-6)
        emb_norm = token_embeddings / (token_embeddings.norm(dim=-1, keepdim=True) + 1e-6)
        logits = torch.matmul(hidden_norm, emb_norm.T)
        return logits * F.softplus(self.output_scale.to(torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = self.pos_encoding(hidden_states, positions)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.blocks[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(hidden_states, residual)

        if not get_pp_group().is_last_rank:
            if residual is None:
                residual = torch.zeros_like(hidden_states)
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states


class GRTForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping: dict = {}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = GRTModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, ignore_unexpected_suffixes=["decay_raw"])
        return loader.load_weights(weights)
