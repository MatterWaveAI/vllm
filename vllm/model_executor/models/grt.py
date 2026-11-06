# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GRT (Geometric Rotation Transformer) model family for vLLM.

Every projection in GRT is a chain of butterfly-permuted Givens rotations, the
token embedding stores planar phase angles plus a log magnitude, and the LM head
is a cosine similarity against the normalized embedding table scaled by a
learned temperature. There are zero `nn.Linear` or `nn.Conv` layers in the
rotational architecture.

Supported architectures:
- `GRTForCausalLM`: Base causal LM (`MatterWaveAI/grt-30m`, `grt-30m-chat`, and
  4-expert Mixture-of-Rotations `grt-30m-mor`).
- `GRTModel` / `GRTEmbeddingModel`: Bidirectional / decoder circular-pooling
  embedding model (`MatterWaveAI/grt-30m-embedding`).
- `GRTForSequenceClassification`: Cross-encoder reranker and sequence classifier
  with Hermitian circular conjugate and multi-vector MaxSim/MinSim late
  interaction (`MatterWaveAI/grt-30m-reranker`).
- `GRTVLForConditionalGeneration`: Vision-language rotational model
  (`MatterWaveAI/grt-30m-vl`).
- `GRTASRForConditionalGeneration`: Automatic speech recognition model
  (`MatterWaveAI/grt-30m-asr`).
- `GRTTTSForConditionalGeneration`: Text-to-speech model with polar STFT vocoder
  (`MatterWaveAI/grt-30m-tts`).
- `GRTOmniForConditionalGeneration`: Unified text + vision + audio + speech
  model (`MatterWaveAI/grt-30m-omni`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.transformers_utils.configs.grt import (
    GRTASRConfig,
    GRTAudioConfig,
    GRTConfig,
    GRTOmniConfig,
    GRTTTSConfig,
    GRTVisionConfig,
    GRTVLConfig,
    load_grt_lora_adapter,
    merge_grt_lora_weights,
)

try:
    from vllm.compilation.decorators import support_torch_compile
except Exception:

    def support_torch_compile(cls: Any) -> Any:  # type: ignore[misc]
        return cls


try:
    from vllm.config import CacheConfig, VllmConfig
except Exception:
    CacheConfig = Any  # type: ignore[misc,assignment]
    VllmConfig = Any  # type: ignore[misc,assignment]

try:
    from vllm.distributed import get_tensor_model_parallel_world_size
except Exception:

    def get_tensor_model_parallel_world_size() -> int:
        return 1


try:
    from vllm.model_executor.layers.attention import Attention
except Exception:
    Attention = None  # type: ignore[misc,assignment]

try:
    from vllm.forward_context import is_forward_context_available
except Exception:

    def is_forward_context_available() -> bool:
        return False


try:
    from vllm.model_executor.layers.quantization import QuantizationConfig
except Exception:
    QuantizationConfig = Any  # type: ignore[misc,assignment]

try:
    from vllm.sequence import IntermediateTensors
except Exception:
    IntermediateTensors = Any  # type: ignore[misc,assignment]

try:
    from .utils import AutoWeightsLoader, _merge_multimodal_embeddings, maybe_prefix
except Exception:

    def maybe_prefix(prefix: str, name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    def _merge_multimodal_embeddings(
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: Any,
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        out = inputs_embeds.clone()
        if isinstance(multimodal_embeddings, (list, tuple)):
            flat_mm = torch.cat(
                [m.reshape(-1, out.shape[-1]) for m in multimodal_embeddings], dim=0
            )
        else:
            flat_mm = multimodal_embeddings.reshape(-1, out.shape[-1])
        out[is_multimodal] = flat_mm.to(out.device, out.dtype)
        return out

    class AutoWeightsLoader:  # type: ignore[no-redef]
        def __init__(self, model: nn.Module):
            self.model = model

        def load_weights(
            self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any
        ) -> set[str]:
            params = dict(self.model.named_parameters())
            buffers = dict(self.model.named_buffers())
            loaded: set[str] = set()
            for name, tensor in weights:
                if name in params:
                    params[name].data.copy_(tensor)
                    loaded.add(name)
                elif name in buffers:
                    buffers[name].data.copy_(tensor)
                    loaded.add(name)
            return loaded


logger = logging.getLogger(__name__)

_GATE_MODES = ("free", "bounded", "bounded_depth", "none")


def _param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=False)


def _raw_scale_init(target_scale: float) -> float:
    if target_scale > 20.0:
        return target_scale
    return math.log(math.exp(target_scale) - 1.0) if target_scale > 0 else 0.0


def _unwrap_vllm_args(
    config: Any = None,
    vllm_config: Any = None,
    quant_config: Any = None,
    cache_config: Any = None,
) -> tuple[Any, Any, Any]:
    if vllm_config is not None:
        if config is None:
            config = getattr(
                getattr(vllm_config, "model_config", None), "hf_config", vllm_config
            )
        if quant_config is None:
            quant_config = getattr(vllm_config, "quant_config", None)
        if cache_config is None:
            cache_config = getattr(vllm_config, "cache_config", None)
    elif config is not None and hasattr(config, "model_config"):
        # Passed vllm_config positionally as the first argument
        vc = config
        config = getattr(vc.model_config, "hf_config", vc)
        if quant_config is None:
            quant_config = getattr(vc, "quant_config", None)
        if cache_config is None:
            cache_config = getattr(vc, "cache_config", None)
    return config, quant_config, cache_config


def _check_config(config: GRTConfig) -> None:
    hidden, heads = config.hidden_size, config.num_attention_heads
    if hidden % 2 or hidden % heads or (hidden // heads) % 2:
        raise ValueError(
            f"GRT needs an even hidden_size and head_dim; got hidden_size={hidden}, "
            f"num_attention_heads={heads}"
        )
    if config.residual_gate_mode not in _GATE_MODES:
        raise ValueError(
            f"Unknown GRT residual_gate_mode {config.residual_gate_mode!r}"
        )
    if config.residual_norm_mode not in ("none", "pearlnorm"):
        raise ValueError(
            f"Unknown GRT residual_norm_mode {config.residual_norm_mode!r}"
        )


def _butterfly_perms(n_stages: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage `k` pairs dims at stride `2 ** (k % log2(dim // 2))`."""
    n_levels = max(1, int(math.log2(dim // 2))) if dim > 2 else 1
    perms = []
    for k in range(n_stages):
        stride = 1 << (k % n_levels)
        perms.append(
            [
                start + i + offset
                for start in range(0, dim, 2 * stride)
                for i in range(stride)
                for offset in (0, stride)
            ]
        )
    perms_t = torch.tensor(perms, dtype=torch.long)
    return perms_t, perms_t.argsort(dim=-1)


def _compose_gathers(perms: torch.Tensor, inv_perms: torch.Tensor) -> torch.Tensor:
    """One gather per stage boundary: each stage's inverse permutation composed
    with the next stage's permutation, `[K + 1, dim]`."""
    return torch.cat([perms[:1], inv_perms[:-1].gather(1, perms[1:]), inv_perms[-1:]])


def _rotate(
    x0: torch.Tensor, x1: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)


def _clifford_features(
    x0: torch.Tensor, x1: torch.Tensor, mixer_bivector: bool = False
) -> torch.Tensor:
    """`[S0, V1, V2, B12]`: pair energy, both vector parts and the bivector
    with the next pair."""
    x0_next, x1_next = x0.roll(-1, dims=-1), x1.roll(-1, dims=-1)
    if mixer_bivector:
        bivector = x0 * x0_next - x1 * x1_next
    else:
        bivector = x0 * x1_next - x1 * x0_next
    return torch.cat([x0 * x0 + x1 * x1, x0, x1, bivector], dim=-1)


def _quantize_symmetric(
    tensor: torch.Tensor, bits: int = 8, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel/per-row weight quantization (`bits` in `{8, 4}`)."""
    if bits not in (4, 8):
        raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")
    qmax = 127.0 if bits == 8 else 7.0
    t_f32 = tensor.to(torch.float32)
    scale = t_f32.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(t_f32 / scale).clamp(-qmax, qmax).to(torch.int8)
    if bits == 8:
        return q, scale
    if q.shape[dim] % 2 != 0:
        raise ValueError(
            f"INT4 packing requires even size along dim={dim}, got {q.shape[dim]}"
        )
    q_u8 = (q + 8).to(torch.uint8)
    idx0 = torch.arange(0, q_u8.shape[dim], 2, device=q.device)
    idx1 = idx0 + 1
    low = q_u8.index_select(dim, idx0)
    high = q_u8.index_select(dim, idx1)
    packed = (low & 0x0F) | ((high & 0x0F) << 4)
    return packed, scale


def _dequantize_symmetric(
    qweight: torch.Tensor, scale: torch.Tensor, bits: int = 8, dim: int = -1
) -> torch.Tensor:
    """Dequantize a weight tensor produced by `_quantize_symmetric`."""
    if bits == 8:
        return qweight.to(torch.float32) * scale.to(torch.float32)
    if bits == 4:
        low = (qweight & 0x0F).to(torch.int8) - 8
        high = ((qweight >> 4) & 0x0F).to(torch.int8) - 8
        d = dim if dim >= 0 else qweight.ndim + dim
        unpacked = torch.stack([low, high], dim=d + 1).flatten(d, d + 1)
        return unpacked.to(torch.float32) * scale.to(torch.float32)
    raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")


def _resolve_quant_bits(
    config: Any, quant_config: QuantizationConfig | None
) -> int | None:
    bits = getattr(config, "quantize_bits", None)
    if bits in (4, 8):
        return int(bits)
    qcfg = getattr(config, "quantization_config", None)
    if isinstance(qcfg, dict):
        b = qcfg.get("bits") or qcfg.get("num_bits")
        if b in (4, 8):
            return int(b)
        if qcfg.get("load_in_4bit"):
            return 4
        if qcfg.get("load_in_8bit"):
            return 8
    if quant_config is not None:
        name = str(getattr(quant_config, "get_name", lambda: "")()).lower()
        if any(k in name for k in ("int4", "fp4", "awq", "gptq", "4bit")):
            return 4
        if any(k in name for k in ("int8", "fp8", "8bit", "online")):
            return 8
        wbits = getattr(quant_config, "weight_bits", None)
        if wbits in (4, 8):
            return int(wbits)
    return None


def _coupled_chain(
    x: torch.Tensor,
    groups: int,
    angles: torch.Tensor,
    coupling: torch.Tensor,
    scales: torch.Tensor,
    gathers: torch.Tensor | None,
    feature_eps: float,
    mixer_bivector: bool = False,
    coupling_scale: torch.Tensor | None = None,
    coupling_bits: int | None = None,
) -> torch.Tensor:
    """Content-modulated rotation chain over `x` of shape `[T, H, D]`."""
    t, h, d = x.shape
    for k in range(angles.shape[0]):
        if gathers is not None:
            x = x[..., gathers[k]]
        g = x.reshape(t, groups, -1)
        x0, x1 = g[..., 0::2], g[..., 1::2]
        features = _clifford_features(x0, x1, mixer_bivector)
        features = features / (features.norm(dim=-1, keepdim=True) + feature_eps)
        ck = (
            _dequantize_symmetric(
                coupling[k], coupling_scale[k], bits=coupling_bits, dim=-2
            )
            if coupling_scale is not None and coupling_bits is not None
            else coupling[k]
        )
        modulation = torch.einsum("tgj,gji->tgi", features, ck)
        theta = angles[k] + torch.tanh(modulation)
        x = (_rotate(x0, x1, theta.cos(), theta.sin()) * scales[k]).view(t, h, d)
    if gathers is not None:
        x = x[..., gathers[-1]]
    return x


def _plain_chain(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scales: torch.Tensor,
    gathers: torch.Tensor | None,
) -> torch.Tensor:
    for k in range(cos.shape[0]):
        if gathers is not None:
            x = x[..., gathers[k]]
        x = _rotate(x[..., 0::2], x[..., 1::2], cos[k], sin[k]) * scales[k]
    if gathers is not None:
        x = x[..., gathers[-1]]
    return x


def _plain_rotate_raw(
    x: torch.Tensor,
    angles: torch.Tensor,
    raw_scales: torch.Tensor,
    perms: torch.Tensor | None,
    inv_perms: torch.Tensor | None,
) -> torch.Tensor:
    """Unfolded plain rotation chain used by vision/audio stems and vocoder."""
    scales = F.softplus(raw_scales)
    orig_shape = x.shape
    out = x.reshape(-1, orig_shape[-1])
    num_tokens, dim = out.shape
    n_pairs = angles.shape[-1]
    for k in range(angles.shape[0]):
        if perms is not None:
            out = out[:, perms[k]]
        pairs = out.view(num_tokens, n_pairs, 2)
        x0, x1 = pairs[:, :, 0], pairs[:, :, 1]
        cos, sin = torch.cos(angles[k]), torch.sin(angles[k])
        out = (
            torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).view(
                num_tokens, dim
            )
            * scales[k]
        )
        if inv_perms is not None:
            out = out[:, inv_perms[k]]
    return out.reshape(orig_shape)


def _mor_chain(
    x: torch.Tensor,
    expert_angles: torch.Tensor,
    expert_gains: torch.Tensor,
    router_weights: torch.Tensor,
    gathers: torch.Tensor | None,
) -> torch.Tensor:
    """Mixture-of-Rotations (`MoR`) chain with composed butterfly gathers."""
    orig_shape = x.shape
    out = x.reshape(-1, orig_shape[-1])
    num_tokens, dim = out.shape
    rw = router_weights.reshape(num_tokens, -1)
    n_stages = expert_angles.shape[1]

    blended_angles = torch.einsum(
        "ne,ekp->nkp", rw.to(dtype=expert_angles.dtype), expert_angles
    )
    blended_scales = torch.einsum(
        "ne,ekd->nkd", rw.to(dtype=expert_gains.dtype), expert_gains
    )

    for k in range(n_stages):
        if gathers is not None:
            out = out[:, gathers[k]]
        x0, x1 = out[:, 0::2], out[:, 1::2]
        theta = blended_angles[:, k, :]
        out = _rotate(x0, x1, theta.cos(), theta.sin()) * blended_scales[:, k, :]
    if gathers is not None:
        out = out[:, gathers[-1]]
    return out.reshape(orig_shape)


def _circular_pool_seq(seq_hidden: torch.Tensor) -> torch.Tensor:
    """Circular mean pooling over a single sequence `[L, D]` -> `[1, D]`."""
    pairs = seq_hidden.float().unflatten(-1, (-1, 2))
    radii = pairs.norm(dim=-1, keepdim=True)
    directions = pairs / torch.clamp(radii, min=1e-9)
    dir_mean = directions.mean(dim=0, keepdim=True)
    rad_mean = radii.mean(dim=0, keepdim=True)
    pooled = (dir_mean * rad_mean).flatten(start_dim=1)
    return F.normalize(pooled, p=2, dim=-1).to(seq_hidden.dtype)


def _circular_pool_batch(
    hidden_states: torch.Tensor,
    prompt_lens: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Circular mean pooling over a batch of sequences (`[B, L, D]` or `[T, D]`)."""
    if hidden_states.ndim == 3 and prompt_lens is None:
        return torch.cat(
            [_circular_pool_seq(hidden_states[i]) for i in range(hidden_states.shape[0])],
            dim=0,
        )
    flat_hs = hidden_states.reshape(-1, hidden_states.shape[-1])
    if prompt_lens is None:
        return _circular_pool_seq(flat_hs)
    if isinstance(prompt_lens, torch.Tensor):
        lens = [int(x) for x in prompt_lens.detach().cpu().tolist()]
    else:
        lens = [int(x) for x in prompt_lens]
    outputs = []
    offset = 0
    for slen in lens:
        if slen <= 0:
            continue
        outputs.append(_circular_pool_seq(flat_hs[offset : offset + slen]))
        offset += slen
    return torch.cat(outputs, dim=0) if outputs else _circular_pool_seq(flat_hs)


class GRTCircularPooler(nn.Module):
    """vLLM-compatible circular mean pooler over planar pairs."""

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: Any = None,
        prompt_lens: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prompt_lens is None and pooling_metadata is not None:
            for attr in ("prompt_lens", "seq_lens", "extend_seq_lens"):
                val = getattr(pooling_metadata, attr, None)
                if val is not None:
                    prompt_lens = val
                    break
        return _circular_pool_batch(hidden_states, prompt_lens=prompt_lens)


class GRTPearlNorm(nn.Module):
    """RMS over pair radii: rescales each pair's magnitude, keeps its phase."""

    def __init__(self, size: int):
        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(size, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        radius_sq = x[..., 0::2].square() + x[..., 1::2].square()
        rms = torch.sqrt(radius_sq.mean(-1, keepdim=True) + 1e-8)
        return x / rms * self.scale


class GRTRadialGELU(nn.Module):
    """GELU on pair magnitudes: each pair is scaled by `gelu(r) / r`."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pairs = x.view(*x.shape[:-1], -1, 2)
        radius = pairs.norm(dim=-1, keepdim=True)
        return (pairs * (F.gelu(radius) / (radius + self.eps))).flatten(-2)


class GRTEmbedding(nn.Module):
    """`hidden_size // 2` phase angles per token plus a log magnitude."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.phase_angles = _param(vocab_size, hidden_size // 2)
        self.log_magnitude = _param(vocab_size)
        self.register_buffer("_embedding_table_q", None, persistent=False)
        self.register_buffer("_embedding_table_scale", None, persistent=False)
        self._quant_bits: int | None = None

    @staticmethod
    def _vectors(angles: torch.Tensor, log_magnitude: torch.Tensor) -> torch.Tensor:
        pairs = torch.stack([angles.cos(), angles.sin()], dim=-1).flatten(-2)
        return pairs * log_magnitude.clamp(-5.0, 5.0).exp().unsqueeze(-1)

    def table(self) -> torch.Tensor:
        if (
            self._embedding_table_q is not None
            and self._embedding_table_scale is not None
            and self._quant_bits is not None
        ):
            return _dequantize_symmetric(
                self._embedding_table_q,
                self._embedding_table_scale,
                bits=self._quant_bits,
                dim=-1,
            )
        return self._vectors(self.phase_angles, self.log_magnitude)

    def unit_table(self) -> torch.Tensor:
        if self.phase_angles.numel() > 0:
            pa = self.phase_angles
            return torch.stack([pa.cos(), pa.sin()], dim=-1).flatten(-2) * (
                pa.shape[-1] ** -0.5
            )
        t = self.table()
        return t / (t.norm(dim=-1, keepdim=True) + 1e-6)

    def quantize_weights(self, bits: int = 8) -> None:
        full_table = self.table()
        qweight, scale = _quantize_symmetric(full_table, bits=bits, dim=-1)
        self.register_buffer("_embedding_table_q", qweight, persistent=False)
        self.register_buffer("_embedding_table_scale", scale, persistent=False)
        self._quant_bits = bits
        self.phase_angles = nn.Parameter(
            torch.empty(0, dtype=torch.float32, device=qweight.device),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if (
            self._embedding_table_q is not None
            and self._embedding_table_scale is not None
            and self._quant_bits is not None
        ):
            q_rows = self._embedding_table_q[input_ids]
            s_rows = self._embedding_table_scale[input_ids]
            return _dequantize_symmetric(q_rows, s_rows, bits=self._quant_bits, dim=-1)
        return self._vectors(
            self.phase_angles[input_ids], self.log_magnitude[input_ids]
        )


class GRTPositionalRotation(nn.Module):
    """Rotates every pair of the stream (or per-head Q/K) by `position * exp(log_freqs)`."""

    def __init__(self, config: Any):
        super().__init__()
        rope_dim = (
            config.hidden_size // config.num_attention_heads
            if getattr(config, "rope_placement", "stream") == "qk"
            else config.hidden_size
        )
        n_pairs = rope_dim // 2
        log_freqs = torch.arange(n_pairs, dtype=torch.float32) / n_pairs
        log_freqs = -log_freqs * math.log(config.pos_freq_base)
        if config.learnable_rope:
            self.log_freqs = nn.Parameter(log_freqs, requires_grad=False)
        else:
            self.register_buffer("log_freqs", log_freqs, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.to(torch.float32).unsqueeze(-1) * self.log_freqs.exp()
        if x.ndim == 3 and angles.ndim == 2 and x.shape[0] == angles.shape[0]:
            angles = angles.unsqueeze(1)
        return _rotate(x[..., 0::2], x[..., 1::2], angles.cos(), angles.sin())


class GRTAttention(nn.Module):
    """Multi-head attention whose Q/K/V/O projections are rotation chains."""

    def __init__(
        self,
        config: Any,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_vllm_attention: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = h = config.num_attention_heads
        self.head_dim = d = config.hidden_size // h
        self.scaling = d**-0.5
        self.is_causal = getattr(config, "is_causal", True)
        k, p = config.n_qkv_rotations, d // 2
        for name in ("q", "k", "v", "o"):
            setattr(self, f"{name}_angles", _param(k, h, p))
            setattr(self, f"{name}_scales", _param(k, h, d))
            setattr(self, f"{name}_coupling", _param(k, h, 4 * p, p))

        f32 = torch.float32
        self.register_buffer(
            "qkv_angles", torch.empty(k, 3 * h, p, dtype=f32), persistent=False
        )
        self.register_buffer(
            "qkv_coupling", torch.empty(k, 3 * h, 4 * p, p, dtype=f32), persistent=False
        )
        self.register_buffer(
            "qkv_gains", torch.empty(k, 3 * h, d, dtype=f32), persistent=False
        )
        self.register_buffer(
            "o_gains", torch.empty(k, h, d, dtype=f32), persistent=False
        )
        self.register_buffer("qkv_coupling_scale", None, persistent=False)
        self.register_buffer("o_coupling_scale", None, persistent=False)
        self._quant_bits: int | None = None
        self._constants_folded: bool = False

        gathers = None
        if k > 1:
            perms, inv_perms = _butterfly_perms(k, d)
            self.register_buffer("_attn_perms", perms)
            self.register_buffer("_attn_inv_perms", inv_perms)
            gathers = torch.empty(k + 1, d, dtype=torch.long)
        self.register_buffer("gathers", gathers, persistent=False)

        self.attn = None
        if use_vllm_attention and Attention is not None:
            try:
                self.attn = Attention(
                    h,
                    d,
                    self.scaling,
                    num_kv_heads=h,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.attn" if prefix else "attn",
                )
            except Exception:
                self.attn = None
        self.attn_dtype = torch.get_default_dtype()

    def fold_constants(self) -> None:
        qkv = ("q", "k", "v")
        self.qkv_angles.copy_(torch.cat([getattr(self, f"{n}_angles") for n in qkv], 1))
        if self._quant_bits is None and self.q_coupling.numel() > 0:
            self.qkv_coupling.copy_(
                torch.cat([getattr(self, f"{n}_coupling") for n in qkv], 1)
            )
        self.qkv_gains.copy_(
            F.softplus(torch.cat([getattr(self, f"{n}_scales") for n in qkv], 1))
        )
        self.o_gains.copy_(F.softplus(self.o_scales))
        if self.gathers is not None:
            self.gathers.copy_(_compose_gathers(self._attn_perms, self._attn_inv_perms))
        self._constants_folded = True

    def quantize_weights(self, bits: int = 8) -> None:
        if not self._constants_folded:
            self.fold_constants()
        if self._quant_bits is not None:
            qkv_full = _dequantize_symmetric(
                self.qkv_coupling,
                self.qkv_coupling_scale,
                bits=self._quant_bits,
                dim=-2,
            )
            o_full = _dequantize_symmetric(
                self.o_coupling, self.o_coupling_scale, bits=self._quant_bits, dim=-2
            )
        else:
            qkv = ("q", "k", "v")
            qkv_full = (
                torch.cat([getattr(self, f"{n}_coupling") for n in qkv], 1)
                if self.q_coupling.numel() > 0
                else self.qkv_coupling
            )
            o_full = self.o_coupling
        qkv_q, qkv_s = _quantize_symmetric(qkv_full, bits=bits, dim=-2)
        o_q, o_s = _quantize_symmetric(o_full, bits=bits, dim=-2)
        self.register_buffer("qkv_coupling", qkv_q, persistent=False)
        self.register_buffer("qkv_coupling_scale", qkv_s, persistent=False)
        self.o_coupling = nn.Parameter(o_q, requires_grad=False)
        self.register_buffer("o_coupling_scale", o_s, persistent=False)
        self._quant_bits = bits
        for name in ("q", "k", "v"):
            setattr(
                self,
                f"{name}_coupling",
                nn.Parameter(
                    torch.empty(0, dtype=torch.float32, device=qkv_q.device),
                    requires_grad=False,
                ),
            )

    def _sdpa_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        """Exact SDPA fallback for standalone execution or bidirectional multimodal towers."""
        t, h, d = q.shape[0], self.n_heads, self.head_dim
        qh = q.view(t, h, d)
        kh = k.view(t, h, d)
        vh = v.view(t, h, d)

        lens = seq_lens if seq_lens else [t]
        outputs = []
        offset = 0
        for slen in lens:
            if slen <= 0:
                continue
            qi = qh[offset : offset + slen].transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
            ki = kh[offset : offset + slen].transpose(0, 1).unsqueeze(0)
            vi = vh[offset : offset + slen].transpose(0, 1).unsqueeze(0)
            outi = (
                F.scaled_dot_product_attention(
                    qi.float(),
                    ki.float(),
                    vi.float(),
                    is_causal=self.is_causal and slen > 1,
                    scale=self.scaling,
                )
                .squeeze(0)
                .transpose(0, 1)
                .reshape(slen, h * d)
            )
            outputs.append(outi)
            offset += slen
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    def forward(
        self,
        x: torch.Tensor,
        qk_rope: tuple[GRTPositionalRotation, torch.Tensor] | None = None,
        seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        if not self._constants_folded:
            self.fold_constants()
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        t, h, d = x_flat.shape[0], self.n_heads, self.head_dim
        qkv = _coupled_chain(
            x_flat.view(t, h, d).repeat(1, 3, 1),
            3 * h,
            self.qkv_angles,
            self.qkv_coupling,
            self.qkv_gains,
            self.gathers,
            1e-6,
            coupling_scale=self.qkv_coupling_scale,
            coupling_bits=self._quant_bits,
        )
        q_part, k_part, v_part = qkv.split(h, dim=1)
        if qk_rope is not None:
            pos_enc, pos_ids = qk_rope
            q_part = pos_enc(q_part, pos_ids)
            k_part = pos_enc(k_part, pos_ids)

        q = q_part.reshape(t, h * d)
        k = k_part.reshape(t, h * d)
        v = v_part.reshape(t, h * d)

        if self.attn is not None and is_forward_context_available():
            out = (
                self.attn(
                    q.to(self.attn_dtype),
                    k.to(self.attn_dtype),
                    v.to(self.attn_dtype),
                )
                .view(t, h, d)
                .to(torch.float32)
            )
        else:
            if seq_lens is None and len(orig_shape) == 3 and orig_shape[0] > 0:
                seq_lens = [orig_shape[1]] * orig_shape[0]
            out = self._sdpa_fallback(q, k, v, seq_lens=seq_lens).view(t, h, d)

        out = _coupled_chain(
            out,
            h,
            self.o_angles,
            self.o_coupling,
            self.o_gains,
            self.gathers,
            1e-6,
            coupling_scale=self.o_coupling_scale,
            coupling_bits=self._quant_bits,
        )
        return out.reshape(orig_shape)


class GRTMixer(nn.Module):
    """Mixes across heads at full width; the coupling is block-diagonal over
    groups of pairs."""

    def __init__(
        self, config: Any, quant_config: QuantizationConfig | None = None
    ):
        super().__init__()
        d = config.hidden_size
        n_pairs, k = d // 2, config.n_mixer_rotations
        self.group = next((g for g in (16, 8, 4, 2) if n_pairs % g == 0), 1)
        self.n_groups = n_pairs // self.group
        self.angles = _param(k, n_pairs)
        self.scales = _param(k, d)
        self.coupling = _param(k, self.n_groups, 4 * self.group, self.group)
        self.register_buffer("coupling_scale", None, persistent=False)
        self._quant_bits: int | None = None
        self._constants_folded: bool = False
        perms, inv_perms = _butterfly_perms(k, d)
        self.register_buffer("_perms_stack", perms)
        self.register_buffer("_inv_perms_stack", inv_perms)
        self.register_buffer(
            "gains", torch.empty(k, d, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gathers", torch.empty(k + 1, d, dtype=torch.long), persistent=False
        )

    def fold_constants(self) -> None:
        self.gains.copy_(F.softplus(self.scales))
        self.gathers.copy_(_compose_gathers(self._perms_stack, self._inv_perms_stack))
        self._constants_folded = True

    def quantize_weights(self, bits: int = 8) -> None:
        if not self._constants_folded:
            self.fold_constants()
        full_c = (
            _dequantize_symmetric(
                self.coupling, self.coupling_scale, bits=self._quant_bits, dim=-2
            )
            if self._quant_bits is not None and self.coupling_scale is not None
            else self.coupling
        )
        q, s = _quantize_symmetric(full_c, bits=bits, dim=-2)
        self.coupling = nn.Parameter(q, requires_grad=False)
        self.register_buffer("coupling_scale", s, persistent=False)
        self._quant_bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._constants_folded:
            self.fold_constants()
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        t, d = x_flat.shape
        k = self.angles.shape[0]
        y = _coupled_chain(
            x_flat.view(t, 1, d),
            self.n_groups,
            self.angles.view(k, self.n_groups, self.group),
            self.coupling,
            self.gains.view(k, self.n_groups, 2 * self.group),
            self.gathers,
            # 1e-2, not the projections' 1e-6: the trained mixer depends on it.
            1e-2,
            mixer_bivector=True,
            coupling_scale=self.coupling_scale,
            coupling_bits=self._quant_bits,
        )
        return y.reshape(orig_shape)


class GRTRotationalRouter(nn.Module):
    """100% rotational MoR router: cosine similarity against `n_experts` planar phase-angle prototypes."""

    def __init__(self, hidden_size: int, n_experts: int):
        super().__init__()
        self.phase_angles = _param(n_experts, hidden_size // 2)
        self.output_scale = _param()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-6)
        unit_table = torch.stack(
            [self.phase_angles.cos(), self.phase_angles.sin()], dim=-1
        ).flatten(-2)
        unit_table = unit_table * (self.phase_angles.shape[-1] ** -0.5)
        return (normed @ unit_table.T) * F.softplus(self.output_scale)


class GRTFeedForward(nn.Module):
    """Zero-pad to `intermediate_size`, rotate, radial GELU on pair magnitudes,
    rotate, truncate back to `hidden_size`. Supports both single-chain and
    Mixture-of-Rotations (`MoR`, `ffn_n_experts >= 2`)."""

    def __init__(self, config: Any):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = width = config.intermediate_size
        k = config.ffn_n_rotations
        self.n_rotations = k
        self.n_experts = int(getattr(config, "ffn_n_experts", 0) or 0)
        self.mor_mode = getattr(config, "mor_mode", "direct")
        self.mor_delta_scale = float(getattr(config, "mor_delta_scale", 0.5))
        self.mor_router_mode = getattr(config, "mor_router_mode", "rotational")
        self.mor_split_routers = bool(getattr(config, "mor_split_routers", False))
        self.mor_top_k = getattr(config, "mor_top_k", None)
        n_pairs = width // 2
        f32 = torch.float32

        if self.n_experts >= 2:
            if self.mor_router_mode == "rotational":
                if self.mor_split_routers:
                    self.expand_router = GRTRotationalRouter(
                        self.hidden_size, self.n_experts
                    )
                    self.contract_router = GRTRotationalRouter(
                        self.hidden_size, self.n_experts
                    )
                else:
                    self.router = GRTRotationalRouter(self.hidden_size, self.n_experts)
            else:
                if self.mor_split_routers:
                    self.expand_router = nn.Linear(
                        self.hidden_size, self.n_experts, bias=False
                    )
                    self.contract_router = nn.Linear(
                        self.hidden_size, self.n_experts, bias=False
                    )
                else:
                    self.router_linear = nn.Linear(
                        self.hidden_size, self.n_experts, bias=False
                    )

            if self.mor_mode == "residual":
                self.expand_base_angles = _param(k, n_pairs)
                self.expand_delta_angles = _param(self.n_experts, k, n_pairs)
                self.expert_expand_scales = _param(self.n_experts, k, width)
                self.contract_base_angles = _param(k, n_pairs)
                self.contract_delta_angles = _param(self.n_experts, k, n_pairs)
                self.expert_contract_scales = _param(self.n_experts, k, width)
            else:
                self.expert_expand_angles = _param(self.n_experts, k, n_pairs)
                self.expert_expand_scales = _param(self.n_experts, k, width)
                self.expert_contract_angles = _param(self.n_experts, k, n_pairs)
                self.expert_contract_scales = _param(self.n_experts, k, width)

            self.register_buffer(
                "expert_expand_gains",
                torch.empty(self.n_experts, k, width, dtype=f32),
                persistent=False,
            )
            self.register_buffer(
                "expert_contract_gains",
                torch.empty(self.n_experts, k, width, dtype=f32),
                persistent=False,
            )
        else:
            self.expand_angles = _param(k, n_pairs)
            self.expand_scales = _param(k, width)
            self.contract_angles = _param(k, n_pairs)
            self.contract_scales = _param(k, width)
            self.register_buffer(
                "rotations",
                torch.empty(2, 2, k, n_pairs, dtype=f32),
                persistent=False,
            )
            self.register_buffer(
                "gains", torch.empty(2, k, width, dtype=f32), persistent=False
            )

        self._constants_folded: bool = False
        gathers = None
        if k > 1:
            perms, inv_perms = _butterfly_perms(k, width)
            self.register_buffer("_ffn_perms", perms)
            self.register_buffer("_ffn_inv_perms", inv_perms)
            gathers = torch.empty(k + 1, width, dtype=torch.long)
        self.register_buffer("gathers", gathers, persistent=False)

    def effective_expand_angles(self) -> torch.Tensor:
        if self.mor_mode == "residual":
            return self.expand_base_angles.unsqueeze(
                0
            ) + self.mor_delta_scale * torch.tanh(self.expand_delta_angles)
        return self.expert_expand_angles

    def effective_contract_angles(self) -> torch.Tensor:
        if self.mor_mode == "residual":
            return self.contract_base_angles.unsqueeze(
                0
            ) + self.mor_delta_scale * torch.tanh(self.contract_delta_angles)
        return self.expert_contract_angles

    def _route(self, logits: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        if (
            self.mor_top_k is None
            or self.mor_top_k <= 0
            or self.mor_top_k >= self.n_experts
        ):
            return F.softmax(logits_f, dim=-1)
        top_vals, top_idx = torch.topk(logits_f, k=self.mor_top_k, dim=-1)
        top_weights = F.softmax(top_vals, dim=-1)
        weights = torch.zeros_like(logits_f)
        return weights.scatter(-1, top_idx, top_weights)

    def fold_constants(self) -> None:
        if self.n_experts >= 2:
            self.expert_expand_gains.copy_(F.softplus(self.expert_expand_scales))
            self.expert_contract_gains.copy_(F.softplus(self.expert_contract_scales))
        else:
            angles = torch.stack([self.expand_angles, self.contract_angles])
            self.rotations.copy_(torch.stack([angles.cos(), angles.sin()], dim=1))
            self.gains.copy_(
                F.softplus(torch.stack([self.expand_scales, self.contract_scales]))
            )
        if self.gathers is not None:
            self.gathers.copy_(_compose_gathers(self._ffn_perms, self._ffn_inv_perms))
        self._constants_folded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._constants_folded:
            self.fold_constants()
        if self.n_experts >= 2:
            if self.mor_split_routers:
                expand_weights = self._route(self.expand_router(x))
                contract_weights = self._route(self.contract_router(x))
            else:
                router_mod = (
                    self.router
                    if self.mor_router_mode == "rotational"
                    else self.router_linear
                )
                expand_weights = self._route(router_mod(x))
                contract_weights = expand_weights

            x = F.pad(x, (0, self.intermediate_size - self.hidden_size))
            x = _mor_chain(
                x,
                self.effective_expand_angles(),
                self.expert_expand_gains,
                expand_weights,
                self.gathers,
            )
            pairs = x.view(*x.shape[:-1], -1, 2)
            radius = pairs.norm(dim=-1, keepdim=True)
            x = (pairs * (F.gelu(radius) / (radius + 1e-8))).flatten(-2)
            x = _mor_chain(
                x,
                self.effective_contract_angles(),
                self.expert_contract_gains,
                contract_weights,
                self.gathers,
            )
            return x[..., : self.hidden_size]

        (expand_cos, expand_sin), (contract_cos, contract_sin) = self.rotations
        x = F.pad(x, (0, self.intermediate_size - self.hidden_size))
        x = _plain_chain(x, expand_cos, expand_sin, self.gains[0], self.gathers)
        pairs = x.view(*x.shape[:-1], -1, 2)
        radius = pairs.norm(dim=-1, keepdim=True)
        x = (pairs * (F.gelu(radius) / (radius + 1e-8))).flatten(-2)
        x = _plain_chain(x, contract_cos, contract_sin, self.gains[1], self.gathers)
        return x[..., : self.hidden_size]


class GRTDecoderLayer(nn.Module):
    """norm -> attention -> norm -> mixer -> norm -> feed-forward, each written
    into the residual stream through a learned gate."""

    def __init__(
        self,
        config: Any,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_vllm_attention: bool = True,
    ) -> None:
        super().__init__()
        d = config.hidden_size
        self.gate_mode = (
            config.residual_gate_mode if config.learnable_residual_gates else "none"
        )
        if self.gate_mode != "none":
            self.res_gate_attn = _param()
            self.res_gate_mixer = _param()
            self.res_gate_ffn = _param()
        self.register_buffer(
            "gates", torch.ones(3, dtype=torch.float32), persistent=False
        )
        self._constants_folded: bool = False

        self.norm1 = GRTPearlNorm(d)
        self.attention = GRTAttention(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention" if prefix else "attention",
            use_vllm_attention=use_vllm_attention,
        )
        self.mixer = (
            GRTMixer(config, quant_config=quant_config)
            if config.n_mixer_rotations > 0
            else None
        )
        self.norm_mixer = GRTPearlNorm(d) if self.mixer is not None else None
        self.norm_ffn = GRTPearlNorm(d)
        self.ffn = GRTFeedForward(config)
        self.post_norm_attn = GRTPearlNorm(d) if config.peri_ln else None
        self.post_norm_ffn = GRTPearlNorm(d) if config.peri_ln else None
        self.res_norm_block = (
            GRTPearlNorm(d) if config.residual_norm_mode == "pearlnorm" else None
        )

    def fold_constants(self) -> None:
        if self.gate_mode != "none":
            raw = torch.stack(
                [self.res_gate_attn, self.res_gate_mixer, self.res_gate_ffn]
            )
            if self.gate_mode == "bounded":
                raw = 1.0 + 0.5 * raw.tanh()
            elif self.gate_mode == "bounded_depth":
                raw = 0.55 + 0.45 * raw.tanh()
            self.gates.copy_(raw)
        self.attention.fold_constants()
        if self.mixer is not None:
            self.mixer.fold_constants()
        self.ffn.fold_constants()
        self._constants_folded = True

    def quantize_weights(self, bits: int = 8) -> None:
        if not self._constants_folded:
            self.fold_constants()
        self.attention.quantize_weights(bits)
        if self.mixer is not None:
            self.mixer.quantize_weights(bits)

    def forward(
        self,
        x: torch.Tensor,
        qk_rope: tuple[GRTPositionalRotation, torch.Tensor] | None = None,
        seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        if not self._constants_folded:
            self.fold_constants()
        y = self.attention(self.norm1(x), qk_rope=qk_rope, seq_lens=seq_lens)
        if self.post_norm_attn is not None:
            y = self.post_norm_attn(y)
        x = x + self.gates[0] * y
        if self.mixer is not None and self.norm_mixer is not None:
            x = x + self.gates[1] * self.mixer(self.norm_mixer(x))
        y = self.ffn(self.norm_ffn(x))
        if self.post_norm_ffn is not None:
            y = self.post_norm_ffn(y)
        x = x + self.gates[2] * y
        if self.res_norm_block is not None:
            x = self.res_norm_block(x)
        return x


class GRTCosineHead(nn.Module):
    """The head's learned temperature and unit embedding table."""

    def __init__(self, vocab_size: int = 0, hidden_size: int = 0) -> None:
        super().__init__()
        self.output_scale = _param()
        self.register_buffer(
            "weight",
            torch.empty(vocab_size, hidden_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_unit_table",
            torch.empty(vocab_size, hidden_size, dtype=torch.float32),
            persistent=False,
        )
        self._has_unit_table: bool = False
        self.register_buffer("_head_q", None, persistent=False)
        self.register_buffer("_head_scale", None, persistent=False)
        self._quant_bits: int | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states / (
            hidden_states.norm(dim=-1, keepdim=True) + 1e-6
        )
        if self._quant_bits is not None and self._head_q is not None:
            head = _dequantize_symmetric(
                self._head_q, self._head_scale, bits=self._quant_bits, dim=-1
            )
            return hidden_states.to(torch.float32) @ head.T
        if self._has_unit_table:
            return (
                hidden_states.to(torch.float32) @ self._unit_table.T
            ) * F.softplus(self.output_scale)
        return hidden_states.to(torch.float32) @ self.weight.T


@support_torch_compile
class GRTBackbone(nn.Module):
    """Core rotational Transformer stack (`embedding`, `pos_encoding`, `blocks`, `final_norm`)."""

    def __init__(
        self,
        config: GRTConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_vllm_attention: bool = True,
    ):
        super().__init__()
        self.config = config
        self.embedding = GRTEmbedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = GRTPositionalRotation(config)
        self.blocks = nn.ModuleList(
            GRTDecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{i}" if prefix else f"blocks.{i}",
                use_vllm_attention=use_vllm_attention,
            )
            for i in range(config.num_hidden_layers)
        )
        self.final_norm = GRTPearlNorm(config.hidden_size)

    def get_input_embeddings(self) -> GRTEmbedding:
        return self.embedding

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def fold_constants(self) -> None:
        for block in self.blocks:
            block.fold_constants()

    def quantize_weights(self, bits: int = 8) -> None:
        self.embedding.quantize_weights(bits)
        for block in self.blocks:
            block.quantize_weights(bits)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        x = inputs_embeds.to(torch.float32)
        qk_rope = None
        if getattr(self.config, "rope_placement", "stream") == "qk":
            qk_rope = (self.pos_encoding, positions)
        else:
            x = self.pos_encoding(x, positions)
        for block in self.blocks:
            x = block(x, qk_rope=qk_rope, seq_lens=seq_lens)
        return self.final_norm(x)


@support_torch_compile
class GRTModel(nn.Module):
    """vLLM `GRTModel` supporting both standalone backbone calls and
    circular-mean pooled embedding requests (`MatterWaveAI/grt-30m-embedding`)."""

    is_pooling_model = True
    default_seq_pooling_type = "MEAN"
    default_tok_pooling_type = "ALL"

    def __init__(
        self,
        config: GRTConfig | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config, quant_config, cache_config = _unwrap_vllm_args(
            config, vllm_config, quant_config, cache_config
        )
        _check_config(config)
        self.config = config
        self.quant_config = quant_config
        self.embedding = GRTEmbedding(config.vocab_size, config.hidden_size)
        self.pos_encoding = GRTPositionalRotation(config)
        self.blocks = nn.ModuleList(
            GRTDecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{i}" if prefix else f"blocks.{i}",
            )
            for i in range(config.num_hidden_layers)
        )
        self.final_norm = GRTPearlNorm(config.hidden_size)
        self.pooler = GRTCircularPooler(normalize=True)

    def get_input_embeddings(self) -> GRTEmbedding:
        return self.embedding

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def fold_constants(self) -> None:
        for block in self.blocks:
            block.fold_constants()

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        self.embedding.quantize_weights(bits)
        for block in self.blocks:
            block.quantize_weights(bits)

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        self.fold_constants()
        bits = _resolve_quant_bits(self.config, self.quant_config)
        if bits is not None:
            self.quantize_weights(bits=bits)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stripped = (
            (name.removeprefix("model."), tensor)
            for name, tensor in weights
            if not name.startswith("lm_head.")
        )
        loaded = AutoWeightsLoader(self).load_weights(stripped)
        self.process_weights_after_loading()
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        prompt_lens: Sequence[int] | torch.Tensor | None = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        x = inputs_embeds.to(torch.float32)
        qk_rope = None
        if getattr(self.config, "rope_placement", "stream") == "qk":
            qk_rope = (self.pos_encoding, positions)
        else:
            x = self.pos_encoding(x, positions)
        seq_lens = (
            [int(s) for s in prompt_lens]
            if prompt_lens is not None
            else None
        )
        for block in self.blocks:
            x = block(x, qk_rope=qk_rope, seq_lens=seq_lens)
        hidden_states = self.final_norm(x)
        if get_embedding:
            return self.pooler(hidden_states, prompt_lens=prompt_lens)
        return hidden_states


class GRTEmbeddingModel(GRTModel):
    """Explicit circular-pooling embedding model wrapper around `GRTModel`."""

    is_pooling_model = True


class GRTForCausalLM(nn.Module):
    """vLLM `GRTForCausalLM` causal language model (`MatterWaveAI/grt-30m`,
    `grt-30m-chat`, `grt-30m-mor`)."""

    def __init__(
        self,
        config: GRTConfig | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config, quant_config, cache_config = _unwrap_vllm_args(
            config, vllm_config, quant_config, cache_config
        )
        _check_config(config)
        try:
            tp_size = get_tensor_model_parallel_world_size()
        except Exception:
            tp_size = 1
        if tp_size != 1:
            raise ValueError(
                "GRT has no dense weights to shard; use tensor_parallel_size=1"
            )
        self.config = config
        self.quant_config = quant_config
        self.model = GRTBackbone(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = GRTCosineHead(config.vocab_size, config.hidden_size)
        self.register_buffer(
            "head",
            torch.empty(config.vocab_size, config.hidden_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("_head_scale", None, persistent=False)
        self._quant_bits: int | None = None
        self.pooler = GRTCircularPooler(normalize=True)

    def get_input_embeddings(self) -> GRTEmbedding:
        return self.model.embedding

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def fold_constants(self) -> None:
        self.model.fold_constants()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        unit_table = self.model.embedding.unit_table()
        full_head = unit_table * F.softplus(self.lm_head.output_scale)
        self.model.quantize_weights(bits)
        head_q, head_s = _quantize_symmetric(full_head, bits=bits, dim=-1)
        self.lm_head.register_buffer("_head_q", head_q, persistent=False)
        self.lm_head.register_buffer("_head_scale", head_s, persistent=False)
        self.lm_head._quant_bits = bits
        dequant = _dequantize_symmetric(head_q, head_s, bits=bits, dim=-1)
        self.lm_head.weight.copy_(dequant)
        self.register_buffer("head", head_q, persistent=False)
        self.register_buffer("_head_scale", head_s, persistent=False)
        self._quant_bits = bits

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        self.model.fold_constants()
        bits = _resolve_quant_bits(self.config, self.quant_config)
        if bits is not None:
            self.quantize_weights(bits=bits)
        else:
            unit_table = self.model.embedding.unit_table()
            full_head = unit_table * F.softplus(self.lm_head.output_scale)
            self.lm_head._unit_table.copy_(unit_table)
            self.lm_head._has_unit_table = True
            self.lm_head.weight.copy_(full_head)
            self.head.copy_(full_head)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = (
            (name, tensor)
            for name, tensor in weights
            if name != "lm_head.phase_angles"
        )
        loaded = AutoWeightsLoader(self).load_weights(filtered)
        self.process_weights_after_loading()
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        hidden_states = self.model(
            input_ids, positions, inputs_embeds=inputs_embeds
        )
        if return_logits:
            return self.compute_logits(hidden_states)
        return hidden_states


class GRTCosineClassificationHead(nn.Module):
    """Zero-`nn.Linear` rotational sequence classification and cross-encoder reranking head."""

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.num_labels = getattr(config, "num_labels", 1)
        n_pairs = config.hidden_size // 2
        self.phase_angles = nn.Parameter(
            torch.zeros(self.num_labels, n_pairs, dtype=torch.float32),
            requires_grad=False,
        )
        self.pair_scales = nn.Parameter(
            torch.full(
                (self.num_labels, n_pairs),
                _raw_scale_init(1.0),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.late_weights = nn.Parameter(
            torch.tensor([[0.45, 0.35, 0.25, 0.35]], dtype=torch.float32)
            .expand(self.num_labels, 4)
            .clone(),
            requires_grad=False,
        )
        self.output_scale = nn.Parameter(
            torch.tensor(_raw_scale_init(4.0), dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(self.num_labels, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer("_unit_table_q", None, persistent=False)
        self.register_buffer("_unit_table_scale", None, persistent=False)
        self._quant_bits: int | None = None

    def _rot_table(self) -> torch.Tensor:
        gains = F.softplus(self.pair_scales)
        return torch.stack(
            [gains * torch.cos(self.phase_angles), gains * torch.sin(self.phase_angles)],
            dim=-1,
        ).flatten(-2)

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        unit_table = (
            _dequantize_symmetric(
                self._unit_table_q,
                self._unit_table_scale,
                bits=self._quant_bits,
                dim=-1,
            )
            if self._unit_table_q is not None
            and self._unit_table_scale is not None
            and self._quant_bits is not None
            else self._rot_table()
        )
        uq, us = _quantize_symmetric(unit_table, bits=bits, dim=-1)
        self.register_buffer("_unit_table_q", uq, persistent=False)
        self.register_buffer("_unit_table_scale", us, persistent=False)
        self._quant_bits = bits
        self.phase_angles = nn.Parameter(
            torch.empty(0, dtype=torch.float32, device=uq.device),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        late_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self._unit_table_q is not None
            and self._unit_table_scale is not None
            and self._quant_bits is not None
        ):
            rot_table = _dequantize_symmetric(
                self._unit_table_q,
                self._unit_table_scale,
                bits=self._quant_bits,
                dim=-1,
            ).to(hidden_states.dtype)
        else:
            rot_table = self._rot_table().to(hidden_states.dtype)
        raw_score = hidden_states @ rot_table.T
        if late_features is not None:
            raw_score = raw_score + (
                late_features.to(hidden_states.dtype)
                @ self.late_weights.to(hidden_states.dtype).T
            )
        return raw_score * F.softplus(self.output_scale) + self.bias


class GRTForSequenceClassification(nn.Module):
    """vLLM `GRTForSequenceClassification` cross-encoder reranker and sequence
    classifier (`MatterWaveAI/grt-30m-reranker`)."""

    is_pooling_model = True
    score_type = "cross-encoder"
    default_seq_pooling_type = "MEAN"
    default_tok_pooling_type = "ALL"

    def __init__(
        self,
        config: GRTConfig | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config, quant_config, cache_config = _unwrap_vllm_args(
            config, vllm_config, quant_config, cache_config
        )
        _check_config(config)
        self.config = config
        self.quant_config = quant_config
        self.num_labels = getattr(config, "num_labels", 1)
        self.model = GRTBackbone(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.score = GRTCosineClassificationHead(config)
        self.pooler = GRTCircularPooler(normalize=True)

    def get_input_embeddings(self) -> GRTEmbedding:
        return self.model.embedding

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @staticmethod
    def _circular_pool_seq(seq_hidden: torch.Tensor) -> torch.Tensor:
        return _circular_pool_seq(seq_hidden)

    @staticmethod
    def _hermitian_pair_features_single(
        h_q: torch.Tensor,
        h_d: torch.Tensor,
        q_lex: torch.Tensor,
        d_lex: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Hermitian conjugate product `u* v` and multi-vector MaxSim/MinSim
        for a single `(query, doc)` pair (`h_q: [L_q, D]`, `h_d: [L_d, D]`)."""
        u = _circular_pool_seq(h_q)  # [1, D]
        v = _circular_pool_seq(h_d)  # [1, D]
        u_pairs = u.float().unflatten(-1, (-1, 2))
        v_pairs = v.float().unflatten(-1, (-1, 2))
        ux, uy = u_pairs[..., 0], u_pairs[..., 1]
        vx, vy = v_pairs[..., 0], v_pairs[..., 1]
        hermitian = (
            torch.stack([ux * vx + uy * vy, ux * vy - uy * vx], dim=-1)
            .flatten(-2)
            .to(h_q.dtype)
        )

        hq_norm = F.normalize(h_q.float(), p=2, dim=-1)
        hd_norm = F.normalize(h_d.float(), p=2, dim=-1)
        sim_mat = torch.matmul(hq_norm, hd_norm.T)  # [L_q, L_d]
        max_q2d = sim_mat.max(dim=1).values
        q2d_mean = max_q2d.mean()
        min_q2d = max_q2d.min()
        max_d2q = sim_mat.max(dim=0).values
        d2q_mean = max_d2q.mean()

        u_lex = _circular_pool_seq(q_lex)
        v_lex = _circular_pool_seq(d_lex)
        lex_cos = (u_lex.float() * v_lex.float()).sum(dim=-1).squeeze(0)

        late_feats = (
            torch.stack([q2d_mean, d2q_mean, min_q2d, lex_cos], dim=-1)
            .unsqueeze(0)
            .to(h_q.dtype)
        )
        return hermitian, late_feats

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        self.model.quantize_weights(bits)
        self.score.quantize_weights(bits)

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        self.model.fold_constants()
        bits = _resolve_quant_bits(self.config, self.quant_config)
        if bits is not None:
            self.quantize_weights(bits=bits)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights)
        self.process_weights_after_loading()
        return loaded

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.score(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        prompt_lens: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        if token_type_ids is not None and bool((token_type_ids == 1).any().item()):
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_input_ids(input_ids)
            flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
            flat_tt = token_type_ids.reshape(-1)
            if prompt_lens is not None:
                seq_lens = [int(x) for x in prompt_lens]
            elif inputs_embeds.ndim == 3:
                seq_lens = [inputs_embeds.shape[1]] * inputs_embeds.shape[0]
            else:
                seq_lens = [flat_embeds.shape[0]]

            scores_list = []
            offset = 0
            for slen in seq_lens:
                emb_i = flat_embeds[offset : offset + slen]
                tt_i = flat_tt[offset : offset + slen]
                q_emb = emb_i[tt_i == 0]
                d_emb = emb_i[tt_i == 1]
                if q_emb.shape[0] > 0 and d_emb.shape[0] > 0:
                    pos_q = torch.arange(q_emb.shape[0], device=q_emb.device)
                    pos_d = torch.arange(d_emb.shape[0], device=d_emb.device)
                    h_q = self.model(
                        None,
                        pos_q,
                        inputs_embeds=q_emb,
                        seq_lens=[q_emb.shape[0]],
                    )
                    h_d = self.model(
                        None,
                        pos_d,
                        inputs_embeds=d_emb,
                        seq_lens=[d_emb.shape[0]],
                    )
                    herm, late = self._hermitian_pair_features_single(
                        h_q, h_d, q_emb, d_emb
                    )
                    scores_list.append(self.score(herm, late))
                else:
                    pos_all = torch.arange(emb_i.shape[0], device=emb_i.device)
                    h_all = self.model(
                        None,
                        pos_all,
                        inputs_embeds=emb_i,
                        seq_lens=[emb_i.shape[0]],
                    )
                    pooled = _circular_pool_seq(h_all)
                    scores_list.append(self.score(pooled, None))
                offset += slen
            return torch.cat(scores_list, dim=0)

        seq_lens = (
            [int(x) for x in prompt_lens]
            if prompt_lens is not None
            else None
        )
        hidden_states = self.model(
            input_ids, positions, inputs_embeds=inputs_embeds, seq_lens=seq_lens
        )
        pooled = _circular_pool_batch(hidden_states, prompt_lens=prompt_lens)
        return self.score(pooled, None)


# ==============================================================================
# Multimodal Rotational Encoders & Vocoder (VL, Audio, ASR, TTS, Omni)
# ==============================================================================


class GRTPolarPatchEmbedding(nn.Module):
    """Zero-`nn.Linear` rotational image patch stem."""

    def __init__(self, config: GRTVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.n_rotations = config.patch_n_rotations
        self.raw_patch_dim = self.num_channels * self.patch_size * self.patch_size
        n_pairs = self.intermediate_size // 2
        self.norm_in = GRTPearlNorm(self.intermediate_size)
        self.angles = _param(self.n_rotations, n_pairs)
        self.scales = _param(self.n_rotations, self.intermediate_size)
        self.norm_out = GRTPearlNorm(self.hidden_size)
        if self.n_rotations > 1:
            perms, inv_perms = _butterfly_perms(
                self.n_rotations, self.intermediate_size
            )
            self.register_buffer("_patch_perms", perms)
            self.register_buffer("_patch_inv_perms", inv_perms)
        else:
            self._patch_perms = self._patch_inv_perms = None

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, channels, height, width = pixel_values.shape
        p = self.patch_size
        grid_h, grid_w = height // p, width // p
        patches = (
            pixel_values.to(torch.float32)
            .view(batch_size, channels, grid_h, p, grid_w, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, grid_h * grid_w, self.raw_patch_dim)
        )
        padded = F.pad(patches, (0, self.intermediate_size - self.raw_patch_dim))
        normed = self.norm_in(padded)
        rotated = _plain_rotate_raw(
            normed, self.angles, self.scales, self._patch_perms, self._patch_inv_perms
        )
        return self.norm_out(rotated[..., : self.hidden_size]), grid_h, grid_w


class GRTRotaryEmbedding2D(nn.Module):
    """2D axial rotational position encoding for planar-pair patch tokens."""

    def __init__(self, config: GRTVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_pairs = config.hidden_size // 2
        self.half_pairs = self.n_pairs // 2
        default_freqs = (
            -math.log(config.pos_freq_base)
            * torch.arange(0, self.half_pairs, dtype=torch.float32)
            / self.half_pairs
        )
        if config.learnable_rope:
            self.log_freqs_y = nn.Parameter(default_freqs.clone(), requires_grad=False)
            self.log_freqs_x = nn.Parameter(default_freqs.clone(), requires_grad=False)
        else:
            self.register_buffer("log_freqs_y", default_freqs.clone(), persistent=False)
            self.register_buffer("log_freqs_x", default_freqs.clone(), persistent=False)

    def forward(
        self, hidden_states: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        device = hidden_states.device
        rows = torch.arange(grid_h, device=device, dtype=torch.float32)
        cols = torch.arange(grid_w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        angles_y = grid_y.reshape(-1, 1) * torch.exp(self.log_freqs_y.float())
        angles_x = grid_x.reshape(-1, 1) * torch.exp(self.log_freqs_x.float())
        angles = torch.cat([angles_y, angles_x], dim=-1).unsqueeze(0)
        return _rotate(
            hidden_states[..., 0::2],
            hidden_states[..., 1::2],
            angles.cos(),
            angles.sin(),
        )


class GRTSpatialRotationalMerger(nn.Module):
    """2x2 spatial pixel-unshuffle merger and Givens butterfly rotational projector."""

    def __init__(self, config: GRTVisionConfig, out_hidden_size: int | None = None):
        super().__init__()
        self.merge_size = config.spatial_merge_size
        self.in_hidden_size = config.hidden_size
        self.out_hidden_size = out_hidden_size or config.hidden_size
        self.merged_dim = max(
            self.in_hidden_size * (self.merge_size**2), self.out_hidden_size
        )
        self.n_rotations = config.merger_n_rotations
        n_pairs = self.merged_dim // 2

        self.norm_in = GRTPearlNorm(self.merged_dim)
        self.expand_angles = _param(self.n_rotations, n_pairs)
        self.expand_scales = _param(self.n_rotations, self.merged_dim)
        self.activation = GRTRadialGELU()
        self.contract_angles = _param(self.n_rotations, n_pairs)
        self.contract_scales = _param(self.n_rotations, self.merged_dim)
        self.norm_out = GRTPearlNorm(self.out_hidden_size)
        if self.n_rotations > 1:
            perms, inv_perms = _butterfly_perms(self.n_rotations, self.merged_dim)
            self.register_buffer("_merger_perms", perms)
            self.register_buffer("_merger_inv_perms", inv_perms)
        else:
            self._merger_perms = self._merger_inv_perms = None

    def forward(
        self, hidden_states: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        batch_size, _, dim = hidden_states.shape
        s = self.merge_size
        out_h, out_w = grid_h // s, grid_w // s
        merged = (
            hidden_states.view(batch_size, grid_h, grid_w, dim)
            .view(batch_size, out_h, s, out_w, s, dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size, out_h * out_w, s * s * dim)
        )
        if merged.shape[-1] < self.merged_dim:
            merged = F.pad(merged, (0, self.merged_dim - merged.shape[-1]))
        merged = self.norm_in(merged)
        merged = _plain_rotate_raw(
            merged,
            self.expand_angles,
            self.expand_scales,
            self._merger_perms,
            self._merger_inv_perms,
        )
        merged = self.activation(merged)
        merged = _plain_rotate_raw(
            merged,
            self.contract_angles,
            self.contract_scales,
            self._merger_perms,
            self._merger_inv_perms,
        )
        return self.norm_out(merged[..., : self.out_hidden_size])


class GRTVisionModel(nn.Module):
    """100% rotational vision tower (`GRTPolarPatchEmbedding` + bidirectional `GRTDecoderLayer` + `GRTSpatialRotationalMerger`)."""

    def __init__(self, config: GRTVisionConfig, out_hidden_size: int | None = None):
        super().__init__()
        self.config = config
        self.patch_embed = GRTPolarPatchEmbedding(config)
        self.pos_encoding = GRTRotaryEmbedding2D(config)
        self.blocks = nn.ModuleList(
            GRTDecoderLayer(config, use_vllm_attention=False)
            for _ in range(config.num_hidden_layers)
        )
        self.final_norm = GRTPearlNorm(config.hidden_size)
        self.merger = GRTSpatialRotationalMerger(
            config, out_hidden_size=out_hidden_size
        )

    def fold_constants(self) -> None:
        for block in self.blocks:
            block.fold_constants()

    def quantize_weights(self, bits: int = 8) -> None:
        for block in self.blocks:
            block.quantize_weights(bits)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states, grid_h, grid_w = self.patch_embed(pixel_values)
        hidden_states = self.pos_encoding(hidden_states, grid_h, grid_w)
        seq_lens = [hidden_states.shape[1]] * hidden_states.shape[0]
        for block in self.blocks:
            hidden_states = block(hidden_states, seq_lens=seq_lens)
        hidden_states = self.final_norm(hidden_states)
        return self.merger(hidden_states, grid_h, grid_w)


class GRTPolarAudioStem(nn.Module):
    """Zero-`nn.Linear`, zero-`nn.Conv1d` polar STFT rotational audio stem."""

    def __init__(self, config: GRTAudioConfig):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.max_audio_frames = config.max_audio_frames
        self.n_freq_bins = config.n_fft // 2 + 1
        self.spec_dim = 2 * self.n_freq_bins
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.n_rotations = config.stem_n_rotations

        n_pairs = self.intermediate_size // 2
        self.norm_in = GRTPearlNorm(self.intermediate_size)
        self.angles = _param(self.n_rotations, n_pairs)
        self.scales = _param(self.n_rotations, self.intermediate_size)
        self.norm_out = GRTPearlNorm(self.hidden_size)
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )
        if self.n_rotations > 1:
            perms, inv_perms = _butterfly_perms(
                self.n_rotations, self.intermediate_size
            )
            self.register_buffer("_stem_perms", perms)
            self.register_buffer("_stem_inv_perms", inv_perms)
        else:
            self._stem_perms = self._stem_inv_perms = None

    def waveform_to_polar_stft(self, audio_values: torch.Tensor) -> torch.Tensor:
        window = self.window.to(device=audio_values.device, dtype=torch.float32)
        if window.numel() != self.win_length:
            window = torch.hann_window(
                self.win_length, device=audio_values.device, dtype=torch.float32
            )
        stft = torch.stft(
            audio_values.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        if stft.shape[-1] > self.max_audio_frames:
            stft = stft[..., : self.max_audio_frames]
        elif stft.shape[-1] < self.max_audio_frames:
            stft = F.pad(stft, (0, self.max_audio_frames - stft.shape[-1]))
        stft = stft.transpose(1, 2)
        r = torch.log1p(stft.abs())
        phase = stft.angle()
        return torch.stack([r * torch.cos(phase), r * torch.sin(phase)], dim=-1).flatten(
            -2
        )

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        polar_frames = (
            self.waveform_to_polar_stft(audio_values)
            if audio_values.ndim == 2
            else audio_values.to(torch.float32)
        )
        if polar_frames.shape[-1] < self.intermediate_size:
            padded = F.pad(
                polar_frames, (0, self.intermediate_size - polar_frames.shape[-1])
            )
        else:
            padded = polar_frames[..., : self.intermediate_size]
        normed = self.norm_in(padded)
        rotated = _plain_rotate_raw(
            normed, self.angles, self.scales, self._stem_perms, self._stem_inv_perms
        )
        return self.norm_out(rotated[..., : self.hidden_size])


class GRTTemporalRotationalMerger(nn.Module):
    """Zero-`nn.Linear`, zero-`nn.Conv1d` temporal frame folding and Givens butterfly rotational projector."""

    def __init__(self, config: GRTAudioConfig, out_hidden_size: int | None = None):
        super().__init__()
        self.merge_size = config.temporal_merge_size
        self.in_hidden_size = config.hidden_size
        self.out_hidden_size = out_hidden_size or config.hidden_size
        self.merged_dim = max(
            self.in_hidden_size * self.merge_size, self.out_hidden_size
        )
        self.n_rotations = config.merger_n_rotations
        n_pairs = self.merged_dim // 2

        self.norm_in = GRTPearlNorm(self.merged_dim)
        self.expand_angles = _param(self.n_rotations, n_pairs)
        self.expand_scales = _param(self.n_rotations, self.merged_dim)
        self.activation = GRTRadialGELU()
        self.contract_angles = _param(self.n_rotations, n_pairs)
        self.contract_scales = _param(self.n_rotations, self.merged_dim)
        self.norm_out = GRTPearlNorm(self.out_hidden_size)
        if self.n_rotations > 1:
            perms, inv_perms = _butterfly_perms(self.n_rotations, self.merged_dim)
            self.register_buffer("_merger_perms", perms)
            self.register_buffer("_merger_inv_perms", inv_perms)
        else:
            self._merger_perms = self._merger_inv_perms = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape
        s = self.merge_size
        merged = hidden_states.reshape(batch_size, seq_len // s, s * dim)
        if merged.shape[-1] < self.merged_dim:
            merged = F.pad(merged, (0, self.merged_dim - merged.shape[-1]))
        merged = self.norm_in(merged)
        merged = _plain_rotate_raw(
            merged,
            self.expand_angles,
            self.expand_scales,
            self._merger_perms,
            self._merger_inv_perms,
        )
        merged = self.activation(merged)
        merged = _plain_rotate_raw(
            merged,
            self.contract_angles,
            self.contract_scales,
            self._merger_perms,
            self._merger_inv_perms,
        )
        return self.norm_out(merged[..., : self.out_hidden_size])


class GRTAudioModel(nn.Module):
    """100% rotational audio encoder tower (`GRTPolarAudioStem` + `GRTDecoderLayer` + `GRTTemporalRotationalMerger`)."""

    def __init__(self, config: GRTAudioConfig, out_hidden_size: int | None = None):
        super().__init__()
        self.config = config
        self.stem = GRTPolarAudioStem(config)
        self.pos_encoding = GRTPositionalRotation(config)
        self.blocks = nn.ModuleList(
            GRTDecoderLayer(config, use_vllm_attention=False)
            for _ in range(config.num_hidden_layers)
        )
        self.final_norm = GRTPearlNorm(config.hidden_size)
        self.merger = GRTTemporalRotationalMerger(
            config, out_hidden_size=out_hidden_size
        )

    def fold_constants(self) -> None:
        for block in self.blocks:
            block.fold_constants()

    def quantize_weights(self, bits: int = 8) -> None:
        for block in self.blocks:
            block.quantize_weights(bits)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.stem(audio_values)
        position_ids = torch.arange(
            hidden_states.shape[1], device=hidden_states.device
        ).unsqueeze(0)
        hidden_states = self.pos_encoding(hidden_states, position_ids)
        seq_lens = [hidden_states.shape[1]] * hidden_states.shape[0]
        for block in self.blocks:
            hidden_states = block(hidden_states, seq_lens=seq_lens)
        hidden_states = self.final_norm(hidden_states)
        return self.merger(hidden_states)


class GRTPolarVocoderHead(nn.Module):
    """Zero-`nn.Linear`, zero-`nn.Conv1d` rotational harmonic speech vocoder head."""

    def __init__(self, config: GRTAudioConfig, in_hidden_size: int | None = None):
        super().__init__()
        self.in_hidden_size = in_hidden_size or config.hidden_size
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.n_freq_bins = config.n_fft // 2 + 1
        self.spec_dim = 2 * self.n_freq_bins
        self.frames_per_token = config.frames_per_token
        self.expanded_dim = self.frames_per_token * self.spec_dim
        self.n_rotations = config.vocoder_n_rotations
        n_pairs = self.expanded_dim // 2

        self.norm_in = GRTPearlNorm(self.expanded_dim)
        self.expand_angles = _param(self.n_rotations, n_pairs)
        self.expand_scales = _param(self.n_rotations, self.expanded_dim)
        self.activation = GRTRadialGELU()
        self.contract_angles = _param(self.n_rotations, n_pairs)
        self.contract_scales = _param(self.n_rotations, self.expanded_dim)
        self.norm_out = GRTPearlNorm(self.spec_dim)
        self.output_scale = _param()
        self.register_buffer(
            "window", torch.hann_window(self.win_length), persistent=False
        )
        if self.n_rotations > 1:
            perms, inv_perms = _butterfly_perms(self.n_rotations, self.expanded_dim)
            self.register_buffer("_voc_perms", perms)
            self.register_buffer("_voc_inv_perms", inv_perms)
        else:
            self._voc_perms = self._voc_inv_perms = None

    def forward(
        self, hidden_states: torch.Tensor, return_waveform: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        orig_2d = hidden_states.ndim == 2
        if orig_2d:
            hidden_states = hidden_states.unsqueeze(0)
        batch_size, seq_len, dim = hidden_states.shape
        if dim < self.expanded_dim:
            reps = (self.expanded_dim + dim - 1) // dim
            padded = hidden_states.repeat(1, 1, reps)[..., : self.expanded_dim]
        else:
            padded = hidden_states[..., : self.expanded_dim]
        normed = self.norm_in(padded)
        rotated = _plain_rotate_raw(
            normed,
            self.expand_angles,
            self.expand_scales,
            self._voc_perms,
            self._voc_inv_perms,
        )
        rotated = self.activation(rotated)
        rotated = _plain_rotate_raw(
            rotated,
            self.contract_angles,
            self.contract_scales,
            self._voc_perms,
            self._voc_inv_perms,
        )
        frames = rotated.reshape(
            batch_size, seq_len * self.frames_per_token, self.spec_dim
        )
        polar_spec = self.norm_out(frames) * F.softplus(self.output_scale)
        waveform = None
        if return_waveform:
            spec_pairs = (
                polar_spec.float()
                .view(batch_size, -1, self.n_freq_bins, 2)
                .transpose(1, 2)
                .contiguous()
            )
            complex_spec = torch.view_as_complex(spec_pairs)
            window = self.window.to(device=polar_spec.device, dtype=torch.float32)
            if window.numel() != self.win_length:
                window = torch.hann_window(
                    self.win_length, device=polar_spec.device, dtype=torch.float32
                )
            waveform = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
            )
        return polar_spec, waveform


class _GRTVLInner(nn.Module):
    def __init__(
        self,
        config: GRTVLConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.visual = GRTVisionModel(
            config.vision_config, out_hidden_size=config.text_config.hidden_size
        )
        self.language_model = GRTBackbone(
            config.text_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )


class GRTVLForConditionalGeneration(nn.Module):
    """vLLM `GRTVLForConditionalGeneration` vision-language model (`MatterWaveAI/grt-30m-vl`)."""

    supports_multimodal = True
    supports_multimodal_embeddings = True

    def __init__(
        self,
        config: GRTVLConfig | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config, quant_config, cache_config = _unwrap_vllm_args(
            config, vllm_config, quant_config, cache_config
        )
        _check_config(config.text_config)
        self.config = config
        self.quant_config = quant_config
        self.model = _GRTVLInner(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = GRTCosineHead(
            config.text_config.vocab_size, config.text_config.hidden_size
        )

    def get_language_model(self) -> GRTBackbone:
        return self.model.language_model

    def get_input_embeddings(self) -> GRTEmbedding:
        return self.model.language_model.embedding

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor]:
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return []
        feats = self.model.visual(pixel_values)
        return [feats[i] for i in range(feats.shape[0])]

    def get_multimodal_embeddings(self, **kwargs: Any) -> list[torch.Tensor]:
        return self.embed_multimodal(**kwargs)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is not None and is_multimodal is not None:
            return _merge_multimodal_embeddings(
                inputs_embeds, multimodal_embeddings, is_multimodal
            )
        return inputs_embeds

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        self.model.visual.fold_constants()
        self.model.language_model.fold_constants()
        bits = _resolve_quant_bits(self.config, self.quant_config)
        if bits is not None:
            self.model.visual.quantize_weights(bits)
            unit_table = self.model.language_model.embedding.unit_table()
            full_head = unit_table * F.softplus(self.lm_head.output_scale)
            self.model.language_model.quantize_weights(bits)
            head_q, head_s = _quantize_symmetric(full_head, bits=bits, dim=-1)
            self.lm_head.register_buffer("_head_q", head_q, persistent=False)
            self.lm_head.register_buffer("_head_scale", head_s, persistent=False)
            self.lm_head._quant_bits = bits
            self.lm_head.weight.copy_(
                _dequantize_symmetric(head_q, head_s, bits=bits, dim=-1)
            )
        else:
            unit_table = self.model.language_model.embedding.unit_table()
            self.lm_head._unit_table.copy_(unit_table)
            self.lm_head._has_unit_table = True
            self.lm_head.weight.copy_(
                unit_table * F.softplus(self.lm_head.output_scale)
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = (
            (name, tensor)
            for name, tensor in weights
            if name != "lm_head.phase_angles"
        )
        loaded = AutoWeightsLoader(self).load_weights(filtered)
        self.process_weights_after_loading()
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if pixel_values is not None and input_ids is not None:
            img_feats = self.model.visual(pixel_values).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            mask = input_ids == self.config.image_token_id
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = img_feats.reshape(-1, inputs_embeds.shape[-1])
        hidden_states = self.model.language_model(
            None, positions, inputs_embeds=inputs_embeds
        )
        if return_logits or pixel_values is not None:
            return self.compute_logits(hidden_states)
        return hidden_states


class _GRTASRInner(nn.Module):
    def __init__(
        self,
        config: GRTASRConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.audio_tower = GRTAudioModel(
            config.audio_config, out_hidden_size=config.text_config.hidden_size
        )
        self.language_model = GRTBackbone(
            config.text_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )


class GRTASRForConditionalGeneration(nn.Module):
    """vLLM `GRTASRForConditionalGeneration` speech-to-text model (`MatterWaveAI/grt-30m-asr`)."""

    supports_multimodal = True
    supports_multimodal_embeddings = True

    def __init__(
        self,
        config: GRTASRConfig | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config, quant_config, cache_config = _unwrap_vllm_args(
            config, vllm_config, quant_config, cache_config
        )
        _check_config(config.text_config)
        self.config = config
        self.quant_config = quant_config
        self.model = _GRTASRInner(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = GRTCosineHead(
            config.text_config.vocab_size, config.text_config.hidden_size
        )

    def get_language_model(self) -> GRTBackbone:
        return self.model.language_model

    def get_input_embeddings(self) -> GRTEmbedding:
        return self.model.language_model.embedding

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor]:
        audio_values = kwargs.get("audio_values")
        if audio_values is None:
            audio_values = kwargs.get("input_features")
        if audio_values is None:
            return []
        feats = self.model.audio_tower(audio_values)
        return [feats[i] for i in range(feats.shape[0])]

    def get_multimodal_embeddings(self, **kwargs: Any) -> list[torch.Tensor]:
        return self.embed_multimodal(**kwargs)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is not None and is_multimodal is not None:
            return _merge_multimodal_embeddings(
                inputs_embeds, multimodal_embeddings, is_multimodal
            )
        return inputs_embeds

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        self.model.audio_tower.fold_constants()
        self.model.language_model.fold_constants()
        bits = _resolve_quant_bits(self.config, self.quant_config)
        if bits is not None:
            self.model.audio_tower.quantize_weights(bits)
            unit_table = self.model.language_model.embedding.unit_table()
            full_head = unit_table * F.softplus(self.lm_head.output_scale)
            self.model.language_model.quantize_weights(bits)
            head_q, head_s = _quantize_symmetric(full_head, bits=bits, dim=-1)
            self.lm_head.register_buffer("_head_q", head_q, persistent=False)
            self.lm_head.register_buffer("_head_scale", head_s, persistent=False)
            self.lm_head._quant_bits = bits
            self.lm_head.weight.copy_(
                _dequantize_symmetric(head_q, head_s, bits=bits, dim=-1)
            )
        else:
            unit_table = self.model.language_model.embedding.unit_table()
            self.lm_head._unit_table.copy_(unit_table)
            self.lm_head._has_unit_table = True
            self.lm_head.weight.copy_(
                unit_table * F.softplus(self.lm_head.output_scale)
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = (
            (name, tensor)
            for name, tensor in weights
            if name != "lm_head.phase_angles"
        )
        loaded = AutoWeightsLoader(self).load_weights(filtered)
        self.process_weights_after_loading()
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if audio_values is not None and input_ids is not None:
            aud_feats = self.model.audio_tower(audio_values).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            mask = input_ids == self.config.audio_token_id
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = aud_feats.reshape(-1, inputs_embeds.shape[-1])
        hidden_states = self.model.language_model(
            None, positions, inputs_embeds=inputs_embeds
        )
        if return_logits or audio_values is not None:
            return self.compute_logits(hidden_states)
        return hidden_states


class GRTTTSForConditionalGeneration(nn.Module):
    """vLLM `GRTTTSForConditionalGeneration` text-to-speech model (`MatterWaveAI/grt-30m-tts`)."""

    def __init__(
        self,
        config: GRTTTSConfig | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config, quant_config, cache_config = _unwrap_vllm_args(
            config, vllm_config, quant_config, cache_config
        )
        _check_config(config.text_config)
        self.config = config
        self.quant_config = quant_config
        self.model = GRTBackbone(
            config.text_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = GRTCosineHead(
            config.text_config.vocab_size, config.text_config.hidden_size
        )
        self.speech_head = GRTPolarVocoderHead(
            config.audio_config, in_hidden_size=config.text_config.hidden_size
        )

    def get_input_embeddings(self) -> GRTEmbedding:
        return self.model.embedding

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def synthesize_speech(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        return_waveform: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if positions is None:
            positions = torch.arange(input_ids.shape[-1], device=input_ids.device)
            if input_ids.ndim == 2:
                positions = positions.unsqueeze(0).expand_as(input_ids)
        hidden_states = self.model(input_ids, positions)
        polar_spec, waveform = self.speech_head(
            hidden_states, return_waveform=return_waveform
        )
        return waveform, polar_spec

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        self.model.fold_constants()
        bits = _resolve_quant_bits(self.config, self.quant_config)
        if bits is not None:
            unit_table = self.model.embedding.unit_table()
            full_head = unit_table * F.softplus(self.lm_head.output_scale)
            self.model.quantize_weights(bits)
            head_q, head_s = _quantize_symmetric(full_head, bits=bits, dim=-1)
            self.lm_head.register_buffer("_head_q", head_q, persistent=False)
            self.lm_head.register_buffer("_head_scale", head_s, persistent=False)
            self.lm_head._quant_bits = bits
            self.lm_head.weight.copy_(
                _dequantize_symmetric(head_q, head_s, bits=bits, dim=-1)
            )
        else:
            unit_table = self.model.embedding.unit_table()
            self.lm_head._unit_table.copy_(unit_table)
            self.lm_head._has_unit_table = True
            self.lm_head.weight.copy_(
                unit_table * F.softplus(self.lm_head.output_scale)
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = (
            (name, tensor)
            for name, tensor in weights
            if name != "lm_head.phase_angles"
        )
        loaded = AutoWeightsLoader(self).load_weights(filtered)
        self.process_weights_after_loading()
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        hidden_states = self.model(
            input_ids, positions, inputs_embeds=inputs_embeds
        )
        if return_logits:
            return self.compute_logits(hidden_states)
        return hidden_states


class _GRTOmniInner(nn.Module):
    def __init__(
        self,
        config: GRTOmniConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.visual = GRTVisionModel(
            config.vision_config, out_hidden_size=config.text_config.hidden_size
        )
        self.audio_tower = GRTAudioModel(
            config.audio_config, out_hidden_size=config.text_config.hidden_size
        )
        self.language_model = GRTBackbone(
            config.text_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )


class GRTOmniForConditionalGeneration(nn.Module):
    """vLLM `GRTOmniForConditionalGeneration` unified multimodal model (`MatterWaveAI/grt-30m-omni`)."""

    supports_multimodal = True
    supports_multimodal_embeddings = True

    def __init__(
        self,
        config: GRTOmniConfig | None = None,
        *,
        vllm_config: VllmConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config, quant_config, cache_config = _unwrap_vllm_args(
            config, vllm_config, quant_config, cache_config
        )
        _check_config(config.text_config)
        self.config = config
        self.quant_config = quant_config
        self.model = _GRTOmniInner(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = GRTCosineHead(
            config.text_config.vocab_size, config.text_config.hidden_size
        )
        self.speech_head = GRTPolarVocoderHead(
            config.audio_config, in_hidden_size=config.text_config.hidden_size
        )

    def get_language_model(self) -> GRTBackbone:
        return self.model.language_model

    def get_input_embeddings(self) -> GRTEmbedding:
        return self.model.language_model.embedding

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            v_feats = self.model.visual(pixel_values)
            out.extend(v_feats[i] for i in range(v_feats.shape[0]))
        audio_values = kwargs.get("audio_values")
        if audio_values is None:
            audio_values = kwargs.get("input_features")
        if audio_values is not None:
            a_feats = self.model.audio_tower(audio_values)
            out.extend(a_feats[i] for i in range(a_feats.shape[0]))
        return out

    def get_multimodal_embeddings(self, **kwargs: Any) -> list[torch.Tensor]:
        return self.embed_multimodal(**kwargs)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is not None and is_multimodal is not None:
            return _merge_multimodal_embeddings(
                inputs_embeds, multimodal_embeddings, is_multimodal
            )
        return inputs_embeds

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def synthesize_speech(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        return_waveform: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if positions is None:
            positions = torch.arange(input_ids.shape[-1], device=input_ids.device)
            if input_ids.ndim == 2:
                positions = positions.unsqueeze(0).expand_as(input_ids)
        inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if pixel_values is not None:
            img_feats = self.model.visual(pixel_values).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            mask = input_ids == self.config.image_token_id
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = img_feats.reshape(-1, inputs_embeds.shape[-1])
        if audio_values is not None:
            aud_feats = self.model.audio_tower(audio_values).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            mask = input_ids == self.config.audio_token_id
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = aud_feats.reshape(-1, inputs_embeds.shape[-1])
        hidden_states = self.model.language_model(
            None, positions, inputs_embeds=inputs_embeds
        )
        polar_spec, waveform = self.speech_head(
            hidden_states, return_waveform=return_waveform
        )
        return waveform, polar_spec

    @torch.no_grad()
    def process_weights_after_loading(self) -> None:
        self.model.visual.fold_constants()
        self.model.audio_tower.fold_constants()
        self.model.language_model.fold_constants()
        bits = _resolve_quant_bits(self.config, self.quant_config)
        if bits is not None:
            self.model.visual.quantize_weights(bits)
            self.model.audio_tower.quantize_weights(bits)
            unit_table = self.model.language_model.embedding.unit_table()
            full_head = unit_table * F.softplus(self.lm_head.output_scale)
            self.model.language_model.quantize_weights(bits)
            head_q, head_s = _quantize_symmetric(full_head, bits=bits, dim=-1)
            self.lm_head.register_buffer("_head_q", head_q, persistent=False)
            self.lm_head.register_buffer("_head_scale", head_s, persistent=False)
            self.lm_head._quant_bits = bits
            self.lm_head.weight.copy_(
                _dequantize_symmetric(head_q, head_s, bits=bits, dim=-1)
            )
        else:
            unit_table = self.model.language_model.embedding.unit_table()
            self.lm_head._unit_table.copy_(unit_table)
            self.lm_head._has_unit_table = True
            self.lm_head.weight.copy_(
                unit_table * F.softplus(self.lm_head.output_scale)
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = (
            (name, tensor)
            for name, tensor in weights
            if name != "lm_head.phase_angles"
        )
        loaded = AutoWeightsLoader(self).load_weights(filtered)
        self.process_weights_after_loading()
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = input_embeds
        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if pixel_values is not None and input_ids is not None:
            img_feats = self.model.visual(pixel_values).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            mask = input_ids == self.config.image_token_id
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = img_feats.reshape(-1, inputs_embeds.shape[-1])
        if audio_values is not None and input_ids is not None:
            aud_feats = self.model.audio_tower(audio_values).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            mask = input_ids == self.config.audio_token_id
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[mask] = aud_feats.reshape(-1, inputs_embeds.shape[-1])
        hidden_states = self.model.language_model(
            None, positions, inputs_embeds=inputs_embeds
        )
        if return_logits or pixel_values is not None or audio_values is not None:
            return self.compute_logits(hidden_states)
        return hidden_states


EntryClass = [
    GRTForCausalLM,
    GRTModel,
    GRTEmbeddingModel,
    GRTForSequenceClassification,
    GRTVLForConditionalGeneration,
    GRTASRForConditionalGeneration,
    GRTTTSForConditionalGeneration,
    GRTOmniForConditionalGeneration,
]

__all__ = [
    "EntryClass",
    "GRTASRForConditionalGeneration",
    "GRTAudioModel",
    "GRTBackbone",
    "GRTCosineClassificationHead",
    "GRTCosineHead",
    "GRTEmbeddingModel",
    "GRTForCausalLM",
    "GRTForSequenceClassification",
    "GRTModel",
    "GRTOmniForConditionalGeneration",
    "GRTPolarAudioStem",
    "GRTPolarPatchEmbedding",
    "GRTPolarVocoderHead",
    "GRTRotaryEmbedding2D",
    "GRTSpatialRotationalMerger",
    "GRTTTSForConditionalGeneration",
    "GRTTemporalRotationalMerger",
    "GRTVLForConditionalGeneration",
    "GRTVisionModel",
    "load_grt_lora_adapter",
    "merge_grt_lora_weights",
]
