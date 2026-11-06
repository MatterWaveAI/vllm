# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GRT model configuration."""

from transformers.configuration_utils import PreTrainedConfig

try:
    from transformers.models.grt.configuration_grt import (
        GRTAudioConfig,
        GRTConfig,
        GRTOmniConfig,
        GRTTTSConfig,
        GRTVisionConfig,
        GRTVLConfig,
    )
except ImportError:

    class GRTConfig(PreTrainedConfig):
        """Configuration for the GRT (Geometric Rotational Transformer) causal LM,
        as Transformers' GRTConfig saves it. The defaults are matterwaveai/grt-30m's.
        Initialization-only fields are kept as extra attributes.
        """

        model_type = "grt"
        keys_to_ignore_at_inference = ["past_key_values"]

        def __init__(
            self,
            vocab_size: int = 151936,
            hidden_size: int = 256,
            intermediate_size: int | None = None,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 4,
            max_position_embeddings: int = 2048,
            ffn_n_rotations: int = 21,
            n_qkv_rotations: int = 11,
            n_mixer_rotations: int = 1,
            pos_freq_base: float = 10000.0,
            learnable_rope: bool = True,
            rope_placement: str = "stream",
            learnable_residual_gates: bool = True,
            residual_gate_mode: str = "bounded_depth",
            residual_norm_mode: str = "none",
            peri_ln: bool = False,
            ffn_n_experts: int = 0,
            mor_mode: str = "direct",
            mor_delta_scale: float = 0.5,
            mor_router_mode: str = "rotational",
            mor_split_routers: bool = False,
            mor_top_k: int | None = None,
            tie_word_embeddings: bool = True,
            pad_token_id: int | None = 151643,
            bos_token_id: int | None = 151643,
            eos_token_id: int | None = 151643,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size or 4 * hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.max_position_embeddings = max_position_embeddings
            self.ffn_n_rotations = ffn_n_rotations
            self.n_qkv_rotations = n_qkv_rotations
            self.n_mixer_rotations = n_mixer_rotations
            self.pos_freq_base = pos_freq_base
            self.learnable_rope = learnable_rope
            self.rope_placement = rope_placement
            self.learnable_residual_gates = learnable_residual_gates
            self.residual_gate_mode = residual_gate_mode
            self.residual_norm_mode = residual_norm_mode
            self.peri_ln = peri_ln
            self.ffn_n_experts = ffn_n_experts
            self.mor_mode = mor_mode
            self.mor_delta_scale = mor_delta_scale
            self.mor_router_mode = mor_router_mode
            self.mor_split_routers = mor_split_routers
            self.mor_top_k = mor_top_k

            kwargs.update(
                tie_word_embeddings=tie_word_embeddings,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )
            super().__init__(**kwargs)

    class GRTVisionConfig(PreTrainedConfig):
        """Configuration for the 100% rotational GRTVisionModel vision tower."""

        model_type = "grt_vision"

        def __init__(
            self,
            image_size: int = 224,
            patch_size: int = 14,
            num_channels: int = 3,
            spatial_merge_size: int = 2,
            hidden_size: int = 256,
            intermediate_size: int | None = None,
            num_hidden_layers: int = 6,
            num_attention_heads: int = 4,
            patch_n_rotations: int = 11,
            merger_n_rotations: int = 21,
            ffn_n_rotations: int = 21,
            n_qkv_rotations: int = 11,
            n_mixer_rotations: int = 1,
            pos_freq_base: float = 10000.0,
            learnable_rope: bool = True,
            learnable_residual_gates: bool = True,
            residual_gate_mode: str = "bounded_depth",
            residual_norm_mode: str = "none",
            peri_ln: bool = False,
            is_causal: bool = False,
            **kwargs,
        ):
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_channels = num_channels
            self.spatial_merge_size = spatial_merge_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size or 4 * hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.patch_n_rotations = patch_n_rotations
            self.merger_n_rotations = merger_n_rotations
            self.ffn_n_rotations = ffn_n_rotations
            self.n_qkv_rotations = n_qkv_rotations
            self.n_mixer_rotations = n_mixer_rotations
            self.pos_freq_base = pos_freq_base
            self.learnable_rope = learnable_rope
            self.learnable_residual_gates = learnable_residual_gates
            self.residual_gate_mode = residual_gate_mode
            self.residual_norm_mode = residual_norm_mode
            self.peri_ln = peri_ln
            self.is_causal = is_causal
            super().__init__(**kwargs)

    class GRTVLConfig(PreTrainedConfig):
        """Configuration for GRTVLForConditionalGeneration (matterwaveai/grt-30m-vl)."""

        model_type = "grt_vl"
        keys_to_ignore_at_inference = ["past_key_values"]
        sub_configs = {"vision_config": GRTVisionConfig, "text_config": GRTConfig}

        def __init__(
            self,
            vision_config: GRTVisionConfig | dict | None = None,
            text_config: GRTConfig | dict | None = None,
            image_token_id: int = 151655,
            vision_start_token_id: int = 151652,
            vision_end_token_id: int = 151653,
            tie_word_embeddings: bool = True,
            **kwargs,
        ):
            if isinstance(vision_config, dict):
                self.vision_config = GRTVisionConfig(**vision_config)
            elif vision_config is None:
                self.vision_config = GRTVisionConfig()
            else:
                self.vision_config = vision_config

            if isinstance(text_config, dict):
                self.text_config = GRTConfig(**text_config)
            elif text_config is None:
                self.text_config = GRTConfig(
                    vocab_size=151669, tie_word_embeddings=tie_word_embeddings
                )
            else:
                self.text_config = text_config

            self.image_token_id = image_token_id
            self.vision_start_token_id = vision_start_token_id
            self.vision_end_token_id = vision_end_token_id
            kwargs.update(tie_word_embeddings=tie_word_embeddings)
            super().__init__(**kwargs)

        @property
        def num_image_tokens(self) -> int:
            grid = self.vision_config.image_size // (
                self.vision_config.patch_size * self.vision_config.spatial_merge_size
            )
            return grid * grid

    class GRTAudioConfig(PreTrainedConfig):
        """Configuration for the 100% rotational GRTAudioModel audio encoder and vocoder."""

        model_type = "grt_audio"

        def __init__(
            self,
            sample_rate: int = 16000,
            n_fft: int = 254,
            hop_length: int = 160,
            win_length: int = 254,
            max_audio_frames: int = 128,
            temporal_merge_size: int = 2,
            hidden_size: int = 256,
            intermediate_size: int | None = None,
            num_hidden_layers: int = 4,
            num_attention_heads: int = 4,
            stem_n_rotations: int = 11,
            merger_n_rotations: int = 15,
            vocoder_n_rotations: int = 15,
            frames_per_token: int = 4,
            ffn_n_rotations: int = 21,
            n_qkv_rotations: int = 11,
            n_mixer_rotations: int = 1,
            pos_freq_base: float = 10000.0,
            learnable_rope: bool = True,
            learnable_residual_gates: bool = True,
            residual_gate_mode: str = "bounded_depth",
            residual_norm_mode: str = "none",
            peri_ln: bool = False,
            is_causal: bool = False,
            **kwargs,
        ):
            self.sample_rate = sample_rate
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length
            self.max_audio_frames = max_audio_frames
            self.temporal_merge_size = temporal_merge_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size or 4 * hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.stem_n_rotations = stem_n_rotations
            self.merger_n_rotations = merger_n_rotations
            self.vocoder_n_rotations = vocoder_n_rotations
            self.frames_per_token = frames_per_token
            self.ffn_n_rotations = ffn_n_rotations
            self.n_qkv_rotations = n_qkv_rotations
            self.n_mixer_rotations = n_mixer_rotations
            self.pos_freq_base = pos_freq_base
            self.learnable_rope = learnable_rope
            self.learnable_residual_gates = learnable_residual_gates
            self.residual_gate_mode = residual_gate_mode
            self.residual_norm_mode = residual_norm_mode
            self.peri_ln = peri_ln
            self.is_causal = is_causal
            super().__init__(**kwargs)

        @property
        def n_freq_bins(self) -> int:
            return self.n_fft // 2 + 1

        @property
        def spec_dim(self) -> int:
            return 2 * (self.n_fft // 2 + 1)

    class GRTTTSConfig(PreTrainedConfig):
        """Configuration for GRTTTSForConditionalGeneration (matterwaveai/grt-30m-tts)."""

        model_type = "grt_tts"
        keys_to_ignore_at_inference = ["past_key_values"]
        sub_configs = {"audio_config": GRTAudioConfig, "text_config": GRTConfig}

        def __init__(
            self,
            audio_config: GRTAudioConfig | dict | None = None,
            text_config: GRTConfig | dict | None = None,
            audio_token_id: int = 151656,
            tie_word_embeddings: bool = True,
            **kwargs,
        ):
            if isinstance(audio_config, dict):
                self.audio_config = GRTAudioConfig(**audio_config)
            elif audio_config is None:
                self.audio_config = GRTAudioConfig()
            else:
                self.audio_config = audio_config

            if isinstance(text_config, dict):
                self.text_config = GRTConfig(**text_config)
            elif text_config is None:
                self.text_config = GRTConfig(
                    vocab_size=151669, tie_word_embeddings=tie_word_embeddings
                )
            else:
                self.text_config = text_config

            self.audio_token_id = audio_token_id
            kwargs.update(tie_word_embeddings=tie_word_embeddings)
            super().__init__(**kwargs)

    class GRTOmniConfig(PreTrainedConfig):
        """Configuration for GRTOmniForConditionalGeneration (matterwaveai/grt-30m-omni)."""

        model_type = "grt_omni"
        keys_to_ignore_at_inference = ["past_key_values"]
        sub_configs = {
            "vision_config": GRTVisionConfig,
            "audio_config": GRTAudioConfig,
            "text_config": GRTConfig,
        }

        def __init__(
            self,
            vision_config: GRTVisionConfig | dict | None = None,
            audio_config: GRTAudioConfig | dict | None = None,
            text_config: GRTConfig | dict | None = None,
            image_token_id: int = 151655,
            vision_start_token_id: int = 151652,
            vision_end_token_id: int = 151653,
            audio_token_id: int = 151656,
            tie_word_embeddings: bool = True,
            **kwargs,
        ):
            if isinstance(vision_config, dict):
                self.vision_config = GRTVisionConfig(**vision_config)
            elif vision_config is None:
                self.vision_config = GRTVisionConfig()
            else:
                self.vision_config = vision_config

            if isinstance(audio_config, dict):
                self.audio_config = GRTAudioConfig(**audio_config)
            elif audio_config is None:
                self.audio_config = GRTAudioConfig()
            else:
                self.audio_config = audio_config

            if isinstance(text_config, dict):
                self.text_config = GRTConfig(**text_config)
            elif text_config is None:
                self.text_config = GRTConfig(
                    vocab_size=151669, tie_word_embeddings=tie_word_embeddings
                )
            else:
                self.text_config = text_config

            self.image_token_id = image_token_id
            self.vision_start_token_id = vision_start_token_id
            self.vision_end_token_id = vision_end_token_id
            self.audio_token_id = audio_token_id
            kwargs.update(tie_word_embeddings=tie_word_embeddings)
            super().__init__(**kwargs)

        @property
        def num_image_tokens(self) -> int:
            grid = self.vision_config.image_size // (
                self.vision_config.patch_size * self.vision_config.spatial_merge_size
            )
            return grid * grid

        @property
        def num_audio_tokens(self) -> int:
            return (
                self.audio_config.max_audio_frames
                // self.audio_config.temporal_merge_size
            )


try:
    from transformers.models.grt.configuration_grt import GRTASRConfig
except ImportError:

    class GRTASRConfig(PreTrainedConfig):
        """Configuration for GRTASRForConditionalGeneration (matterwaveai/grt-30m-asr)."""

        model_type = "grt_asr"
        keys_to_ignore_at_inference = ["past_key_values"]
        sub_configs = {"audio_config": GRTAudioConfig, "text_config": GRTConfig}

        def __init__(
            self,
            audio_config: GRTAudioConfig | dict | None = None,
            text_config: GRTConfig | dict | None = None,
            audio_token_id: int = 151656,
            tie_word_embeddings: bool = True,
            **kwargs,
        ):
            if isinstance(audio_config, dict):
                self.audio_config = GRTAudioConfig(**audio_config)
            elif audio_config is None:
                self.audio_config = GRTAudioConfig()
            else:
                self.audio_config = audio_config

            if isinstance(text_config, dict):
                self.text_config = GRTConfig(**text_config)
            elif text_config is None:
                self.text_config = GRTConfig(
                    vocab_size=151669, tie_word_embeddings=tie_word_embeddings
                )
            else:
                self.text_config = text_config

            self.audio_token_id = audio_token_id
            kwargs.update(tie_word_embeddings=tie_word_embeddings)
            super().__init__(**kwargs)

        @property
        def num_audio_tokens(self) -> int:
            return (
                self.audio_config.max_audio_frames
                // self.audio_config.temporal_merge_size
            )


def merge_grt_lora_weights(
    base_state_dict: dict,
    lora_state_dict: dict,
    r: int = 8,
    lora_alpha: float = 16.0,
    wrap_mod_2pi: bool = True,
) -> dict:
    """Pre-merge toroidal GRTRotationalLoRA weights (`lora_A`, `lora_B`) into `base_state_dict`.

    Computes `theta = (base_angles + (lora_alpha / r) * (lora_A @ lora_B)) mod 2*pi`
    with zero inference overhead.
    """
    import math
    import torch

    merged = dict(base_state_dict)
    lora_a_keys: dict[str, str] = {}
    lora_b_keys: dict[str, str] = {}
    for k in lora_state_dict:
        if k.endswith(".lora_A"):
            lora_a_keys[k[:-7]] = k
        elif k.endswith(".lora_a"):
            lora_a_keys[k[:-7]] = k
        elif k.endswith(".lora_B"):
            lora_b_keys[k[:-7]] = k
        elif k.endswith(".lora_b"):
            lora_b_keys[k[:-7]] = k
        elif k in merged:
            merged[k] = lora_state_dict[k].to(
                device=merged[k].device, dtype=merged[k].dtype
            )

    for prefix, key_a in lora_a_keys.items():
        if prefix not in lora_b_keys:
            continue
        lora_a = lora_state_dict[key_a].float()
        lora_b = lora_state_dict[lora_b_keys[prefix]].float()
        inferred_r = int(lora_a.shape[-1]) if lora_a.ndim >= 2 else r
        scale = float(lora_alpha) / float(max(1, inferred_r))
        delta = torch.matmul(lora_a, lora_b) * scale

        candidates = [prefix]
        if not prefix.endswith(".angles"):
            candidates.append(f"{prefix}.angles")
        if prefix.startswith("base_model.model."):
            stripped = prefix[len("base_model.model.") :]
            candidates.extend([stripped, f"{stripped}.angles"])

        target_key = next((c for c in candidates if c in merged), None)
        if target_key is None:
            continue
        base_angles = merged[target_key]
        delta_dev = delta.to(device=base_angles.device, dtype=torch.float32)
        if base_angles.ndim == 3 and delta_dev.ndim == 2:
            if (
                delta_dev.shape[0] == base_angles.shape[0]
                and delta_dev.shape[1] == base_angles.shape[2]
            ):
                delta_dev = delta_dev.unsqueeze(1).expand(base_angles.shape)
            else:
                delta_dev = delta_dev.reshape(base_angles.shape)
        elif tuple(delta_dev.shape) != tuple(base_angles.shape):
            delta_dev = delta_dev.reshape(base_angles.shape)
        updated = base_angles.float() + delta_dev
        if wrap_mod_2pi:
            updated = torch.remainder(updated, 2.0 * math.pi)
        merged[target_key] = updated.to(dtype=base_angles.dtype)

    return merged


def load_grt_lora_adapter(
    model_or_state_dict,
    lora_adapter_path_or_dict,
    r: int = 8,
    lora_alpha: float = 16.0,
    wrap_mod_2pi: bool = True,
):
    """Load and pre-merge a toroidal GRTRotationalLoRA adapter into a vLLM/HF GRT model or state_dict."""
    import json
    import os

    if isinstance(lora_adapter_path_or_dict, str):
        path = lora_adapter_path_or_dict
        if os.path.isdir(path):
            for cfg_name in ("grt_lora_config.json", "adapter_config.json"):
                cfg_path = os.path.join(path, cfg_name)
                if os.path.exists(cfg_path):
                    with open(cfg_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    r = int(meta.get("r", r))
                    lora_alpha = float(meta.get("lora_alpha", lora_alpha))
                    break
            st_file = os.path.join(path, "grt_lora_adapter.safetensors")
            if not os.path.exists(st_file):
                st_file = os.path.join(path, "adapter_model.safetensors")
            from safetensors.torch import load_file

            lora_sd = load_file(st_file)
        else:
            from safetensors.torch import load_file

            lora_sd = load_file(path)
    else:
        lora_sd = dict(lora_adapter_path_or_dict)

    if isinstance(model_or_state_dict, dict):
        return merge_grt_lora_weights(
            model_or_state_dict,
            lora_sd,
            r=r,
            lora_alpha=lora_alpha,
            wrap_mod_2pi=wrap_mod_2pi,
        )

    state = model_or_state_dict.state_dict()
    merged = merge_grt_lora_weights(
        state,
        lora_sd,
        r=r,
        lora_alpha=lora_alpha,
        wrap_mod_2pi=wrap_mod_2pi,
    )
    model_or_state_dict.load_state_dict(merged, strict=False)
    if hasattr(model_or_state_dict, "process_weights_after_loading"):
        model_or_state_dict.process_weights_after_loading()
    return model_or_state_dict


__all__ = [
    "GRTASRConfig",
    "GRTAudioConfig",
    "GRTConfig",
    "GRTOmniConfig",
    "GRTTTSConfig",
    "GRTVisionConfig",
    "GRTVLConfig",
    "load_grt_lora_adapter",
    "merge_grt_lora_weights",
]
