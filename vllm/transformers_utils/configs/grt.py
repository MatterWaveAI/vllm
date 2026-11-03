# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GRT model configuration."""

from transformers.configuration_utils import PreTrainedConfig

try:
    from transformers.models.grt.configuration_grt import (
        GRTConfig,
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
            learnable_residual_gates: bool = True,
            residual_gate_mode: str = "bounded_depth",
            residual_norm_mode: str = "none",
            peri_ln: bool = False,
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
            self.learnable_residual_gates = learnable_residual_gates
            self.residual_gate_mode = residual_gate_mode
            self.residual_norm_mode = residual_norm_mode
            self.peri_ln = peri_ln

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


__all__ = ["GRTConfig", "GRTVisionConfig", "GRTVLConfig"]
