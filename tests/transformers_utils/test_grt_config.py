# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that GRT's config.json loads without a GRT-aware Transformers, and
verify that TransformersForCausalLM dispatches GRT attention and cosine head."""

import json
import sys
import types

import pytest
import torch

try:
    import vllm.config.utils  # noqa: F401
except ModuleNotFoundError:
    for parent in ("vllm.config", "vllm.config.utils"):
        if parent not in sys.modules:
            m = types.ModuleType(parent)
            m.__path__ = []
            sys.modules[parent] = m
    sys.modules["vllm.config.utils"].getattr_iter = (
        lambda obj, names, default=None, **kw: next(
            (getattr(obj, n) for n in names if hasattr(obj, n)), default
        )
    )

from vllm.transformers_utils.config import get_config
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

# matterwaveai/grt-30m's config.json, as Transformers' GRTConfig saves it.
_GRT_30M_CONFIG = {
    "angle_init_scale": 0.1,
    "architectures": ["GRTForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "coupling_init_mode": "diagonal",
    "coupling_init_scale": 0.01,
    "dtype": "float32",
    "eos_token_id": 151643,
    "ffn_n_rotations": 21,
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "learnable_residual_gates": True,
    "learnable_rope": True,
    "log_magnitude_init": 0.0,
    "max_position_embeddings": 2048,
    "model_type": "grt",
    "n_mixer_rotations": 1,
    "n_qkv_rotations": 11,
    "num_attention_heads": 4,
    "num_hidden_layers": 12,
    "output_scale_init": 30.0,
    "pad_token_id": 151643,
    "peri_ln": False,
    "phase_init_scale": 0.1,
    "pos_freq_base": 10000.0,
    "residual_gate_mode": "bounded_depth",
    "residual_norm_mode": "none",
    "tie_word_embeddings": True,
    "transformers_version": "5.21.0.dev0",
    "use_cache": True,
    "vocab_size": 151936,
}

# Sizes away from the defaults, so a value the config ignored would show.
_RESIZED = {
    "hidden_size": 512,
    "num_attention_heads": 8,
    "num_hidden_layers": 6,
    "max_position_embeddings": 4096,
    "intermediate_size": 1536,
    "vocab_size": 1000,
}


@pytest.mark.parametrize(
    "config_json",
    [_GRT_30M_CONFIG, {**_GRT_30M_CONFIG, **_RESIZED}],
    ids=["grt-30m", "resized"],
)
def test_config_loads(tmp_path, config_json):
    if config_json is not _GRT_30M_CONFIG:
        config_json = {
            **config_json,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
    (tmp_path / "config.json").write_text(json.dumps(config_json))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, GRTConfig)
    for key in (*_RESIZED, "n_qkv_rotations", "ffn_n_rotations", "n_mixer_rotations"):
        assert getattr(config, key) == config_json[key], key
    assert config.residual_gate_mode == "bounded_depth"
    assert config.residual_norm_mode == "none"
    assert not config.peri_ln
    assert config.learnable_residual_gates
    assert config.tie_word_embeddings


def test_grt_transformers_backend_parity():
    pytest.importorskip("transformers.models.grt")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.grt import GRTConfig as HfGRTConfig
    from transformers.models.grt import GRTForCausalLM, GRTModel

    def _mock_vllm_attn(
        module, query, key, value, attention_mask, scaling=None, **kwargs
    ):
        t = query.shape[-2]
        scale = scaling or (query.shape[-1] ** -0.5)
        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=query.device), diagonal=1
        )
        probs = torch.softmax(scores + mask, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        out = torch.matmul(probs, value)
        return out.transpose(1, 2).reshape(t, -1), None

    ALL_ATTENTION_FUNCTIONS.register("vllm", _mock_vllm_attn)
    cfg = HfGRTConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        n_qkv_rotations=3,
        ffn_n_rotations=3,
        n_mixer_rotations=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    assert GRTForCausalLM.is_backend_compatible()
    ref_model = GRTForCausalLM(cfg).eval()

    vllm_cfg = HfGRTConfig(**cfg.to_dict())
    vllm_cfg._attn_implementation = "vllm"
    vllm_base = GRTModel(vllm_cfg)
    vllm_head = GRTForCausalLM(vllm_cfg).lm_head
    vllm_base.load_state_dict(ref_model.model.state_dict())
    vllm_head.load_state_dict(ref_model.lm_head.state_dict())
    vllm_head.phase_angles = vllm_base.embedding.phase_angles
    vllm_base.eval()
    vllm_head.eval()

    input_ids = torch.tensor([[1, 5, 12, 33, 100]], dtype=torch.long)
    positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        ref_logits = ref_model(input_ids=input_ids, position_ids=positions).logits[0]
        vllm_hidden = vllm_base(
            input_ids=input_ids,
            position_ids=positions,
            use_cache=False,
            return_dict=False,
        )[0][0]
        vllm_logits = vllm_head(vllm_hidden)

    torch.testing.assert_close(vllm_logits, ref_logits, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "model_type,arch,expected_cls",
    [
        ("grt_vl", "GRTVLForConditionalGeneration", GRTVLConfig),
        ("grt_tts", "GRTTTSForConditionalGeneration", GRTTTSConfig),
        ("grt_omni", "GRTOmniForConditionalGeneration", GRTOmniConfig),
        ("grt_asr", "GRTASRForConditionalGeneration", GRTASRConfig),
    ],
)
def test_grt_multimodal_and_speech_configs_load(tmp_path, model_type, arch, expected_cls):
    cfg_dict = {
        "architectures": [arch],
        "model_type": model_type,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "grt",
            "vocab_size": 1000,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
    }
    if model_type in ("grt_vl", "grt_omni"):
        cfg_dict["vision_config"] = {
            "model_type": "grt_vision",
            "image_size": 28,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        }
    if model_type in ("grt_tts", "grt_omni", "grt_asr"):
        cfg_dict["audio_config"] = {
            "model_type": "grt_audio",
            "n_fft": 30,
            "hop_length": 16,
            "win_length": 30,
            "max_audio_frames": 8,
            "temporal_merge_size": 2,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
        }

    sub_dir = tmp_path / model_type
    sub_dir.mkdir()
    (sub_dir / "config.json").write_text(json.dumps(cfg_dict))
    config = get_config(str(sub_dir), trust_remote_code=False)
    assert isinstance(config, expected_cls)
    assert config.text_config.hidden_size == 64
    if hasattr(config, "audio_config"):
        assert isinstance(config.audio_config, GRTAudioConfig)
        assert config.audio_config.hidden_size == 64
    if hasattr(config, "vision_config"):
        assert isinstance(config.vision_config, GRTVisionConfig)
        assert config.vision_config.hidden_size == 64


def test_grt_rotational_lora_adapter_premerge(tmp_path):
    import math
    from safetensors.torch import save_file

    base_angles = torch.full((3, 32), 0.5, dtype=torch.float32)
    state_dict = {
        "model.layers.0.self_attn.q_proj.angles": base_angles.clone(),
        "lm_head.output_scale": torch.tensor(30.0),
    }
    r = 4
    lora_alpha = 8.0
    lora_a = torch.full((3, r), 0.25, dtype=torch.float32)
    lora_b = torch.full((r, 32), 0.5, dtype=torch.float32)
    tuned_scale = torch.tensor(33.5)

    adapter_dir = tmp_path / "lora_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "grt_lora_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": lora_alpha})
    )
    save_file(
        {
            "model.layers.0.self_attn.q_proj.angles.lora_A": lora_a,
            "model.layers.0.self_attn.q_proj.angles.lora_B": lora_b,
            "lm_head.output_scale": tuned_scale,
        },
        str(adapter_dir / "grt_lora_adapter.safetensors"),
    )

    merged = load_grt_lora_adapter(state_dict, str(adapter_dir))
    expected_angles = torch.remainder(
        base_angles + (lora_alpha / r) * (lora_a @ lora_b), 2.0 * math.pi
    )
    torch.testing.assert_close(
        merged["model.layers.0.self_attn.q_proj.angles"],
        expected_angles,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(merged["lm_head.output_scale"], tuned_scale)

