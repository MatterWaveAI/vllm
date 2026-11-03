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
from vllm.transformers_utils.configs.grt import GRTConfig

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
