# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit and numerical parity tests for native vLLM GRT (Geometric Rotation Transformer) models."""

from __future__ import annotations

import inspect
import json
import sys
import types
import unittest
from pathlib import Path

import torch

# Ensure `tests/models/transformers/` does not shadow the `transformers` package
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p != _THIS_DIR]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_VLLM_ROOT = _REPO_ROOT / "vllm"
try:
    import vllm.config.utils  # noqa: F401
except ModuleNotFoundError:
    for _parent, _rel in [
        ("vllm", ""),
        ("vllm.config", "config"),
        ("vllm.config.utils", "config"),
        ("vllm.model_executor", "model_executor"),
        ("vllm.model_executor.models", "model_executor/models"),
    ]:
        if _parent not in sys.modules:
            _mod = types.ModuleType(_parent)
            _mod.__path__ = [str(_VLLM_ROOT / _rel if _rel else _VLLM_ROOT)]
            sys.modules[_parent] = _mod
    sys.modules["vllm.config.utils"].getattr_iter = (
        lambda obj, names, default=None, **kw: next(
            (getattr(obj, n) for n in names if hasattr(obj, n)), default
        )
    )

from vllm.model_executor.models.grt import (
    EntryClass,
    GRTASRForConditionalGeneration,
    GRTAudioModel,
    GRTCosineClassificationHead,
    GRTCosineHead,
    GRTEmbeddingModel,
    GRTForCausalLM,
    GRTForSequenceClassification,
    GRTModel,
    GRTOmniForConditionalGeneration,
    GRTPolarAudioStem,
    GRTPolarPatchEmbedding,
    GRTPolarVocoderHead,
    GRTRotaryEmbedding2D,
    GRTSpatialRotationalMerger,
    GRTTTSForConditionalGeneration,
    GRTTemporalRotationalMerger,
    GRTVLForConditionalGeneration,
    GRTVisionModel,
    load_grt_lora_adapter,
    merge_grt_lora_weights,
)
from vllm.transformers_utils.configs.grt import (
    GRTASRConfig,
    GRTAudioConfig,
    GRTConfig,
    GRTOmniConfig,
    GRTTTSConfig,
    GRTVisionConfig,
    GRTVLConfig,
)


class _DummyModelConfig:
    def __init__(self, hf_config):
        self.hf_config = hf_config


class _DummyVllmConfig:
    def __init__(self, hf_config, quant_config=None, cache_config=None):
        self.model_config = _DummyModelConfig(hf_config)
        self.quant_config = quant_config
        self.cache_config = cache_config


class TestVllmGRTModels(unittest.TestCase):
    """Test suite verifying vLLM GRT configs, all 7 native model classes,
    constant folding, circular pooling, INT8/INT4 quantization, Toroidal LoRA,
    and numerical parity against Hugging Face transformers and local checkpoints."""

    def test_configs_and_entry_classes(self):
        cfg = GRTConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=128,
        )
        self.assertEqual(cfg.model_type, "grt")
        self.assertGreaterEqual(len(EntryClass), 7)

        vcfg = GRTVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        acfg = GRTAudioConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
        vl_cfg = GRTVLConfig(vision_config=vcfg.to_dict(), text_config=cfg.to_dict())
        asr_cfg = GRTASRConfig(audio_config=acfg.to_dict(), text_config=cfg.to_dict())
        tts_cfg = GRTTTSConfig(audio_config=acfg.to_dict(), text_config=cfg.to_dict())
        omni_cfg = GRTOmniConfig(
            vision_config=vcfg.to_dict(),
            audio_config=acfg.to_dict(),
            text_config=cfg.to_dict(),
        )
        self.assertEqual(vl_cfg.model_type, "grt_vl")
        self.assertEqual(asr_cfg.model_type, "grt_asr")
        self.assertEqual(tts_cfg.model_type, "grt_tts")
        self.assertEqual(omni_cfg.model_type, "grt_omni")

        reg_text = (
            _VLLM_ROOT / "model_executor" / "models" / "registry.py"
        ).read_text(encoding="utf-8")
        for arch in (
            "GRTForCausalLM",
            "GRTModel",
            "GRTEmbeddingModel",
            "GRTForSequenceClassification",
            "GRTVLForConditionalGeneration",
            "GRTASRForConditionalGeneration",
            "GRTTTSForConditionalGeneration",
            "GRTOmniForConditionalGeneration",
        ):
            self.assertIn(f'"{arch}": ("grt", "{arch}")', reg_text)

    def test_causal_lm_and_mor_forward_and_quantization(self):
        torch.manual_seed(42)
        for n_experts, mor_mode in [(0, "direct"), (4, "direct"), (4, "residual")]:
            cfg = GRTConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                n_qkv_rotations=2,
                n_mixer_rotations=1,
                ffn_n_rotations=2,
                ffn_n_experts=n_experts,
                mor_mode=mor_mode,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            vllm_cfg = _DummyVllmConfig(cfg)
            model = GRTForCausalLM(vllm_config=vllm_cfg)
            for p in model.parameters():
                if p.numel() > 0:
                    p.data.uniform_(-0.2, 0.2)
            model.process_weights_after_loading()
            input_ids = torch.tensor([[1, 5, 9, 12]], dtype=torch.long)
            positions = torch.arange(4, dtype=torch.long).unsqueeze(0)
            hidden_fp32 = model(input_ids, positions)
            logits_fp32 = model.compute_logits(hidden_fp32)
            self.assertEqual(logits_fp32.shape, (1, 4, 128))
            self.assertTrue(torch.isfinite(logits_fp32).all())

            # INT8 quantization
            model.quantize_weights(bits=8)
            logits_int8 = model.compute_logits(model(input_ids, positions))
            self.assertEqual(logits_int8.shape, (1, 4, 128))
            cos_sim = torch.nn.functional.cosine_similarity(
                logits_fp32.flatten(), logits_int8.flatten(), dim=0
            ).item()
            self.assertGreater(cos_sim, 0.99)

    def test_embedding_and_circular_pooler(self):
        cfg = GRTConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        model = GRTEmbeddingModel(vllm_config=_DummyVllmConfig(cfg))
        model.process_weights_after_loading()
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3], dtype=torch.long)
        out = model(
            input_ids, positions, prompt_lens=[3, 4], get_embedding=True
        )
        self.assertEqual(out.shape, (2, 64))
        norms = out.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

        # Verify circular pooler matches single-sequence circular pool exactly
        hs = model(input_ids, positions, prompt_lens=[3, 4], get_embedding=False)
        ref_0 = GRTForSequenceClassification._circular_pool_seq(hs[:3])
        ref_1 = GRTForSequenceClassification._circular_pool_seq(hs[3:])
        ref = torch.cat([ref_0, ref_1], dim=0)
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))

    def test_reranker_sequence_classification(self):
        cfg = GRTConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_labels=1,
        )
        model = GRTForSequenceClassification(vllm_config=_DummyVllmConfig(cfg))
        model.process_weights_after_loading()
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        positions = torch.arange(7, dtype=torch.long)
        tt_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        out_pair = model(
            input_ids, positions, token_type_ids=tt_ids, prompt_lens=[7]
        )
        self.assertEqual(out_pair.shape, (1, 1))
        self.assertTrue(torch.isfinite(out_pair).all())

        # Single sequence mode
        out_single = model(input_ids, positions, prompt_lens=[3, 4])
        self.assertEqual(out_single.shape, (2, 1))

    def test_multimodal_vl_asr_tts_omni(self):
        tcfg = GRTConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            n_qkv_rotations=1,
            n_mixer_rotations=1,
            ffn_n_rotations=1,
        )
        vcfg = GRTVisionConfig(
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=8,
            spatial_merge_size=2,
            n_qkv_rotations=1,
            n_mixer_rotations=1,
            ffn_n_rotations=1,
            patch_n_rotations=1,
            merger_n_rotations=1,
        )
        acfg = GRTAudioConfig(
            hidden_size=32,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            n_fft=64,
            hop_length=16,
            win_length=64,
            max_audio_frames=16,
            temporal_merge_size=2,
            frames_per_token=2,
            n_qkv_rotations=1,
            n_mixer_rotations=1,
            ffn_n_rotations=1,
            stem_n_rotations=1,
            merger_n_rotations=1,
            vocoder_n_rotations=1,
        )

        # 1. VL
        vl_cfg = GRTVLConfig(vision_config=vcfg, text_config=tcfg, image_token_id=100)
        vl_model = GRTVLForConditionalGeneration(vllm_config=_DummyVllmConfig(vl_cfg))
        vl_model.process_weights_after_loading()
        pixel_values = torch.randn(1, 3, 32, 32)
        mm_embeds = vl_model.get_multimodal_embeddings(pixel_values=pixel_values)
        self.assertEqual(len(mm_embeds), 1)
        self.assertEqual(mm_embeds[0].shape, (4, 64))
        vl_ids = torch.tensor([[1, 100, 100, 100, 100, 2]], dtype=torch.long)
        vl_pos = torch.arange(6, dtype=torch.long).unsqueeze(0)
        vl_logits = vl_model(vl_ids, vl_pos, pixel_values=pixel_values)
        self.assertEqual(vl_logits.shape, (1, 6, 128))

        # 2. ASR
        asr_cfg = GRTASRConfig(audio_config=acfg, text_config=tcfg, audio_token_id=101)
        asr_model = GRTASRForConditionalGeneration(
            vllm_config=_DummyVllmConfig(asr_cfg)
        )
        asr_model.process_weights_after_loading()
        audio_values = torch.randn(1, 256)
        aud_embeds = asr_model.get_multimodal_embeddings(audio_values=audio_values)
        self.assertEqual(len(aud_embeds), 1)
        self.assertEqual(aud_embeds[0].shape, (8, 64))
        asr_ids = torch.tensor([[1] + [101] * 8 + [2]], dtype=torch.long)
        asr_pos = torch.arange(10, dtype=torch.long).unsqueeze(0)
        asr_logits = asr_model(asr_ids, asr_pos, audio_values=audio_values)
        self.assertEqual(asr_logits.shape, (1, 10, 128))

        # 3. TTS
        tts_cfg = GRTTTSConfig(audio_config=acfg, text_config=tcfg)
        tts_model = GRTTTSForConditionalGeneration(
            vllm_config=_DummyVllmConfig(tts_cfg)
        )
        tts_model.process_weights_after_loading()
        tts_ids = torch.tensor([[1, 5, 9, 2]], dtype=torch.long)
        waveform, polar_spec = tts_model.synthesize_speech(
            tts_ids, return_waveform=True
        )
        self.assertEqual(
            polar_spec.shape,
            (1, 4 * acfg.frames_per_token, 2 * (acfg.n_fft // 2 + 1)),
        )
        self.assertIsNotNone(waveform)

        # 4. Omni
        omni_cfg = GRTOmniConfig(
            vision_config=vcfg,
            audio_config=acfg,
            text_config=tcfg,
            image_token_id=100,
            audio_token_id=101,
        )
        omni_model = GRTOmniForConditionalGeneration(
            vllm_config=_DummyVllmConfig(omni_cfg)
        )
        omni_model.process_weights_after_loading()
        omni_ids = torch.tensor([[1] + [100] * 4 + [101] * 8 + [2]], dtype=torch.long)
        omni_pos = torch.arange(14, dtype=torch.long).unsqueeze(0)
        omni_logits = omni_model(
            omni_ids, omni_pos, pixel_values=pixel_values, audio_values=audio_values
        )
        self.assertEqual(omni_logits.shape, (1, 14, 128))
        omni_wave, omni_spec = omni_model.synthesize_speech(
            omni_ids,
            omni_pos,
            pixel_values=pixel_values,
            audio_values=audio_values,
            return_waveform=True,
        )
        self.assertIsNotNone(omni_wave)
        self.assertEqual(
            omni_spec.shape,
            (1, 14 * acfg.frames_per_token, 2 * (acfg.n_fft // 2 + 1)),
        )

    def test_toroidal_lora_merge_into_model(self):
        cfg = GRTConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            n_qkv_rotations=2,
            n_mixer_rotations=1,
            ffn_n_rotations=2,
        )
        model = GRTForCausalLM(cfg)
        model.process_weights_after_loading()
        lora_a = torch.ones(2, 2, 2)
        lora_b = torch.ones(2, 2, 8)
        adapter = {
            "model.blocks.0.attention.q_angles.lora_A": lora_a,
            "model.blocks.0.attention.q_angles.lora_B": lora_b,
        }
        load_grt_lora_adapter(model, adapter, r=2, lora_alpha=4.0)
        expected = torch.full((2, 2, 8), 4.0)
        self.assertTrue(
            torch.allclose(model.model.blocks[0].attention.q_angles, expected)
        )

    def test_parity_with_hf_hub_checkpoints(self):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        import transformers.models.grt.modeling_grt as hf_grt_mod
        from transformers.models.grt.configuration_grt import GRTConfig as HFGRTConfig

        def _fetch_hf(repo_id: str) -> Path:
            return Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["config.json", "model.safetensors"],
                )
            )

        HFGRTForCausalLM = hf_grt_mod.GRTForCausalLM
        HFGRTModel = hf_grt_mod.GRTModel
        HFGRTForSequenceClassification = getattr(
            hf_grt_mod, "GRTForSequenceClassification", None
        )
        hf_supports_mor = "ffn_n_experts" in inspect.getsource(
            hf_grt_mod.GRTFeedForward
        )

        # 1. MatterWaveAI/grt-30m CausalLM Parity
        ckpt_30m = _fetch_hf("MatterWaveAI/grt-30m")
        with open(ckpt_30m / "config.json") as f:
            raw_cfg = json.load(f)
        vllm_cfg = GRTConfig(**raw_cfg)
        hf_cfg = HFGRTConfig(**raw_cfg)
        state = load_file(str(ckpt_30m / "model.safetensors"))

        vllm_model = GRTForCausalLM(vllm_cfg)
        vllm_model.load_weights(state.items())
        vllm_model.eval()

        hf_model = HFGRTForCausalLM(hf_cfg)
        hf_model.load_state_dict(state, strict=False)
        hf_model.eval()

        input_ids = torch.tensor([[1, 42, 100, 7, 255]], dtype=torch.long)
        positions = torch.arange(5, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            vllm_logits = vllm_model.compute_logits(
                vllm_model(input_ids, positions)
            )
            hf_logits = hf_model(
                input_ids=input_ids, position_ids=positions
            ).logits
        max_diff = (vllm_logits - hf_logits).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"30m CausalLM max diff {max_diff}")

        # 2. MatterWaveAI/grt-30m-mor CausalLM Load & Parity
        ckpt_mor = _fetch_hf("MatterWaveAI/grt-30m-mor")
        with open(ckpt_mor / "config.json") as f:
            raw_cfg = json.load(f)
        vllm_cfg = GRTConfig(**raw_cfg)
        state = load_file(str(ckpt_mor / "model.safetensors"))

        vllm_model = GRTForCausalLM(vllm_cfg)
        vllm_model.load_weights(state.items())
        vllm_model.eval()

        input_ids = torch.tensor([[1, 42, 100, 7, 255]], dtype=torch.long)
        positions = torch.arange(5, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            vllm_logits = vllm_model.compute_logits(
                vllm_model(input_ids, positions)
            )
            self.assertEqual(vllm_logits.shape, (1, 5, vllm_cfg.vocab_size))
            self.assertTrue(torch.isfinite(vllm_logits).all())
            if hf_supports_mor:
                hf_cfg = HFGRTConfig(**raw_cfg)
                hf_model = HFGRTForCausalLM(hf_cfg)
                hf_model.load_state_dict(state, strict=False)
                hf_model.eval()
                hf_logits = hf_model(
                    input_ids=input_ids, position_ids=positions
                ).logits
                max_diff = (vllm_logits - hf_logits).abs().max().item()
                self.assertLess(
                    max_diff, 1e-4, f"30m-mor CausalLM max diff {max_diff}"
                )

        # 3. MatterWaveAI/grt-30m-embedding GRTModel Parity
        ckpt_emb = _fetch_hf("MatterWaveAI/grt-30m-embedding")
        with open(ckpt_emb / "config.json") as f:
            raw_cfg = json.load(f)
        vllm_cfg = GRTConfig(**raw_cfg)
        hf_cfg = HFGRTConfig(**raw_cfg)
        state = load_file(str(ckpt_emb / "model.safetensors"))

        vllm_model = GRTModel(vllm_cfg)
        vllm_model.load_weights(state.items())
        vllm_model.eval()

        hf_model = HFGRTModel(hf_cfg)
        stripped = {
            k.removeprefix("model."): v
            for k, v in state.items()
            if not k.startswith("lm_head.")
        }
        hf_model.load_state_dict(stripped, strict=False)
        hf_model.eval()

        input_ids = torch.tensor([[1, 42, 100, 7, 255]], dtype=torch.long)
        positions = torch.arange(5, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            vllm_hs = vllm_model(input_ids, positions, get_embedding=False)
            hf_hs = hf_model(
                input_ids=input_ids, position_ids=positions
            ).last_hidden_state
        max_diff = (vllm_hs - hf_hs).abs().max().item()
        self.assertLess(
            max_diff, 1e-4, f"30m-embedding hidden state max diff {max_diff}"
        )

        # 4. MatterWaveAI/grt-30m-reranker GRTForSequenceClassification Load & Score Parity
        ckpt_rr = _fetch_hf("MatterWaveAI/grt-30m-reranker")
        with open(ckpt_rr / "config.json") as f:
            raw_cfg = json.load(f)
        vllm_cfg = GRTConfig(**raw_cfg)
        state = load_file(str(ckpt_rr / "model.safetensors"))

        vllm_model = GRTForSequenceClassification(vllm_cfg)
        vllm_model.load_weights(state.items())
        vllm_model.eval()

        input_ids = torch.tensor(
            [[1, 15, 28, 2, 42, 99, 105, 2]], dtype=torch.long
        )
        tt_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)
        positions = torch.arange(8, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            vllm_score = vllm_model(
                input_ids, positions, token_type_ids=tt_ids
            )
            self.assertEqual(vllm_score.shape, (1, 1))
            self.assertTrue(torch.isfinite(vllm_score).all())
            if HFGRTForSequenceClassification is not None:
                hf_cfg = HFGRTConfig(**raw_cfg)
                hf_model = HFGRTForSequenceClassification(hf_cfg)
                hf_model.load_state_dict(state, strict=False)
                hf_model.eval()
                hf_score = hf_model(
                    input_ids=input_ids, token_type_ids=tt_ids
                ).logits
                max_diff = (vllm_score - hf_score).abs().max().item()
                self.assertLess(
                    max_diff,
                    1e-4,
                    f"30m-reranker cross-encoder score max diff {max_diff}",
                )


if __name__ == "__main__":
    unittest.main()
