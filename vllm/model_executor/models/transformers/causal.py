# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
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
"""Transformers modeling backend mixin for causal language models."""

from collections.abc import Iterable
from operator import attrgetter
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.transformers.utils import attrsetter
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from .base import Base

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class CausalMixin(VllmModelForTextGeneration, Base):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip VllmModelForTextGeneration.__init__ and call the next class in MRO
        super(VllmModelForTextGeneration, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        tie_word_embeddings = self._get_tie_word_embeddings()

        if self.pp_group.is_last_rank:
            vocab_emb = next(
                (
                    module
                    for module in self.model.get_input_embeddings().modules()
                    if isinstance(module, VocabParallelEmbedding)
                ),
                None,
            )
            custom_lm_head = None
            if vocab_emb is None:
                with torch.device("meta"):
                    causal_model = AutoModelForCausalLM.from_config(
                        **self._from_config_kwargs
                    )
                candidate = getattr(causal_model, "lm_head", None)
                if candidate is not None and not isinstance(candidate, nn.Linear):
                    custom_lm_head = candidate

            if custom_lm_head is not None:
                self.lm_head = custom_lm_head
                self.init_parameters(self.lm_head, dtype=self.model_config.head_dtype)
                if tie_word_embeddings:
                    bmp_prefix = f"{self.model.base_model_prefix}."
                    for dst_key, src_key in (
                        getattr(causal_model, "_tied_weights_keys", None) or {}
                    ).items():
                        if dst_key.startswith("lm_head.") and src_key.startswith(
                            bmp_prefix
                        ):
                            attrsetter(dst_key.removeprefix("lm_head."))(
                                self.lm_head,
                                attrgetter(src_key.removeprefix(bmp_prefix))(
                                    self.model
                                ),
                            )
            else:
                self.lm_head = ParallelLMHead(
                    self.text_config.vocab_size,
                    self.text_config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
                if tie_word_embeddings and vocab_emb is not None:
                    self.lm_head = self.lm_head.tie_weights(vocab_emb)

            self.logits_processor = LogitsProcessor(
                self.text_config.vocab_size,
                scale=getattr(self.text_config, "logit_scale", 1.0),
                logits_as_input=custom_lm_head is not None,
                soft_cap=getattr(self.text_config, "final_logit_softcapping", None),
            )
        else:
            self.lm_head = PPMissingLayer()

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        """A thin wrapper around `Base.load_weights` to handle the lm_head bias."""
        lm_head_bias = set()
        lora_a_tensors: dict[str, torch.Tensor] = {}
        lora_b_tensors: dict[str, torch.Tensor] = {}
        load_device = self.vllm_config.load_config.device or self.device_config.device

        def auto_load_lm_head_bias(weights):
            for name, weight in weights:
                if name.endswith(".lora_A"):
                    lora_a_tensors[name.removesuffix(".lora_A")] = weight
                    lm_head_bias.add(name)
                elif name.endswith(".lora_B"):
                    lora_b_tensors[name.removesuffix(".lora_B")] = weight
                    lm_head_bias.add(name)
                elif (
                    name.endswith("lm_head.bias")
                    and self.pp_group.is_last_rank
                    and isinstance(self.lm_head, ParallelLMHead)
                ):
                    # load_weights is called outside vLLM's device context,
                    # so we must enter it again when registering the bias.
                    with torch.device(load_device):
                        self.lm_head._register_bias()
                    self.lm_head.bias.weight_loader(self.lm_head.bias, weight)
                    lm_head_bias.add(name)
                else:
                    yield name, weight

        loaded = super().load_weights(auto_load_lm_head_bias(weights)) | lm_head_bias
        if lora_a_tensors and lora_b_tensors:
            lora_alpha = float(getattr(self.text_config, "lora_alpha", 16.0))
            self._apply_grt_lora_pairs(lora_a_tensors, lora_b_tensors, lora_alpha=lora_alpha)
        return loaded

    @torch.no_grad()
    def _apply_grt_lora_pairs(
        self,
        lora_a_tensors: dict[str, "torch.Tensor"],
        lora_b_tensors: dict[str, "torch.Tensor"],
        lora_alpha: float = 16.0,
    ) -> None:
        """Fold toroidal `GRTRotationalLoRA` phase-shift deltas into loaded GRT angle parameters."""
        named_params = dict(self.named_parameters())
        for base_key, a_weight in lora_a_tensors.items():
            if base_key not in lora_b_tensors or base_key not in named_params:
                continue
            b_weight = lora_b_tensors[base_key]
            target = named_params[base_key]
            r = max(1, int(a_weight.shape[-1]))
            scaling = float(lora_alpha) / float(r)
            delta = (a_weight.to(device=target.device, dtype=torch.float32) @ b_weight.to(device=target.device, dtype=torch.float32)) * scaling
            if target.ndim == 3 and delta.ndim == 2:
                if delta.shape[0] == target.shape[0] and delta.shape[1] == target.shape[2]:
                    delta = delta.unsqueeze(1).expand(target.shape)
                else:
                    delta = delta.reshape(target.shape)
            elif tuple(delta.shape) != tuple(target.shape):
                delta = delta.reshape(target.shape)
            merged = torch.remainder(target.data.float() + delta, 2.0 * torch.pi)
            target.data.copy_(merged.to(dtype=target.dtype))

    @torch.no_grad()
    def load_grt_lora_adapter(
        self,
        adapter_dir: "str | dict[str, torch.Tensor]",
        lora_alpha: float = 16.0,
    ) -> None:
        """Load and fold a `GRTRotationalLoRA` adapter directory (`grt_lora_adapter.safetensors`) or state dict into this model."""
        import json
        import os

        if isinstance(adapter_dir, dict):
            sd = dict(adapter_dir)
        elif os.path.isdir(adapter_dir):
            cfg_path = os.path.join(adapter_dir, "grt_lora_config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    lora_alpha = float(json.load(f).get("lora_alpha", lora_alpha))
            weights_path = os.path.join(adapter_dir, "grt_lora_adapter.safetensors")
            if not os.path.isfile(weights_path):
                weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
            from safetensors.torch import load_file

            sd = load_file(weights_path)
        else:
            from safetensors.torch import load_file

            sd = load_file(adapter_dir)

        lora_a = {}
        lora_b = {}
        for k, v in sd.items():
            if k.endswith(".lora_A"):
                lora_a[k.removesuffix(".lora_A")] = v
            elif k.endswith(".lora_a"):
                lora_a[k.removesuffix(".lora_a")] = v
            elif k.endswith(".lora_B"):
                lora_b[k.removesuffix(".lora_B")] = v
            elif k.endswith(".lora_b"):
                lora_b[k.removesuffix(".lora_b")] = v
        self._apply_grt_lora_pairs(lora_a, lora_b, lora_alpha=lora_alpha)
        named_params = dict(self.named_parameters())
        for k, v in sd.items():
            if not k.endswith((".lora_A", ".lora_B", ".lora_a", ".lora_b")) and k in named_params:
                target = named_params[k]
                target.data.copy_(v.to(device=target.device, dtype=target.dtype))

    def compute_logits(self, hidden_states: "torch.Tensor") -> "torch.Tensor | None":
        if not isinstance(self.lm_head, ParallelLMHead):
            return self.logits_processor(None, self.lm_head(hidden_states))
        logits = self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)
        return logits
