# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP admits only locally available DSpark inputs and static verification."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.dspark.utils import validate_dspark_pipeline_config


def _config(
    *, pp_size=3, adaptive=False, model_type="deepseek_v41", layers=(37, 38, 39)
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        speculative_config=SimpleNamespace(
            enable_adaptive_verification=adaptive,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type=model_type,
                    num_hidden_layers=40,
                    dspark_target_layer_ids=layers,
                )
            ),
        ),
    )


def test_last_stage_contains_all_auxiliary_captures(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "8,12,20")
    validate_dspark_pipeline_config(_config())


@pytest.mark.parametrize("layers", [(20, 38, 39), (37, 38, 41), ()])
def test_rejects_unavailable_auxiliary_captures(monkeypatch, layers):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "8,12,20")
    with pytest.raises(ValueError, match="auxiliary layers"):
        validate_dspark_pipeline_config(_config(layers=layers))


def test_rejects_unsynchronized_adaptive_layouts():
    with pytest.raises(ValueError, match="enable_adaptive_verification=false"):
        validate_dspark_pipeline_config(_config(adaptive=True))


def test_rejects_architectures_without_pipeline_embedding_loading():
    with pytest.raises(ValueError, match="requires DeepSeek V4.1"):
        validate_dspark_pipeline_config(_config(model_type="deepseek_v4"))


def test_single_stage_preserves_other_dspark_configurations():
    validate_dspark_pipeline_config(
        _config(pp_size=1, adaptive=True, model_type="qwen3")
    )


def test_draft_parallel_config_retains_tp_without_partitioning_the_draft():
    target = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=3)
    draft = SpeculativeConfig.create_draft_parallel_config(
        target, 2, pipeline_parallel_size=1
    )
    assert draft.tensor_parallel_size == 2
    assert draft.pipeline_parallel_size == 1
    assert draft.world_size == 2
    assert target.pipeline_parallel_size == 3
    assert target.world_size == 6


@pytest.mark.parametrize("include_embedding", [True, False])
def test_pipeline_draft_requires_real_embedding_weights(monkeypatch, include_embedding):
    """The last stage must load a real embedding, never a PP placeholder."""
    from vllm.models.deepseek_v4_1.nvidia import dspark

    draft = dspark.DSparkDeepseekV4ForCausalLM.__new__(
        dspark.DSparkDeepseekV4ForCausalLM
    )
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(n_routed_experts=1, num_attention_heads=2)
    draft.model = nn.Module()
    draft.model.layers = [SimpleNamespace(ffn=SimpleNamespace(use_mega_moe=True))]
    draft.model.embed_tokens = nn.Embedding(4, 3)
    draft.model.confidence_head = None
    draft.has_own_embed_tokens = True
    draft.pad_shared_expert = False
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(draft, "process_weights_after_loading", lambda: None)
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    if include_embedding:
        loaded = draft.load_weights([("embed.weight", weight)])
        assert loaded == {"model.embed_tokens.weight"}
        torch.testing.assert_close(draft.model.embed_tokens.weight, weight)
    else:
        with pytest.raises(ValueError, match="draft embedding was not loaded"):
            draft.load_weights([])
