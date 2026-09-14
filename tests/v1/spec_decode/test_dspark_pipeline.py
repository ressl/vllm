# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP admits only locally available DSpark inputs and static verification."""

from types import SimpleNamespace

import pytest

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
