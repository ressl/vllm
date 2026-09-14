# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from unittest.mock import Mock

import numpy as np
import torch

from vllm.v1.worker.gpu import pp_utils


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_deferred_drafts_exclude_reused_slots_and_unfinished_prefill(monkeypatch):
    """A late broadcast must not overwrite another request's draft block."""
    handler = pp_utils.PPHandler.__new__(pp_utils.PPHandler)
    handler.device = torch.device("cpu")
    handler.main_stream = Mock()
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)
    draft_tokens = torch.tensor([[11, 12], [21, 22], [31, 32]])
    slot = pp_utils.PendingRecv(
        event=Mock(),
        sampled_tokens=torch.tensor([[10, -1, -1]] * 3),
        num_sampled=torch.ones(3, dtype=torch.int32),
        num_rejected=torch.zeros(3, dtype=torch.int32),
        idx_mapping=torch.tensor([2, 0, 3]),
        idx_mapping_np=np.array([2, 0, 3]),
        need_sampled_mask=np.array([True, True, False]),
        gen_at_receive_np=np.zeros(3, dtype=np.int32),
        draft_tokens=draft_tokens,
    )
    handler.queue = deque([None, None, slot])
    handler.on_req_idx_freed(0)
    monkeypatch.setattr(
        pp_utils, "async_copy_to_gpu", lambda values, device: torch.from_numpy(values)
    )

    assert handler.get_prev_sampled_outputs() is None
    assert handler.get_prev_sampled_outputs() is None
    outputs = handler.get_prev_sampled_outputs()

    assert outputs["idx_mapping"].tolist() == [2, -1, -1]
    assert outputs["draft_tokens"] is draft_tokens
    handler.main_stream.wait_event.assert_called_once_with(slot.event)
    assert handler.get_prev_sampled_outputs() is None
