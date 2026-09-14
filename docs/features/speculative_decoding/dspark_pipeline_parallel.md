# Experimental DSpark with pipeline parallelism

This fork adds a restricted DeepSeek V4.1 path to the V2 model runner. It builds
on upstream `dsv41-optimized` at `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.
End-to-end qualification is in progress; no DSpark speedup is claimed yet.

## What changes

- Keep the draft model on the last stage's tensor-parallel group.
- Load its input embedding from the target checkpoint. Earlier pipeline stages
  do not own this draft model, and the last stage does not own the target's input
  embedding. Reuse the last stage's target output head.
- Broadcast the next draft block alongside accepted samples and rejection counts.
  Filter cancelled requests and reused slots before updating request state.
- Collect auxiliary target streams only on the last stage, and reject a partition
  that places a required capture on an earlier stage.
- Reject adaptive verification with pipeline parallelism. This first version
  requires a common static verification layout across stages.

## Scope of the experiment

The initial topology is six RTX PRO 6000 Blackwell GPUs, TP2/PP3, with a
`8,12,20` layer partition. Draft capture IDs are `37,38,39`. The seventh GPU used
for other services is outside the experiment.

The intended speculative configuration is:

```json
{
  "method": "dspark",
  "num_speculative_tokens": 5,
  "draft_sample_method": "probabilistic",
  "rejection_sample_method": "block",
  "enable_adaptive_verification": false
}
```

Keep the default draft tensor-parallel size equal to the target TP size. This
branch does not add arbitrary cross-stage auxiliary capture or claim support
for other DSpark architectures with PP.

The 1,048,576-token request limit, native vision and 24 concurrent scheduler
slots are qualification targets. Draft weights, embeddings, cache and working
buffers consume additional memory. Existing target-only capacity is not proof
that the same capacity is available with DSpark.

## Validation so far

- 17 focused tests passed on a Blackwell GPU, including masked draft updates,
  rejection accounting and protection against reused request slots.
- The additional configuration test verifies that the draft remains TP2/PP1
  while its target remains TP2/PP3.
- Embedding-loading tests verify real checkpoint data and fail if it is absent.
- A full model startup exposed an additional draft configuration check; the
  draft parallel configuration now represents a single stage.
- Hardware startup, output correctness and comparative performance remain under
  evaluation. Passing unit tests is not a production qualification.

The design and acceptance criteria are recorded in the
[design document](../../superpowers/specs/2026-09-14-dspark-pipeline-parallel-design.md).
Existing hardware runtime adjustments are documented separately in
[llm-inference-recipes](https://github.com/ressl/llm-inference-recipes).

## Focused tests

Use an installed vLLM environment matching this preview branch:

```bash
.venv/bin/python -m pytest \
  tests/v1/spec_decode/test_dspark_pipeline.py \
  tests/v1/worker/test_pp_utils.py \
  tests/v1/worker/test_gpu_input_batch_v2.py
```

The input-batch kernel test requires a GPU. The complete upstream test harness
also needs the upstream test dependencies. No upstream pull request is being
submitted until a human has reviewed the changes and the relevant evaluations.
