# Experimental DSpark with pipeline parallelism

This fork adds a restricted DeepSeek V4.1 path to the V2 model runner. It builds
on upstream `dsv41-optimized` at `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.
This is an experimental prototype. It has not completed end-to-end qualification;
no DSpark speedup is claimed, and the serving system was returned to its baseline.

## What changes

- Keep the draft model on the last stage's tensor-parallel group.
- Load its input embedding from the target checkpoint. Earlier pipeline stages
  do not own this draft model, and the last stage does not own the target's input
  embedding. Reuse the last stage's target output head.
- Broadcast the next draft block alongside accepted samples and rejection counts.
  Filter cancelled requests and reused slots before updating request state.
- Pad prefill samples to the receiver's fixed verification width so that every
  pipeline stage uses the same collective size.
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

## Validation on September 14, 2026

The focused tests cover configuration, checkpoint embedding loading, request-slot
reuse, padding of sampled tokens, and GPU rejection/draft-state updates. The
padding regression reproduces mismatched collective sizes before the fix and
passes after it. The GPU update tests pass both with draft tokens and without
them; the latter caught a Triton compilation failure during initial warmup.

Full-scale testing reached the following points:

- All target shards and the three draft layers loaded on TP2/PP3. Last-stage
  model memory increased from about 70.29 GiB to 75.10 GiB per GPU.
- The baseline's 1.8 GiB KV allocation was insufficient for the requested 1M
  context with DSpark: initialization required about 1.87 GiB. A 2 GiB allocation
  passed that capacity check. This does not validate processing a 1M-token input.
- After correcting the draft parallel configuration and the zero-draft kernel
  path, startup stalled during warmup. Inspection found a sampled-token broadcast
  width mismatch. The fix passes its regression tests but has not yet been
  retried with the complete model.
- The test window ended before a candidate served requests. The original serving
  configuration was restored. DSpark text, reasoning, tools, vision, long-context
  correctness and throughput remain unverified.

Target-only reference measurements used a 1,024-token prompt and 1,024 generated
tokens per request, temperature 0 and seed 42. A single request measured 103.85
tokens/s during streamed decode (101.28 tokens/s including time to first token).
An isolated run of 24 requests measured 494.18 generated tokens/s in aggregate
and a median 21.29 decode tokens/s per request. These are baseline measurements,
not results from this DSpark branch; each configuration has only one retained run.

The next hardware check must first demonstrate a completed startup and correct
multi-step generation with the padding fix, then rerun text/tools, mixed vision,
the exact context limit and concurrency comparisons before promotion. Passing
unit tests is not a production qualification.

The design and acceptance criteria are recorded in the
[design document](../../superpowers/specs/2026-09-14-dspark-pipeline-parallel-design.md).
Existing hardware runtime adjustments are documented separately in
[llm-inference-recipes](https://github.com/ressl/llm-inference-recipes).

## Focused tests

The test build applies the source changes as an overlay on the existing pinned
preview runtime. This preserves the hardware runtime patches from the recipes
repository; it is not a validation of a full rebuild of every upstream component.
To reproduce an overlay from a clean checkout of this branch:

```bash
mkdir -p /tmp/dspark-build
git diff e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba HEAD -- vllm \
  > /tmp/dspark-build/dspark-pp.patch
cp tools/dspark_pipeline/overlay.Dockerfile /tmp/dspark-build/Dockerfile
docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE=YOUR_EXISTING_PINNED_RUNTIME \
  -t vllm-dspark-pp:experimental /tmp/dspark-build
```

Replace `YOUR_EXISTING_PINNED_RUNTIME` with the digest of your working DeepSeek
V4.1 preview runtime. The runtime needs `patch` and the expected Python package
layout. Patch application stops on context mismatches. Keep your existing
hardware configuration and add the speculative configuration above only in an
isolated test. An experimental patch is not a general compatibility guarantee.

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
