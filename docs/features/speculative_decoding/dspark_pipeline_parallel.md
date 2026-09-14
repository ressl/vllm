# Experimental DSpark with pipeline parallelism

This fork adds a restricted DeepSeek V4.1 path to the V2 model runner. It builds
on upstream `dsv41-optimized` at `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.
The restricted six-GPU configuration completed full-model qualification on
September 14, 2026. This remains a hardware-specific experimental fork, not an
upstream-supported release or a guarantee for other topologies.

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

The tested speculative configuration is:

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

The tested configuration keeps native vision, a 1,048,576-token request limit
and 24 concurrent scheduler slots. It uses `--kv-cache-memory-bytes 2147483648`
and target CUDA-graph capture sizes `[6,12,24,48,96,144]`, maximum 144, with
`FULL_DECODE_ONLY`. Other runtime settings match the six-GPU recipe, including
its 1,536-token prefill budget. The cache pool is shared: 24 scheduler slots do
not provide 24 independent million-token contexts. Draft weights, embeddings,
cache and working buffers consume additional memory.

## Validation on September 14, 2026

18 CPU tests and six GPU input-batch tests passed. The focused tests cover
configuration, checkpoint embedding loading, request-slot
reuse, padding of sampled tokens, and GPU rejection/draft-state updates. The
padding regression reproduces mismatched collective sizes before the fix and
passes after it. The GPU update tests pass both with draft tokens and without
them; the latter caught a Triton compilation failure during initial warmup.

Full-model checks passed for normal answers, separate reasoning, automatic tool
round trips, streaming, JSON-schema output, native image understanding, image
follow-ups, four-image requests, image-limit rejection and mixed image/text
concurrency. Four streaming requests were cancelled, followed by 24 simultaneous
arithmetic requests; every answer was correct and the queue drained.

The exact context-boundary test used 1,048,448 input tokens plus 128 forced
output tokens and recovered all three separated records in 129.01 seconds.
That synthetic recall test does not establish general long-context model quality.
It ignores EOS to exercise the full output budget; continuation beyond the answer
is not evaluated as a normal chat response.

Memory observations:

- All target shards and the three draft layers loaded on TP2/PP3. Last-stage
  model memory increased from about 70.29 GiB to 75.10 GiB per GPU.
- The baseline's 1.8 GiB KV allocation did not satisfy the 1M request limit with
  DSpark (initialization required about 1.87 GiB). The tested profile uses 2 GiB.
- Weights remain resident on the GPUs; CPU and storage offload are disabled.

The successful retry used the sampled-token padding fix at source commit
`607cc00e8cc5d9c6786a422237391c6cc74c94b1`. Earlier trials had failed during
configuration or warmup and were rolled back; the retry completed without
additional inference-code changes. A diagnostic image added signal-triggered
stack dumps, which were removed from the final runtime image.

### First isolated performance comparison

Every request used a 1,024-token prompt and 1,024 generated tokens, temperature 0,
fixture seed 42, with no prefix-cache hits or preemptions. These are single
retained runs of the target-only baseline and the diagnostic DSpark build.
They are workload-specific observations, not confidence intervals.

| Measurement | Target only | DSpark | Change |
| --- | ---: | ---: | ---: |
| Single-request streamed decode, tokens/s | 103.85 | 290.14 | +179% |
| Single-request end-to-end output, tokens/s | 101.28 | 271.19 | +168% |
| 24-request aggregate output, tokens/s | 494.18 | 729.29 | +48% |
| 24-request median streamed decode, tokens/s/request | 21.29 | 33.20 | +56% |

The final image, without diagnostics, independently passed the full suite again.
It measured **285.52 decode tokens/s** for one request and **725.59 aggregate
output tokens/s** at 24 requests, respectively **175%** and **47%** above the
baseline. Its exact-context run took 132.86 seconds. JSON output, generation at
temperature 0.7, cancellation and request-slot reuse also passed.

[Machine-readable results](../../../tools/dspark_pipeline/results-2026-09-14.json)
retain both candidate runs. Greedy response hashes differed across runs;
bitwise reproducibility and broad accuracy parity were not established.

The eight-request DSpark run measured 394.47 aggregate output tokens/s. No matching
eight-request baseline was retained for this comparison. Single-request decode
excludes time to first token; aggregate output rates include it. Draft acceptance
and speed depend on the workload, prompt length and concurrency.

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
isolated test. Retest the complete workload on your hardware before promoting the overlay.

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

To exercise JSON output, cancellation and request-slot reuse against an isolated
server after the core recipe acceptance tests:

```bash
.venv/bin/python tools/dspark_pipeline/acceptance.py \
  --url http://127.0.0.1:30000 --output /tmp/dspark-acceptance.json
```

Related upstream work: [vLLM PR #53577](https://github.com/vllm-project/vllm/pull/53577)
implements broader DSpark pipeline support, including warmup safeguards and
other models. This fork is a restricted overlay on the V4.1 preview runtime;
it does not replace that upstream review process.
