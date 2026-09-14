# DeepSeek V4.1 DSpark with pipeline parallelism

## Goal

Enable an experimental DSpark path for DeepSeek V4.1 Flash with a tensor-parallel
target distributed across pipeline stages. Keep the target's vision support,
quantization, and full resident weights. Measure correctness and performance
before recommending the feature for serving.

## Approach

Extend vLLM's V2 runner on the `dsv41-optimized` upstream branch at
`e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`. This branch's DSpark loader and draft
model match the preview runtime being evaluated. The preview wheel's reported
commit could not be resolved upstream; do not claim these are identical builds.

SGLang would require the same draft-state synchronization work plus porting the
existing target runtime optimizations. A tensor-parallel-only deployment avoids
pipeline integration but changes placement and memory requirements. Keep the
existing TP2/PP3 layout for this experiment.

## Supported first version

- DeepSeek V4.1 only, with every requested auxiliary capture on the last stage.
- The drafter runs on the last stage's existing tensor-parallel group.
- Static verification of five draft tokens, using probabilistic drafting and
  block rejection sampling. Reject adaptive verification with PP explicitly until
  verification layouts can be synchronized across stages.
- The last stage loads its own sharded input embedding from the target checkpoint
  because the target's embedding exists only on the first stage. Reuse its output
  head locally. Never alias a pipeline placeholder as a real embedding.
- Carry both accepted output tokens and the next draft block through the existing
  deferred PP broadcast. Retain request generation filtering for cancellation,
  preemption and slot reuse. All stages must verify the same draft token IDs.
- Collect auxiliary hidden states only on the last stage. Validate placement
  before model allocation and retain all other unsupported-method checks.

## Isolation and acceptance

Build an overlay of the reviewed source diff on the pinned runtime, preserving
existing hardware and vision patches. Keep deployment details outside this fork.
The seventh GPU used for speech is outside the experiment.

Run unit tests for configuration rejection, embedding loading, PP receive queue
lifetime and draft updates after rejection. Then exercise real multi-GPU serving:
startup, deterministic fixtures, sampled text, structured output, images, mixed
prefill/decode, cancelled requests, slot reuse, long context and concurrency.
Compare identical baseline/candidate workloads, including acceptance length,
decode throughput, end-to-end latency, memory usage and errors.

The long-context target is 1,048,576 tokens and the concurrency target is 24;
these are acceptance requirements, not yet measured DSpark capabilities. A
candidate that cannot preserve them must remain experimental with its limits
documented. Restore the baseline after a failed test or a performance regression.

## Implementation sequence

1. Add restricted configuration validation and last-stage auxiliary capture.
2. Load the draft embedding on the last stage and preserve TP1/PP1 sharing.
3. Extend the PP broadcast and masked state update with next-step draft tokens.
4. Run focused tests and build a pinned candidate overlay.
5. Run hardware correctness tests before performance comparisons.
6. Publish measured results and the reproducible patch through the fork's CI.

## Review

Scope is restricted to the selected topology and architecture. Adaptive
verification and arbitrary cross-stage auxiliary captures are deferred, with
explicit errors. No speedup or memory capacity is asserted before measurement.
