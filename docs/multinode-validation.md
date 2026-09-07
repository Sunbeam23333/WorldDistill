# Multi-node and NVIDIA compatibility follow-up — 2026-09-07

This follow-up extends the preceding [operator/training release](operator-training-audit.md).
It does **not** claim completion of every NVIDIA device, model or distributed
combination. The [expected-results matrix](hardware-expectations.md) is the
feature-by-feature acceptance plan, with external numbers clearly separated
from WorldDistill measurements.

## Implemented changes

- Fixed-size manual/Slurm `torchrun`, static/c10d rendezvous, inherited GPU/MIG
  visibility, explicit backend/timeouts, and node-aware FSDP shard/replica groups.
  Invalid rank layouts, uneven workers and changing world size fail explicitly.
- Native noise-routed experts are kept in one collective leaf for FSDP/ZeRO-3,
  preventing divergent per-rank expert choices from reordering child collectives.
  All experts in that leaf are gathered together; full-leaf peaks can still OOM.
- Actual-device GEMM/math-attention precision probes and all-rank common dtype
  selection. Explicit unavailable precision/TF32 requests are not downgraded.
  Legacy NVIDIA and ARM64 devices have separate conservative installation lanes;
  ROCm's CUDA-compatible namespace is not NVIDIA qualification.
- Fused Triton supervision checks the input tensor's actual GPU, admitting
  only known NVIDIA SM80+ targets. Older/unknown devices use the PyTorch reference;
  illegal-access/device failures are not swallowed or reported as successful fusion.
- DeepSpeed uses exclusive Torch AMP with version/API/scaler admission, not a
  native-half/outer-autocast combination that can mismatch FP32 keyword inputs.
  CPUAdam preserves fresh Adam(W) groups and scheduler association for optimizer
  offload; initialized optimizer state is never silently discarded. Base and
  consistency EMA tensors follow CPU-staged student device placement.
- DeepSpeed FP16 checkpoints explicitly capture and restore each rank's actual
  Torch GradScaler, separately from the native ZeRO scaler. Missing, malformed,
  wrong-world-size and nonfinite states fail rather than restart a fresh scaler.
- A synthetic distributed training checker exercises collectives, rank replicas,
  seven DDP trainer loops, progressive stage transitions and four-step continuous
  versus two-plus-two resumed execution. It compares parameter **updates**, all
  optimizer/RNG/cursor state, engine partitions and method sidecars, rejecting
  frozen updates and nonfinite tensors. FSDP/ZeRO require their selected GPU gates.
- A separate real-runtime CUDA checker compares serial/DPP outputs, loss and
  gradients, observes actual streams/events, stresses temporary tensor lifetimes,
  and optionally exports a trace. It makes no overlap or throughput promise.

## Evidence boundary

Final local full-suite runs, 2026-09-07:

| Dependency environment | Tests | Subtests | GPU-only skips |
|---|---:|---:|---:|
| Diffusers 0.36.0 / Transformers 4.57.1 | 640 passed | 68 passed | 31 |
| Diffusers 0.40.0 / Transformers 5.16.1 | 640 passed | 68 passed | 31 |

These are repeat runs of the same suite, not 1,280 independent features.
Python compilation, shell syntax, package metadata and whitespace checks also
passed. The 31 skips comprise one device-precision case, eight fused-loss cases,
sixteen quantization cases and six stream/lifetime cases; none count as GPU passes.

All local execution used macOS ARM64 with Python 3.11.14 and PyTorch 2.14.0,
without CUDA. Tests include real tiny random-initialized Diffusers models and
actual local multi-process Gloo execution; DeepSpeed/FSDP/CUDA interface cases
are mocks or CPU references where hardware is required. A real CPU GradScaler
test does not qualify a CUDA GradScaler or a DeepSpeed engine.

Logical two-agent tests exercised static rendezvous with two total workers and
c10d with four total workers, including node-aware group collectives. Both agents
ran on the **same physical computer**. This is not evidence for multi-machine
NCCL, InfiniBand/RoCE, hybrid FSDP, ZeRO, or any GPU scaling number.

The precision, stream and distributed CUDA checkers were also invoked on this
CPU host: each returned exit 2 with `status=unavailable`, `qualified=false` and
no GPU cases. Their unavailable results are expected negative controls.

## Remaining qualification and implementation

Run the [hardware commands](training.md#hardware-qualification-commands) on each
exact device/build and then test the actual pretrained model, shape, training
recipe, save/restart/export, quality and long-run stability. Start at two ranks
before larger node counts and retain failures as well as successes.

No A100/H20/B200/B300 GPU measurement, native-model video-quality result or
sanitizer trace is supplied by this release. Generic TP/PP/SP, changed-world-size
elastic recovery, teacher-parameter offload, native action-model adapters,
MeanFlow dual-time/JVP and joint LTX-2 AV training remain separate incomplete
workflows. A reference fallback or an inference catalog entry is not their
implementation.
