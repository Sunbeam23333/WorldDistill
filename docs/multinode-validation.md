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
  all registered trainer loops, progressive stage transitions and four-step continuous
  versus two-plus-two resumed execution. It compares parameter **updates**, all
  optimizer/RNG/cursor state, engine partitions and method sidecars, rejecting
  frozen updates and nonfinite tensors. FSDP/ZeRO require their selected GPU gates.
- A separate real-runtime CUDA checker compares serial/DPP outputs, loss and
  gradients, observes actual streams/events, stresses temporary tensor lifetimes,
  and optionally exports a trace. It makes no overlap or throughput promise.

## Regression coverage

The release regression suite covers local multi-process collectives, node-aware groups, checkpoint state and continuous-versus-resumed updates in both supported Diffusers dependency environments. The [training guide](training.md) lists distributed launch and runtime checks; [hardware expectations](hardware-expectations.md) defines the resource and quality measurements for a workload.
