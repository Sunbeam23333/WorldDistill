# Implementation audit

This document records the repository-to-README audit performed for the 0.2.0
release candidate. It distinguishes fixed execution contracts from claims that
still need hardware or model evidence.

## Release-blocking contracts fixed

| Area | Prior failure | Current behavior | Evidence |
|---|---|---|---|
| Source packaging | Root `data/` and `models/` ignore rules also hid Python packages below `training/` and `inference/lightx2v/` | Ignore rules are root-anchored; required source is tracked and checked in CI | clean-clone source check |
| Inference inputs | S2V/I2I examples omitted required files; the shell wrapper dropped audio and several conditioning fields | Wrapper forwards task inputs; Animate requires pose, face, and reference images | CPU CLI tests |
| RS2V | Unified runner bypassed SekoTalk's clip state, reference-state sequence, and overlap latent | Public RS2V routes through `shot_runner.rs2v_infer` | command-routing test; checkpoint run pending |
| Multi-GPU inference | `torchrun -n 8` with a serial config could create independent workers, typically all targeting the default device | Shell and Python runtime reject world-size/config-mesh mismatch | serial/8-way topology tests |
| Public configs | Audio/RS2V/VSR examples contained workstation-local absolute paths | Internal LoRA defaults are removed; caller model path wins overlays; optional VSR fails with a setup message | JSON/static checks |
| Cached training data | Missing/corrupt latents silently became a fixed-shape zero tensor | All declared paths are checked at startup; load corruption fails with sample/path context | dataset regression tests |
| CUDA dense attention | Config overlays could restore an unsupported backend; registry/import presence was mistaken for device compatibility | Resolution runs after overlays, imports callables, respects device profiles, and keeps SM100/103 on SDPA until an optimized kernel is device-validated | simulated A100/H20/B200/B300 tests |
| Low-bit CUDA extension | The vendored CUTLASS operators could be selected outside their compiled target | NVFP4/MXFP constructors require the exact `sm_120a` target and callable operators; B200/B300 fail before weight loading | simulated capability tests and source target audit |
| Hybrid-Sparse Memory | A nominal sparse policy could select only from a recent prefix and diverge between context paths | Full-history anchors plus contiguous recent tail share one selector | deterministic selector/tail tests |
| Action/context slicing | Chunked latents retained full-length camera/action tensors, misaligning conditions after sparse memory packing | Temporal conditions are indexed by the exact packed frame indices; ambiguous temporal axes fail early | packed-condition regression tests |
| Checkpoint ordering | Resume happened before distributed wrapping and collective saves were rank-zero gated | Engines wrap before resume and collective save/load paths are rank-consistent | CPU/static tests; multi-rank GPU gate pending |
| Exact resume | Checkpoints omitted RNG state and the consumed dataloader position, so mid-epoch restart replayed data and noise | Per-rank RNG state, sampler epoch, batch cursor, and world size are saved and restored; legacy checkpoints warn and use weights-only semantics | interrupted-vs-continuous CPU loop test |
| Model outputs and student loading | Diffusers `.sample` outputs were not normalized; missing student paths silently cloned the teacher | A shared strict extractor accepts tensor/tuple/mapping/`.sample`; explicit student checkpoints fail on missing, directory, or incompatible state | output/loading contract tests |
| Loss selection | LPIPS choices were advertised without a decoded perceptual path, while generic Huber silently executed MSE | Public choices are MSE/Huber, both have tested implementations, and unknown preset values fail fast | CPU masked-loss tests |
| Optimizer routing | Muon's nested AdamW state and dual-student parameters were not fully represented in save/step/sync paths | Nested AdamW state round-trips; dual branches share one optimizer and synchronized routing; unsupported sharded combinations fail fast | CPU optimizer/DDP contract tests |

## Claims intentionally held back

- No A100, H20, B200, or B300 row is marked hardware-validated without a signed
  host manifest, native-architecture report, kernel comparison, training step,
  inference result, and checkpoint restart.
- Runner/config presence is not called end-to-end distillation support. Image
  and audio-video families currently belong primarily to inference.
- CUDA-stream overlap is an implemented event dependency, but no profiler trace
  or throughput claim is published yet.
- Asynchronous prefetch and heterogeneous placement remain plans/metadata, not a
  background transfer executor.
- Adversarial and DMD-style trainers remain research scaffolds until objective
  parity and real-model results are established.
- No WorldDistill speedup, memory, or quality number is presented without a
  checked-in run manifest and artifact. The vendored LightX2V README retains
  upstream results under an explicit not-reproduced-by-WorldDistill notice.

## Next validation gate

For each supported host tuple, run the checklist in
[reproducibility.md](reproducibility.md), add the immutable artifacts, then
upgrade only the specific model/task/device cell supported by that evidence.
