# Implementation audit

This document records the repository-to-README audit performed for the 0.2.0
release candidate and summarizes fixed execution contracts.

The current [training guide](training.md) is the executable configuration
reference; historical fixes below retain their original context.

## Training/runtime completion — 2026-10-13

| Area | Implemented contract | Regression |
|---|---|---|
| Objectives | Exact teacher-guided MeanFlow JVP; checkpoint-native flow/VP conversion; two-stream LTX2 loss with modality masks | Analytic derivatives, prediction conversion, real tiny Wan/LTX2 updates |
| Teacher placement | CPU or private memory-mapped parameters, tied weights, dynamic buffers, bounded stream-safe device leases | Resident/offloaded parity, revision handoff, CPU/NVMe CLI train/resume |
| Parallel state | FSDP/ZeRO EMA, collective FP16 overflow handling, full-state progressive teacher handoff; replicated auxiliary optimizers and true GAS | Continuous/resumed optimizer, scaler, RNG and target state |
| Progressive LoRA | Full CPU snapshot and safe merge into existing teacher parameter objects | LoRA/DoRA/rsLoRA, stage transitions and resume |
| Context parallelism | Wan Ulysses model-layer token/head exchange, DP×CP data/RNG semantics and chained-rollout input gradients | Two/four-process full-model gradient and training-update comparisons |
| Tensor/pipeline parallelism | Wan TP×PP×DP, real stage ownership, native declarative plans and Matrix/LingBot PP; canonical Adam/EMA and LoRA handoff | Full-model numerical comparisons, seven first-order trainers, accumulation and restart |
| Elastic recovery | Committed global batch cursor, complete-checkpoint discovery, portable DDP/FSDP/TP-PP state and DeepSpeed Universal conversion | DDP/FSDP 2→1→2; fixed-TP DP resizing; corruption and publication checks |
| Model interfaces | Hunyuan1.5 T2V/I2V raw conditions, LTX statistics, LTX2 AV heads, trusted installed native factories | Exact forward inputs, condition preservation and frame alignment |
| Native/control models | Concrete Matrix-Game2 and LingBot-CAM architectures; dedicated Diffusers VACE/Animate/SkyReels; licensed WorldPlay installed-source bridge | Actual tiny architecture backward, strict reload and native condition contracts; WorldPlay bridge uses the external author runtime |
| Export and sampling | LoRA merge, native/dual experts, MeanFlow and AV; strict reload; pipeline or recorded-timetable adapter sampling | Parameter inventory, prediction parity and full CLI train/resume/export |
| Data and evaluation | Raw→cached encoding with provenance; official VBench bridge, matched-I3D FVD, recovered-action/trajectory metrics | Cache round-trip, invalid-input tests and numerical references |

Supported combinations and architectural restrictions are listed in
[model support](model-support.md), [training](training.md), and
[CUDA compatibility](cuda-compatibility.md). Regression CI runs both pinned
Diffusers dependency environments and a minimal core environment.

## Release-blocking contracts fixed

| Area | Prior failure | Current behavior | Evidence |
|---|---|---|---|
| Source packaging | Root `data/` and `models/` ignore rules also hid Python packages below `training/` and `inference/lightx2v/` | Ignore rules are root-anchored; required source is tracked and checked in CI | clean-clone source check |
| Inference inputs | S2V/I2I examples omitted required files; the shell wrapper dropped audio and several conditioning fields | Wrapper forwards task inputs; Animate requires pose, face, and reference images | CPU CLI tests |
| RS2V | Unified runner bypassed SekoTalk's clip state, reference-state sequence, and overlap latent | Public RS2V routes through `shot_runner.rs2v_infer` | command-routing test |
| Multi-GPU inference | `torchrun -n 8` with a serial config could create independent workers, typically all targeting the default device | Shell and Python runtime reject world-size/config-mesh mismatch | serial/8-way topology tests |
| Public configs | Audio/RS2V/VSR examples contained workstation-local absolute paths | Internal LoRA defaults are removed; caller model path wins overlays; optional VSR fails with a setup message | JSON/static checks |
| Cached training data | Missing/corrupt latents silently became a fixed-shape zero tensor | All declared paths are checked at startup; load corruption fails with sample/path context | dataset regression tests |
| CUDA dense attention | Config overlays could restore an unsupported backend; registry/import presence was mistaken for device compatibility | Resolution runs after overlays, imports callables, respects device profiles, and keeps SM100/103 on SDPA until an optimized kernel is device-validated | simulated A100/H20/B200/B300 tests |
| Low-bit CUDA extension | The vendored CUTLASS operators could be selected outside their compiled target | NVFP4/MXFP constructors require the exact `sm_120a` target and callable operators; B200/B300 fail before weight loading | simulated capability tests and source target audit |
| Hybrid-Sparse Memory | A nominal sparse policy could select only from a recent prefix and diverge between context paths | Full-history anchors plus contiguous recent tail share one selector | deterministic selector/tail tests |
| Action/context slicing | Chunked latents retained full-length camera/action tensors, misaligning conditions after sparse memory packing | Temporal conditions are indexed by the exact packed frame indices; ambiguous temporal axes fail early | packed-condition regression tests |
| Checkpoint ordering | Resume happened before distributed wrapping and collective saves were rank-zero gated | Engines wrap before resume and collective save/load paths are rank-consistent | CPU/static tests |
| Exact resume | Checkpoints omitted RNG state and the consumed dataloader position, so mid-epoch restart replayed data and noise | Per-rank RNG state, sampler epoch, batch cursor, and world size are saved and restored; legacy checkpoints warn and use weights-only semantics | interrupted-vs-continuous CPU loop test |
| Model outputs and student loading | Diffusers `.sample` outputs were not normalized; missing student paths silently cloned the teacher | A shared strict extractor accepts tensor/tuple/mapping/`.sample`; explicit student checkpoints fail on missing, directory, or incompatible state | output/loading contract tests |
| Loss selection | LPIPS choices were advertised without a decoded perceptual path, while generic Huber silently executed MSE | Public choices are MSE/Huber, both have tested implementations, and unknown preset values fail fast | CPU masked-loss tests |
| Optimizer routing | Muon's nested AdamW state and dual-student parameters were not fully represented in save/step/sync paths | Nested AdamW state round-trips; high/low branches share one collective container, optimizer and EMA with GAS and FSDP/ZeRO. Muon retains serial/DDP placement | Optimizer, divergent-expert, checkpoint and accumulation tests |

## Current runtime contracts

See [training](training.md), [model support](model-support.md), and
[CUDA compatibility](cuda-compatibility.md) for current APIs and configuration requirements.
The vendored LightX2V documentation attributes its performance results to the
original upstream implementation and workloads.
