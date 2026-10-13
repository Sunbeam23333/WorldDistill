<div align="center">

# WorldDistill

**A unified distillation runtime for video generators and world models.**

[![CPU contract tests](https://github.com/Sunbeam23333/WorldDistill/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunbeam23333/WorldDistill/actions/workflows/ci.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10--3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch core ≥2.5.1](https://img.shields.io/badge/PyTorch_core-%E2%89%A52.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-0B6B45.svg)](LICENSE)

[Installation](docs/installation.md) · [Inference](docs/inference.md) · [Training](docs/training.md) · [Multi-node launch](docs/distributed-launch.md) · [Model support](docs/model-support.md) · [CUDA policy](docs/cuda-compatibility.md) · [Qualification & expected results](docs/hardware-expectations.md) · [Implementation audit](docs/implementation-audit.md)

</div>

WorldDistill couples a research-oriented training/runtime layer for video and
action-conditioned world models with a vendored, LightX2V-based multimodal
inference backend. The repository contains executable trainer loops, cache and
memory policies, CUDA-stream primitives, model/config metadata, and inference
runners. Joint audio-video distillation uses a dedicated two-stream trainer; model-specific inputs are documented in the training guide.


## Architecture

<div align="center">
  <img src="assets/readme/worlddistill_overview.png" alt="WorldDistill teacher-student runtime and inference architecture" width="100%">
</div>

The bow-tie view separates task inputs, teacher/student execution, runtime
mechanisms, and deployment outputs. The trainer registry includes Step,
Progressive, Consistency, Stream, Context Forcing, adversarial, and DMD-style
paths, together with teacher-guided MeanFlow and joint audio-video distillation. [Figure provenance](assets/readme/PROVENANCE.md).

## What is implemented

| Component | Repository state | Validation in this repository |
|---|---|---|
| Shared model metadata | YAML registry plus train/infer resolvers and aliases | CPU catalog/config consistency tests |
| Step distillation | Fixed-timestep trainer, target-only supervision, checkpoint-native dual teacher/student experts | Real tiny Wan forward/backward and exact export/reload |
| MeanFlow | Explicit two-time student, exact directional JVP and teacher-guided target | Analytic derivatives, gradient isolation and exact restart |
| VP diffusion | Checkpoint-native alpha schedule and epsilon/v/sample conversion | CogVideoX DDIM/DPM references and Step gradients |
| Joint audio-video | LTX2 video/audio branches and modality-balanced losses | Two-head gradients, runtime payloads and export/reload |
| Teacher offload | CPU or file-backed frozen weights with stream-safe device leases | Tied weights, buffers, cache hits and revision handoff |
| Raw-video cache builder | Checkpoint VAE/text/image/AV encoders, all tensor conditions and source-frame alignment | Raw→cache→dataset round-trip and fail-safe publication |
| Stream / Progressive / Consistency | Generated-overlap multi-step unroll, complete progressive intervals, EMA consistency | CPU optimizer/restart tests; real tiny Wan tests where its forward contract supports the method |
| Context Forcing | Teacher context, target mask, frame-aligned action/camera packing, tail-safe chunking | CPU memory, temporal-index, and non-divisible-tail tests |
| Hybrid-Sparse Memory | Full-history sparse anchors plus contiguous recent tail | Deterministic selector tests |
| DistillCache | Memory/disk/hybrid stores; original-age-preserving promotion; bounded background CPU prefetch | CPU freshness, worker and runtime regressions |
| CUDA stream runtime | Independent teacher stream, event recording, explicit wait | Serial/DPP output, gradient, event and lifetime checks |
| Dense attention fallback | Device-aware FA2/FA3/Sage/SDPA selection after config overlays | Simulated A100/H20/B200/B300 dispatch tests |
| Fused target-only MSE | Triton kernel with PyTorch reference path | PyTorch numerical reference and CUDA JIT checker |
| DDP / FSDP / DeepSpeed | Wrapper and collective checkpoint control flow | Collective train/resume and optimizer-state checks |
| Wan context parallelism | Model-layer Ulysses token/head exchange, DP×CP sampling and synchronized conditions | Real tiny-Wan forward, chained-rollout gradients and optimizer-update parity |
| Multi-node launch | Manual/Slurm, fixed-size static/c10d rendezvous, node-aware hybrid FSDP groups | Collective launch, node-aware groups and restart checks |
| Automatic precision | Real GEMM/math-attention probes and all-rank common dtype selection; explicit unsupported requests fail | Local precision probes and cross-rank dtype agreement |
| Adversarial / DMD-style paths | Latent discriminator; fake-distribution DSM and detached teacher-minus-fake generator gradients | Tiny-model updates and numerical contracts |

## Core mechanisms

### CUDA stream runtime

<div align="center">
  <img src="assets/readme/cuda_stream_runtime.png" alt="CUDA teacher and student stream execution schematic" width="100%">
</div>

The teacher may launch on a dedicated CUDA stream while the student advances on
the default stream. An event establishes the supervision dependency before the
loss. Diagram spacing is schematic, not a measured profiler trace.
[PDF](assets/readme/cuda_stream_runtime.pdf) · [TeX source](assets/readme/source/cuda_stream_runtime.tex)

### Hybrid-Sparse Memory

<div align="center">
  <img src="assets/readme/hybrid_sparse_memory.png" alt="Hybrid-sparse memory selector over full history" width="100%">
</div>

Under budget `B`, the selector keeps `r = max(1, round(rho * B))` recent frames
contiguously and samples the remaining anchors across the complete earlier
history. Teacher-generated and ground-truth context now consume the same selected
indices. [PDF](assets/readme/hybrid_sparse_memory.pdf) · [TeX source](assets/readme/source/hybrid_sparse_memory.tex)

### Action-conditioned context forcing

<div align="center">
  <img src="assets/readme/action_conditioned_rollout.png" alt="Schematic action-conditioned context-forcing dataflow" width="100%">
</div>

The lower panel reflects the implemented training dataflow: teacher-denoised
context, packed context/target timesteps, action conditioning, and future-only
supervision. The street frames are a generated schematic visualization, not
released-checkpoint output or rollout benchmark evidence.
[PDF](assets/readme/action_conditioned_rollout.pdf) · [TeX source](assets/readme/source/action_conditioned_rollout.tex)

## Quick verification

Create and activate a Python 3.10–3.12 environment, install a PyTorch build for
your target GPU from the [official selector](https://pytorch.org/get-started/locally/),
then install WorldDistill:

```bash
git clone https://github.com/Sunbeam23333/WorldDistill.git
cd WorldDistill
bash scripts/setup_env.sh --dev

python tools/check_cuda_compat.py --probe-precision --json
python -m pytest -q training/tests
```

Optimized kernels and DeepSpeed are opt-in because their builds are tied to the
active CUDA/PyTorch environment:

```bash
bash scripts/setup_env.sh --install-kernels --install-deepspeed
python tools/check_cuda_compat.py --strict
```

See [installation and hardware notes](docs/installation.md) before using
Blackwell. A CUDA version table is a compatibility policy, not a substitute for
an actual kernel forward/backward run on the target host.

Training defaults to `--mixed_precision auto`: choose the common supported
BF16/FP16/FP32 mode after actual device probes and trainer-specific scaler
checks. This does not qualify optional attention/quantization kernels. Older
NVIDIA cards may require an archived software stack; unknown architectures are
not certified by extrapolation.

For multi-node manual/Slurm launch and fixed-size restarts, see the
[launch guide](docs/distributed-launch.md). For executable GPU gates, feature
acceptance targets, memory formulas and carefully scoped published reference
numbers, see [qualification and expected results](docs/hardware-expectations.md)
and the [training verification commands](docs/training.md#hardware-qualification-commands).

## Inference CLI

Resolve configs and validate shell quoting without loading a checkpoint:

```bash
bash scripts/run_infer.sh \
  --model_cls wan2.2_moe \
  --task t2v \
  --model_path ./models/Wan2.2-T2V-A14B \
  --prompt "A paper boat crossing a rain-soaked city street." \
  --gpus 1 \
  --dry-run
```

Run after the dry-run command and CUDA probe are correct:

```bash
bash scripts/run_infer.sh \
  --model_cls wan2.2_moe \
  --task t2v \
  --model_path ./models/Wan2.2-T2V-A14B \
  --prompt "A paper boat crossing a rain-soaked city street." \
  --gpus 1
```

The default JSON above is serial. The wrapper now rejects a mismatched
`--gpus` value instead of launching independent workers on one device. For
multi-GPU inference, pass a config whose `parallel` mesh has the same world
size. Dry-run checks command/config/topology construction only; it does not
check that input files or checkpoints exist and does not execute a kernel.

Conditioned tasks require their actual inputs. For example:

```bash
# Speech-to-video
bash scripts/run_infer.sh --model_cls seko_talk --task s2v \
  --model_path ./models/SekoTalk --image_path /path/to/portrait.png \
  --audio_path /path/to/speech.wav --prompt "The speaker addresses the camera." \
  --gpus 1 --dry-run

# Image editing
bash scripts/run_infer.sh --model_cls qwen_image --task i2i \
  --model_path ./models/Qwen-Image-Edit --image_path /path/to/input.png \
  --prompt "Render the scene as a watercolor." --gpus 1 --dry-run

# Action-conditioned world model
export LIGHTX2V_WORLDPLAY_POSE_PROVIDER=your_package.worldplay_pose_adapter
bash scripts/run_infer.sh --model_cls worldplay_distill --task game \
  --model_path ./models/HY-WorldPlay --image_path /path/to/start.png \
  --prompt "Drive through the intersection while keeping the scene coherent." \
  --pose "w-3, right-0.5" --action_ckpt ./models/HY-WorldPlay/action.safetensors \
  --gpus 1 --dry-run
```

The wrapper validates required argument presence and also forwards
`last_frame_path`, `audio_path`, `image_strength`, VACE/Animate inputs,
camera/action files, and explicit output paths. See
[docs/inference.md](docs/inference.md).

## Distillation training

The fastest executable training check uses tiny CPU models and exercises forward,
backward, optimizer, checkpoint save, and resume:

```bash
python -m pytest -q training/tests/test_training_loop_smoke.py
```

The unified real-model entry accepts Diffusers-style directories whose
transformer forward contract matches the selected trainer:

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
  --distill_method step_distill \
  --teacher_model_path ./models/teacher-diffusers \
  --model_cls wan2.2_moe \
  --data_json data/train_cached.json \
  --data_mode cached \
  --cache_dir data/cached_latents \
  --output_dir results/step_distill
```

Native differentiable models use a strict `worlddistill_model.json` adapter with explicit action/camera inputs. Loading its installed factory requires `--allow_native_model_code`. Method contracts and data schemas are documented in [docs/training.md](docs/training.md).

## Model integration status

“Runner + config” means concrete inference source and a default config are in the
repository. “CLI contract” means argument/config construction is CPU-tested.
Training entries refer to the exact architecture and conditioning contract.

| Family | Inference surface | Unified training | Notes |
|---|---|---|---|
| Wan 2.1 / 2.2 and compatible distilled checkpoints | Runner + config | Exact Diffusers adapter, native dual experts | Wan Ulysses context parallelism; method-specific objective contracts |
| Wan MeanFlow | Runner + config | Teacher-guided `mean_flow_distill` | Explicit two-time student and exact JVP target |
| Wan VACE / Animate / Self-Forcing variants | Runner + config | No dedicated native training integration | Inference variants do not imply a Self-Forcing training algorithm or causal adapter |
| HunyuanVideo 1.5 | Runner + config | Diffusers T2V/I2V adapter | VAE, reference-frame and image-encoder conditioning |
| HY-WorldPlay | Runner + config; external licensed pose adapter required | Author-supplied native training adapter required | Generic Context Forcing is available; no bundled native WorldPlay training adapter |
| Matrix-Game 2.0 | Runner + config; game CLI contract | Not integrated | Image-conditioned game routing is wired |
| LingBot-CAM | Runner + config | Not integrated | Checkpoint location is external to the catalog |
| Wan 2.2 Audio | S2V runner + public config; CLI contract | Not integrated | Requires image and audio inputs |
| SekoTalk | S2V runner; stateful shot-based RS2V wrapper | Not integrated | RS2V reuses overlap latents across clips |
| LTX-Video 2 | Runner + config | Joint audio-video adapter | Packed audio latents and both output heads |
| Qwen Image / LongCat / Z-Image / BAGEL | Runner + config; image CLI contract | Not integrated | Inference only in WorldDistill today |
| SkyReels-V2 and nine catalogued world-model families | Stub/interface only | Not integrated | Ten stub families in total; contributions require a concrete runner, config, and smoke evidence |

The detailed per-model/task table and checkpoint availability are in
[docs/model-support.md](docs/model-support.md). The downloader intentionally does
not imply that every catalog entry has a downloadable checkpoint.

## CUDA compatibility policy

| Target | Compute capability | Native-toolchain policy | Preferred dense attention order |
|---|---:|---:|---|
| A100 | 8.0 | CUDA 11.0+ | FA2 → Sage2 → Torch SDPA |
| H100 / H200 | 9.0 | CUDA 11.8+ | FA3 → FA2 → Sage2 → Torch SDPA |
| H20 | 9.0 | CUDA 12.2+ | FA2 → Sage2 → Torch SDPA; FA3 opt-in with device probe |
| B200 | 10.0 | CUDA 12.8+ | Torch SDPA; architecture-specific optimized backends |
| B300 | 10.3 | CUDA 12.9+ policy; newer vendor builds recommended | Torch SDPA; architecture-specific optimized backends |

Resolution happens after JSON and checkpoint config overlays. Only dense backend
fields are changed; sparse/distributed algorithms such as Ulysses, ring, SVG,
and neighborhood attention are preserved. B200/B300 deliberately do not
auto-select the currently imported FA2/FA3/Sage3 callables: those public kernel
surfaces do not establish `sm_100`/`sm_103` compatibility. A strict probe also checks whether
PyTorch lists the native `sm_*` target instead of silently treating PTX JIT as
native validation. The vendored low-bit CUTLASS extension is explicitly
`sm_120a`-only and is not presented as B200/B300 support. More details:
[docs/cuda-compatibility.md](docs/cuda-compatibility.md).

## Known limitations

- Fixed-size homogeneous worker counts are supported by the launcher. Uneven
  workers, changed-world-size elastic recovery and generic model-layer TP/PP
  are not implemented; a working rendezvous is not a claim of universal topology support.
- Native adapters require the model author's differentiable architecture and licensed weights. Catalog-only entries do not substitute for those assets.
- Multi-rank FSDP/DeepSpeed now receive CPU-staged student parameters instead of
  an already materialized full GPU student. Teacher weights can stay on CPU or file-backed storage; a complete teacher forward must fit the execution device.
- Adversarial training uses a latent projection discriminator; DMD/DMD2 use
  trained fake-score DSM and their separately selected regression/GAN terms.
  FSDP/ZeRO partition the student; auxiliary models remain full replicas with
  independent optimizers, accumulation and AMP scalers.
- Wan uses real Ulysses context parallelism with `sp_size>1`. Other architectures
  require a matching model-layer adapter. MeanFlow keeps `sp_size=1` because its
  JVP needs higher-order derivatives beyond the Ulysses backward contract.
- Cached-latent manifests fail fast on missing declared files and corrupt
  tensors. `tools/cache_training_data.py` uses the selected pipeline's encoders;
  model-specific audio preprocessing and licensed assets remain explicit inputs.
- Benchmark reports identify their model, precision, shape, topology and software
  revisions. Vendored LightX2V results retain their original upstream attribution.

See [docs/reproducibility.md](docs/reproducibility.md) for benchmark protocols. The release
[implementation audit](docs/implementation-audit.md) records fixed contracts and
the remaining gates. The [operator/training follow-up](docs/operator-training-audit.md)
documents the latest fixes, executable checks, and deliberately unclaimed capabilities.

## Repository map

```text
WorldDistill/
├── configs/                 model registry and distillation presets
├── inference/lightx2v/      vendored multimodal inference backend and runners
├── training/                trainers, runtime, data, distributed utilities, tests
├── scripts/                 setup, inference, training, and paper-suite wrappers
├── tools/                   compatibility probe, downloader, sync, conversion tools
├── assets/readme/           current figures, vector PDFs, TeX sources, provenance
└── docs/                    installation, support, CUDA, training, reproducibility
```

## Contributing

New integrations should include a real runner (not a stub), a default config,
input-contract tests, and an evidence record for every hardware claim. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

- [LightX2V](https://github.com/ModelTC/LightX2V) — inference backend foundation
- [Open-Sora](https://github.com/hpcaitech/Open-Sora) — training infrastructure reference
- [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) — world-model/context-forcing reference
- [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) — diffusion-forcing reference

## Citation

```bibtex
@misc{worlddistill2026,
  title        = {WorldDistill: A Unified Distillation Runtime for Video Generators and World Models},
  year         = {2026},
  url          = {https://github.com/Sunbeam23333/WorldDistill}
}
```

## License

The code distributed in this repository is licensed under the
[Apache License 2.0](LICENSE), except where an individual file states otherwise.
WorldDistill does not bundle Tencent HY-WorldPlay source code or model weights;
its optional provider interface can load a user-supplied HY-WorldPlay
integration, which remains subject to Tencent's separate terms. Review
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES), the repository [NOTICE](NOTICE), and
the included [Tencent HY-WorldPlay Community License Agreement](third_party/licenses/TENCENT_HY_WORLDPLAY_COMMUNITY_LICENSE.txt)
before enabling or redistributing that optional integration.
