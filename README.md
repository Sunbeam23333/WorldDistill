<div align="center">

<img src="pics/worldDistill_banner.jpg" alt="WorldDistill Banner" width="100%">

# WorldDistill

**A unified toolkit for distilling and accelerating video, image, and audio-video generation models and world models.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

</div>

---

## Overview

WorldDistill provides a **unified framework** for distilling video generation models, audio-video models, image generation/editing models, and interactive world models into faster, smaller variants. It integrates:

- **Multimodal Inference Engine** — High-performance multi-GPU inference for **video, image, and audio-video** tasks, with automatic config discovery and modality-aware output handling. Adapted from [LightX2V](https://github.com/ModelTC/lightx2v)
- **Training Pipeline** — Multiple distillation methods (step, stream, progressive, consistency, context forcing, adversarial ADD/LADD, DMD), referencing [Open-Sora](https://github.com/hpcaitech/Open-Sora) and [HY-WorldPlay](https://github.com/Tencent/HunyuanVideo)
- **Train-Infer Unified Metadata** — Shared `model_catalog` across training and inference: model family, architecture, supported tasks, modality I/O profile, and distillation stage (base/teacher/student) are resolved from a single `model_zoo.yaml`
- **Teacher-Student Runtime** — Built-in `TeacherStudentRuntime` with distillation cache, fused supervision, and experiment tracking for tightly coupled train-infer workflows (e.g., OPD, context forcing)
- **Distributed/Acceleration** — DDP/FSDP/DeepSpeed (ZeRO), Sequence Parallelism, gradient checkpointing, mixed precision
- **Extensible Architecture** — Registry-based Runner/Scheduler system for easy model integration

### Architecture

<div align="center">
<img src="pics/FlowChart.png" alt="WorldDistill Architecture" width="100%">
</div>

---

## Supported Models

### Video Generation Models

| Model | Architecture | Tasks | Distillation Methods | Status |
|:------|:------------|:------|:-------------------|:------:|
| **Wan 2.1 / 2.2** | Wan DiT | T2V, I2V | Step, LoRA, Full, FP8 | ✅ |
| **Wan 2.2 MoE (A14B)** | Wan MoE DiT | T2V, I2V | Step (4-step, dual model) | ✅ |
| **Wan 2.1 Self-Forcing** | Wan DiT | T2V | Self-Forcing / Streaming | ✅ |
| **Wan 2.1 MeanFlow Distill** | Wan DiT | T2V | Mean-Flow Distillation | ✅ |
| **HunyuanVideo 1.5** | HY DiT | T2V, I2V | Step | ✅ |
| **LTX-Video 2** | DiT | T2V, I2V, T2AV, I2AV | Step | ✅ |
| **SkyReels-V2** | Wan + Diffusion Forcing | T2V, I2V | Stream (Diffusion Forcing) | 🔧 |

### Audio-Video & Talking Head Models

| Model | Architecture | Tasks | Features | Status |
|:------|:------------|:------|:---------|:------:|
| **Wan 2.2 Audio** | Wan Dense | S2V, RS2V | Audio conditioning, streaming | ✅ |
| **SekoTalk** | Wan | S2V, RS2V | Audio conditioning, reference speech | ✅ |

### Image Generation & Editing Models

| Model | Architecture | Tasks | Features | Status |
|:------|:------------|:------|:---------|:------:|
| **Qwen Image Edit 2511** | Qwen | T2I, I2I | Image generation & editing | ✅ |

### Video Editing & Animation Models

| Model | Architecture | Tasks | Features | Status |
|:------|:------------|:------|:---------|:------:|
| **Wan 2.1 VACE** | Wan DiT | VACE | Video editing | ✅ |
| **Wan 2.2 MoE VACE** | Wan MoE | VACE | Video editing | ✅ |
| **Wan 2.2 Animate** | Wan MoE | Animate | Motion guidance, pose conditioning | ✅ |

### World Models (Interactive / Game Generation)

| Model | Architecture | Tasks | Distillation Methods | Status |
|:------|:------------|:------|:-------------------|:------:|
| **WorldPlay** | HY DiT + Action + PRoPE | T2V, I2V, Game | Step, Context Forcing | ✅ |
| **LingBot-CAM** | Wan MoE + Plucker Camera | I2V (Camera Ctrl) | — | ✅ |
| **GameFactory** | Custom | Game Video | Step | 🔧 |
| **Hunyuan-GameCraft** | HY DiT | Game Video | Step, Context Forcing | 🔧 |
| **Infinite-World** | Custom | Open-World Game | Step | 🔧 |
| **Matrix-Game 2.0** | Wan + Self-Forcing | Game | Step | 🔧 |
| **Genie / Genie 2** | Autoregressive | Game | Progressive | 🔧 |
| **GameGen-X** | Custom | Game | Step | 🔧 |
| **V-Mem / SPMem** | Memory-augmented | Long Video | Context Forcing | 🔧 |
| **CAM** | Memory-augmented | Long Video | Context Forcing | 🔧 |
| **Mirage (Decart)** | Custom | Game | Step | 🔧 |

> ✅ = Fully supported &nbsp;|&nbsp; 🔧 = Runner interface defined, implementation in progress — contributions welcome!

---

## Distillation Methods

| Method | Target Steps | Key Idea | Config Preset |
|:-------|:----------:|:---------|:-------------|
| **Step Distillation** | 2 / 4 / 8 | Fixed timestep schedule, no CFG. Supports MoE dual-model (high/low noise). | `step_distill_4step.json` |
| **Stream Distillation** | Per-frame | Diffusion Forcing: per-frame independent noise with monotonic schedule. Sliding window for infinite length. | `stream_distill.json` |
| **Progressive Distillation** | N → N/2 | Iteratively halve steps: teacher does 2 steps, student matches in 1. | — |
| **Consistency Distillation** | 1–4 | Trajectory Consistency (TCD). EMA target + Huber loss for stability. | `consistency_distill.json` |
| **Context Forcing** | 4 | Memory-aware: teacher-generated context prevents student drift. Curriculum training. | `context_forcing.json` |
| **Adversarial Distillation (ADD/LADD)** | 1–4 | Score distillation + adversarial discriminator in latent space. | — |
| **Distribution Matching Distillation (DMD)** | 1 | Fake score network aligns student distribution with teacher. | — |
| **Distribution Matching Distillation (DMD2)** | 1 | GAN-augmented distillation on real data (no fake-score regression). | — |

---

## Feature Checklist (功能清单)

- **Multimodal Inference**: unified pipeline for video, image, and audio-video tasks with modality-aware output labeling and automatic config discovery from `model_zoo.yaml`.
- **Train-Infer Unified Metadata**: shared `model_catalog` resolves architecture, model family, modality I/O profile, distillation stage (base/teacher/student), and teacher-student pairing across both training and inference.
- **Training Pipeline**: unified trainer loop, gradient accumulation, EMA, checkpointing, resume.
- **Teacher-Student Runtime**: `TeacherStudentRuntime` with `DistillCache`, `FusedSupervision`, and `ExperimentTracker` for OPD / context forcing / adversarial distillation loops.
- **Distillation Methods**: Step, Stream (Diffusion Forcing), Progressive, Consistency (TCD), Context Forcing, Adversarial (ADD/LADD), DMD, Mean-Flow.
- **Distributed/Acceleration**: DDP, FSDP (FULL/HYBRID), DeepSpeed ZeRO (1/2/3), Sequence Parallelism, mixed precision, gradient checkpointing.
- **Inference Engine**: LightX2V-based multi-GPU inference, step-distill schedulers, stream/consistency schedulers.
- **Data & Sampling**: cached latent dataset, bucket sampler for variable resolution/frames.
- **Utilities**: model downloader, fast sync, weight conversion, camera pose generator.
- **Logging / Dashboards**: console, TensorBoard, and optional W&B with runtime/cache/performance metrics.
- **Evaluation**: periodic validation via `val_data_json` and `eval_every`.
- **Smoke Tests**: lightweight `test_runtime_smoke.py` validates catalog resolution, trainer args, and inference metadata without GPU.

## Quick Start

### 1. Environment Setup

```bash
# Clone & install
git clone https://github.com/Sunbeam23333/WorldDistill.git
cd WorldDistill
bash scripts/setup_env.sh
```

Or manually:

```bash
conda create -n worlddistill python=3.10 -y && conda activate worlddistill
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
cd inference && pip install -e . && cd ..
pip install -r requirements.txt
pip install "transformers==4.57.1"
```

### 2. Download Models

```bash
# Configure credentials (if behind proxy)
cp .env.example .env
# Edit .env with your proxy/HF token settings

# List available models
python tools/download_models.py --list

# Download a specific model
python tools/download_models.py --model wan2.2_moe --target-root ./models
```

### 3. Inference

**Single GPU:**

```bash
python -m lightx2v.infer \
    --model_cls wan2.2_moe \
    --task t2v \
    --model_path ./models/Wan2.2-T2V-A14B \
    --prompt "A white cat wearing sunglasses on a surfboard at a summer beach."
```

**Multi-GPU (8×H20):**

```bash
bash scripts/run_infer.sh \
    --model wan2.2_moe \
    --task t2v \
    --prompt "A futuristic cityscape at sunset with flying cars." \
    --gpus 8
```

**Step-distilled (4-step, 8× faster):**

```bash
bash scripts/run_infer.sh \
    --model wan2.2_moe_distill \
    --task t2v \
    --prompt "An astronaut floating in space, Earth in the background." \
    --gpus 8
```

**Audio-Video (SekoTalk, speech-to-video):**

```bash
bash scripts/run_infer.sh \
    --model seko_talk \
    --task s2v \
    --gpus 1 \
    --save_path results/talking_head.mp4
```

**Image Editing (Qwen, image-to-image):**

```bash
bash scripts/run_infer.sh \
    --model qwen-image-edit-2511 \
    --task i2i \
    --prompt "Turn the scene into a watercolor painting." \
    --gpus 1 \
    --save_path results/edited.png
```

**Self-Forcing streaming generation:**

```bash
bash scripts/run_infer.sh \
    --model wan2.1_sf \
    --task t2v \
    --prompt "A river flowing through a forest in autumn." \
    --gpus 4
```

### 4. Distillation Training

**Step Distillation (4-step):**

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
    --distill_method step_distill \
    --teacher_model_path ./models/Wan2.2-T2V-A14B \
    --model_cls wan2.2_moe \
    --distill_preset configs/distill_presets/step_distill_4step.json \
    --data_json data/train.json \
    --cache_dir data/cached_latents \
    --output_dir results/step_distill_4step \
    --learning_rate 1e-5 \
    --max_train_steps 50000 \
    --batch_size 1
```

**Stream Distillation (Diffusion Forcing):**

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
    --distill_method stream_distill \
    --teacher_model_path ./models/SkyReels-V2 \
    --model_cls skyreels_v2 \
    --distill_preset configs/distill_presets/stream_distill.json \
    --data_json data/train.json \
    --output_dir results/stream_distill
```

**Context Forcing (for world models):**

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
    --distill_method context_forcing \
    --teacher_model_path ./models/HunyuanVideo-WorldPlay \
    --model_cls worldplay \
    --distill_preset configs/distill_presets/context_forcing.json \
    --data_json data/train.json \
    --optimizer muon \
    --output_dir results/context_forcing
```

**Adversarial Distillation (ADD/LADD):**

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
    --distill_method adversarial_distill \
    --teacher_model_path ./models/Wan2.2-T2V-A14B \
    --model_cls wan2.2_moe \
    --data_json data/train.json \
    --output_dir results/adversarial_distill
```

**DMD (Distribution Matching Distillation):**

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
    --distill_method dmd_distill \
    --teacher_model_path ./models/Wan2.2-T2V-A14B \
    --model_cls wan2.2_moe \
    --data_json data/train.json \
    --output_dir results/dmd_distill
```

**DMD2 (GAN-augmented):**

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
    --distill_method dmd_distill \
    --dmd_variant dmd2 --dmd_use_gan \
    --teacher_model_path ./models/Wan2.2-T2V-A14B \
    --model_cls wan2.2_moe \
    --data_json data/train.json \
    --output_dir results/dmd2_distill
```

**With LoRA (parameter-efficient):**

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
    --distill_method step_distill \
    --teacher_model_path ./models/Wan2.2-T2V-A14B \
    --use_lora --lora_rank 64 \
    --distill_preset configs/distill_presets/step_distill_4step.json \
    --data_json data/train.json \
    --output_dir results/step_distill_lora
```

### Parallel Training Options

- **DDP**: `--parallel_mode ddp` (default)
- **FSDP**: `--parallel_mode fsdp --fsdp_shard_strategy full|hybrid`
- **DeepSpeed ZeRO**: `--parallel_mode deepspeed --deepspeed_stage 1|2|3`
- **Sequence Parallel**: `--sp_size 2` (set group size, combine with any mode)

### Validation / Evaluation

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
    --distill_method step_distill \
    --teacher_model_path ./models/Wan2.2-T2V-A14B \
    --data_json data/train.json \
    --val_data_json data/val.json \
    --eval_every 5000 \
    --eval_batches 4
```

### Training Monitoring / Dashboards

```bash
bash scripts/run_train.sh \
    --method context_forcing \
    --teacher_model ./models/HunyuanVideo-WorldPlay \
    --data_json data/train.json \
    --report_to console,tensorboard,wandb \
    --wandb_project worlddistill \
    --wandb_run_name context-forcing-dpp
```

Tracked metrics now include:

- training loss / eval loss / learning rate / grad norm
- optimizer skip count for unstable steps
- step time, steps/sec, samples/sec, ETA
- GPU allocated/reserved/peak memory (when running on CUDA)
- runtime cache hits/misses, prefetch count, DPP async launch/wait stats

Useful commands:

```bash
tensorboard --logdir results/training/tensorboard --port 6006
wandb sync ./results/training/wandb
```

### 5. Benchmarking

```bash
# Wan2.2 MoE T2V benchmark (5 prompts, 8 GPUs)
bash scripts/benchmarks/bench_wan22.sh

# LingBot camera-controlled I2V benchmark
bash scripts/benchmarks/bench_lingbot.sh

# HunyuanVideo 1.5 T2V benchmark
bash scripts/benchmarks/bench_hunyuan.sh
```

---

## Project Structure

```
WorldDistill/
├── inference/                  # Inference engine (adapted from LightX2V)
│   ├── lightx2v/
│   │   ├── infer.py            # Inference entry point (prints modality & distill profile)
│   │   ├── pipeline.py         # Unified pipeline with modality-aware result labeling
│   │   ├── models/
│   │   │   ├── runners/        # Model-specific runners
│   │   │   │   ├── wan/        # Wan 2.1/2.2/MoE/Audio/VACE/SF/Animate runners
│   │   │   │   ├── hunyuan_video/ 
│   │   │   │   ├── worldplay/  # WorldPlay runner
│   │   │   │   ├── ltx2/       # LTX-Video 2 runner (T2V/I2V/T2AV/I2AV)
│   │   │   │   └── world_models/  # World model stubs (10 models)
│   │   │   └── schedulers/
│   │   │       ├── wan/step_distill/        # Step distillation scheduler
│   │   │       ├── stream_distill/          # Stream (Diffusion Forcing) scheduler
│   │   │       └── consistency_distill/     # Consistency (TCD) scheduler
│   │   └── utils/
│   │       ├── model_catalog.py  # Inference-side multimodal & distill metadata resolver
│   │       ├── set_config.py     # Auto config discovery from catalog when no explicit JSON
│   │       └── env_compat.py     # Version-tolerant import utilities
│   └── configs/                # Model-specific inference configs
│       ├── wan22/              # Wan 2.2 (video, audio, MoE)
│       ├── seko_talk/          # SekoTalk (S2V, RS2V)
│       ├── self_forcing/       # Self-Forcing streaming
│       ├── meanflow/           # Mean-Flow distillation
│       ├── wan22_vace/         # VACE video editing
│       └── ...
│
├── training/                   # Distillation training pipeline
│   ├── train_distill.py        # Training entry point
│   ├── trainer_args.py         # All configurable hyperparameters (+ modality & distill fields)
│   ├── model_catalog.py        # Training-side model metadata (mirrors inference catalog)
│   ├── env_compat.py           # Version-tolerant import utilities
│   ├── trainers/
│   │   ├── base_distill_trainer.py      # Base class (flow matching, DDP/FSDP/DS, grad accum)
│   │   ├── step_distill_trainer.py      # Fixed N-step + dual-model MoE
│   │   ├── stream_distill_trainer.py    # Diffusion Forcing + sliding window
│   │   ├── progressive_distill_trainer.py # Iterative step halving
│   │   ├── consistency_distill_trainer.py # TCD/LCD + EMA
│   │   ├── context_forcing_trainer.py   # Memory-aware + curriculum
│   │   ├── adversarial_distill_trainer.py # ADD/LADD adversarial distillation
│   │   └── dmd_distill_trainer.py       # Distribution Matching Distillation (DMD/DMD2)
│   ├── runtime/                # Train-infer coupled runtime modules
│   │   ├── teacher_student_runtime.py   # Teacher-Student execution runtime
│   │   ├── world_model_runtime.py       # World model specific runtime
│   │   ├── distill_cache.py             # Distillation cache for teacher outputs
│   │   └── fused_supervision.py         # Fused multi-loss supervision
│   ├── tests/
│   │   └── test_runtime_smoke.py  # Smoke tests (catalog, args, metadata — no GPU needed)
│   ├── data/
│   │   ├── video_dataset.py    # Raw video + cached latent datasets
│   │   └── bucket_sampler.py   # Resolution/frame bucketed sampling
│   └── utils/
│       ├── optimizers.py       # AdamW + Muon optimizer
│       ├── schedulers.py       # LR schedulers (cosine, linear, etc.)
│       ├── distributed.py      # DDP + FSDP + DeepSpeed utilities
│       ├── batch_encoder.py    # Batch encoding utilities
│       └── experiment_tracking.py # Experiment tracker (console/TB/W&B)
│
├── tools/                      # Utility scripts
│   ├── download_models.py      # Model downloader (HuggingFace)
│   ├── fast_sync.py            # High-speed parallel file sync
│   ├── generate_camera_poses.py # Camera trajectory generator
│   └── convert_weights.py      # Weight format converter
│
├── scripts/                    # Shell scripts
│   ├── setup_env.sh            # Environment setup
│   ├── run_infer.sh            # Unified inference runner (task-aware output: .mp4/.png/.wav)
│   ├── run_train.sh            # Unified training runner
│   └── benchmarks/             # Model-specific benchmark scripts
│
├── configs/
│   ├── model_zoo.yaml          # Model registry (architectures, tasks, modalities, distill pairs)
│   └── distill_presets/        # Distillation method presets (JSON)
│
├── pics/                       # Project images
│   ├── worldDistill_banner.jpg # Project banner
│   └── FlowChart.png           # Architecture flowchart
│
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
├── setup.py                    # pip install -e .
└── LICENSE                     # Apache 2.0
```

---

## Data Preparation

WorldDistill supports two data modes:

### Cached Latent Mode (Recommended)

Pre-compute latents and text embeddings for faster training:

```json
[
  {
    "latent_path": "latents/video_001.pt",
    "text_embed_path": "text_embeds/video_001.pt",
    "image_cond_path": "image_conds/video_001.pt",
    "num_frames": 49,
    "resolution": [60, 107],
    "text": "A cat playing with a ball..."
  }
]
```

### Raw Video Mode

Load and encode videos on-the-fly (slower, but requires no preprocessing):

```json
[
  {
    "path": "videos/video_001.mp4",
    "text": "A cat playing with a ball...",
    "resolution": [480, 854],
    "num_frames": 49,
    "fps": 24
  }
]
```

---

## Adding a New Model

WorldDistill uses a **registry-based** Runner system. To add a new model:

1. **Create a Runner** in `inference/lightx2v/models/runners/your_model/`:

```python
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.models.runners.default_runner import DefaultRunner

@RUNNER_REGISTER("your_model_key")
class YourModelRunner(DefaultRunner):
    def __init__(self, config):
        super().__init__(config)
        # Load your model architecture
    
    def run(self, *args, **kwargs):
        # Implement inference pipeline
        ...
```

2. **Register** in `inference/lightx2v/infer.py` — add `"your_model_key"` to the `model_cls` choices list and import.

3. **Add config** in `inference/configs/your_model/` and register in `configs/model_zoo.yaml`.

4. **(Optional)** Add a custom Scheduler in `inference/lightx2v/models/schedulers/` for specialized distillation.

---

## Adding a New Distillation Method

1. **Create a Trainer** in `training/trainers/your_method_trainer.py`:

```python
from training.trainers.base_distill_trainer import BaseDistillTrainer

class YourMethodTrainer(BaseDistillTrainer):
    def compute_distill_loss(self, teacher_output, student_output, batch, timesteps):
        # Define your loss function
        ...
    
    def prepare_teacher_input(self, batch, noisy_latents, timesteps):
        # Prepare model inputs
        ...
```

2. **Register** in `training/trainers/__init__.py`:

```python
from training.trainers.your_method_trainer import YourMethodTrainer
TRAINER_REGISTRY["your_method"] = YourMethodTrainer
```

3. **Add preset** in `configs/distill_presets/your_method.json`.

---

## Known Limitations

- **Runner-based training**: Training-side Runner integration is a placeholder; use diffusers or explicit weight files.
- **FSDP checkpoints**: Optimizer state is not fully captured in FSDP saves (model weights only).
- **VLM/LLM distillation**: Not currently supported in the `LightX2V` inference runtime. The multimodal metadata layer is designed to be modality-agnostic, so a future `vlm_llm` runtime can be added without disrupting the existing diffusion/video pipeline.
- **Unused args**: `denoising_steps_per_frame`, `generator_update_interval`, and `loss_type` are currently not wired into training loops.

## Acknowledgments

- [LightX2V](https://github.com/ModelTC/lightx2v) — Inference acceleration framework (basis of our inference engine)
- [Open-Sora 2.0](https://github.com/hpcaitech/Open-Sora) — Training infrastructure (flow matching, bucket sampling)
- [HY-WorldPlay](https://github.com/Tencent/HunyuanVideo) — World model training pipeline (context forcing, Muon optimizer, curriculum)
- [SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) — Diffusion Forcing reference
- [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — ADD/LADD distillation
- [Latent Adversarial Diffusion Distillation](https://arxiv.org/abs/2403.12015) — latent-space adversarial distillation
- [Distribution Matching Distillation](https://arxiv.org/abs/2311.18828) — DMD
- [Improved DMD](https://arxiv.org/abs/2405.14867) — DMD2
- [Consistency Models](https://arxiv.org/abs/2303.01469) — Trajectory consistency distillation

---

## Citation

If you find WorldDistill useful, please consider citing:

```bibtex
@misc{worlddistill2025,
  title={WorldDistill: A Unified Toolkit for Video Generation Model Distillation},
  year={2026},
  url={https://github.com/Sunbeam23333/WorldDistill}
}
```

## License

[Apache License 2.0](LICENSE)
