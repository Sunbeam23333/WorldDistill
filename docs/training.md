# Training

WorldDistill exposes one trainer registry and a common runtime boundary. Method
registration is not the same as real-model objective parity; the table below
states the current evidence.

| Method | Implemented surface | Current evidence |
|---|---|---|
| Step | Fixed timetable, single/dual student routing, target loss | Two-step CPU loop and exact single-rank checkpoint resume; real adapter pending |
| Stream | Per-frame schedule, temporal window/overlap, aligned action/camera slicing | CPU schedule/window contracts; `denoising_steps_per_frame` is not yet an inner solver |
| Progressive | Adjacent schedule pairs, two teacher substeps to one student target | CPU schedule/checkpoint contracts; real-model run pending |
| Consistency | EMA target and Huber/MSE options | Implementation/registry checks |
| Context Forcing | Teacher context, hybrid memory, action/camera passthrough, target mask | CPU selector/chunk tests; closed-loop rollout evaluation pending |
| Adversarial | Latent projection discriminator plus distillation term | Experimental serial/DDP scaffold; GAS=1; not separate certified ADD and LADD paths |
| DMD-style | Fake-score and DMD2 GAN auxiliaries | Experimental serial/DDP scaffold; GAS=1; reverse-KL/objective parity is not yet established |

The shared target-supervision path supports MSE and Huber loss. The optional
Triton fusion applies only to masked MSE; Huber always uses the PyTorch path.
LPIPS is intentionally not exposed because latent-space trainers do not decode
predictions to a calibrated perceptual image space.

## Executable CPU smoke test

```bash
python -m pytest -q training/tests/test_training_loop_smoke.py
```

This runs two optimizer steps with tiny teacher/student modules, saves a normal
checkpoint, restores it, and compares the student parameter.

## Cached latent data

```json
[
  {
    "sample_id": "clip-0001",
    "latent_path": "latents/clip-0001.pt",
    "text_embed_path": "text/clip-0001.pt",
    "image_cond_path": "image/clip-0001.pt",
    "action_path": "actions/clip-0001.pt",
    "num_frames": 49,
    "resolution": [60, 107],
    "text": "A camera turns left at an intersection."
  }
]
```

Paths may be relative to `--cache_dir`. Action/camera tensors are loaded when
present. Dataset construction checks every declared local path before workers
start, and tensor deserialization failures raise with the sample ID and path;
missing/corrupt latents are never replaced by synthetic zeros. There is not yet
a universal preprocessing command that can create all fields for every model
family; use the matching teacher tokenizer/VAE and record their revisions in the
manifest.

For Context Forcing, frame-wise `actions` and `camera_poses` are indexed with
the exact packed latent order (selected sparse memory first, then the target
chunk). Ambiguous tensor layouts fail fast. The stock curriculum reaches 160
frames, so `run_train.sh` defaults this method to `--num_frames 160`; cached or
raw clips must actually expose enough frames for the intended stages.

## Raw video data

```json
[
  {
    "sample_id": "clip-0001",
    "path": "videos/clip-0001.mp4",
    "text": "A camera turns left at an intersection.",
    "resolution": [480, 854],
    "num_frames": 49,
    "fps": 24
  }
]
```

Raw mode uses the Diffusers batch encoder. Its checkpoint must expose the
expected text encoder and VAE components.

## Real-model entry

```bash
torchrun --nproc_per_node=8 training/train_distill.py \
  --distill_method step_distill \
  --teacher_model_path ./models/teacher-diffusers \
  --model_cls wan2.2_moe \
  --data_json data/train_cached.json \
  --data_mode cached \
  --cache_dir data/cached_latents \
  --output_dir results/step
```

The current loader reliably handles compatible Diffusers directories. Bare
state dictionaries and inference Runner objects do not yet provide a universal
training adapter. Test the exact model forward signature before a long run.

## Runtime presets

- `step_distill_4step.json`: DPP/fused-supervision requests and dual-model step
  routing. CUDA-unavailable paths fall back safely.
- `context_forcing.json`: world-model runtime, hybrid teacher-context cache, and
  full-history Hybrid-Sparse Memory.
- `world_model_runtime.json`: more explicit cache/memory research knobs.

Asynchronous prefetch and heterogeneous offload are planning metadata today,
not a background transfer executor.

Muon and the dual high/low-noise student mode are supported in serial or DDP
training. They fail fast under FSDP or DeepSpeed until sharded optimizer and
dual-model state transitions have dedicated multi-rank implementations.

`sp_size>1` is accepted only when both model adapters explicitly declare
`supports_worlddistill_sequence_parallel`. Generic Diffusers modules fail fast:
splitting latent frames without model-layer attention communication would be a
silent mathematical error, not valid sequence parallelism.

## Distributed checkpoints

DeepSpeed and FSDP initialization now precede resume. All ranks participate in
collective saves; DeepSpeed stores client step/epoch state and FSDP gathers model
plus optimizer state on rank zero. These paths still require the multi-rank GPU
restart gate in [reproducibility](reproducibility.md) before being labelled
hardware-validated.

New checkpoints also record each rank's Python, NumPy, PyTorch and CUDA RNG
state plus its deterministic sampler epoch/cursor. The iterator is rebuilt and
advanced before RNG restoration, so the CPU smoke resumes at the exact next
batch. Exact restart requires the same dataset, batch size, sampler, and world
size; legacy checkpoints without this metadata fall back to weights/optimizer
resume with an explicit warning.

FSDP and DeepSpeed are currently initialized only after the loader has placed a
full teacher and a cloned FP32 student on each GPU. Their sharding reduces
steady-state training memory, but it is not a load-time out-of-core path: the
unsharded construction peak must fit. Models that require CPU/meta
initialization plus sharded checkpoint loading are not supported yet.
