# Training

WorldDistill exposes one trainer registry and a common runtime boundary. Method
registration is not the same as real-model objective parity; the table below
states the current evidence.

| Method | Implemented surface | Current evidence |
|---|---|---|
| Step | Fixed timetable, native high/low-noise teacher AND student routing | Tiny real Wan optimizer, export/reload and CPU restart tests |
| Stream | Differentiable per-frame Euler unroll; generated overlap passed to the next window | CPU multi-window/solver and restart tests; requires a causal/per-frame model adapter |
| Progressive | All schedule intervals including the final interval to zero | Sampling regression, real tiny Wan update and restart tests |
| Consistency | EMA target and Huber/MSE options | Real tiny Wan update and CPU restart tests |
| Context Forcing | Teacher context, hybrid memory, action/camera passthrough, target mask | CPU selector/chunk tests; closed-loop rollout evaluation pending |
| Adversarial | Latent projection discriminator plus distillation term | Experimental serial/DDP scaffold; GAS=1; not separate certified ADD and LADD paths |
| DMD-style | Fake-distribution DSM; teacher-minus-fake distribution gradient; DMD online teacher regression, DMD2 GAN without regression | Numerical target tests, tiny real Wan DMD/DMD2 updates and CPU/Gloo restart; pretrained reproduction pending |

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

Cold-cache prefetch now runs in a bounded background CPU worker. It preserves
the original production step through promotion and validates freshness again
at consumption. It never runs the teacher or CUDA kernels in that worker.
Teacher-parameter heterogeneous offload is still not implemented.

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

Multi-rank FSDP and DeepSpeed receive CPU-staged student parameters. The loader
does not first create a full FP32 student on every GPU. Teachers remain
replicated; this is not an out-of-core teacher or fully sharded file reader.
GPU initialization peaks and FSDP/ZeRO restarts still require hardware runs.

## Exact architecture and conditioning

The loader instantiates the checkpoint's actual Diffusers `_class_name` rather
than substituting `Transformer2DModel`. A Wan2.2 pipeline with `transformer_2`
must provide its boundary; both teacher and student retain both experts.
Encoded I2V `image_cond` is concatenated only after shape checks. Text embeddings,
pooled embeddings, a second text encoder, masks, and image embeddings remain
distinct inputs. Unsupported action/camera or temporal-attention inputs raise
instead of being dropped. An inference-only runner is not made differentiable
merely by wrapping its name in a training adapter.

Cached samples may provide `conditioning_path`, a trusted tensor-only `.pt`
dictionary with the model's named conditioning tensors. Raw video controls
must provide one entry per original source frame (`action_temporal_axis` and
`camera_temporal_axis` default to zero). Video sampling and repeat padding use
the same indices for controls; causal VAE compression selects corresponding
latent anchor frames. Timestamped or interval-integrated actions require an
explicit upstream conversion, not guessed interpolation. Wan VAE channel-wise
mean/std and multi-encoder prompt outputs are preserved. This is not universal
raw I2V preprocessing: the HunyuanVideo 1.5 image/mask channels and vision
embeddings (including its T2V zero conditions) must be explicitly supplied;
LTX-family raw VAE normalization requires a native adapter. Non-1000 declared
scheduler time scales are rejected by the current training schedules. Stock
CogVideoX uses variance-preserving diffusion, not this trainer's linear flow
noise process; its layout adapter does not make those objectives equivalent,
and it is explicitly rejected until a matching training objective is provided.

## Train → export → sample

```bash
python tools/export_student.py --base_model /models/teacher-diffusers \
  --checkpoint results/step --output_dir results/step/student_export --num_steps 4
python tools/sample_student.py --bundle results/step/student_export \
  --prompt "A paper boat crosses the rain." --save_path results/step/student.mp4
```

The bundle contains trained denoisers and checkpoint provenance. Sampling loads
them into the original Diffusers pipeline, reusing its licensed VAE/text assets.
It does not sample an unrelated catalog student. Native runner conversion,
legacy separately saved dual students and unmerged PEFT weights are not silently
treated as full Diffusers exports. The base pipeline must support every supplied
control. This sampling bridge is single-device and separate from LightX2V's
optimized runner. The paper scripts reject `INFER_NUM_GPUS` other than one for
this student bridge. JSON conditions retain their frame count and resolution
unless explicitly overridden by command-line flags.

Bundle schema 2 verifies the complete denoiser configuration/weight inventory.
It supports a distinct student architecture and declared `torch.compile`
checkpoint prefixes, but does not promise sampling-objective parity: the
training timetable/shift is recorded, while the base pipeline still supplies
its scheduler and default guidance. Validate these model-specific settings
before reporting a few-step quality result. Checkpoint identity alone does not
establish that equivalence.

Export derives a default step count only from an unambiguous method/checkpoint
record (including the current trained progressive stage). Otherwise supply
`--num_steps`. The world-model paper script consequently requires
`STUDENT_NUM_STEPS` and `STUDENT_PIPELINE_CONDITIONS` for a genuinely compatible
pipeline; its stock native WorldPlay path is not a completed training adapter.

Every training run records local `metrics.jsonl`, `host_manifest.json`,
`config_snapshot.json` and `run_card.md`, even without W&B/TensorBoard. Use
`tools/summarize_run.py RUN_DIR --warmup 50 --measured 200` with `log_every=1`
to aggregate observed step timings. `profile_warmup_steps` and
`profile_active_steps` enable per-rank Chrome traces; tracing itself adds overhead.
Neither these records nor tiny-model tests establish pretrained quality.
