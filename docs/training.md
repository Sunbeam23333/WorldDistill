# Training

WorldDistill exposes one trainer registry and a common runtime boundary. Method
registration is not the same as real-model objective parity; the table below
states the executable contracts.

| Method | Implemented surface | Regression coverage |
|---|---|---|
| Step | Fixed timetable, native high/low-noise teacher AND student routing | Tiny real Wan optimizer, export/reload and CPU restart tests |
| Stream | Differentiable per-frame Euler unroll; generated overlap passed to the next window | CPU multi-window/solver and restart tests; requires a causal/per-frame model adapter |
| Progressive | All schedule intervals including the final interval to zero | Sampling regression, real tiny Wan update and restart tests |
| Consistency | EMA target and Huber/MSE options | Real tiny Wan update and CPU restart tests |
| Context Forcing | Teacher context, hybrid memory, action/camera passthrough, target mask | CPU selector/chunk tests |
| Adversarial | Latent projection discriminator plus distillation term; independent AMP and accumulated auxiliary updates | Serial/DDP and sharded-student/full-replica-critic contracts, checkpoint and resume |
| DMD-style | Fake-distribution DSM; teacher-minus-fake distribution gradient; DMD online teacher regression, DMD2 GAN without regression | Numerical target tests, tiny real Wan DMD/DMD2 updates and CPU/Gloo restart |
| MeanFlow | Trainable two-time endpoint adapter and exact JVP teacher-guided target | Analytic JVP, stop-gradient, optimizer/RNG and exact restart |
| Joint AV | Native LTX2 video/audio streams, independent mean-reduced weighted losses | Both heads receive gradients; joint runtime payload and export/reload |

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
missing/corrupt latents are never replaced by synthetic zeros.

Build a cache with the matching local Diffusers pipeline:

```bash
python tools/cache_training_data.py --data_json data/raw.json \
  --model_path /models/teacher-pipeline --output_dir data/encoded-run \
  --device cuda --dtype bfloat16 --seed 42
```

The output directory must be new. Train with
`--data_json data/encoded-run/manifest.json --cache_dir data/encoded-run`.
The encoder retains every tensor condition, aligned controls, AV latents and
fps; records source/latent frame indices, per-sample seeds and model-config
hashes; and validates the resulting cached dataset before publishing its
manifest. Audio features must already match the selected checkpoint's
preprocessor and sampled-video timeline.

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
Teacher parameters can independently use `--runtime_teacher_offload cpu|nvme`. CPU mode uses pinned host weights; NVMe mode uses a private memory-mapped checkpoint under `--runtime_teacher_offload_dir`. This is filesystem paging plus H2D transfers, not GPUDirect Storage. Whole-teacher execution must fit one device. Stream events bound in-flight allocations and synchronize stage changes and checkpoint operations.

Muon requires a complete matrix gradient for its orthogonalization and currently
uses serial/DDP training. FSDP/DeepSpeed use AdamW. Native high/low-noise routing
keeps both experts in one student container and one optimizer lifecycle.

Explicit `--use_dual_model` uses the same unified high/low container for a
single-expert base checkpoint, including gradient accumulation, DDP, FSDP and
ZeRO. Both branches participate in EMA and checkpointing. The launcher preserves
this selection under sharding instead of disabling the low-noise branch.
Unified checkpoints include `dual_student_state.pt`; resume and export validate
its layout, native time scale and routing timetable. Legacy high-weight/low-sidecar
checkpoints can migrate, with low-side EMA initialized from saved weights because
that historical format only saved high-side EMA. An incompatible legacy optimizer
parameter layout requires a weights-only restart.

The bucket sampler groups actual tensor shapes and dtypes, all condition fields,
and batch-global values such as fps. Video length, packed-audio length and text
token count can differ across buckets, not within one batch. Explicit bucket
configurations are subdivided by the same exact signature; they do not silently
resize or pad tensors. Cached data headers are inspected without retaining full
tensor payloads in the sampler.

With Adversarial/DMD, FSDP and ZeRO partition the student while the fake-score
model and discriminator remain full replicas. Include those auxiliary weights,
gradients and optimizer states in the per-rank memory budget. Their optimizers,
FP16 scalers and accumulated update counts are saved separately. Progressive
LoRA stage handoff merges a full CPU snapshot without replacing live teacher
parameters; budget CPU memory for that snapshot and merge workspace.

`sp_size>1` is accepted only when both model adapters explicitly declare
`supports_worlddistill_sequence_parallel`. Generic Diffusers modules fail fast:
splitting latent frames without model-layer attention communication would be a
silent mathematical error, not valid sequence parallelism.

## Distributed checkpoints

Use the [multi-node guide](distributed-launch.md) for manual, Slurm or
containerized launch. `--mixed_precision auto` runs real per-device precision
probes, intersects their supported modes across ranks and applies method/scaler
constraints before model loading. Explicit unsupported precision/TF32 requests
fail. Small probe success is not full-model or optional-kernel certification.

DeepSpeed and FSDP initialization now precede resume. All ranks participate in
collective saves; DeepSpeed stores client step/epoch state and FSDP gathers model
plus optimizer state on rank zero. FP16 FSDP uses a world-group ShardedGradScaler; base EMA stores rank-local shards, including ZeRO-3 FP32 master shards. Consistency targets use matching FSDP shards or a CPU-backed ZeRO-3 target.

New checkpoints also record each rank's Python, NumPy, PyTorch and CUDA RNG
state plus its deterministic sampler epoch/cursor. The iterator is rebuilt and
advanced before RNG restoration, so the CPU smoke resumes at the exact next
batch. Exact restart requires the same dataset, batch size, sampler, and world
size; legacy checkpoints without this metadata fall back to weights/optimizer
resume with an explicit warning.

Multi-rank FSDP and DeepSpeed receive CPU-staged student parameters. The loader
does not first create a full FP32 student on every GPU. Teacher CPU/file-backed parameter offload is independent of student sharding and supervision caching.

Native noise-routed experts are collective leaves: FSDP must not split their
conditionally executed descendants, and ZeRO-3 registers the routed group with
DeepSpeed's leaf-module API. Otherwise ranks choosing different experts can
enter different all-gather sequences. A leaf gathers **all of its experts** as
one unit; budget that peak in addition to the teacher and optimizer state.

DeepSpeed mixed precision uses its supported Torch AMP path (0.19.6+), with the
engine owning backward/loss scaling. The outer AMP context also covers the
frozen teacher and consistency target. Native low-precision conversion is not
combined with outer AMP: newer DeepSpeed disables that outer context inside the
student engine, which would otherwise leave FP32 keyword latents incompatible
with converted weights. Base and consistency EMA shadows follow CPU-staged
students onto the execution device. CPU optimizer offload uses CPUAdam with
preserved optimizer groups and scheduler association, not a silent discarded
optimizer state. Every stage/precision/offload combination still needs its gate.

## Hardware qualification commands

Run the following on the allocated NVIDIA host, preserving its assigned GPU
visibility. These commands neither allocate remote resources nor download
pretrained models:

```bash
python tools/check_cuda_compat.py --probe-precision --json --output results/device-probe.json
python tools/check_attention_kernels.py
python tools/check_quant_kernels.py
python tools/check_runtime_streams.py --precision all --output results/runtime-streams.json
python tools/check_runtime_streams.py --precision all --profile results/runtime-trace.json
compute-sanitizer --tool memcheck python tools/check_quant_kernels.py
compute-sanitizer --tool memcheck python tools/check_runtime_streams.py --precision all
```

The stream check compares serial/DPP teacher outputs, student loss and gradients,
checks real streams/events and exercises input/output allocator lifetimes on a
separate consumer stream. Its trace is evidence for inspection, **not** a speedup
or overlap assertion. `all` means all locally probe-supported precisions, not
every theoretical dtype. Never count an unavailable/skipped backend as a pass.

Run an actual two-rank training/restart gate on a single GPU node:

```bash
torchrun --standalone --nproc-per-node=2 -m tools.check_distributed_training \
  --device cuda --parallel ddp --methods all --mixed-precision auto \
  --output-dir /shared/qualification/ddp-unique-run
```

For two nodes with four GPUs each, run once per node, changing `--node-rank`
to 1 on the second node. The output must be the **same new empty shared path**:

```bash
torchrun --nnodes=2 --nproc-per-node=4 --node-rank=0 \
  --rdzv-backend=static --master-addr=node0.example --master-port=29500 \
  -m tools.check_distributed_training --device cuda --parallel ddp \
  --methods all --mixed-precision auto --timeout 600 \
  --output-dir /shared/qualification/multinode-ddp-unique-run
```

Use the same launch topology and a **different empty output path** per case:

| Strategy | Additional/replacement checker arguments | Explicit coverage boundary |
|---|---|---|
| DDP | `--parallel ddp --methods all` | All registered methods; auxiliary losses have independent scaler and GAS handling |
| FSDP full | `--parallel fsdp --fsdp-strategy full --methods all` | FP32/BF16/FP16; rank-local EMA and collective scaler; auxiliary models replicated |
| FSDP hybrid | `--parallel fsdp --fsdp-strategy hybrid --methods all` | At least two agents with at least two workers each; auxiliary models replicated |
| ZeRO-1 / ZeRO-2 | `--parallel deepspeed --zero-stage 1 --methods all` | Run stages 1 and 2 separately, then separately with `--cpu-offload` |
| ZeRO-3 | `--parallel deepspeed --zero-stage 3 --methods all` | Rank-local EMA; auxiliary models replicated; also test CPU offload separately |
| Accumulation | `--gradient-accumulation-steps 2` with a supported method subset | Microbatch losses are normalized before each optimizer step |
| High/low students | `--methods step_distill --dual-student --gradient-accumulation-steps 2` | Both experts update and preserve routing/EMA; run each selected parallel strategy |

This gate uses small synthetic Conv3d denoisers and actual optimizers. It checks
collectives, rank replicas and continuous four-step training against a two-step
checkpoint plus resumed next two steps, including optimizer/RNG/cursor and
method sidecars. Progressive runs cross a stage boundary. Its strict update
comparison rejects a run that never updates after resume. JSON records commit,
dirty state, devices, ranks and exact checked scope. An unsupported requested
combination remains `partial` with nonzero exit. Each report records the actual
device and distributed world size.

These correctness checks complement pretrained-model quality, memory, bandwidth and scaling benchmarks. See
the real-model stability, export, quality and performance matrix in
[hardware expectations](hardware-expectations.md).

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
mean/std and multi-encoder prompt outputs are preserved. HunyuanVideo 1.5 builds its T2V zero mask/image conditions and I2V reference-frame VAE/vision conditions independently per batch item. LTX normalization reads the VAE's registered channel mean/std and scaling factor.

Training reads the checkpoint's time scale (`--num_train_timesteps 0` means auto). CogVideoX StepDistill uses its native VP cumulative alpha schedule and epsilon/v/sample prediction conversions. Trajectory methods require flow velocity from both teacher and student.

## Train → export → sample

```bash
python tools/export_student.py --base_model /models/teacher-diffusers \
  --checkpoint results/step --output_dir results/step/student_export --num_steps 4
python tools/sample_student.py --bundle results/step/student_export \
  --prompt "A paper boat crosses the rain." --save_path results/step/student.mp4
```

The bundle contains trained denoisers and checkpoint provenance. Sampling loads
them into the original Diffusers pipeline, reusing its licensed VAE/text assets.
It does not sample an unrelated catalog student. LoRA full-training checkpoints are reconstructed with their recorded adapter settings, loaded strictly, and safely merged before export. The base pipeline must support every supplied
control. This sampling bridge is single-device and separate from LightX2V's
optimized runner. The paper scripts reject `INFER_NUM_GPUS` other than one for
this student bridge. JSON conditions retain their frame count and resolution
unless explicitly overridden by command-line flags.

Bundles verify the complete denoiser configuration/weight inventory, including
distinct student architectures, declared compile wrappers, LoRA merges,
MeanFlow endpoint conditioning and high/low-noise experts. The `pipeline` backend
uses the original pipeline's scheduler and guidance. The `adapter` backend
accepts encoded conditions, replays the recorded flow timetable/shift or VP
DDIM schedule, and supports native/MeanFlow students. Its latent output can be
decoded by the base VAE or a trusted native decoder. Joint AV output preserves
both modalities. Select the backend and solver explicitly when comparing runs.

Export derives a default step count only from an unambiguous method/checkpoint
record (including the current trained progressive stage). Otherwise supply
`--num_steps`. The world-model paper script consequently requires
`STUDENT_NUM_STEPS` and `STUDENT_PIPELINE_CONDITIONS` for a genuinely compatible
pipeline. Native models can instead use encoded-condition adapter sampling;
WorldPlay requires the separate licensed author training runtime described in
[native model setup](native-world-models.md).

Every training run records local `metrics.jsonl`, `host_manifest.json`,
`config_snapshot.json` and `run_card.md`, even without W&B/TensorBoard. Use
`tools/summarize_run.py RUN_DIR --warmup 50 --measured 200` with `log_every=1`
to aggregate observed step timings. `profile_warmup_steps` and
`profile_active_steps` enable per-rank Chrome traces; tracing itself adds overhead.
Quality evaluation uses the generated-output workflow below.


## MeanFlow and joint audio-video recipes

Use `configs/distill_presets/mean_flow_distill.json` for teacher-guided MeanFlow.
The student receives both endpoints `timestep` and `timestep_r`; a new
zero-initialized time conditioner is trained with the original denoiser when
the checkpoint has no native two-time input. The target uses the exact
directional JVP along the frozen teacher velocity. This is a teacher-guided
variant, not a relabelled Step loss. LoRA keeps the time conditioner trainable.

Use `configs/distill_presets/joint_av_distill.json` for LTX2. Cached samples add
`audio_latent_path` with packed `L,D` audio and `conditioning_path` with the
checkpoint's audio/video text features. See
[`joint_av_cached.template.json`](../configs/data_templates/joint_av_cached.template.json).
Both modality losses are mean-reduced separately, then weighted by
`joint_av_video_weight` and `joint_av_audio_weight`. Video-only trainers reject
joint AV models. Raw audio input must already use the checkpoint's expected
log-mel preprocessing; arbitrary waveforms are not interpreted as latent tensors.

Native action models use `worlddistill_model.json`. Matrix-Game2 and LingBot-CAM
have built-in concrete architectures; HY-WorldPlay binds the separately
installed licensed author training class. See [native setup](native-world-models.md).
External custom factories additionally require `--allow_native_model_code`.
Explicit input maps, required conditions and strict weights prevent silently
dropping controls.

For example, a local model directory can contain `model.safetensors` and this
descriptor (replace the factory and argument names with those of the installed
author implementation):

```json
{
  "schema_version": 1,
  "factory": "my_world_model:build_model",
  "constructor_kwargs": {},
  "weights": "model.safetensors",
  "latent_layout": "BCTHW",
  "noise_process": "flow",
  "prediction_type": "velocity",
  "num_train_timesteps": 1000,
  "input_map": {"hidden_states": "x", "timestep": "t", "actions": "actions"},
  "required_conditions": ["actions"]
}
```

The factory returns the real `torch.nn.Module`; its forward must explicitly
accept each mapped input. VP descriptors additionally provide their complete
`alphas_cumprod` and prediction convention. Export and adapter sampling require
the same explicit native-code opt-in.

## Wan context parallelism

For tensor/pipeline partitioning and changed-DP-size checkpoint recovery, see
[model parallelism and elastic training](parallel-elastic-native.md).

Set `--sp_size 2` with a distributed Wan training job to split transformer tokens
and exchange attention heads through Ulysses. `world_size` and the number of
attention heads must be divisible by `sp_size`; the patched video token count
must also divide evenly. The built-in path uses native SDPA, zero dropout and
`ring_degree=1`. Other model families require their own model-layer plan.

Ranks in a CP group consume the same encoded sample and synchronized noise.
Different DP groups consume disjoint samples. Effective batch size is
`batch_size × gradient_accumulation_steps × (world_size / sp_size)`.
Student gradients use CP summation and DP averaging; differentiable rollout
inputs sum their CP contributions. Frozen teacher and EMA targets have their
own groups; full-replica auxiliary models retain ordinary data-parallel updates.

The regression suite compares real Wan forward outputs, chained rollout
gradients, activation checkpointing and optimizer updates at SP=1/2, and checks
DP×CP with four processes. MeanFlow requires `sp_size=1`: its exact JVP needs
higher-order attention derivatives, while this Ulysses backend has first-order
backward. Ulysses does not turn a bidirectional model into a causal model.

## Generated-output evaluation

`tools/evaluate_generated.py` writes a fresh result file and input hashes:

```bash
python tools/evaluate_generated.py --output results/fvd.json fvd \
  --reference_metadata data/reference-i3d.json --generated_metadata results/student-i3d.json
python tools/evaluate_generated.py --output results/vbench.json vbench \
  --videos results/videos --full_info /datasets/VBench_full_info.json \
  --result_dir results/vbench-detail --dimensions subject_consistency motion_smoothness
python tools/evaluate_generated.py --output results/drift.json trajectory \
  --tensors results/recovered-tracks.safetensors --coordinate_units meters
```

FVD reads caller-supplied I3D feature matrices in safetensors. Each JSON identifies
`features`, a `videos` list in row order, and an `extractor` object containing
`name: "i3d"`, `weights_sha256`, and explicit `preprocessing`. Reference and
generated extractor metadata must match. This follows the
[official FVD feature-distance definition](https://github.com/google-research/google-research/tree/master/frechet_video_distance);
it does not replace I3D with an untrained feature network.

VBench invokes the installed [official evaluator](https://github.com/Vchitect/VBench)
in custom-input mode, preserving dimension scores and prompt mappings. It uses
the evaluator's local-checkpoint selection by default; install its required
models separately. Trajectory ADE/FDE and action classification accuracy operate
on supplied recovered tracks/actions in the declared coordinate units.
ADE averages valid trajectory points; FDE averages each trajectory's last valid
point. The result records these aggregation conventions for mixed-length data.
