# Model parallelism, elastic recovery, and native training

## Tensor and pipeline ownership

Wan training accepts `--tp_size` and `--pp_size` with `--parallel_mode ddp`.
The rank layout is DP × PP × TP; `world_size` must divide into whole model
groups. All members of a model group consume the same sample and noise, while
different DP groups consume different samples. Effective batch size is
`batch_size × gradient_accumulation_steps × world_size / (tp_size × pp_size)`.

```bash
torchrun --standalone --nproc_per_node=8 -m training.train_distill \
  --teacher_model_path /models/wan-diffusers --data_json /data/cached.json \
  --output_dir /shared/runs/wan --tp_size 2 --pp_size 2 \
  --parallel_mode ddp --mixed_precision auto
```

TP partitions attention Q/K/V and FFN projections, including the global
normalization required by Wan's Q/K RMSNorm. PP assigns transformer blocks
and embedding/output modules to real stage owners; other stages do not keep
their parameters. Its current schedule executes each owner in order and
broadcasts activations. It does not overlap microbatches or implement 1F1B.
Small top-level parameters remain replicated and are counted once in clipping.

Teacher, student, and consistency target use separate collective groups.
Gradient accumulation, DP reduction, global norm clipping, activation
checkpointing, EMA, canonical AdamW moments, and full-model export preserve
their parameter ownership. Checkpoint recomputation runs inside the owner
stage, not around a distributed boundary. Standard LoRA and rsLoRA use
partition-aware factors; Progressive merges a full CPU snapshot against a
weight-free full architecture before updating the existing teacher objects.

Requirements: eager execution, AdamW, `sp_size=1`, first-order objectives.
MeanFlow's higher-order JVP, DoRA, nonzero LoRA dropout, and simultaneous
FSDP/ZeRO wrapping are rejected for this plan. Wan Ulysses remains a separate
context-parallel option. No flag automatically discovers a correct plan for an
arbitrary architecture.

Native adapters can declare `parallel_plan` in `worlddistill_model.json`:

```json
{
  "schema_version": 1,
  "tensor": {
    "modules": {"mlp.up": "column", "mlp.down": "row"},
    "divide_attributes": []
  },
  "pipeline": {
    "blocks": "blocks",
    "first_stage": ["embedding"],
    "last_stage": ["head"],
    "gradient_checkpointing_attribute": "gradient_checkpointing"
  }
}
```

These illustrative paths must be replaced with the actual author architecture.
Paths are relative to the adapter's model. Supported tensor layouts are
`column`, `row`, and `rms_norm`; head/width attributes are divided only when
explicitly listed. PP can alternatively declare `modules: {"path": stage}`.
Matrix-Game2 and LingBot-CAM include audited PP plans. Their action/camera
blocks stay intact on their owner; native TP requires an explicit valid plan.

## Elastic data-parallel restart

`--elastic_resume` saves a global consumed-batch cursor and portable training
state. `--elastic_auto_resume` selects the latest completely published
checkpoint from the shared output directory on torchrun restart. Checkpoints
become eligible only after all method-specific sidecars have been written.
Incomplete writes do not become automatic resume candidates.

Run one agent on each node, with the same visible worker count per node:

```bash
bash scripts/run_train.sh --teacher_model /models/teacher-diffusers \
  --data_json /data/cached.json --output_dir /shared/runs/elastic \
  --nnodes 1:2 --nproc_per_node 4 --rdzv_backend c10d \
  --rdzv_endpoint rendezvous.example:29500 --rdzv_id unique-elastic-job \
  --max_restarts 5 --elastic_resume --elastic_auto_resume --parallel fsdp
```

The launcher forwards both elastic flags and validates the worker range.
DP degree may change; SP/TP/PP degrees must remain fixed. Every permitted
membership must contain whole model groups. An explicit initial `resume_from`
wins on the first launch; after a restart, automatic recovery prefers the
latest complete checkpoint. A restarted job without a complete checkpoint or
an explicit resume path fails instead of silently starting over.

The sampler commits only consumed batches, not prefetched ones. After a
membership change it reassigns the unconsumed global suffix. Final-group
padding repeats only from that suffix. Dataset/bucket identity, seed, and
batch size are checked. New workers receive deterministic generation-specific
RNG streams; changed global batch size changes the optimization trajectory,
so elastic resizing is not bitwise equivalent to a fixed-size run.

- DDP loads complete model/Adam state on the new replicas.
- FSDP uses canonical full model/optimizer state and PyTorch's state-dict
  resharding APIs; EMA and consistency targets have portable full states.
- TP/PP keeps canonical full weights, Adam moments, and EMA, then restores the
  same model plan on the new number of DP replicas.
- DeepSpeed uses its official Universal Checkpoint converter and loader,
  plus separate optimizer-group options/steps, frozen tensors, persistent
  buffers, and AMP state. Use DeepSpeed ≥0.19.6 with initialized Adam state.
  Optimizer NVMe swapping is not combined with Universal resizing; frozen
  teacher file-backed storage is a separate supported mechanism.

ZeRO-1/2 loading includes a scoped compatibility implementation for the official
Universal tensor format, covering padding-only shards and master/moment view
ownership. The regression invokes real CPU DeepSpeed engines for each ZeRO
stage and checks 2→1→2 continuation with unequal optimizer-group settings.

Checkpoint serialization and conversion require shared storage and sufficient
host memory/disk for full states. TP/PP uses pre-created Gloo checkpoint groups
so full CPU snapshots are not staged through NCCL's GPU object buffers. The
Gloo host network must therefore also be reachable between participating ranks.
Unequal workers per node and changing model
parallel degrees are not supported. Fixed-size launch remains the default.

## Concrete native models

The built-in native loader uses `architecture` rather than an arbitrary
installed factory. Architecture config, checkpoint prefix/index, source
revision and strict weight inventory are retained through export.

| Architecture | Native training inputs | Built-in parallel plan |
|---|---|---|
| Matrix-Game2 causal/foundation | Encoded image condition, CLIP image features, pixel-rate keyboard and mouse actions | PP |
| LingBot-CAM | T5 text features, camera features or latent-aligned C2W poses with pixel intrinsics, checkpoint-specific image condition | PP |
| Diffusers Wan VACE | Encoded source/mask/reference `vace_context` and optional layer scales | Single-device/DDP/FSDP/ZeRO |
| Diffusers Wan Animate | Reference-prefixed latents/image condition, pose latents and normalized face pixels | Single-device/DDP/FSDP/ZeRO |
| Diffusers SkyReels-V2 | Scalar or framewise timesteps, text/image features and checkpoint-dependent physical FPS | Single-device/DDP/FSDP/ZeRO |

VACE/Animate/SkyReels use their exact Diffusers classes. A causal timestep
interface is not by itself a causal attention mask. Pre-encoded conditioning
must follow each checkpoint's native protocol; these adapters do not invent
pose/face extractors or convert arbitrary actions into native controls.
See [model support](model-support.md) for licensed external models and
[training](training.md) for data, sampling and objective contracts.

Sources: [PyTorch distributed state dictionaries](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html),
[torchrun elastic launch](https://docs.pytorch.org/docs/stable/elastic/run.html),
[DeepSpeed Universal Checkpointing](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/universal-checkpointing.md).
