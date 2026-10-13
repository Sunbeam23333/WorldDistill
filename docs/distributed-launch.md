# Multi-node launch and elastic recovery

WorldDistill launches one `torchrun` agent per node, with the same integer
worker count on every node. `--gpus` remains an alias for
`--nproc_per_node`; it is **not** the total GPU count across the job.
The launcher never rewrites `CUDA_VISIBLE_DEVICES`, GPU UUIDs, or MIG masks.
`LOCAL_RANK` indexes the scheduler-visible device set.

All participating nodes need compatible software, their licensed model assets,
the training data, and a consistent checkpoint path. Elastic recovery requires
shared checkpoint storage accessible under the same path on every node.
No SSH launcher or remote command execution is provided. Fixed-size execution
is the default; data-parallel membership changes require explicit opt-in.

### Dynamic expert collective boundaries

`NoiseRoutedDenoiser` marks its high/low-noise expert group as a collective
leaf. Different ranks may select different experts, but FSDP auto-wrapping
must not split that group into independently gathered child blocks. Both
default and custom wrapping policies retain its entire subtree inside the
nearest ancestor FSDP unit. PyTorch mixed-precision BatchNorm override
wrappers inside such a leaf are explicitly rejected; use FP32 or a model
without those overridden module types.

Before ZeRO-3 initialization, WorldDistill uses DeepSpeed's official
`set_z3_leaf_modules` API for each marked instance. Missing or ineffective
registration fails early. ZeRO-1/2 and DDP are unchanged. This follows
[DeepSpeed's dynamic expert leaf guidance](https://deepspeed.readthedocs.io/en/latest/training.html#configuring-zero-leaf-modules).

This guard gathers **all experts in each routed leaf**, including inactive
ones, on every rank. Peak memory includes that complete group (and potentially
the larger enclosing FSDP unit), activations and communication buffers.
Include that complete group when sizing the job's memory budget.

## Manual launch: run once on each node

On node 0:

```bash
bash scripts/run_train.sh --teacher_model /models/teacher-diffusers \
  --data_json /data/train.json --output_dir /shared/runs/example \
  --nnodes 2 --node_rank 0 --nproc_per_node 4 \
  --rdzv_backend static --rdzv_endpoint node0.example:29500 \
  --rdzv_id unique-job-2026-09-07 --rdzv_timeout 600 --dist_timeout 600 \
  --dist_backend nccl --mixed_precision auto --parallel ddp
```

Run the identical command on node 1, changing only `--node_rank 1` (and any
node-local path that intentionally points at identical assets). The
rendezvous endpoint must be reachable from every node. Use a unique job ID
and an unoccupied port; do not share endpoints between unrelated jobs.
`--rdzv_backend c10d` is also supported for a fixed node count. Its current
agent ranks may be reassigned after a restart; rank groups are reconstructed
from current agent/local-rank metadata, not saved host-rank assumptions.
For c10d on hosts whose default hostname is not reachable/resolvable from peers,
set `--local_addr THIS_NODE_REACHABLE_IP` separately on each node. This controls
the node address advertised for bootstrap; setting the shared rendezvous
endpoint alone does not change an unreachable advertised hostname. Do not use
loopback addresses for a real cross-machine job. Required data-plane ports and
interfaces must also be reachable; a successful rendezvous is not a fabric test.

Single-node launch without an endpoint uses `torchrun --standalone`, which
allocates an isolated rendezvous port. Add `--dry_run` to print the command
and validate launch arguments without importing PyTorch/models, running the
dependency checker, creating output directories, or opening network sockets.
It cannot establish that a remote endpoint, GPU, model, or data file works.

Both preflight and training honor `--required_transformers_version VERSION`.
This does not assert that an arbitrary version is model-compatible; it makes
the explicit version requirement consistent. `--mixed_precision auto` is
resolved by training against the participating devices; an explicit precision
must be supported by all workers.

## Slurm: one agent task per node

The allocation must expose all of that node's assigned GPUs to its one agent.
Do not start one Slurm task per GPU and then ask each task to spawn another
full node of workers.

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

set -euo pipefail
export RDZV_ENDPOINT="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):29500"
export RDZV_ID="$SLURM_JOB_ID"
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --gpu-bind=none \
  bash scripts/run_train.sh --launcher slurm --nproc_per_node 4 \
  --teacher_model /models/teacher-diffusers --data_json /data/train.json \
  --output_dir /shared/runs/example --dist_backend nccl
```

The adapter derives the node count and node index from `SLURM_NNODES` and
`SLURM_PROCID`, requires `SLURM_LOCALID=0` and exactly one task per node, and
uses `SLURM_JOB_ID` as the default rendezvous ID. The endpoint remains explicit
(argument or `RDZV_ENDPOINT`); the script does not guess a network interface.
The exact partition, GPU resource syntax and binding settings remain
cluster-specific. Preserve the allocation's GPU mask.

## Kubernetes / other schedulers

Use one agent Pod per participating node and homogeneous GPU requests/limits
(for example four `nvidia.com/gpu` devices per Pod). Place agents on distinct
physical nodes when a node-sharded topology is intended. Supply stable
headless-service DNS for the rendezvous host and assign each agent a unique
`NODE_RANK` in `[0, NNODES)`, for example from an indexed workload controller.
Run the manual command once inside each Pod with the matching node rank.
Keep container-injected device visibility; do not replace UUID-based masks
with host ordinals. A GPU count alone does not establish peer access, fabric
connectivity, or the legality of a particular MIG/NCCL configuration.

## Backend, timeouts and restarts

- `auto` selects NCCL when CUDA is available, otherwise Gloo. NCCL requires a
  NVIDIA CUDA build (`torch.version.cuda`), NCCL, and enough visible NVIDIA GPUs.
  ROCm/HIP's `torch.cuda` and RCCL's `nccl` compatibility names are not accepted
  as NVIDIA support. CPU/Gloo remains available when GPUs are explicitly hidden.
  See the official [HIP/CUDA identification guidance](https://docs.pytorch.org/docs/stable/notes/hip.html#checking-for-hip).
- Gloo is the CPU validation path. To use it on a GPU host, explicitly set
  `CUDA_VISIBLE_DEVICES=''` before launch; the framework does not hide devices
  on the user's behalf.
- `--rdzv_timeout` bounds rendezvous joining/store operations. `--dist_timeout`
  bounds process-group initialization and collectives. A process that cannot
  join or a failed collective exits rather than falling back to another backend.
- `--max_restarts N` enables torchrun retries, default zero.
  `--resume_from /shared/run/checkpoint-K` explicitly chooses the checkpoint
  used on each invocation. Automatic latest-checkpoint selection additionally
  requires `--elastic_resume --elastic_auto_resume`, described below.
- Fixed-size launch exports expected global/local worker counts and every
  restarted worker checks them. Elastic launch replaces the fixed global count
  with an allowed node range. Unequal worker counts across nodes are rejected.

Direct `torchrun -m training.train_distill` users can retain that interface.
For restart safety, declare the fixed worker counts explicitly, for example:

```bash
WORLD_DISTILL_EXPECTED_WORLD_SIZE=8 \
WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE=4 \
WORLD_DISTILL_DIST_BACKEND=nccl WORLD_DISTILL_DIST_TIMEOUT=600 \
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  --rdzv_backend=static --rdzv_endpoint=node0.example:29500 \
  --rdzv_id=example --max_restarts=0 -m training.train_distill \
  --teacher_model_path /models/teacher-diffusers --data_json /data/train.json
```

Without explicit elastic opt-in, a restarted worker without the expected
world-size declaration fails instead of assuming unchanged size.

## Elastic DP membership: explicit opt-in

Run one manual agent per node with the same command and shared rendezvous ID:

```bash
bash scripts/run_train.sh --teacher_model /models/teacher-diffusers \
  --data_json /data/train.json --output_dir /shared/runs/elastic \
  --nnodes 1:2 --nproc_per_node 4 --rdzv_backend c10d \
  --rdzv_endpoint rendezvous.example:29500 --rdzv_id unique-elastic-job \
  --max_restarts 5 --elastic_resume --elastic_auto_resume --parallel fsdp
```

`--elastic_resume` enables portable training checkpoints and the global data
cursor. It must already be enabled when the source checkpoint is saved.
`--elastic_auto_resume` selects the latest complete checkpoint on restart;
an explicit `--resume_from` takes precedence on the first launch. A restarted
job with neither a complete checkpoint nor an explicit resume path fails
instead of silently beginning a new run. A checkpoint is published only after
model, optimizer and all method-specific sidecars have finished writing;
partial checkpoint directories are not automatic recovery candidates.

The sampler records consumed global batches, not DataLoader-prefetched batches.
On resize, it distributes the unconsumed suffix across the new DP replicas.
The last incomplete replica group pads only from that remaining suffix.
Dataset manifest/bucket identity, seed, batch size and sampler settings must
match. Resizing creates deterministic generation-specific RNG streams and
changes the effective global batch size; it does not promise the bitwise
trajectory of an unchanged-size run. Same-topology recovery retains per-rank
RNG and the saved optimizer/EMA state.

Portable state paths are backend-specific:

| Backend | Resized restore |
|---|---|
| DDP | Full model, Adam state, scheduler, scaler and EMA on each new replica |
| FSDP | PyTorch canonical full model/optimizer state reshards onto new workers; portable EMA and consistency target states |
| TP/PP with DDP | Canonical model, Adam and EMA state restores the same model plan with a new DP degree |
| DeepSpeed ZeRO-1/2/3 | Official Universal Checkpoint conversion/loading, plus preserved optimizer-group options/steps, frozen weights, persistent buffers and AMP/EMA state |

DeepSpeed Universal resizing uses DeepSpeed ≥0.19.6 and initialized Adam state
at an optimizer-step boundary. CPU offload is allowed; rank-local ZeRO
optimizer/parameter NVMe swapping is rejected for this path. This restriction
does not apply to frozen teacher NVMe storage. Full-state serialization and
Universal conversion require sufficient host memory and shared disk capacity.
The ZeRO-1/2 load compatibility layer handles padding-only shards and restores
master/moment inspection views while retaining the official Universal format,
engine metadata, scheduler restoration and low-precision synchronization.

Only the DP degree may change: `--sp_size`, `--tp_size` and `--pp_size` must
remain fixed, and every allowed membership must contain whole model groups.
For a variable node range the launcher therefore requires workers per node to
be divisible by the product of these degrees. `MIN:MAX` requires c10d and
manual agents; the Slurm adapter remains fixed-allocation. Hybrid FSDP's range
must keep at least two nodes. See [model-parallel ownership and native plans](parallel-elastic-native.md).

For a direct `torchrun` invocation, pass the same training flags and declare
the node-range bounds explicitly (the child cannot read its parent's flags):

```bash
WORLD_DISTILL_MIN_NODES=1 WORLD_DISTILL_MAX_NODES=2 \
WORLD_DISTILL_EXPECTED_LOCAL_WORLD_SIZE=4 \
torchrun --nnodes=1:2 --nproc_per_node=4 --rdzv_backend=c10d \
  --rdzv_endpoint=rendezvous.example:29500 --rdzv_id=unique-elastic-job \
  --max_restarts=5 -m training.train_distill \
  --teacher_model_path /models/teacher-diffusers --data_json /data/train.json \
  --output_dir /shared/runs/elastic --elastic_resume --elastic_auto_resume
```

Do not retain `WORLD_DISTILL_EXPECTED_WORLD_SIZE` from a fixed-size job when
enabling elastic membership. The entry point enables the elastic environment
contract only when `--elastic_resume` is explicit.

## Hybrid FSDP is explicitly node-aware

`--parallel fsdp --fsdp_strategy hybrid` requires at least two nodes and two
workers per node. It constructs two sets of process groups from the validated
current worker topology: each agent's host-local workers form a shard group;
the same local worker index across nodes forms a replica group. The pair is
passed directly to FSDP as `(shard_group, replica_group)`. Every rank creates
the groups in the same order. A single node, incomplete local group, conflicting
host metadata, heterogeneous local sizes, unavailable FSDP, or CPU-only FSDP
request fails clearly; it does not quietly turn into full sharding or DDP.

This is post-load wrapping. It is not distributed/meta checkpoint loading,
teacher sharding, or proof that a large model fits during initialization.
Sequence parallelism still requires a model-specific adapter; increasing
node count does not implement missing sequence-parallel model collectives.
Wan has a concrete Ulysses context-parallel adapter and separate tensor/
pipeline partition plans. Native architectures require a declared valid plan;
setting a larger world size alone does not partition layers or tensors.
TP/PP use `--tp_size` / `--pp_size` with DDP; they are not interchangeable
with FSDP/ZeRO wrapping or with `--sp_size`. See the
[supported plans and combination constraints](parallel-elastic-native.md).

## Reproducible recovery checks

```bash
python -m pytest training/tests/test_elastic_resume.py training/tests/test_elastic_entry.py -q
```

The recovery suite launches real Gloo process groups through a 2 → 1 → 2
sequence and checks model weights, optimizer moments, EMA/target/auxiliary
state, data cursor and continued training. It also launches a real c10d agent
with a `1:2` range. FSDP's CPU-device cases exercise checkpoint resharding;
normal training retains its CUDA placement requirements.

The optional DeepSpeed cases instantiate actual ZeRO engines and run the
official Universal converter. A CI lane can install DeepSpeed ≥0.19.6 in its
own environment and run them with the official CPU accelerator; no CUDA
extension build is needed for these CPU/AdamW checkpoint tests:

```bash
DS_BUILD_OPS=0 python -m pip install 'deepspeed==0.19.6'
DS_ACCELERATOR=cpu python -m pytest training/tests/test_elastic_resume.py \
  -k real_cpu_zero -q
```

When DeepSpeed is absent those three optional cases are skipped; the ordinary
DDP/FSDP recovery suite still runs. GPU collective/kernel tests remain separate
from the restart protocol tests.

## Upstream contracts

The launch shape and membership/restart policy follow the official
[torchrun documentation](https://docs.pytorch.org/docs/stable/elastic/run.html).
Hybrid group ordering follows the official
[FSDP process-group contract](https://docs.pytorch.org/docs/stable/fsdp.html).
Timeout handling follows the official
[distributed initialization and collective documentation](https://docs.pytorch.org/docs/stable/distributed.html).
Portable sharded state follows [PyTorch distributed checkpoint APIs](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
and [DeepSpeed Universal Checkpointing](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/universal-checkpointing.md).
