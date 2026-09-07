# Fixed-size multi-node launch

WorldDistill launches one `torchrun` agent per node, with the same integer
worker count on every node. `--gpus` remains an alias for
`--nproc_per_node`; it is **not** the total GPU count across the job.
The launcher never rewrites `CUDA_VISIBLE_DEVICES`, GPU UUIDs, or MIG masks.
`LOCAL_RANK` indexes the scheduler-visible device set.

This implements launch, initialization and node-aware grouping contracts.
CPU/Gloo tests exercise local multi-agent rendezvous, collective communication
and tiny-model DDP training. They do not certify multiple physical machines,
NCCL, GPU FSDP/ZeRO, InfiniBand/RoCE, or any A100/H20/B200/B300 topology.
All participating nodes need compatible software, their licensed model assets,
the training data, and a consistent checkpoint path. No SSH launcher or remote
command execution is provided.

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
the larger enclosing FSDP unit), activations and communication buffers: it is
not a free memory optimization or evidence that a particular checkpoint fits.
CPU policy/mock tests do not qualify routed FSDP or ZeRO-3 on GPU; different
per-rank routing still needs real multi-GPU forward/backward verification.

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
- `--max_restarts N` enables fixed-size torchrun retries, default zero.
  `--resume_from /shared/run/checkpoint-K` explicitly chooses the checkpoint
  used on each invocation. Retrying does **not** automatically discover the
  newest checkpoint; without a resume path the command starts from its initial
  state. Do not describe retries as automatic latest-checkpoint recovery.
- `--nnodes MIN:MAX`, heterogeneous workers per node, and world-size changes
  are rejected. The launcher exports expected global/local worker counts and
  every restarted worker checks them. No elastic membership or resharding claim
  is made.

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

The direct API cannot inspect the parent torchrun command's node-range flags.
Using an elastic parent launcher is unsupported; a restarted worker without
the expected-world-size declaration fails instead of assuming unchanged size.
DDP, FSDP and ZeRO checkpoint compatibility also depends on the training
checkpoint contract, not just successful rendezvous.

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
Stock training adapters do not provide generic SP, tensor parallelism (TP),
or pipeline parallelism (PP). TP/PP are not launch modes; setting a larger
world size does not partition model layers or tensors automatically. A custom
SP adapter must implement the full model-side communication contract before
`--sp_size > 1` can be used; a dry run does not validate that implementation.

## Upstream contracts

The launch shape and fixed-size restart policy follow the official
[torchrun documentation](https://docs.pytorch.org/docs/stable/elastic/run.html).
Hybrid group ordering follows the official
[FSDP process-group contract](https://docs.pytorch.org/docs/stable/fsdp.html).
Timeout handling follows the official
[distributed initialization and collective documentation](https://docs.pytorch.org/docs/stable/distributed.html).
