# Hardware expectations and real-device acceptance matrix

Review date: **2026-09-07**. This is an engineering acceptance plan and a
primary-source reference sheet, **not a WorldDistill benchmark result**.
Implementation descriptions follow the current source and
[operator audit](operator-training-audit.md), [training contracts](training.md),
[model support](model-support.md), [fixed-size distributed launch](distributed-launch.md),
and [CUDA policy](cuda-compatibility.md).
Those documents remain authoritative if implementation changes after this review.

中文结论：代码可构造、CPU 测试通过、某个算子在 GPU 上运行、真实预训练模型训练成功、
达到论文质量，是五件不同的事。目前没有 A100/H20/B200/B300 的完整真机证据链。
下面的数字只有三种身份：**已公开的外部结果、明确假设下的理论估算、待执行的验收门槛**；
没有一项被冒充为本项目实测或交付速度承诺。未实现的功能先补实现，再运行验收，
不能通过跳过测试、静默回退或更换模型来填绿表格。

## 1. How to read and fill the tables

| Label | Meaning |
|---|---|
| C | Checked-in implementation with CPU contract/tiny-model evidence; not full-model GPU qualification |
| P | Partial/research integration; the remaining boundary is stated explicitly |
| U | Not implemented for the stated end-to-end workflow; a name, interface, or inference runner is insufficient |
| pending | No WorldDistill real-device result for this exact cell |
| unsupported | Deliberately rejected combination; this is not a failed performance experiment |
| NR / N/A | Not reported by the cited external source / not applicable; never silently inferred |

Every cell is a tuple, not a GPU name alone:
`commit × model revision × task × method × dtype × kernel build × shape × topology`.
An A100 single-GPU forward pass does not qualify A100 multi-node restart, and
an H100 result does not qualify H20 merely because both are Hopper.
All WorldDistill measured fields below start as **pending**, including features
with CPU evidence. External records in Section 6 never fill these fields.

## 2. Seven trainer acceptance matrix

Shared minimum gate: real checkpoint and real data/conditions; finite loss and
gradients; intended student parameters change while frozen teacher parameters
do not; no dropped conditions; optimizer/scheduler state is restored; one
deterministic uninterrupted-versus-resumed next update agrees within Gate R.
Run at least 20 optimizer updates for an executable smoke, then the longer
stability/quality protocol below. This small smoke is not convergence evidence.

| Method | Current implementation level and boundary | Real-device acceptance target | Memory/time observation; external reference | Still to measure |
|---|---|---|---|---|
| Step | C: fixed timetable; native Wan2.2 high/low routing. P: pretrained few-step quality and scheduler parity | Exercise both experts and both sides of the boundary; inspect selected timestep/teacher; compare supervised loss to unfused reference; exported count matches recorded timetable | Teacher + student residency, expert activation peaks, teacher/student time separately; E1 | Per-GPU loss/gradient error, expert updates, peak VRAM, step ms, 1/2/4/8-step quality |
| Stream | C: differentiable per-frame Euler unroll and generated overlap. P: requires a genuinely causal/per-frame adapter; generic bidirectional Wan is not one | Perturb generated history and verify next-window response; compare every inner Euler step to reference; non-divisible window tail, bounded history, long rollout; verify causal no-future-leakage | Record unroll count, retained graph length and KV/latent memory separately; E3/E4 are different trained architectures | TTFF, sustained generated FPS, time per chunk, drift/failure over 5/20/60 s |
| Progressive | C: exact halving stages, final interval to zero, stage state/restart. P: FSDP/ZeRO teacher handoff and optimizer rebuilding not integrated | Train through at least one stage transition; both teacher handoff and optimizer policy survive restart; export current trained stage, not the final configured target; new-stage boundary with no updates requires explicit count | Stage-local loss and time; count reduction is not an E2E speed guarantee | Per-stage quality, transition/restart error, memory peak during teacher replacement |
| Consistency | C: EMA target, MSE/Huber. P: model-specific consistency/preconditioning and pretrained-quality reproduction | Frozen target has no gradient; EMA follows its specified recurrence; consistency boundary/timestep and Huber paths match reference; EMA checkpoint preserved | Include EMA storage; sharded base EMA currently restricted; no borrowed image FID | EMA numerical error, 1/few-step video quality, convergence at fixed compute |
| Context Forcing | C: full-prefix memory selection, aligned action/camera slicing and target-only mask. P: native world-model adapter and context-teacher training | Packed indices and every condition refer to the same original frames; context-only locations have zero direct supervised gradient; evaluate fixed-size history with real generated context | Current Hybrid-Sparse selects latent/context frames, not the Context Forcing paper's key-similarity KV consolidation; E5 | Teacher robustness, action response, long-horizon consistency and reset rate |
| Adversarial | C/P: latent projection discriminator + distillation; serial/DDP, GAS=1. Not separately certified ADD/LADD recipes; sharded auxiliary state incomplete | Discriminator and generator alternate correctly; real and fake inputs differ; generator gradient reaches student; all auxiliary parameters/optimizer/RNG restore | Report generator/discriminator updates separately; a lower GAN loss alone is not quality improvement | GAN stability, collapsed/static clip rate, video quality and resume parity |
| DMD / DMD2 (one registry method) | C: fake-score DSM, real-minus-fake distribution gradient; DMD online regression; DMD2 GAN without substituted regression. P: serial/DDP; FSDP/ZeRO auxiliary models and pretrained reproduction incomplete | Check fake score is trained on current generated samples, gradient sign/normalization against reference, critic-to-generator update ratio, teacher freeze, DMD2 no unwanted paired-regression term | Teacher + student + trainable fake score + optional discriminator/EMA; not ordinary two-model memory. E2/E4/E6 | Per-objective gradients, update counts, critic stability, checkpoint state, few-step video quality |

Evidence entry points: [trainer registry](../training/trainers/__init__.py),
[algorithm regressions](../training/tests/test_algorithm_regressions.py),
[distributed algorithm tests](../training/tests/test_distributed_algorithms.py),
[training entry integration](../training/tests/test_training_entry_integration.py),
[training-loop smoke](../training/tests/test_training_loop_smoke.py).
An unsupported distributed method must fail before a long training job starts;
successful DDP of one method is not evidence for the other six.

## 3. Runtime, operator, model and lifecycle acceptance matrix

Gate names refer to Section 4. Memory formula names refer to Section 5.
The final column names required **unfilled** observations, not anticipated wins.

| Function | Current implementation / explicit limitation | Real-device gate and expected invariant | Pending record |
|---|---|---|---|
| Teacher-output/context cache | C: memory/disk/hybrid payload cache, content/condition/revision keys, freshness checks. Reuse rate depends on exact workload | Changed teacher, latent, noise/timestep or condition must miss; valid same-key reuse equals uncached output; stale prefetch never becomes fresh on promotion | Hit/miss/stale rate, teacher calls avoided, key/hash cost, CPU RSS, disk bytes, transfer time |
| Cold-cache prefetch | C: bounded CPU worker; not a teacher forward worker or GPU parameter offloader | Fault/cancel/close paths do not hang; bounded outstanding requests; prefetched and synchronous payloads agree | Read/promotion latency, overlap, queue occupancy, pinned-memory high-water mark |
| Hybrid-Sparse Memory | C: recent latent tail + chronological anchors from complete older prefix. P: no learned importance policy or per-layer persistent KV implementation | Count stays within budget; no future/duplicate frames; action/camera order exact; compare dense-recent, strided and hybrid at equal budget | Selected indices, GPU/CPU memory separately, quality vs budget, drift and motion metrics; F2 |
| DPP / teacher CUDA stream | C: separate stream, events, allocator lifetime recording; [real runtime checker](../tools/check_runtime_streams.py) exercises actual teacher/student APIs, same/cross-consumer streams and released-temporary stress. P: GPU correctness, true overlap and benefit unmeasured | Gate R against serial output, loss and parameter gradients on identical inputs; real stream/event/counter checks; no race/use-after-free under sanitizer; trace shows actual concurrent work where dependencies permit | Teacher/student timelines, overlap fraction, tail stalls, peak VRAM; F3. CPU runs are unavailable, not a GPU pass |
| Fused supervision | C: Triton masked MSE forward/backward on known NVIDIA SM80+ targets; actual tensor-device guard, FP32 intermediates, unmasked fast path. Older/unknown targets and Huber stay PyTorch | Gate K-loss including FP16 loss scaling, FP32/BF16, fractional/zero/broadcast masks, non-tile tails; reference fallback does not qualify Triton | Max loss/gradient error, dtype, real selected kernel, kernel time, allocations |
| Dense SDPA / FA2 / FA3 / optional FA4 | C: shared mask/GQA/packed/scale/causal contract and dispatch. P: exact build/device forward AND backward certification | Gate K-attn; test cross-attention, unequal lengths, packed batches, empty/all-masked edges, family head dimensions; explicit FA4 requires probe | Backend actually used, max/relative errors, backward availability, time by sequence length; E7 |
| SageAttention / quantized attention | C/P: optional inference integrations and guards; inference availability is not autograd support | Forward-only qualification stays inference-only; Gate K-attn + full-model quality. Do not route training through a detached/inference-only kernel | Per-layer/end-to-end error, outlier cases, quantization overhead, quality; E8 |
| INT8 GEMM | C: local Triton quantization + tail-safe GEMM; optional extensions separately guarded. Not INT8 training | Gate K-gemm vs dequantized reference; all M/N/K tails, noncontiguous scales, zero rows, bias/GELU; sanitizer clean | Kernel error, original-BF16 quantization error separately, layer coverage and E2E throughput |
| FP8 GEMM | C/P: same local GEMM contracts; eligible architectures and available callable required; no native A100 FP8 claim | Gate K-gemm; FP32 scales remain finite on zero/tiny inputs; calibrate outliers against original BF16 model | Format/granularity/scale metadata, clipping rate, error and quality, actual FP8 instruction path |
| NVFP4 / MXFP4 / MXFP6-MXFP8 / MXFP8 | P: vendored extension paths have architecture-specific guards. U: those SM120-specific paths are not B200/B300 native implementations | Add correct target kernel/layout/scales first; then dequantized-reference + original-model quality + sanitizer gates. BF16 fallback must not be labelled FP4 | Extension revision, SASS target, actual packed format, padding/scales bytes, quality; external path E12 is not integrated |
| Q8F / DeepGEMM / vLLM / SGL / TorchAO / Marlin | P: imported-callable and architectural prerequisites, not qualification of each wheel | Independent per-backend matrix; missing symbol/unsupported device fails before launch. Current Q8F SM89; current DeepGEMM entry SM90/100; Marlin INT4 is not FP4 | Exact binary/package revision, flags, numerical coverage, memory and sanitizer logs |
| torch.compile | C: scopes/fallback controls and strict compiled-checkpoint export. CPU eager-compiler tests are not CUDA Inductor tests | Gate R eager vs compiled forward/update; dynamic frame shapes and both experts; preserve gradients/state, record graph breaks and recompilations | Cold compile time, warm time, graph-break count, extra memory; no speedup promised |
| Activation checkpointing | C/P: supported model hooks; must validate every model/backend combination | Gate R with checkpointing off/on; RNG/dropout and gradients agree; no nested checkpoint/compile failure | Peak activation memory, recomputation time, useful batch/sequence increase |
| AdamW / Muon / gradient accumulation | C: optimizer contracts. Muon currently serial/DDP only; adversarial/DMD constraints remain method-specific | Global effective batch and update frequency identical; FP16 scaling/overflow handled; no accidental extra update on partial accumulation | Per-group update norm, overflow/skipped steps, GAS and exact batch IDs |
| Automatic training precision | C: real local GEMM/math-attention probes, all-rank intersection and method/scaler restrictions. Explicit unsupported dtype/TF32 requests fail; CPU controls remain CPU-only | Mixed-card workers select one common dtype; if BF16 is unavailable and FSDP/auxiliary trainers lack an FP16 scaler, auto selects FP32 (`no`), not unsafe FP16 | Requested/selected precision, all-rank probe records, controls, actual full-model numerics; probe success is not optional-kernel or model qualification |
| Fixed-size multi-node launcher | C: manual/Slurm, static/c10d, one agent/node, homogeneous worker counts, visibility preservation, group checks and timeouts. P: real GPU cluster validation | Reject uneven/elastic memberships and invalid visible-device mappings; reconstruct groups from current rank metadata; explicit resume checkpoint on fixed-size retries | Logical CPU multi-agent controls separately from distinct physical hosts, node/rank counts, NCCL/fabric and recovery logs |
| DDP | C: wrapper/collective paths and actual two-rank CPU/Gloo evidence. P: multi-node NCCL qualification | Gate R at fixed effective batch; same final gradient/parameter on all ranks; unused native experts do not hang; accumulation uses correct synchronization | All-reduce time, worst-rank step time, scaling curve, failure propagation |
| FSDP | C/P: CPU-staged student, full/node-aware hybrid grouping, model/optimizer collective state. Generic teacher remains replicated; FP16 scaler/base EMA and some algorithms explicitly restricted | Supported-method BF16 or explicitly selected FP32 Gate R at 2 then 8 ranks; measure initialization/full-state-save peaks; shard optimizer correctly; ensure every required rank participates | F1 model-state bytes, gather peak, save/load time, next-update error; E10 |
| DeepSpeed ZeRO-1/2/3 | C/P: exclusive engine Torch AMP with version/API admission, external teacher/EMA forward autocast, per-rank FP16 scaler checkpoint; replicated ZeRO-1/2 EMA follows CPU-staged student placement. Not blanket support for auxiliary models, Muon or dual legacy students | Independently test each stage/offload configuration; preserve RNG/cursor, both native and Torch AMP scaler state, all optimizer partitions and root method sidecars; no false successful export of shards. CPU/mock tests do not certify a DeepSpeed engine | Engine config, CPU/NVMe use, collective logs, consolidated-student identity; E9 |
| Training sequence parallelism | P: communication helpers and explicit model-adapter capability gate. U: generic Diffusers model-layer SP | Local frame slicing alone must fail; a future adapter must reproduce global attention/RoPE/masks and backward vs unsharded reference | 2/4/8-way forward/gradient error; Ulysses divisibility, ring tails, all-to-all volume |
| Inference ring / Ulysses / sparse attention | P: vendored, model-specific implementations; not automatically rewritten by dense fallback policy | Compare each actual runner at fixed checkpoint/seed against its single-device reference; validate masks and global positions | Latent error, decoded quality, collective time, real sample latency; E11 |
| Teacher parameter offload / heterogeneous teacher placement | U in unified training; placement plans and CPU payload cache are not executed parameter offload | Implement ownership/prefetch/synchronization first; demonstrate correctness and a measured GPU-memory decrease without missing teacher states | Transfer bytes/time, CPU pinned peak, PCIe/NUMA placement, stalls |
| Save → restart → next update | C: rank-local RNG and sampler cursor, optimizer/scheduler plus method sidecars; fixed-size launcher retries use an explicitly chosen checkpoint. P: exact GPU restart; changed-world-size elastic restart unsupported | Gate R; crash immediately before/after save, nonzero sampler cursor, auxiliary updates and stage boundary; corrupt/missing shard fails collectively; retries do not auto-discover latest checkpoint | Next batch IDs, RNG draws, parameters, optimizer/EMA/critic states, recovery time |
| Student export → real video | C: full trained denoiser identity/hash/config inventory, native dual and distinct architecture, declared compile wrapper. P: one-device Diffusers sampling; scheduler/guidance parity not automatic | Gate E; explicit unsupported LoRA/legacy-dual cases; manifest current-stage/count provenance; confirm pipeline uses exported student, not catalog/base teacher | Checkpoint/output hashes, exact solver/timetable/guidance, latent and video comparison |
| Raw video / cached conditions | C/P: normalization, source-frame controls and multi-encoder contracts. Hunyuan1.5 zero/image conditions and LTX normalization need native preprocessing | Compare raw-produced latent/condition tensors to family upstream pipeline on same source; no synthetic zero fallback; temporal anchors/masks correct | VAE/text revisions, preprocessing error, latent layout, cached-vs-raw loss |
| Wan2.1 / native Wan2.2 MoE | C: real tiny-model training/export and both native experts. P: pretrained T2V/I2V quality | Real teacher/student task at target shape; both experts' weights survive save/export; I2V first-frame and image features retained | Per-task videos, losses, quality, peak memory including both experts; E1/E11 |
| HunyuanVideo / HunyuanVideo1.5 | C/P: actual tiny forward/backward and explicit dual-encoder/image conditions. Not universal raw I2V | Full family checkpoint with required masks/pooled/vision conditions and guidance; Gate E | Encoder/condition shapes, raw-preprocess parity, actual output video |
| Action-conditioned world models | P: HY-WorldPlay/Matrix-Game/LingBot inference surfaces. U: universal native action-training adapter | Requires real action-conditioned checkpoint and licensed assets; left/right/stop/camera perturbations must causally alter expected next rollout; measure action latency and scene drift | Closed-loop trajectory/action metrics, failure/reset frequency, native adapter provenance |
| MeanFlow training | U: inference/catalog variant is not a dual-time/JVP trainer | Implement both time arguments, average-velocity target, JVP/autograd and boundary tests before quality/one-step claims | JVP reference error, real one-step video quality, memory/time; no imported result target |
| LTX-Video 2 / LTX-2 joint audio-video | P: inference runner/config. U: unified native joint AV training; earlier LTX layout adapter is not LTX-2 | Native audio/video latents, separate times/scales, connectors and synchronized loss/decoders; every trained branch gets gradient | Audio-video sync, audio quality, video quality, full AV save/export; E13 |
| SkyReels-V2 and other catalog stubs | U for advertised native training; some runners remain interfaces | Implement each model/task loader and forward before any benchmark. Include GameFactory/GameCraft/Infinite-World/Genie/GameGen-X/V-Mem/SPMem/CAM/Mirage separately | Per-family task/input/license/checkpoint and real forward artifacts |
| Evaluation / paper tables | P: measured run logging exists; VBench/FVD/action drift pipelines and reproductions remain separate work | Real generated clips only; fixed prompt/data/seed set, evaluator version and bootstrap CIs; no simulated fill-ins | Quality, diversity, dynamic degree, failure/OOM rate; results with and without speed optimizations |

Sources for implementation rows: [runtime](../training/runtime/),
[model adapter](../training/model_adapter.py),
[distributed trainer](../training/trainers/base_distill_trainer.py),
[quantization policy](../quant_compat.py),
[export](../training/student_export.py), and the linked audit documents above.
Stock CogVideoX variance-preserving diffusion is explicitly incompatible with
the current linear-flow training schedules; tensor-layout compatibility does
not fix the objective. Non-1000 training time scales are also rejected.

The repository's DeepSpeed FP16/BF16 route requires **DeepSpeed ≥0.19.6** and
the required callable Torch AMP APIs; older builds may only use explicitly
selected FP32 here. This is our admission policy, not a claim that every older
release lacks AMP. Native half-weight conversion is disabled: the engine owns
autocast and scaling, while the outer autocast context also covers frozen
teacher/EMA forwards. This matches the upstream [Torch AMP and nested-context
contract](https://deepspeed.readthedocs.io/en/latest/training.html#mixed-precision-training).
WorldDistill separately saves and validates the FP16 Torch GradScaler's state
for every rank, since it is not the native ZeRO loss-scaler object. Missing,
corrupt or changed-world-size scaler checkpoints fail; a CPU GradScaler/mock
engine test proves only these state-management contracts, not GPU convergence.

## 4. Prespecified numerical and measurement gates

These are initial engineering thresholds, not claims about perceptual equality.
Record errors even when they pass. Do not loosen a threshold after observing a
failure without documenting the reason and rerunning baseline/candidate together.

| Gate | Initial acceptance criterion | Boundary |
|---|---|---|
| K-loss (existing checker) | Loss `atol=2e-5, rtol=2e-5`; prediction/target gradients `atol=2e-4, rtol=3e-3`; all finite | [Quant/fused checker](../tools/check_quant_kernels.py), including upstream scale 65536 and 2050-element non-tile shape; calibrate larger real tensors separately |
| K-gemm (existing checker) | FP32 output vs FP64 dequantized GEMM reference: `atol=2e-4, rtol=3e-4`; all finite; correct zero rows | Tests kernel implementation on already quantized operands, **not** closeness to original BF16 weights or perceptual quality |
| K-attn (existing checker) | Relative L2 forward ≤0.025 for dense backends, ≤0.08 for Sage; supported Q/K/V backward ≤0.05; all finite | [Attention checker](../tools/check_attention_kernels.py); permissive smoke, not a universal precision specification. Backend without backward is inference-only |
| R (new real-model acceptance plan) | Deterministic FP32 small reference: `atol=1e-5, rtol=1e-4` next loss/update; initial BF16/FP16 same-workload target: relative L2 parameter-update and loss error ≤1e-2; exact batch IDs, step counts and discrete state | Compare update vectors, not total parameter norm which can hide broken updates. Report absolute error near zero. Distributed reduction order can differ; justify any revised tolerance |
| E (export) | Same stored dtype: exact equality of every exported trained tensor before any intentional conversion; hashes complete; same-device/same-solver eager latent replay passes R | Encoded video-byte equality is not required. Different scheduler, VAE, guidance or dtype is a separate experiment |
| S (stability/performance plan) | No NaN/Inf/OOM/hang over ≥200 measured updates after ≥50 warmup updates, across ≥3 seeds for training; inference ≥5 warmups + ≥20 requests for each shape where affordable | Log failures and cold start separately. Report median/p95/worst-rank step, samples/s and generated frames/s; tracing runs separate from final timing |
| Q (quality plan) | Declare non-inferiority margin before the run; paired prompts/seeds and 95% bootstrap CI. Suggested initial screen: ≤1.0 absolute VBench point drop AND no new action/audio-sync failure mode | Proposed screen, not established acceptable quality or implemented evaluator. FVD/FID require evaluator/sample-count protocol; lower training loss cannot substitute |

Run memory safety checks separately from timing, including
`compute-sanitizer --tool memcheck python tools/check_quant_kernels.py`.
Archive sanitizer output and exit status: the checker itself does not certify
that sanitizer was used. A skipped FP8 case on A100 is expected architecture
exclusion, not FP8 success; a partial suite must remain partial. For an exact
backend claim, a quiet fallback to SDPA is **not** a successful backend test.

## 5. Memory and speed expectations: formulas, not measured predictions

### F1. Per-rank state accounting

Let `P` be student parameter count, `n` the actual sharding-group size,
`p` parameter bytes/element, `g` gradient bytes/element and `o` optimizer/master
bytes/element. These formulas describe steady-state **student model states**,
not total VRAM. They follow the partitioning distinction in
[DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) and
[PyTorch FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

| Strategy | Ideal resident student state bytes/rank | Important excluded terms |
|---|---|---|
| Serial / DDP | `P × (p + g + o)` | Activations, workspace, communication buckets, teachers, EMA and critics |
| ZeRO-1 | `P × (p + g + o/n)` | Gradients/parameters remain replicated |
| ZeRO-2 | `P × (p + (g+o)/n)` | Parameters remain replicated; communication/accumulation buffers |
| ZeRO-3 / full-shard FSDP | `P × (p+g+o)/n` | Active all-gather unit, prefetch, full-state save and initialization peaks |
| Quantized inference weights | Approximately `P × b/8 + scale/zero-point/padding bytes` | Activations, high-precision unquantized layers, conversion copies; not a training-state formula |

For **the specific** FP16/BF16 copy + 16-bit gradients + FP32 master and two
FP32 Adam moments convention, `p=2,g=2,o=12`, so DDP has `16P` bytes of these
states. Native PyTorch BF16/FP32 optimizers may store states differently: inspect
actual tensors instead of assuming this convention. Frozen replicated teachers
add `sum(P_teacher × teacher_bytes)` to **every rank**, regardless of student
sharding. Include both Wan2.2 experts, DMD fake-score parameters/optimizer,
discriminator, EMA, VAE/text encoders and runtime cache where resident.

Use `max_memory_allocated`, `max_memory_reserved`, device-reported used VRAM,
CPU RSS and pinned memory together. A fit estimate must include measured
initialization, optimizer-step and checkpoint-save peaks; `16P/n` alone cannot
justify saying a full training job fits a particular GPU.

### F2. Context and attention

Selected latent payload alone uses `batch × channels × selected_frames ×
latent_height × latent_width × bytes`. A fixed selected-frame budget bounds
this payload, **not** the original full-video tensor, teacher activations or
disk/hot-tier copies. If a native per-layer KV cache is later implemented, its
uncompressed tensor accounting is `2 × layers × batch × cached_tokens ×
KV_heads × head_dim × bytes`; the factor two is K and V. This is not a statement
that the current latent selector already owns such a KV cache.

Attention's dense score matrix scales with `batch × heads × query_tokens ×
key_tokens`; FlashAttention avoids materializing that full matrix, while exact
dense attention arithmetic remains quadratic. Sparse selection changes the
attended information and requires quality validation; it is not just a
lossless memory allocator optimization.

### F3. Overlap, caching and scaling

- Ideal two-task overlap: independent teacher duration `T` and student duration
  `S` can reduce `T+S` toward `max(T,S)`, giving a bound of at most 2× **for those
  two tasks under ideal resource/dependency assumptions**. Actual DPP shares SMs,
  memory bandwidth and capacity; optimizer/communication work is not free.
- Ideal cache cost: `(1-hit_rate) × teacher_time + hit_rate × fetch_time +
  key/management_cost`. Fresh random timesteps/noise can yield almost no valid
  hits. Never assume that caching always accelerates training.
- If a fraction `f` of baseline time speeds up by `r`, ideal Amdahl speedup is
  `1 / ((1-f) + f/r)`. Reducing sampling from `K` to `k` only gives an idealized
  `K/k` denoiser-count factor when per-step work and guidance are identical;
  text encoding, VAE decoding, I/O and rollout changes still count.
- Measure strong scaling at fixed global effective batch, or weak scaling at
  fixed per-rank batch, and label which. Record `throughput(n)/throughput(1)`
  and efficiency `/n` only if the one-rank model fits under the same protocol;
  otherwise use the smallest fitting baseline and name it. No assumed
  A100→H20→B200→B300 multiplier, linear multi-node gain or target MFU is assigned.

## 6. Primary-source public reference numbers — NOT WorldDistill results

Accessed **2026-09-07**. Dates below are publication/release dates when available;
rolling documentation without a stated experiment date is marked as such.
`NR` is intentional: do not infer a benchmark's precision, frame count or
interconnect from a modern default config. Training costs exclude upstream
pretraining/data generation unless the source explicitly includes them.

| ID / source and date | Published result | Exact scope: model, hardware, GPU count, dtype, frames/resolution | Baseline / non-transferability |
|---|---|---|---|
| E1 [FastWan blog](https://haoailab.com/blogs/fastvideo_post_training/), 2025-08-04 | 1.3B: H200 E2E 5 s, denoise ~1 s; RTX4090 E2E 21 s, denoise 2.8 s. Table: H200 denoise FA2 95.21 s → VSA+DMD+compile 0.98 s. 5B: H200 E2E 16 s | **Inference**; Wan2.1-1.3B, 5 s/480P; Wan2.2-5B, 5 s/720P; one GPU/request; dtype/frame count NR in cited table | Joint checkpoint, sparse attention and compilation changes; 0.98 s excludes encoder/VAE. Not a single-kernel gain |
| E2 [Same FastWan training report](https://haoailab.com/blogs/fastvideo_post_training/), 2025-08-04 | 4k steps, 64 H200, 768 GPU-hours | **Training**; Wan2.1-1.3B sparse distillation; dtype/exact training clip shape NR in this cost statement | Different data/recipe; not WorldDistill convergence time or 64× single-GPU speed |
| E3 [Self Forcing paper, Table 1](https://arxiv.org/html/2506.08009v1), 2025-06-09 | Chunk-wise 17.0 generated FPS, 0.69 s latency; VBench 84.31 | **Inference**; Wan2.1-1.3B, one H100, 4 steps, 832×480, 5 s at playback 16 FPS; exact encoded count/dtype NR in table | Table Wan2.1: 0.78 FPS/103 s/84.26. Causal initialization + self-generated history + rolling KV; not generic Stream training |
| E4 [Self Forcing official training](https://github.com/guandeh17/Self-Forcing#training), 2025 release; rolling README | 600 iterations, under 2 h on 64 H100 | **Training**; DMD self-forcing Wan2.1-1.3B; 8 nodes×8 GPUs; dtype/clip shape NR in timing statement | README's under-16 h on 8 H100 is an **author projection**, not a second measured run; excluded from target table |
| E5 [Context Forcing paper, §4/Table 2](https://arxiv.org/html/2602.06028v1), 2026-02-05 | 21 latent-frame KV budget, >20 s context; student 17 FPS; 60 s VBench total 82.45 | **Training+inference metrics**; Wan2.1-1.3B; stage 1: 81 frames/600 iterations; stage 2: 10–30 s/500 iterations; batch 64. Hardware/count/dtype/resolution NR | Genuine KV slow/fast memory, context-teacher preparation and context DMD. Our chronological latent selector does not reproduce this algorithm or quality |
| E6 [DMD2 authors' page](https://tianweiy.github.io/dmd2/), NeurIPS 2024 | One-step FID 1.28 on ImageNet-64×64 and 8.35 on zero-shot COCO2014; SDXL preference study: student 4 steps vs teacher 50 | **Image**, not video; SDXL for COCO/image examples; hardware/count/dtype and COCO resolution NR on page | No corresponding WorldDistill FVD/VBench target. Do not call sampling-step reductions a measured GPU training speedup |
| E7 [FlashAttention-3 author blog](https://tridao.me/blog/2024/flash3/), 2024 | FP16 1.5–2.0× vs FA2, up to 740 TFLOPS; FP8 near 1.2 PFLOPS | **Attention kernel**, one H100; FP16/FP8 as stated; sequence/head shapes vary; frames/resolution N/A | Kernel throughput, not video E2E or backward certification; H20/B200/B300 must be tested independently |
| E8 [SageAttention official H20 table](https://github.com/thu-ml/SageAttention), update 2025-01-28 | CogVideoX1.5-5B: FA2 25m34s; FA3 17m32s; FA3-FP8 12m14s; Sage 12m07s | **Inference**; H20; exact GPU count/dtype/frames/resolution NR in table (FA3-FP8 label is explicit) | Useful H20 reference, incomplete reproducibility metadata. CogVideoX's objective differs from our flow trainers; not our speed |
| E9 [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/), rolling docs; experiment date NR | GPT-2 1.5B optimizer states: 18 GB → 2.25 GB per device with ZeRO-1 across 8 GPUs | **Training**, 8×V100-SXM3 32 GB, batch 1/device, mixed-precision model/FP32 Adam state convention; frames/resolution N/A | Optimizer-state saving only, not 8× total VRAM saving; teachers/activations excluded |
| E10 [Megatron-LM performance](https://github.com/NVIDIA/Megatron-LM#performance-benchmarking), rolling README; experiment date NR | 462B benchmark on 6,144 H100; up to ~47% MFU; GPT-3 strong scaling 96→4,608 H100, MFU 47→42% | **LLM training throughput**, sequence 4096, GPT-3 global batch 1152; dtype/topology NR in summary; frames/resolution N/A | Not trained to convergence; TP/PP/DP overlap and different GEMM shapes. No transferable video MFU or scaling promise |
| E11 [Wan2.2 official inference](https://github.com/Wan-Video/Wan2.2#run-wan22), release 2025-07-28 | A14B example needs ≥80 GB; TI2V-5B example runs ≥24 GB; 8-GPU FSDP+Ulysses commands provided | **Inference**; A14B 1280×720; 5B 1280×704; offload/convert dtype/T5 CPU options; count 1 or command example 8; exact dtype/frames NR | Fit statements/launch examples, **not** measured training capacity or 8-GPU speedup; MoE has both experts |
| E12 [FastVideo FP4 examples](https://haoailab.com/FastVideo/inference/examples/optimizations/), rolling docs | No independent latency number adopted here; explicit SM100a/SM103a FP4-linear path and BF16 baseline available | **Inference**; Wan2.1-1.3B; B200/B300, FlashInfer + optional TAEHV; NVFP4, optional QAD checkpoint | Concrete upstream implementation direction; not compatibility of our SM120 extension or an already integrated WorldDistill feature |
| E13 [FastVideo LTX-2.3 blog](https://haoailab.com/blogs/fastvideo_realtime_1080p/), **2026-03-11** on page | ~4.55 s E2E for 5 s synchronized AV, 1088×1920 at playback 24 FPS | **Inference**, LTX-2.3, one B200; NVFP4 DiT linear layers + full-stack optimization; exact frame count/other dtypes NR | Not LTX-2 training. Serving fleet has 36 independent replicas, not 36 GPUs per sample. No transfer of the latency to WorldDistill |

For FSDP implementation guidance, the
[PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
describes pre-forward/backward all-gather and backward reduce-scatter, and
distinguishes FSDP2 from FSDP1. WorldDistill's current wrapper is not automatically
FSDP2 because another project uses it. Model-layer wrapping/prefetch and a
sharded checkpoint format require implementation and independent qualification.

## 7. GPU-family qualification plan

This is **coverage to test**, not an assertion that every model fits every SKU.
Read the latest [CUDA compatibility policy](cuda-compatibility.md) for exact
toolkit/platform bounds. Product-to-CC identities are cross-checked against
[NVIDIA's GPU list](https://developer.nvidia.com/cuda/gpus); H20-specific policy
uses the additional sources linked in the CUDA document. Device memory and
link bandwidth must be read from the actual host, not assigned from family name.

| Family / representative products | CC | Safe first qualification path / important exclusions | Real-device state |
|---|---|---|---|
| Legacy Maxwell / Pascal | 5.x / 6.x | Compatible archived Torch/CUDA stack first; FP32/math fallback; no BF16/TF32/FP8/FP4 claim; unsupported wheels may make even this unavailable | pending; legacy feasibility only |
| Volta / Turing, V100 / T4 | 7.0/7.2 / 7.5 | Native-compatible Torch SDPA/math, model memory permitting; no automatic FA2/BF16/FP8 qualification | pending |
| A100 / A30; A800 family where available | 8.0 | BF16/FP16, SDPA → eligible FA2/Sage2; INT8 separately; no native FP8/FP4 | pending; A100 priority |
| A10 / A40 / RTX3090 / RTX A6000 | 8.6 | BF16/FP16, eligible FA2/Sage2 after build probe; no native FP8/FP4; check PCIe peer paths | pending |
| Jetson Orin | 8.7 | JetPack/ARM64-native installation; conservative SDPA; no implied optional quant/attention-extension support | pending; separate platform lane |
| L4 / L40 / L40S / RTX4090 | 8.9 | BF16 reference → eligible FP8/INT8 inference; Q8F only with its exact SM89 build; no native FP4 | pending |
| H100 / H200 / GH200; H800 where available | 9.0 | BF16 reference → eligible FA3/FA2/Sage2 and FP8; GH200 ARM64 and memory topology are separate build/host cases | pending |
| H20 | 9.0 | Product-aware runtime minimum; conservative FA2/Sage2/SDPA policy, verify installed kernels; not an H100 throughput equivalent | pending; priority |
| B200 / GB200 | 10.0 | Native Torch build first; SDPA safe default; explicit qualified FA4; current SM120-only low-bit extensions rejected | pending; priority |
| B300 / GB300 | 10.3 | Independently target SM103; do not assume SM100 binary/DeepGEMM entry qualifies; same conservative default | pending; priority |
| Jetson Thor | 11.0 | JetPack/ARM64/platform-specific toolchain; SDPA first; no generic server-kernel whitelist | pending; separate platform lane |
| RTX5090 / RTX PRO Blackwell | 12.0 | Native build + SDPA first; explicitly qualify applicable SM120 low-bit/Sage/FA4 paths, each separately | pending |
| DGX Spark / GB10 | 12.1 | ARM64 wheel/toolchain and host-memory constraints; do not relabel it B200; exact SM121 extensions require testing | pending; separate platform lane |
| Any unlisted or future CC | unknown | Conservative probe only; no extrapolated optional-kernel capability or dtype guarantee | unverified, not supported by analogy |

Minimal priority ladder: A100 single-device BF16 correctness → H20 product-aware
BF16/FP8 → B200 and B300 independent native builds → each target family's
single-node multi-GPU run → matching multi-node run. Full model weights may
require more memory than a single GPU; use a real smaller model or a documented
sharding recipe, and never label a tiny substitute as qualification of the large one.

## 8. Single-node / multi-node topology scenarios

The launch shape is a **planned test scenario**, not a resource allocation or
approved cloud purchase. Check actual links with topology/P2P probes and NCCL
collective tests before diagnosing model performance. NVIDIA recommends
separating GPU/NIC topology, networking, runtime and performance diagnosis in
its [NCCL troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).

| Scenario | Planned scale / topology | What must work | What to record; expected limitation |
|---|---|---|---|
| T1: one GPU | 1 node×1 device, each target family | Real forward/backward, operators, export, local restart | Establish reference memory/quality and separate cold/warm latency |
| T2: PCIe multi-GPU | 1×2 then 1×4/8 where installed | DDP; supported FSDP/ZeRO; no invalid P2P assumption | PCIe generation, NUMA/CPU affinity, P2P matrix, all-reduce/gather throughput; sharding can be communication-bound |
| T3: NVLink / NVSwitch | 1×2 then 1×8 homogeneous GPUs | Same tests plus native inference SP where implemented | Actual switch/link topology, transport selected, useful overlap; no automatic 8× gain |
| T4: basic multi-node Ethernet | 2×2 then 2×8, homogeneous GPUs | Rendezvous, collective error propagation, DDP smoke and distributed save/restart | NIC/interface/MTU, sockets/RDMA selection, inter-node latency; intended correctness lane, not fast scaling promise |
| T5: InfiniBand / RoCE | 2×8 then 4×8 or available scale | BF16 DDP and supported FSDP/ZeRO, real workload and collective checkpoints | NIC/GPU affinity, fabric rate, GPUDirect path, congestion, per-rank p95; all-to-all/ring only when native model supports it |
| T6: GB200/GB300 rack-scale fabric | Actual allocated subset, topology recorded | Independent build/kernel qualification plus collectives over the provided fabric | Do not assume every B200/B300 server is an NVL rack or that memory is one flat pool |
| T7: mixed GPU families or unequal memory | Separate test only, e.g. A100+H20 | Not a baseline support claim; identical dtype/layout/collective schedule and lowest-common backend required | Slowest rank and smallest VRAM bound progress; segregated homogeneous jobs preferred for reproducible claims |
| T8: partitioned/shared devices | MIG or shared GPU if explicitly available | First verify supported P2P/NCCL/runtime operations and capacity | Instance profile, tenancy, contention; whole-GPU results do not qualify an instance |
| T9: node/rank failure and restart | Reuse T3/T5, fixed world size first | Missing rank/file/corrupt state is visible to all ranks; valid checkpoint restores exact next update | Save atomicity, durable shared storage, recovery latency. Changed-world-size elastic restart is a separate unimplemented/unqualified contract |
| T10: ARM64 / Jetson / Spark | Single node, then only documented available interconnect | Compatible Python/Torch/NCCL/kernel binaries and model preprocessing | CPU/RAM/pinned-memory limits, platform BSP; x86 server wheels and throughput do not transfer |

A useful coverage record is one row per
`feature ID × GPU family × T-scenario × model/task × precision/backend`.
Do not collapse this cross-product into a single “all cards supported” badge.

The launcher now provides executable manual/Slurm and static/c10d paths plus
host-local shard/inter-node replica group construction. Logical multi-agent
CPU/Gloo tests run agents on the same physical host; even a four-worker result
does not fill any multi-machine NCCL, hybrid-FSDP or ZeRO hardware cell.
[Distributed training smoke](../tools/check_distributed_training.py) uses tiny
synthetic Conv3d denoisers, not pretrained videos, and its selected scope must
be retained in any evidence report. It checks all-rank full student/auxiliary
parameter replicas (including FSDP full-state and ZeRO-3 gathered parameters),
nonzero continuous-versus-resumed parameter deltas, optimizer/scaler/RNG/cursor
trees, engine optimizer partitions, per-rank Torch AMP scalers, client EMA and
method-specific sidecars.
Matching unchanged weights do not qualify a resumed update. These checks still
need execution on every claimed GPU/parallel combination, and are not full
Gate R for a real pretrained model, its next loss and output quality. CPU and
single-rank controls always retain `qualified=false`; even a GPU pass only
qualifies the explicitly selected synthetic cases and observed host topology.

## 9. Required result row and honest completion rule

| Field group | Must be stored for each real run |
|---|---|
| Identity | UTC time, git commit/dirty status, source/kernel hashes, model/tokenizer/VAE revisions, checkpoint and exported-weight hashes |
| Hardware/software | Exact product + memory/CC, GPU UUID or anonymized stable ID, count/nodes, driver/CUDA/Torch/Triton/extension/NCCL versions, topology/fabric, power/clock settings, OS/CPU architecture |
| Workload | Trainer/objective, dataset/prompt/control revision, input and latent frames/resolution, batch/GAS/world size, seed, solver/timetable/guidance, dtype/TF32/quant format, memory budget, cache state |
| Correctness | Actual backend/fallback status, max absolute/relative errors, gradients/updates, finite counts, sample IDs, save/resume comparison, sanitizer result and logs |
| Resource/time | Initialization and steady-state allocated/reserved/device VRAM, CPU RSS/pinned/disk, cold compile/load, encoder/denoiser/VAE/save time, median/p95/max-rank training step, samples/s, TTFF and generated FPS |
| Quality | Real output paths/hashes, evaluator revision and sample counts, paired-seed scores/CIs, action/audio-sync errors, scene drift/reset/static-collapse/failure rates |
| Decision | `measured-pass`, `measured-fail`, `pending`, `unsupported`, or `blocked`; explicit scope and reason; paired baseline ID for every speed or memory claim |

Current logger/summary and kernel tools populate only part of this schema;
topology, real quality evaluators, profiler traces and native-model qualification
still need to be supplied. A host manifest or 200 finite steps does not imply
that all fields are implemented or that quality is acceptable.

中文验收顺序：先让**真实模型的一步**数学正确，再验证**同一模型的训练、保存、恢复和导出**，
然后看**长时间稳定性与质量**，最后才比较加速和显存。只有跑过的精确组合才能填
`measured-pass`；未完成的原生 action、MeanFlow、LTX-2、通用 SP、异构 teacher offload
仍明确保留未完成标签。外部论文表格说明“哪些方向可能有效”，不能替代这条证据链。
