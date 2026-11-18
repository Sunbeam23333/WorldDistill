# Method and model support matrix

This matrix separates an executable objective from a catalog label or an
inference-only runner. The shared WorldDistill contribution is the runtime
surface around these objectives. It covers model resolution, cache-aware
teacher/student overlap, chunk-aware context selection, and reproducible
supervision and checkpoint contracts.

| Family | Current status | Code surface | Boundary |
|---|---|---|---|
| Step distillation | Implemented | `StepDistillTrainer` | Native Wan2.2 high/low-noise routing is supported when the checkpoint exposes it. |
| CFG distillation | Implemented | `CFGDistillTrainer`, `cfg_distill.json` | Requires explicit `uncond_*` text conditions. The student uses one conditional forward. |
| Progressive distillation | Implemented | `ProgressiveDistillTrainer` | Two-step teacher interval halving and stage handoff. |
| Consistency / TCD-LCD style | Implemented | `ConsistencyDistillTrainer` | EMA adjacent-point flow-space consistency. |
| PCM | Implemented core phase contract | `PCMTrainer`, `pcm_distill.json` | Phase-aware sub-trajectory sampling is exposed. The original repository's full guided/adversarial recipe is not implied by the name. |
| TCM | Implemented core two-stage contract | `TCMTrainer`, `tcm_distill.json` | Stage 1 full-range training, stage 2 truncation, and frozen stage-1 boundary loss. |
| Stream / Diffusion Forcing | Implemented generic path | `StreamDistillTrainer` | Requires a causal/per-frame model adapter. Self-Forcing and Causal-Forcing are separate registered trainers. |
| Self-Forcing | Implemented core rollout contract | `SelfForcingTrainer`, `self_forcing.json` | Generated overlap is fed into later windows with configurable rollout repeats and teacher/ground-truth context mixing. |
| Causal-Forcing | Implemented core causal rollout contract | `CausalForcingTrainer`, `causal_forcing.json` | Enforces causal temporal masks and configurable generated context frames. |
| rCM / sCM | Implemented core continuous consistency contracts | `RCMTrainer`, `SCMTrainer` | Continuous pair sampling, endpoint anchoring, and score-consistency weighting are exposed. |
| Diagonal distillation | Implemented | `DiagonalDistillTrainer`, `diagonal_distill.json` | Monotone diagonal frame/noise schedule with jitter control. |
| Reflow | Implemented | `ReflowTrainer`, `reflow.json` | Straight-line flow target with optional teacher endpoint mixing. |
| TMD | Implemented core transition contract | `TMDTrainer`, `tmd_distill.json` | Adjacent-state teacher transition matching with configurable gap and target weighting. |
| SiD-DiT | Implemented core score-identity contract | `SiDDiTTrainer`, `sid_dit_distill.json` | Pointwise score matching plus finite-difference identity supervision. |
| VSD / RL | Implemented core reward-aware contract | `VSDRLTrainer`, `vsd_rl_distill.json` | Teacher KL surrogate plus reward or preference advantage policy term. |
| Data-free DMD | Implemented synthetic-source contract | `DataFreeDMDTrainer`, `data_free_dmd_distill.json` | Synthesizes latent sources when batches omit latents and reuses DMD/DMD2 objectives. |
| Context Forcing | Implemented runtime path | `ContextForcingTrainer`, `WorldModelRuntime` | Hybrid-sparse memory and teacher context are available. The paper's slow/fast key-similarity KV algorithm is model-owned. |
| ADD/LADD | Implemented | `AdversarialDistillTrainer` | Latent projection discriminator and auxiliary optimizer state. |
| DMD/DMD2 | Implemented | `DMDDistillTrainer` | Fake-score DSM, DMD regression, and DMD2 GAN paths. |
| MeanFlow | Implemented | `MeanFlowDistillTrainer` | Teacher-guided dual-time JVP path with explicit higher-order derivative constraints. |
| Joint audio/video | Implemented for LTX2 | `JointAVDistillTrainer` | Separate video/audio heads and packed audio latent contract. |
| Generic sparse MoE | Implemented routing adapter | `training/utils/moe.py` | Top-k token routing, capacity-limited local dispatch/combine, and expert-parallel `all_to_all_single` helpers. Model adapters own expert module forward and checkpoint layout. |

The remaining boundary is model-native coverage for SkyReels and other
catalogued stub world models, plus paper-specific extras that require their
original model architecture, data engine, or reward service. The unified
trainers expose executable core objectives and keep those external contracts
explicit.

## What is novel in WorldDistill

The project does not claim a new distillation loss. Its contribution is the
runtime and evaluation substrate that makes different losses comparable under
one explicit interface.

1. A shared train/infer model catalog resolves model identity, task, checkpoint,
   conditioning, and export metadata.
2. A cache-aware teacher/student runtime exposes teacher-output reuse,
   teacher-context reuse, CUDA event ordering, and overlap counters.
3. World-model rollout uses a chunk-aware memory policy with dense-recent,
   strided-history, and hybrid-sparse selection while preserving action/camera
   alignment and target-only supervision masks.
4. Fused masked supervision, precision policy, elastic state, and checkpoint
   identity are runtime-visible controls that can be audited independently of
   model quality.

TP/PP, elastic recovery, and native Matrix-Game2/LingBot adapters extend the
release engineering surface. They are implementation coverage, not additional
paper-level algorithmic novelty.

Primary references: [Phased Consistency Models](https://arxiv.org/abs/2405.18407),
[Truncated Consistency Models](https://arxiv.org/abs/2410.14895),
[Transition Matching Distillation](https://openaccess.thecvf.com/content/CVPR2026/html/Nie_Transition_Matching_Distillation_for_Fast_Video_Generation_CVPR_2026_paper.html),
[Motion Consistency Model](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c859b99b5d717c9035e79d43dfd69435-Abstract-Conference.html).
