# Model support

Status vocabulary:

- **Integrated**: a concrete runner and default configuration are checked in.
- **Contract-tested**: metadata/config/CLI construction has a CPU test.
- **Diffusers adapter**: loads the exact denoiser architecture and its supported conditioning and noise process.
- **Stub**: interface only; execution may raise `NotImplementedError`.

Use the training guide for objective-specific input contracts.

## Video and audio-video inference

| Catalog family | Tasks | Runner/config | Unified training |
|---|---|---|---|
| Wan 2.1 | T2V, I2V, FLF2V | Integrated where a default config is listed | Diffusers adapter |
| Wan 2.1 distilled | T2V, I2V as catalogued | Integrated inference variants | Compatible Diffusers checkpoints use the matching training objective |
| Wan 2.1 MeanFlow | T2V as catalogued | Integrated inference variant | Teacher-guided `mean_flow_distill`; explicit dual-time conditioning |
| Wan Self-Forcing | T2V as catalogued | Integrated inference variant | No dedicated native Self-Forcing training integration; Stream requires a causal/per-frame adapter |
| Wan 2.1 VACE | VACE | Integrated | Exact Diffusers VACE control adapter |
| Wan 2.2 MoE / distilled | T2V, I2V, selected FLF2V configs | Integrated | Diffusers adapter |
| Wan 2.2 VACE / Animate | VACE, Animate | Integrated | VACE context/scales and Animate reference/pose/face adapters |
| Wan 2.2 Audio | S2V | Integrated; internal LoRA defaults removed; CLI contract-tested | Not integrated |
| SekoTalk | S2V; stateful RS2V | Integrated; RS2V uses the shot wrapper and overlap-latent path | Not integrated |
| HunyuanVideo 1.5 | T2V, I2V | Integrated | Diffusers adapter |
| HunyuanVideo 1.5 distilled | T2V | Integrated; no distilled I2V default is advertised | Diffusers adapter |
| LTX-Video 2 | T2V, I2V, T2AV, I2AV | Integrated | `joint_av_distill`; native two-stream LTX2 adapter |
| SkyReels-V2 | T2V, I2V | Stub LightX2V runner | Exact Diffusers adapter; framewise diffusion forcing and FPS |

## Image inference

| Catalog family | Tasks | Runner/config | Unified training |
|---|---|---|---|
| Qwen Image | T2I, I2I | Integrated; I2I CLI contract-tested | Not integrated |
| LongCat Image | T2I, I2I | Integrated | Not integrated |
| Z-Image | T2I (I2I is catalogued but lacks a default config) | Integrated where configured | Not integrated |
| BAGEL | T2I, I2I | Integrated | Not integrated |

## World models

| Catalog family | Tasks | Runner/config | Unified training |
|---|---|---|---|
| HY-WorldPlay distilled / AR / BI | I2V, Game; selected T2V metadata | Runner/config integrated; external licensed pose provider required; game CLI contract-tested | Explicit bridge to the separately installed official training class; licensed source, distributed runtime and native shape restrictions apply |
| Matrix-Game 2.0 | Game | Integrated; game routing/CLI contract | Built-in author causal/foundation architectures, pixel-rate controls, strict checkpoint loading and PP |
| LingBot-CAM | Camera-controlled I2V | Integrated; external checkpoint mapping | Built-in author camera architecture, relative camera/ray features and PP |
| GameFactory, GameCraft, Infinite-World, Genie, GameGen-X, V-Mem, SPMem, CAM, Mirage | Catalogued tasks | Stub/interface only | Not integrated |

## Checkpoint availability

`tools/download_models.py --list` reports only download recipes that actually
exist in the tool. A catalog entry without an `hf_repo`, or a runner whose
checkpoint license requires a separate flow, is not downloadable through that
command. Runner/config presence and checkpoint availability are intentionally
separate facts.

## Native differentiable model adapter

A local `worlddistill_model.json` can select the concrete `matrix_game2`,
`lingbot_cam`, or externally installed `hy_worldplay` architecture. The Matrix
and LingBot implementations retain their original licenses and pinned source
revisions. See [native setup and descriptors](native-world-models.md).

External custom models may instead name an installed `module:factory`,
constructor arguments, canonical-to-native `input_map`, required conditions,
latent layout, and noise process. Installed external code requires
`--allow_native_model_code`. Safetensors loading is strict; export preserves
the descriptor and trained weights. Pixel-rate actions and latent-rate camera
features have distinct temporal contracts, not a shared guessed layout.

[Parallel and elastic training](parallel-elastic-native.md) documents the
audited Wan TP/PP plans, native PP plans, explicit custom plans and portable
training state. New Diffusers control adapters use the exact public model
classes and consume checkpoint-compatible encoded features.

## Adding or upgrading an integration

Required evidence:

1. canonical catalog entry and aliases;
2. concrete runner registration;
3. default config for each advertised task;
4. required-input CLI test;
5. checkpoint/source revision and license;
6. at least one real forward result manifest;
7. target-GPU kernel and memory record if hardware support is claimed.
