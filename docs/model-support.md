# Model support

Status vocabulary:

- **Integrated**: a concrete runner and default configuration are checked in.
- **Contract-tested**: metadata/config/CLI construction has a CPU test.
- **Experimental training**: the generic training entry may work with a matching
  Diffusers transformer, but no family-specific end-to-end artifact is checked in.
- **Stub**: interface only; execution may raise `NotImplementedError`.

None of these labels imply output quality or A100/H20/B200/B300 validation.

## Video and audio-video inference

| Catalog family | Tasks | Runner/config | Unified training |
|---|---|---|---|
| Wan 2.1 | T2V, I2V, FLF2V | Integrated where a default config is listed | Experimental training |
| Wan 2.1 distilled / MeanFlow / Self-Forcing | T2V, I2V as catalogued | Integrated inference variants | MeanFlow is not a trainer-registry method |
| Wan 2.1 VACE | VACE | Integrated | Not integrated |
| Wan 2.2 MoE / distilled | T2V, I2V, selected FLF2V configs | Integrated | Experimental training |
| Wan 2.2 VACE / Animate | VACE, Animate | Integrated | Not integrated |
| Wan 2.2 Audio | S2V | Integrated; internal LoRA defaults removed; CLI contract-tested | Not integrated |
| SekoTalk | S2V; stateful RS2V | Integrated; RS2V uses the shot wrapper and overlap-latent path | Not integrated |
| HunyuanVideo 1.5 | T2V, I2V | Integrated | Experimental training |
| HunyuanVideo 1.5 distilled | T2V | Integrated; no distilled I2V default is advertised | Experimental training |
| LTX-Video 2 | T2V, I2V, T2AV, I2AV | Integrated | Not integrated |
| SkyReels-V2 | T2V, I2V | Stub runner | Stream trainer is generic, not a SkyReels adapter |

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
| HY-WorldPlay distilled / AR / BI | I2V, Game; selected T2V metadata | Runner/config integrated; external licensed pose provider required; game CLI contract-tested | Context-forcing research path |
| Matrix-Game 2.0 | Game | Integrated; game routing/CLI contract | Not integrated |
| LingBot-CAM | Camera-controlled I2V | Integrated; external checkpoint mapping | Not integrated |
| GameFactory, GameCraft, Infinite-World, Genie, GameGen-X, V-Mem, SPMem, CAM, Mirage | Catalogued tasks | Stub/interface only | Not integrated |

## Checkpoint availability

`tools/download_models.py --list` reports only download recipes that actually
exist in the tool. A catalog entry without an `hf_repo`, or a runner whose
checkpoint license requires a separate flow, is not downloadable through that
command. Runner/config presence and checkpoint availability are intentionally
separate facts.

## Adding or upgrading an integration

Required evidence:

1. canonical catalog entry and aliases;
2. concrete runner registration;
3. default config for each advertised task;
4. required-input CLI test;
5. checkpoint/source revision and license;
6. at least one real forward result manifest;
7. target-GPU kernel and memory record if hardware support is claimed.
