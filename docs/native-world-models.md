# Native world-model training adapters

WorldDistill includes the actual parameterized Matrix-Game 2 and LingBot-CAM
architectures, plus model-specific conditioning adapters for Diffusers VACE,
Wan Animate and SkyReels V2. These are separate from inference-only runners and
from the generic installed-factory extension API. Catalog membership alone does
not install a native training architecture.

| Family | Architecture implementation | Training input |
| --- | --- | --- |
| Matrix-Game 2 | Bundled author foundation `WanModel` and causal `CausalWanModel`, with keyboard/mouse action modules | Encoded image/mask, CLIP image features, pixel-rate actions |
| LingBot-CAM | Bundled author camera-conditioned `WanModel`, including every camera injector | T5 features, camera features or aligned C2W poses plus intrinsics, checkpoint-specific image/mask |
| Wan VACE | Installed Diffusers `WanVACETransformer3DModel` | VACE source/mask/reference context and per-layer scale |
| Wan Animate | Installed Diffusers `WanAnimateTransformer3DModel` | Reference-prefixed latents, encoded reference/mask, pose latents, normalized face crops |
| SkyReels V2 | Installed Diffusers `SkyReelsV2Transformer3DModel` | T5 features, scalar or per-frame noise times, optional encoded image condition and physical FPS |
| HY-WorldPlay | Specific bridge to separately installed official **training** class | Author runtime initialization, encoded image/mask, text, discrete actions, W2C cameras and normalized intrinsics |

The Diffusers adapters require an installed release exposing their respective
classes; the architecture numerical tests use Diffusers 0.40.0. They do not make
older Diffusers versions contain newer model classes.

## Loading native checkpoints

Place `worlddistill_model.json` beside the checkpoint's safetensors. It contains
`schema_version: 1`, an explicit `architecture`, the complete author constructor
`model_config`, and a relative `weights` path. Both a single `.safetensors` file
and a `.safetensors.index.json` with local shards are accepted. Missing/unexpected
keys, shape mismatches, invalid shard inventories and paths escaping the model
directory fail strictly. `.pt`/`.pth` author checkpoints must first be converted
by their trusted producer; the native loader does not execute pickled weights.

Examples are in [`configs/native_models`](../configs/native_models/):

- `matrix_game2_universal.example.json` reproduces the official universal causal
  constructor config (omitting upstream's unused `inject_sample_info: false`,
  which is not a constructor argument). The example explicitly strips the author's generator
  wrapper `model.` prefix. For weights saved directly from `CausalWanModel`, set
  `weights_prefix` to `""`. Select `variant: "foundation"` only with that model's
  own constructor config and checkpoint.
- `lingbot_cam_tiny.example.json` is a **small architecture example**, not the
  released 14B checkpoint config. For released checkpoints copy the exact
  `config.json` constructor fields from the desired high/low/fast component,
  removing Diffusers metadata fields beginning with `_`, into `model_config`.
- `hy_worldplay_tiny.example.json` illustrates the installed-source bridge's
  schema. It is **not** the production checkpoint configuration and does not
  replace the official runtime requirements below.

The tiny examples are not pretrained models. Never relabel production weights
with a tiny configuration or change model dimensions to make loading succeed.

Load with `training.model_adapter.load_diffusers_training_model(model_directory)`;
the same directory can be a teacher/student model path in the training CLI.
Built-in Matrix/LingBot do not require permission to import arbitrary native
code. HY-WorldPlay does require `allow_native_model_code=True` (CLI
`--allow_native_model_code`).

For a high/low LingBot pair, place each complete native component in `high/` and
`low/`, then place this `worlddistill_adapter.json` at their parent:

```json
{"schema_version":1,"type":"noise_routed","boundary_ratio":0.947,"num_train_timesteps":1000}
```

Use the boundary and time scale recorded by the selected checkpoint; do not
reuse this example's boundary for a different model variant.

## Canonical conditioning and data

All shapes below include batch `B`; cached per-sample `.pt` files omit it. Clean
video latents are `B,C,T,H,W`. Store named conditioning tensors together in a
`conditioning_path` dictionary in the cached manifest. Data collation preserves
their exact shapes, dtypes and names; export sampling accepts the same named
batched tensors via `--conditioning_tensors conditions.safetensors`.

| Adapter | Required named tensors / semantics |
| --- | --- |
| Matrix2 | `image_cond`: `B,in_dim-C,T,H,W`; `image_embeds`: `B,N,1280`; `keyboard_actions`: `B,P,K`; `mouse_actions`: `B,P,2`, where `P=(T-1)*vae_time_compression_ratio+1` and `K` is the checkpoint keyboard dimension |
| LingBot-CAM | `encoder_hidden_states`: `B,L,text_dim`; `camera_features`: `B,384,T,H,W`, **or** `camera_poses`: `B,T,4,4` C2W plus `camera_intrinsics`: `B,T,4` (`fx,fy,cx,cy`) / `B,T,3,3`; optional `image_cond` only when required by `in_dim` |
| VACE | `encoder_hidden_states`; `vace_context`: `B,vace_in_channels,T,H,W`; optional `vace_scale`: `B,len(vace_layers)` or a shared per-layer vector |
| Animate | `encoder_hidden_states`; checkpoint-specific `image_embeds` and `image_cond`; `pose_latents`: `B,C,T-1,H,W`; `face_pixels`: `B,3,4*(T-2)+1,S,S` in `[-1,1]`, `S=motion_encoder_size` |
| SkyReels V2 | `encoder_hidden_states`; optional checkpoint-specific `image_cond` / `image_embeds`; `fps_condition`: `B` with positive physical FPS when `inject_sample_info=true` |

Matrix causal models accept `B,T` noise times; foundation models require one
noise level per sample. Actions retain **pixel-frame rate** because the actual
action encoder constructs causal multi-frame windows. They are not generic
latent-frame `actions` and are not text embeddings.

LingBot's camera conversion follows the author's relative-frame pose and
translation normalization, pixel-centered unit rays, concatenated **origin plus
direction**, and 8x8 pixel unshuffle. It does not substitute cross-product
Pluecker moments. Input poses must already match latent frames. Intrinsics use
pixels at decoded `8H x 8W`; do not pass normalized `[0,1]` intrinsics here.

Animate's clean training latent includes the extra reference position. The
adapter does not fabricate a reference prefix or silently crop its prediction.
For decoded samples, apply the official Animate reference-prefix removal in the
chosen decoder. Pose extraction, face detection and their associated licenses
remain explicit preprocessing inputs, not guessed raw-video transformations.

SkyReels maps physical FPS using the official convention: 16 FPS -> category 0,
other positive FPS -> category 1. `B,T` time inputs enable the real per-frame
time embedding. The installed architecture supplies block-causal attention
when `num_frame_per_block>1` and the latent length is divisible by that block
size; its default block value 1 is **not** advertised as a causal mask.

The native adapters consume cached, author-preprocessed conditions. The generic
raw encoder does not automatically become a Matrix/LingBot/VACE/Animate
preprocessor. Its existing raw-control transport additionally supports:

- `camera_intrinsics_path` with `camera_intrinsics_resolution: [source_H,source_W]`:
  intrinsics are resized with the video and temporally aligned to VAE anchors.
- `keyboard_action_path` / `mouse_action_path` with
  `matrix_control_alignment: "sampled_video"`: controls must already be produced
  for the sampled pixel-frame sequence; the cache encoder preserves pixel rate.
  It does not sum or interpolate mouse deltas or discard keyboard transitions.

Chunked latent selection supports VACE context and precomputed LingBot camera
features on their explicit time axes. For LingBot chunks, compute relative-pose
features on the **full** clip before selecting frames. Matrix action windows
and Animate's prefix/face protocol currently require full-clip training; the
chunked combination is rejected rather than misaligning controls.

## Export, precision and parallelism

Native components retain their source revision, exact constructor config,
weights and control protocol. Student export rebuilds the actual architecture,
strictly loads the trained state, and supports the MeanFlow wrapper without
flattening it into a different model. Use the adapter sampling backend for
native bundles; image/video decoding requires the original compatible VAE or
an explicit author decoder. No VAE weights are included here.

Matrix/LingBot CPU tests execute real author-architecture forward/backward,
BF16/FP16 storage, activation checkpointing, strict checkpoint reload and
multi-step sampling. VACE/Animate/SkyReels tests execute their real Diffusers
architectures, including control-branch gradients. These are small randomly
initialized architecture tests, not generated-video quality scores.

Built-in native pipeline plans partition the actual block stacks; their
two-rank tests compare prediction, gradients, updates and state restoration
against the unpartitioned architectures. Native TP/SP is not inferred from a
model name or a list of linear layers: a model-specific head/layout contract is
required. See the distributed-training documentation for supported plans.

## HY-WorldPlay external training runtime

HY-WorldPlay is separately licensed under the Tencent Hunyuan Community
License. WorldDistill does **not** redistribute its architecture implementation
or weights. Obtain the official repository and dependencies under its terms,
use reviewed revision `1588e1336e842b03b0a7860c654ebd7c46bb065e`, and set
`WORLDDISTILL_HY_WORLDPLAY_SOURCE` to that checkout. `source_root` in the
descriptor is an alternative, but the environment variable makes exported
bundles independent of an absolute source path.

The bridge imports exactly:

```text
trainer.models.hyvideo.models.transformers.ar_action_hunyuanvideo_1_5_transformer.
ARHunyuanVideo_1_5_DiffusionTransformer
```

It calls `add_discrete_action_parameters()` before strict loading. It does not
substitute the `hyvideo` inference model or an arbitrary factory.

Initialize the official training distributed runtime before forward with its
SP size set to 1. Its internal SP groups are not WorldDistill's context-parallel
protocol. The installed training dependencies and CUDA runtime remain required.
The pinned author's gradient-checkpointing flag is inert in this forward path;
the bridge rejects WorldDistill's flag, so disable it or use the author's
external activation-checkpoint wrapper. No wrapper is silently claimed active.

Additional cached inputs are `worldplay_actions: B,T` discrete author action
codes, `camera_view_matrices: B,T,4,4` W2C and
`camera_intrinsics_normalized: B,T,3,3`. Supply the checkpoint's text mask,
image features and ByT5 features (`byt5_text_states` / `byt5_text_mask`) when
configured. Text time is zero and video/action modulation follows the author's
latent-frame layout. The reviewed causal attention implementation fixes 1560
spatial tokens and four-latent chunks; the bridge rejects incompatible shapes.
Its `torch` mode is bidirectional, not a replacement causal training policy.

HY bridge unit tests cover permission, pinned class/source and input contracts;
they do not pretend to be numerical executions of the separately installed
author runtime. Matrix/LingBot and the three Diffusers adapters have separate
real-architecture numerical tests.

## Sources and licensing

- [Matrix-Game 2 source and universal config](https://github.com/SkyworkAI/Matrix-Game/tree/756776d516233027631008761d4d40ddaea3e23a/Matrix-Game-2)
- [LingBot-World camera architecture and preprocessing](https://github.com/Robbyant/lingbot-world/tree/a43bec7f8091c83e9b30b16b912f6fc906236fa6/wan)
- [HY-WorldPlay official training source](https://github.com/Tencent-Hunyuan/HY-WorldPlay/tree/1588e1336e842b03b0a7860c654ebd7c46bb065e/trainer)
- [Diffusers transformer implementations](https://github.com/huggingface/diffusers/tree/v0.40.0/src/diffusers/models/transformers)

Source revisions, upstream licenses and modifications are recorded in
[`THIRD_PARTY_NOTICES`](../THIRD_PARTY_NOTICES). The Matrix MIT and LingBot
Apache-2.0 license files are included in both source and wheel distributions.
