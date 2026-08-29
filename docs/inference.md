# Inference

The public wrapper resolves model aliases and default configs, validates the GPU
count, and forwards task-specific inputs to `lightx2v.infer`.

## Dry-run first

```bash
bash scripts/run_infer.sh \
  --model_cls wan2.2_moe \
  --task t2v \
  --model_path ./models/Wan2.2-T2V-A14B \
  --prompt "A small robot tending a rooftop garden." \
  --gpus 1 \
  --dry-run
```

Dry-run mode prints a shell-escaped `torchrun` command. It intentionally skips
heavy Python imports and does not create an output directory. It validates the
default model/task mapping (when used), required argument presence, JSON
presence/readability, shell construction, and declared parallel topology. It
does not validate input/checkpoint file existence, checkpoint contents, runner
imports, or CUDA kernels.

## Required task inputs

| Task | Required input |
|---|---|
| `t2v`, `t2i`, `t2av` | `--prompt` in the public wrapper |
| `i2v`, `i2i`, `i2av` | `--image_path` |
| `flf2v` | `--image_path`, `--last_frame_path` |
| `s2v` | `--image_path`, `--audio_path` |
| `rs2v` | `--model_cls seko_talk`, `--image_path`, `--audio_path`; routed through the stateful shot pipeline |
| `vace` | `--src_ref_images`; source video/mask as required by the config |
| `animate` | `--src_pose_path`, `--src_face_path`, `--src_ref_images`; optional background/mask for replacement |
| `game` | `--image_path`; pose/action fields depend on the runner |

The wrapper also supports `--image_strength`, `--pose`, `--action_path`,
`--action_ckpt`, `--transformer_model_name`, Animate source fields, and explicit
`--save_path`.

HY-WorldPlay pose conversion is intentionally not redistributed in this
repository because the upstream community license has separate territory and
use terms. Eligible users can provide an importable adapter exposing
`pose_to_input`, `parse_pose_string`, `pose_string_to_json`, and
`generate_camera_trajectory_local`, then set
`LIGHTX2V_WORLDPLAY_POSE_PROVIDER=<module.name>`. See
[`THIRD_PARTY_NOTICES`](../THIRD_PARTY_NOTICES) before enabling it.

## Parallel launch contract

The launcher world size must equal the config mesh. With no enabled `parallel`
object, use `--gpus 1`. For example, the repository's H20-specific Wan 2.2
config declares `cfg_p_size=2` and `seq_p_size=4`, so it requires eight workers:

```bash
bash scripts/run_infer.sh --model_cls wan2.2_moe --task t2v \
  --model_path ./models/Wan2.2-T2V-A14B --gpus 8 \
  --config_json inference/configs/wan22/wan_moe_t2v_h20_8gpu.json --dry-run
```

This checks topology only; the config is labelled for H20 and must not be used
as evidence for another GPU without a real kernel and memory smoke run.

## Config resolution

Resolution order is:

1. CLI and default values;
2. explicit `--config_json`, or catalog default config;
3. compatible checkpoint `config.json` overlays;
4. canonical model metadata;
5. CUDA dense-attention backend resolution.

The final stage can change only dense backend values (`flash_attn2`,
`flash_attn3`, `sage_attn2`, `sage_attn3`, `torch_sdpa`, `auto`). It does not
rewrite Ulysses, ring, SVG, neighborhood, or other sparse/distributed algorithms.
Set `"strict_cuda_backend": true` in a config to reject fallback.

## Output types

- image tasks default to `output.png`;
- video and audio-video tasks default to `output.mp4`;
- an explicit `--save_path` always wins.

## Integration status

Concrete runners/configs are available for the integrated families listed in
[model support](model-support.md). Catalog entries marked `stub` are interface
placeholders and may raise `NotImplementedError`. A dry-run for a stub does not
upgrade it to runnable status.
