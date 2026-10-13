# Operator and training follow-up — 2026-09-07

This release fixes executable correctness contracts identified after the 0.2.0
audit.

For the subsequent launcher, mixed-precision, DeepSpeed scaler/CPUAdam and
expanded hardware-qualification work, see the
[multi-node follow-up](multinode-validation.md). The latest changes are summarized
in the [October completion audit](implementation-audit.md#trainingruntime-completion--2026-10-13).

## Correctness changes

- Dense attention uses a shared batch/packed/GQA/mask/scale/causal contract.
  Matrix-Game2, BAGEL, audio adapters and Hunyuan no-pad layers no longer rely
  on hardcoded or conditionally undefined FA symbols. Ring merging has a
  public-interface LSE path and a bounded-memory exact reference fallback.
- FA3 supports old/new import namespaces. FA4 is available as an explicit,
  device-probed option; default Blackwell dispatch remains conservative.
  A successful import is not a kernel qualification.
- Fused MSE retains FP32 intermediate gradients, including loss scaling.
  Its normalization no longer calls `.item()` to synchronize with the host,
  and unmasked loss no longer allocates a full all-ones mask.
- INT8/FP8 GEMM uses masked tail reads for M/N/K, scales and bias. Quantized
  linear layers validate architecture prerequisites and actual callables.
  Zero-row FP8 reference quantization keeps finite FP32 scales.
- Cache promotion preserves original freshness. A bounded CPU prefetch worker
  fetches future teacher context, with freshness checked again at consumption.
  CUDA-stream inputs/outputs have explicit allocator lifetime recording.
- Progressive training includes the last interval. Stream training uses its
  inner step count and previous generated overlap. DMD/DMD2 train a fake score
  on generated samples and use the detached real-minus-fake distribution
  gradient; DMD2 does not substitute teacher-regression MSE for that gradient.
- Exact Diffusers architecture loading retains both Wan2.2 experts. Student
  export/reload links the sampled weights to the actual training checkpoint.
  Raw normalization/control sampling and multi-encoder conditioning are explicit.
  Complete denoiser configuration/weight inventories are hashed; pipelines that
  ignore the exported student are rejected. Declared compiled checkpoints retain
  their original resume keys while export removes the compiler wrapper prefix.

## Regression coverage

Automated regressions include real tiny Diffusers denoisers, optimizer steps, export/reload, numerical references and multi-process Gloo restart. Use the attention, quantization, stream and distributed checkers for installation-specific diagnostics.

## Current integration contracts

The current implementation adds CPU/file-backed teacher parameters, exact MeanFlow JVP, checkpoint-native VP noise, HunyuanVideo 1.5 T2V/I2V encoding, LTX2 joint audio-video training and explicit native model descriptors. See [training](training.md) and [model support](model-support.md) for configuration details.

SM120a low-bit binaries retain their architecture guards; installing a compatible extension is separate from model loading. Native model factories and weights must be supplied under their original licenses.

Upstream algorithm/backend references: [DMD2](https://tianweiy.github.io/dmd2/),
[FlashAttention](https://github.com/Dao-AILab/flash-attention),
[NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus).
