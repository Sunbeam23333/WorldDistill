# Operator and training follow-up — 2026-09-07

This release fixes executable correctness contracts identified after the 0.2.0
audit. It is an initial validation release, **not completion of every model and
hardware cell**. The paper's simulated results are not promoted to measurements.

For the subsequent launcher, mixed-precision, DeepSpeed scaler/CPUAdam and
expanded hardware-qualification work, see the
[multi-node follow-up](multinode-validation.md). The counts below describe the
preceding release, not the newer suite.

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

## Initial validation

The CPU test suite includes real tiny random-initialized Wan denoisers (not a
pretrained video-quality experiment), optimizer steps across registered methods,
student export/reload equivalence, regression references and two-process Gloo
training/restart checks. CUDA-only tests are skipped on CPU, not counted as
successful GPU validation. CI runs CPU contract tests and separate Diffusers
integration environments.

Final local run on 2026-09-07 (macOS arm64, Python 3.11.14, PyTorch 2.14.0;
CPU only):

| Environment | Main tests | Subtests | CUDA-only skips |
|---|---:|---:|---:|
| Diffusers 0.36.0 / Transformers 4.57.1 | 330 passed | 71 passed | 22 |
| Diffusers 0.40.0 / Transformers 5.16.1 | 330 passed | 71 passed | 22 |

These counts include actual two-rank Gloo processes, real tiny Wan and
HunyuanVideo 1.5 forward/backward, CLI training/resume/export, and compiler
wrapper export. The compiled test uses the eager compiler backend, not CUDA
Inductor performance. Static Python compilation, shell syntax, package metadata
and whitespace checks passed. Both GPU qualification tools returned exit 2 /
unavailable on this host. The two dependency runs are repeat validation of the
same tests, not 660 distinct features or GPU measurements.

```bash
python -m pytest -q training/tests
python tools/check_cuda_compat.py --strict
python tools/check_attention_kernels.py --help
python tools/check_quant_kernels.py --help
```

On a GPU host, run the selected backend's forward/backward and quantization
checks; run the quantization script under `compute-sanitizer --tool memcheck`
for non-tile-aligned shapes. Then test the real checkpoint's training step,
save/resume and video output. Archive exact driver/Torch/CUDA/kernel versions,
device architecture, configuration, tolerances, errors and output provenance.
The GPU checkers return an unavailable status on a CPU host; no green hardware
badge is inferred from that result. A partial quantization run (for example,
FP8 skipped on A100) also has `qualified=false` and exits nonzero. A ROCm build
exposing the `torch.cuda` namespace is not accepted as NVIDIA CUDA evidence.

## Boundaries still requiring work or resources

- No A100/H20/B200/B300 pretrained-model or kernel run was performed on the
  macOS development host. GPU performance, numerical tolerance and sanitizer
  records must come from those devices.
- SM120a-only low-bit extensions are rejected on B200/SM100 and B300/SM103;
  they are not a new data-center Blackwell implementation. External Q8F,
  DeepGEMM, SGL/vLLM and Marlin binaries still require device-side qualification.
- Native action-conditioned world-model training adapters, the ten catalog
  stubs, LTX2 joint audio/video training and a dual-time/JVP MeanFlow model are
  not implemented by an architecture name or a method alias. Invalid contracts
  fail explicitly. A causal stream recipe requires a model that implements its
  temporal contract; a bidirectional pretrained Wan is not relabelled causal.
- Teacher-parameter offload is not the CPU cache-prefetch worker. Teachers
  remain replicated; FSDP/ZeRO use CPU-staged students but still need GPU
  construction-memory and collective restart validation.
- The export bridge verifies weight identity, not scheduler/objective parity.
  It records training sampling settings without automatically replaying them.
  Native raw I2V/HunyuanVideo 1.5 zero-condition construction and LTX raw VAE
  normalization need further model-specific work. Incompatible declared noise
  time scales are rejected rather than silently routed on the wrong threshold.
  Stock CogVideoX VP diffusion is also rejected by the linear-flow trainers;
  merely matching its tensor layout would silently train the wrong objective.
- VBench/FVD, action consistency, task-specific drift/failure metrics and
  pretrained ADD/LADD/DMD/DMD2 reproduction remain separate evaluation work.
  Runtime timing scripts use measured observations, three default seeds and
  constant TF32 settings; they do not generate synthetic quality metrics.

Upstream algorithm/backend references: [DMD2](https://tianweiy.github.io/dmd2/),
[FlashAttention](https://github.com/Dao-AILab/flash-attention),
[NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus).
