# CUDA compatibility

WorldDistill separates four questions that are often conflated:

1. does the GPU have a known compute capability;
2. does the PyTorch build include a native architecture target;
3. do the chosen precision and core operations actually launch and produce correct finite results;
4. is the requested optional attention/quantization kernel independently qualified.

An architecture profile, successful import, or metadata-only exit code is **not**
a hardware validation result. All profiles below remain untested on real GPUs
in this development environment (CPU only).

## Probe

```bash
python tools/check_cuda_compat.py
python tools/check_cuda_compat.py --json
python tools/check_cuda_compat.py --strict
python tools/check_cuda_compat.py --probe-precision --json --output results/precision.json
python tools/check_cuda_compat.py --probe-precision --device 0 --precision fp16
python tools/check_cuda_compat.py --strict-native-arch --json
```

The report contains the Torch and CUDA runtime versions, `torch.cuda.get_arch_list`,
per-device profiles, expected `sm_*`, and compatibility issues. Metadata-only
reports always have `qualified=false`. Without `--device`, the precision probe
tests every visible GPU; with it, qualification is scoped only to that index.

`--probe-precision` executes small GEMM and causal **math SDPA** forward/backward
cases, synchronizes launches, and compares against CPU FP32 references using
the same quantized inputs. Candidate FP32/FP16/BF16 modes must pass finite-value
and relative-L2 checks (0.005/0.015/0.06 respectively). These are smoke tolerances,
not quality/accuracy guarantees for a full model. The probe uses a local RNG and
caches results per process/device/Torch/CUDA/build-architecture tuple; it does
not change the training RNG or global TF32 settings. It does not validate AMP
gradient scaling, convolutions, optimized attention, NCCL, or model convergence.

Exit 2 means NVIDIA CUDA unavailable (including CPU and ROCm); exit 1 means a
requested probe is incomplete/failed or an explicit precision/TF32 requirement
cannot be met. Probe exit 0 only qualifies its stated small-operation scope.
`--strict` also fails on runtime-policy issues or unknown architectures;
`--strict-native-arch` independently requires the exact native `sm_*` build
target. Missing native SASS is not itself a failed launch: compatible cubin or
PTX may run, but only execution establishes that. This distinction follows
[NVIDIA's compiler compatibility documentation](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-compiler-driver-nvcc/index.html).

## Precision and mixed GPU ranks

`probe_cuda_precision(torch, device)` returns `supported_precisions` from
successful launches, separate from `policy_precisions`. `auto` selects BF16,
then FP16, then FP32 (`no`) from the observed set. An explicit `bf16`/`fp16`/`no`
request raises if it cannot be honored; it never silently changes precision.
In a mixed-GPU distributed job, gather each rank's live report and call
`common_supported_precisions(reports)` before `select_mixed_precision(...)`.
For example, successfully probed V100+A100 ranks can share FP16, not BF16.
An unavailable/failed rank makes the common set empty. This does not attest to
cross-node NCCL transport or distributed training throughput.

Native BF16 and TF32 are disabled on Volta/Turing even if PyTorch can create or
emulate a BF16 tensor. Ampere introduced those formats; desktop SM86 also has
them, as documented in the [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
`validate_tf32_request(...)` checks every actual device's known feature profile
and successful FP32 smoke, without mutating settings; it does not assert that a
particular matmul selected Tensor Cores. Explicit TF32 on unsupported ranks is
an error, not an ignored toggle. Unknown architectures only attempt FP32 and
remain `unverified`, even after that limited smoke passes.

## Runtime attention resolution

`inference/lightx2v/utils/set_config.py` invokes the resolver only after explicit,
catalog, and checkpoint configuration overlays. Availability is determined by
importing the actual callable, not by seeing a registry key or module spec.

For an explicit dense backend:

- strict mode raises if it is missing or outside the device preference set;
  explicit FA4 additionally requires its on-device numerical smoke;
- non-strict mode chooses the first installed compatible backend;
- Torch SDPA is the final safe dense fallback when the PyTorch build supports it.

Backend constructors also fail immediately with an actionable message if a
selected FlashAttention or SageAttention callable is missing. This avoids a
later `NoneType is not callable` failure inside a long forward pass.

The optional installer does not make every backend available:

| Resolver name | Python callable/import | Installation status |
|---|---|---|
| `flash_attn2` | `flash_attn.flash_attn_interface` | `--install-kernels` installs the PyPI package |
| `flash_attn3` | `flash_attn_3.flash_attn_interface` or legacy top-level `flash_attn_interface` | separate Hopper build; CUDA >=12.3 |
| `flash_attn4` | `flash_attn.cute` | separate FA4 CuTeDSL install; explicit request and device probe |
| `sage_attn2` | `sageattention` | manual architecture-compatible install required |
| `sage_attn3` | `sageattn3` | manual architecture-compatible install required |
| `torch_sdpa` | `torch.nn.functional.scaled_dot_product_attention` | ships with compatible PyTorch |

When an optional callable is absent, non-strict resolution moves to the next
compatible installed backend; strict mode fails.

Sparse/distributed algorithms (`ulysses`, `ring`, `svg`, neighborhood attention,
and similar) are not rewritten by this dense-backend policy.
Their genuinely dense action/audio/text sublayers now use the shared policy.
Ring's dense LSE sub-operation has a bounded-memory exact reference fallback;
this does not establish optimized distributed throughput.

## Vendored low-bit CUTLASS extension

`inference/lightx2v_kernel` is a separate optional extension for `nvfp4`,
`mxfp4`, `mxfp6-mxfp8`, and `mxfp8`. Its current sources and CMake target are
specifically `sm_120a`. That binary is **not** a generic Blackwell fallback and
does not cover B200 (`sm_100`) or B300 (`sm_103`), nor A100/H20. Selecting one
of these schemes now checks both the exact capability and imported callables at
construction time, before weights or a long forward are launched. Use another
quantization scheme or unquantized weights on the four data-center targets
until a separately validated architecture-specific implementation exists.

## Policy profiles

| Profile / example | CC | Native toolkit/platform floor | Precision candidates | Dense preference | Real GPU result |
|---|---:|---:|---|---|---|
| Maxwell M40/M60 | 5.0 / 5.2 | 6.5; legacy CUDA <=12.x | FP32 only | SDPA | Not run |
| Jetson Nano / TX1 | 5.3 | 7.0; legacy JetPack | FP32, FP16 | SDPA | Not run |
| Pascal P100/P40 / TX2 | 6.0 / 6.1 / 6.2 | 8.0; legacy CUDA <=12.x | FP32, FP16 | SDPA | Not run |
| Volta V100 / Xavier | 7.0 / 7.2 | 9.0; legacy CUDA <=12.x | FP32, FP16 | SDPA | Not run |
| Turing T4 / RTX 20 | 7.5 | 10.0 | FP32, FP16 | SDPA | Not run |
| Ampere A100/A30 | 8.0 | 11.0 | FP32, FP16, BF16 | FA2, Sage2, SDPA | Not run |
| Ampere A10/A40 / RTX 30 | 8.6 | 11.1 | FP32, FP16, BF16 | FA2, Sage2, SDPA | Not run |
| Jetson Orin | 8.7 | JetPack with CUDA >=11.4 | FP32, FP16, BF16 | SDPA | Not run |
| Ada L4/L40 / RTX 40 | 8.9 | 11.8 | FP32, FP16, BF16 | Sage2, FA2, SDPA | Not run |
| Hopper H100/H200 | 9.0 | 11.8 | FP32, FP16, BF16 | FA3, FA2, Sage2, SDPA | Not run |
| H20 | 9.0 | 12.2 product policy | FP32, FP16, BF16 | FA2, Sage2, SDPA | Not run |
| B200/GB200 | 10.0 | 12.8 | FP32, FP16, BF16 | SDPA; FA4 explicit + smoke | Not run |
| B300/GB300 | 10.3 | 12.9 | FP32, FP16, BF16 | SDPA; FA4 explicit + smoke | Not run |
| Thor T5000/T4000 | 11.0 | CUDA 13 / JetPack 7 | FP32, FP16, BF16 | SDPA | Not run |
| RTX 50 / RTX PRO Blackwell | 12.0 | 12.8 | FP32, FP16, BF16 | SDPA | Not run |
| GB10 / DGX Spark | 12.1 | 12.9 | FP32, FP16, BF16 | SDPA | Not run |
| Any unlisted capability | other | unverified | FP32 probe only | SDPA | Unverified |

The CC/product mappings follow NVIDIA's [current GPU table](https://developer.nvidia.com/cuda/gpus)
and [legacy GPU table](https://developer.nvidia.com/cuda/gpus/legacy). Toolkit
floors are native support guardrails, not a promise that a Python wheel bundles
every required architecture/library. SM86 was added by
[CUDA 11.1](https://docs.nvidia.com/cuda/archive/11.1.0/cuda-toolkit-release-notes/index.html);
[CUDA 12.9](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)
added SM103 and SM121. Thor's former SM101 name became SM110 in CUDA 13,
as recorded in [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/13.1.0/nvcompdx/0.1.1/release_notes.html).
We do not treat arbitrary SM101 devices as an alias or enable desktop kernels
on Thor. Orin's CUDA 11.4 platform is documented in
[JetPack 5.0.2](https://developer.nvidia.com/embedded/jetpack-sdk-502), and TX1's
CUDA 7.0 platform in [Tegra R23.1](https://developer.nvidia.com/embedded/linux-tegra-r231).
All Jetson profiles additionally require the matching
[JetPack-specific PyTorch package](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
and the project's Python/PyTorch dependency floor. Older JetPack wheels may not
satisfy that floor; a known CC does not remove that installation blocker.

### Legacy wheel and toolkit lifecycle

CUDA 13 removed offline compilation/library support for Maxwell, Pascal and
Volta. A current driver does not restore those libraries. NVIDIA lists their
last toolkit family as CUDA 12.x and the relevant data-center driver branch as
R580 in its [architecture lifecycle matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html).
Kepler and older devices are outside the supported modern training stack;
an `unverified` metadata entry does not imply their old wheel can satisfy this
project's Torch >=2.5.1 / Python >=3.10 dependencies.

There is also a **wheel-specific** restriction below CUDA 13: the published
[PyTorch 2.11 packaging matrix](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)
keeps pre-Turing targets in CUDA 12.6 builds, while its CUDA 12.8/13.0 builds
start at Turing. Thus, installing CUDA 12.8 alone is not a V100 fix. An example
legacy candidate is Torch 2.11's `cu126` wheel; choose the correct Python/OS
variant from [official previous-version instructions](https://pytorch.org/get-started/previous-versions/)
and run the local probe. This is an installation candidate, not a tested
WorldDistill configuration. For B300/SM121, choose a build with a suitable
native target or demonstrably runnable compatible cubin/PTX and suitable CUDA
libraries; the version string alone is insufficient.

These are dispatch/install guardrails. The current public repository does not
contain real A100/H20/B200/B300 run manifests.

The callable currently detected as `sage_attn3` is an SM120/121 implementation,
not a B200/B300 implementation. The imported FA2 and FA3 interfaces likewise do
not advertise data-center Blackwell support. Non-strict resolution therefore
falls back to Torch SDPA even if one of those packages happens to import; strict
selection rejects it. A future optimized backend must add a capability-aware
probe and a reviewable host result before entering an automatic preference set.
SM120/121 also stay on SDPA: importing `sageattn3_blackwell` is insufficient
evidence that the installed wheel contains a runnable kernel for the exact
device. Sage3 requires a real forward smoke on that host. Every capability not
listed in the table—including intermediate and future CCs—fails closed to
native SDPA until an explicit policy and kernel-launch probe are added. The
new old-GPU/Jetson policies do **not** expand any INT8/FP8/FP4 backend target.
`quant_compat.py` still checks exact per-backend CC sets, callable availability,
NVIDIA CUDA (not ROCm's `torch.cuda` namespace), and the runtime floor.

## Hardware validation gate

Without `--probe-precision`, the metadata probe above does not execute a CUDA kernel. Use
`tools/check_attention_kernels.py` for selected attention forward/backward cases
and `tools/check_quant_kernels.py` for local GEMM/fused-loss numerical cases.
Both report GPU-unavailable (nonzero exit) on CPU. External extension guards in
`quant_compat.py` validate architecture prerequisites and callable symbols only;
they do not attest to the installed binary's numerical accuracy.

`tools/check_runtime_streams.py` separately checks the real
`TeacherStudentRuntime` serial and DPP APIs with a small differentiable synthetic
model. It compares teacher outputs, student outputs/losses and parameter
gradients; asserts a distinct teacher stream, recorded/completed ready events,
launch/wait counters, and real delegated `record_stream` calls. Both original
and separate consumer streams are exercised. Temporary inputs/outputs are
released before queued consumers finish, with same-size allocations on their
original streams to stress allocator lifetime protection. This is a bounded
correctness stress, not exhaustive race/sanitizer coverage.

```bash
python tools/check_runtime_streams.py --precision all --consumer-stream both --output results/streams.json
python tools/check_runtime_streams.py --precision fp16 --iterations 10 --profile results/streams-trace.json --output results/streams-fp16.json
```

Here `all` means all precisions that passed the actual-device precision probe;
the JSON includes that probe and the exact selected modes. It never claims a
pass for excluded modes. An explicit unavailable precision fails without a
downgrade. CPU/ROCm returns exit 2 and `qualified=false`; GPU failures or serial
fallback return exit 1. Optional profiling exports a scheduling trace without
asserting that the GPU overlapped work or delivered a speedup. Neither this
synthetic check nor its trace establishes full-model DPP throughput.

For each GPU/PyTorch/CUDA tuple, record:

- `nvidia-smi`, driver, CUDA toolkit, Torch version, Torch CUDA version;
- native `sm_*` list and exact optional-kernel package commits;
- one attention forward/backward comparison against SDPA;
- one serial vs CUDA-stream numerical comparison;
- one training step and one representative inference;
- peak memory and any fallback warnings;
- commit SHA, config, seed, resolution, frames, and output hash.

Only after those artifacts are committed should the README matrix use the word
“validated” for that device.
