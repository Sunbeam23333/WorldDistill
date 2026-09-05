# CUDA compatibility

WorldDistill separates three questions that are often conflated:

1. does the GPU have a known compute capability;
2. does the PyTorch build include a native architecture target;
3. is the requested optional attention callable installed and compatible.

## Probe

```bash
python tools/check_cuda_compat.py
python tools/check_cuda_compat.py --json
python tools/check_cuda_compat.py --strict
```

The report contains the Torch and CUDA runtime versions, `torch.cuda.get_arch_list`,
per-device profiles, expected `sm_*`, and compatibility issues. `--strict` is
appropriate for release/benchmark jobs.

## Runtime attention resolution

`inference/lightx2v/utils/set_config.py` invokes the resolver only after explicit,
catalog, and checkpoint configuration overlays. Availability is determined by
importing the actual callable, not by seeing a registry key or module spec.

For an explicit dense backend:

- strict mode raises if it is missing or outside the device preference set;
- non-strict mode chooses the first installed compatible backend;
- Torch SDPA is the final safe dense fallback when the PyTorch build supports it.

Backend constructors also fail immediately with an actionable message if a
selected FlashAttention or SageAttention callable is missing. This avoids a
later `NoneType is not callable` failure inside a long forward pass.

The optional installer does not make all four backends available:

| Resolver name | Python callable/import | Installation status |
|---|---|---|
| `flash_attn2` | `flash_attn.flash_attn_interface` | `--install-kernels` installs the PyPI package |
| `flash_attn3` | top-level `flash_attn_interface` | separate Hopper/Blackwell-compatible source build required |
| `sage_attn2` | `sageattention` | manual architecture-compatible install required |
| `sage_attn3` | `sageattn3` | manual architecture-compatible install required |
| `torch_sdpa` | `torch.nn.functional.scaled_dot_product_attention` | ships with compatible PyTorch |

When an optional callable is absent, non-strict resolution moves to the next
compatible installed backend; strict mode fails.

Sparse/distributed algorithms (`ulysses`, `ring`, `svg`, neighborhood attention,
and similar) are not rewritten by this dense-backend policy.

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

| Profile | Capability | Minimum runtime policy | Preference |
|---|---:|---:|---|
| Ampere data center | 8.0 | CUDA 11.0 | FA2, Sage2, SDPA |
| H100 / H200 | 9.0 | CUDA 11.8 | FA3, FA2, Sage2, SDPA |
| H20 | 9.0 | CUDA 12.2 | FA2, Sage2, SDPA; FA3 is not auto-selected |
| Blackwell data center | 10.0 | CUDA 12.8 | SDPA only until an `sm_100` kernel is device-validated |
| Blackwell Ultra policy | 10.3 | CUDA 12.9 | SDPA only until an `sm_103` kernel is device-validated |
| Blackwell client | 12.0 / 12.1 | CUDA 12.8 | SDPA until Sage3 passes an exact-device runtime smoke |
| Any unlisted capability | any other CC | unverified | SDPA only; optional kernels are rejected automatically |

These are dispatch/install guardrails. The current public repository does not
contain real A100/H20/B200/B300 run manifests.

The callable currently detected as `sage_attn3` is an SM120/121 implementation,
not a B200/B300 implementation. The imported FA2 and FA3 interfaces likewise do
not advertise data-center Blackwell support. Non-strict resolution therefore
falls back to Torch SDPA even if one of those packages happens to import; strict
selection rejects it. A future optimized backend must add a capability-aware
probe and a signed host result before entering an automatic preference set.
SM120/121 also stay on SDPA: importing `sageattn3_blackwell` is insufficient
evidence that the installed wheel contains a runnable kernel for the exact
device. Sage3 requires a real forward smoke on that host. Every capability not
listed in the table—including intermediate and future CCs—fails closed to
native SDPA until an explicit policy and kernel-launch probe are added.

## Hardware validation gate

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
