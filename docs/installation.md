# Installation

WorldDistill deliberately does not pin one CUDA wheel for every GPU. A100,
Hopper, B200, and B300 need different native architecture targets, and optional
attention kernels must be compiled against the active PyTorch/CUDA pair.

## 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Python 3.10–3.12 is covered by CPU CI. Install the PyTorch build recommended by
the [official selector](https://pytorch.org/get-started/locally/) for the driver
and target GPU. For example, the setup wrapper can install from an official
wheel index supplied by the user:

```bash
bash scripts/setup_env.sh \
  --torch-index-url https://download.pytorch.org/whl/cu128 \
  --dev
```

The URL above is an example, not a universal choice. In particular, do not use a
CUDA 12.1 wheel as evidence of native B200/B300 support.

If PyTorch is already installed, keep it and run:

```bash
bash scripts/setup_env.sh --dev
```

The wrapper installs WorldDistill, registers the vendored LightX2V package,
imports the training entry, and compiles the inference sources as a portable
dependency/syntax smoke test. The full inference registry eagerly imports
GPU-specific backends such as Triton, so it is checked by `run_infer.sh` on the
target CUDA host rather than during a CPU install. Core and vendored metadata
both accept PyTorch 2.5.1 or newer; a newer runtime is still unvalidated until a
target-GPU run record is checked in.

## 2. Optional components

```bash
# FA2 plus sgl-kernel (not FA3 or SageAttention)
bash scripts/setup_env.sh --install-kernels

# ZeRO support
bash scripts/setup_env.sh --install-deepspeed
```

The distributed extra requires DeepSpeed 0.19.6+ for its explicit Torch AMP
engine/scaler contract. BF16/FP16 are configured as `torch_autocast`, not native
low-precision model conversion: this preserves FP32 timesteps and keyword inputs
while the engine manages operation precision and scaling. Older manually
installed engines are rejected for mixed precision; FP32 still requires its
own compatible engine/build test. CPU optimizer offload additionally needs a
working DeepSpeedCPUAdam extension. No DeepSpeed GPU run has been performed in
the CPU development environment. See the official
[mixed-precision API](https://deepspeed.readthedocs.io/en/latest/training.html#mixed-precision-training).

`--install-kernels` installs the PyPI `flash-attn` package (the FA2 import path)
and `sgl-kernel`. FA3 accepts the separate `flash_attn_3.flash_attn_interface`
package and the older top-level `flash_attn_interface` build;
Sage2 and Sage3 use `sageattention` and `sageattn3`. Install those manually from
a revision documented for the target GPU and record that revision. Kernel
installation may require a compiler, CUDA toolkit, and Ninja. A package
importing successfully on one GPU does not establish compatibility on another.

### Isolated modern training integration

The vendored inference engine retains its Transformers 4.57.1 pin. The optional
`requirements-validation.txt` instead describes the separate Diffusers 0.40.0 /
Transformers 5.16.1 **training** environment tested by CI. In a fresh Python 3.11
environment, install a target-appropriate PyTorch build first, then:

```bash
python -m pip install -r requirements-validation.txt
python -m pytest -q training/tests
```

Run training from the repository with `python -m training.train_distill` and
`--required_transformers_version 5.16.1` (plus the model/data arguments). Do not
install this requirements file over a working inference environment, or treat
these CPU integration tests as qualification of a CUDA wheel or kernel binary.

### Server frontend assets

Deployment and web-app dependencies are intentionally separate from the model
runner environment:

```bash
python -m pip install -e 'inference[server]'
# Optional Gradio app or MThreads platform support:
python -m pip install -e 'inference[app]'
python -m pip install -e 'inference[mthreads]'
```

The `deploy/server/static/{assets,index.html}` symlinks intentionally point to
the generated Vite output and may be dangling in a fresh source checkout. Build
that output before starting the web server:

```bash
cd inference/lightx2v/deploy/server/frontend
npm install
npm run build
```

Before starting the API server, configure independent, non-placeholder secrets:

```bash
export JWT_SECRET_KEY='replace-with-a-long-random-value'
export WORKER_SECRET_KEY='replace-with-a-different-long-random-value'
# Optional; when omitted, refresh tokens reuse JWT_SECRET_KEY.
export REFRESH_JWT_SECRET_KEY='replace-with-a-third-long-random-value'
```

Those literal values are rejected placeholders. Generate each value once with
a cryptographic secret generator, store it in the deployment secret manager,
and keep the values stable across restarts.

GitHub and Google login are optional. Each provider must be either completely
unset or configured with all three values, including a trusted callback URI
registered with that provider:

```bash
export GITHUB_CLIENT_ID='...'
export GITHUB_CLIENT_SECRET='...'
export GITHUB_REDIRECT_URI='https://worlddistill.example/auth/callback/oauth'

export GOOGLE_CLIENT_ID='...'
export GOOGLE_CLIENT_SECRET='...'
export GOOGLE_REDIRECT_URI='https://worlddistill.example/auth/callback/oauth'
```

The default CORS policy is same-origin. If the frontend is hosted on another
origin, enumerate exact origins rather than using a wildcard:

```bash
export LIGHTX2V_CORS_ALLOWED_ORIGINS='https://app.example.com,http://localhost:5173'
```

OAuth state is browser-bound, one-time, and stored in the server process. The
bundled entry point therefore runs one worker. A multi-process or multi-host
deployment must replace that store with a shared atomic backend (for example,
Redis `GETDEL`) before increasing the worker count.

## 3. Verify the environment

```bash
python tools/check_cuda_compat.py
python tools/check_cuda_compat.py --json
python tools/check_cuda_compat.py --strict
python tools/check_cuda_compat.py --probe-precision --json --output results/device-probe.json
python -m pytest -q training/tests
SKIP_PLATFORM_CHECK=1 python -c 'import training.train_distill; import lightx2v.infer'
```

`--strict` fails if CUDA is unavailable, the reported CUDA runtime is below the
policy minimum, or the architecture is unverified. `--strict-native-arch`
separately requires a listed native architecture; compatible PTX/cubin execution
is not the same as native SASS qualification. `--probe-precision` executes small
GEMM/math-attention forward/backward tests; it does not certify full models or
optional extensions. CPU-only hosts return `qualified=false` and exit 2.

## Hardware guidance

| GPU | Policy profile | What must still be checked on the host |
|---|---|---|
| A100 | `sm_80`, CUDA 11.0+ | FA2/SDPA forward-backward and distributed restart |
| H20 | `sm_90`, CUDA 12.2+ | FA2/Sage2/SDPA selection and actual kernel execution; FA3 opt-in only after device validation |
| B200 | `sm_100`, CUDA 12.8+ | Native PyTorch SDPA target; optimized kernels remain opt-in/pending |
| B300 | `sm_103`, CUDA 12.9+ policy | Native PyTorch SDPA target; verify every custom extension |

The repository currently has simulated policy tests, not signed results for
these four systems. See [CUDA compatibility](cuda-compatibility.md).
Older server/consumer NVIDIA families and ARM64 platforms have explicit policy
lanes, not one universal wheel. See the complete
[GPU and topology qualification tables](hardware-expectations.md) and
[fixed-size multi-node launcher](distributed-launch.md).

## Minimal and legacy requirement files

- `requirements.txt`: base training/inference Python dependencies.
- `requirements-kernels.txt`: FA2 and sgl-kernel only; FA3/Sage are manual.
- `requirements-distributed.txt`: optional DeepSpeed runtime.
- `setup.py` extras: `train`, `inference`, `cuda-kernels`, `distributed`, `dev`.

Avoid installing every optional package until the base CPU tests and CUDA probe
pass. This makes architecture or ABI failures much easier to isolate.
