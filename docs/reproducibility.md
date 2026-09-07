# Reproducibility and evidence levels

WorldDistill uses explicit evidence labels for code, figures, and claims.

| Label | Meaning |
|---|---|
| `measured` | Produced by a checked-in trace, experiment, or dataset row |
| `implemented` | A runnable code path exists, but it is not a scientific result |
| `simulation` | Produced by a declared analytical/simulation procedure |
| `illustrative` | Explains topology or intent; must not be read as model output |
| `failed-gate` | A completed test missed its prespecified criterion |

The README mechanism diagrams are described in
[`assets/readme/PROVENANCE.md`](../assets/readme/PROVENANCE.md).

## CPU release gate

```bash
python -m pytest -q training/tests
python -m compileall -q cuda_compat.py distill_capabilities.py training tools inference/lightx2v
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

CI runs the contract suite on Python 3.10, 3.11, and 3.12. It also verifies that
critical `training/data` and `inference/lightx2v/models` source files are tracked,
preventing broad ignore patterns from producing a broken clean clone.

## GPU release gate

Run the hardware record in [CUDA compatibility](cuda-compatibility.md), followed
by:

1. optimized attention vs Torch SDPA numerical test;
2. Triton fused-loss forward/backward vs PyTorch reference;
3. serial vs DPP output and gradient comparison;
4. DDP, FSDP, and DeepSpeed train-save-restart-next-loss comparison;
5. real inference for every advertised model/task pair;
6. benchmark manifest with warmup, repeats, synchronization, and failure logs.

Unrun cells remain “pending”; they are not silently inferred from a neighboring
GPU architecture.

Executable single-device stream/lifetime and multi-rank optimizer/restart
commands are in [training qualification](training.md#hardware-qualification-commands).
The complete [expected-results matrix](hardware-expectations.md) separates
acceptance criteria, theoretical resource accounting and upstream published
numbers from actual WorldDistill measurements. Its topology coverage must not
be inferred from local multi-agent CPU controls.

## Benchmark manifest minimum fields

```json
{
  "status": "measured",
  "git_commit": "<sha>",
  "gpu": "<exact product>",
  "driver": "<version>",
  "torch": "<version+cuda>",
  "kernel_revisions": {},
  "model_revision": "<immutable revision>",
  "task": "t2v",
  "resolution": [480, 854],
  "frames": 49,
  "batch_size": 1,
  "warmup_runs": 2,
  "measured_runs": 5,
  "seed": 42,
  "result_files": []
}
```

Do not publish an “× faster” statement without both baseline and candidate
records under the same protocol.
