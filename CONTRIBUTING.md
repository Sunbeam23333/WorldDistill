# Contributing

Thank you for improving WorldDistill. Keep implementation, documentation, and
evidence status synchronized.

## Development setup

```bash
bash scripts/setup_env.sh --dev
python -m pytest -q training/tests
```

## Pull-request checklist

- Add or update tests for every changed CLI/config/runtime contract.
- Run `python -m pytest -q training/tests`,
  `python -m compileall -q cuda_compat.py distill_capabilities.py training tools inference/lightx2v`,
  shell syntax checks, and `git diff --check`.
- Do not mark a stub or metadata entry as supported.
- Keep inference integration, training integration, checkpoint availability, and
  hardware validation as separate fields.
- Include immutable model/config/kernel revisions for measured results.
- Add a real forward artifact before claiming a new runner is runnable.
- Add target-host logs before claiming A100/H20/B200/B300 validation.
- Label generated or schematic imagery in its caption/provenance record.

## New model integration

Add the catalog entry, aliases, concrete runner registration, default task
config, task-input test, and checkpoint acquisition instructions. If training is
supported, add an adapter-level forward/backward smoke test rather than relying
on the generic loader alone.

## New distillation method

Document the exact objective, add a trainer registry entry and preset, and write
a numerical test that maps every equation term to code. A method inspired by a
paper should be called a scaffold until objective and update-schedule parity are
demonstrated.
