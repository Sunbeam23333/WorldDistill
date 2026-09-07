"""Summarize recorded observations only; no projected quality/speedup values."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


RUNTIME_COUNTERS = frozenset({
    "hits", "misses", "puts", "evictions", "hot_hits", "cold_hits", "promotions",
    "planned_prefetches", "prefetch_submitted", "prefetch_consumed", "prefetch_failed",
    "prefetch_completed", "prefetch_hits", "async_teacher_launches", "async_teacher_waits",
    "async_teacher_cache_hits",
})


def _runtime_snapshot(row: dict) -> dict:
    values = {k.removeprefix("runtime/"): v for k, v in row.items() if k.startswith("runtime/")}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite/non-numeric runtime metric {key} at step {row['step']}")
        if key in RUNTIME_COUNTERS and (value < 0 or int(value) != value):
            raise ValueError(f"Runtime counter {key} must be a non-negative integer")
    return values


def summarize(root: Path, warmup: int, measured: int) -> dict:
    if type(warmup) is not int or type(measured) is not int or warmup < 0 or measured < 1:
        raise ValueError("warmup must be an integer >=0 and measured an integer >=1")
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
    host = json.loads((root / "host_manifest.json").read_text())
    # A resumed/repeated run shares a directory but not a run_id. Never average
    # duplicate steps or mix measurements from an older host invocation.
    records = [row for row in records if row.get("run_id") == host["run_id"] and "step_time_sec" in row]
    selected = records[warmup:warmup + measured]
    if len(selected) != measured:
        raise ValueError(f"Need {warmup + measured} logged steps, have {len(records)}; use log_every=1")
    observed = records[:warmup + measured]
    for index, row in enumerate(observed):
        step, duration = row.get("step"), row["step_time_sec"]
        if type(step) is not int or step < 0:
            raise ValueError("Each logged training step must have a non-negative integer step index")
        if index and step != observed[index - 1]["step"] + 1:
            raise ValueError("Training step indices must be consecutive without duplicates; use log_every=1")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError(f"step_time_sec must be finite and positive at step {step}")
    times = [row["step_time_sec"] for row in selected]
    result = {"evidence_type": "observed", "run_id": host["run_id"], "warmup_observations": warmup,
              "measured_observations": measured, "mean_step_seconds": statistics.mean(times),
              "std_step_seconds": statistics.stdev(times) if len(times) > 1 else 0.,
              "measurement_start_step": selected[0]["step"], "measurement_end_step": selected[-1]["step"],
              "cuda_available": host["cuda_available"], "quality_metrics": "not evaluated"}
    snapshots = [_runtime_snapshot(row) for row in observed]
    final = snapshots[-1]
    baseline = snapshots[warmup - 1] if warmup else None
    # Unknown metrics may be instantaneous gauges; never subtract them as if
    # they were counters. With no pre-window observation, do not invent zero.
    window = {}
    for key in RUNTIME_COUNTERS & final.keys():
        values = [row[key] for row in snapshots if key in row]
        if any(after < before for before, after in zip(values, values[1:])):
            raise ValueError(f"Runtime counter {key} reset within the observed window")
        if baseline is not None and key in baseline and all(key in row for row in snapshots[warmup:]):
            window[key] = final[key] - baseline[key]
    total = window.get("hits", 0) + window.get("misses", 0)
    runtime = {
        "run_id": host["run_id"],
        "measurement_start_step": selected[0]["step"], "measurement_end_step": selected[-1]["step"],
        "window_counters": window,
        "window_cache_hit_rate": window["hits"] / total if {"hits", "misses"} <= window.keys() and total else None,
        "window_counters_unavailable": sorted((RUNTIME_COUNTERS & final.keys()) - window.keys()),
        "counter_baseline_step": observed[warmup - 1]["step"] if warmup else None,
        "cumulative_through_window_end": {key: value for key, value in final.items()
                                          if key in RUNTIME_COUNTERS or key == "cache_hit_rate"},
        "gauges_at_window_end": {key: value for key, value in final.items()
                                 if key not in RUNTIME_COUNTERS and key != "cache_hit_rate"},
    }
    (root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "runtime_stats.json").write_text(json.dumps(runtime, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measured", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 0 or args.measured < 1:
        parser.error("warmup must be >=0 and measured >=1")
    print(json.dumps(summarize(args.root, args.warmup, args.measured), indent=2))


if __name__ == "__main__":
    main()
