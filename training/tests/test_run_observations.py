import json
from pathlib import Path

import pytest

from training.trainer_args import TrainerArgs
from training.utils.experiment_tracking import ExperimentTracker
from tools.summarize_run import summarize


def test_local_observations_without_optional_trackers(tmp_path):
    tracker = ExperimentTracker(TrainerArgs(output_dir=str(tmp_path), report_to="none"))
    tracker.log_config({"seed": 42})
    for step in range(4):
        tracker.log_metrics({"step_time_sec": float(step + 1), "runtime/hits": step}, step)
    tracker.close()
    result = summarize(tmp_path, 1, 3)
    assert result["mean_step_seconds"] == 3.0
    assert result["evidence_type"] == "observed"
    runtime = json.loads((tmp_path / "runtime_stats.json").read_text())
    assert runtime["window_counters"]["hits"] == 3
    assert runtime["counter_baseline_step"] == 0
    assert result["measurement_start_step"] == 1
    assert result["measurement_end_step"] == 3
    # A second invocation cannot reuse old measurements as its own evidence.
    next_tracker = ExperimentTracker(TrainerArgs(output_dir=str(tmp_path), report_to="none"))
    next_tracker.log_config({"seed": 43})
    next_tracker.close()
    with pytest.raises(ValueError, match="have 0"):
        summarize(tmp_path, 1, 3)


def _write_observations(tmp_path, records):
    (tmp_path / "host_manifest.json").write_text(json.dumps({"run_id": "test", "cuda_available": False}))
    (tmp_path / "metrics.jsonl").write_text("\n".join(
        json.dumps({"run_id": "test", **record}) for record in records
    ))


def test_runtime_window_excludes_warmup_and_preserves_gauges(tmp_path):
    _write_observations(tmp_path, [
        {"step": 10, "step_time_sec": 100, "runtime/hits": 10, "runtime/misses": 0,
         "runtime/cache_hit_rate": 1, "runtime/resident_bytes": 1000},
        {"step": 11, "step_time_sec": 2, "runtime/hits": 10, "runtime/misses": 10,
         "runtime/cache_hit_rate": .5, "runtime/resident_bytes": 250},
    ])
    result = summarize(tmp_path, 1, 1)
    runtime = json.loads((tmp_path / "runtime_stats.json").read_text())
    assert result["mean_step_seconds"] == 2
    assert runtime["window_counters"] == {"hits": 0, "misses": 10}
    assert runtime["window_cache_hit_rate"] == 0
    assert runtime["cumulative_through_window_end"] == {"hits": 10, "misses": 10, "cache_hit_rate": .5}
    assert runtime["gauges_at_window_end"] == {"resident_bytes": 250}


def test_no_counter_baseline_does_not_invent_window_totals(tmp_path):
    _write_observations(tmp_path, [{"step": 100, "step_time_sec": 1, "runtime/hits": 900}])
    summarize(tmp_path, 0, 1)
    runtime = json.loads((tmp_path / "runtime_stats.json").read_text())
    assert runtime["window_counters"] == {}
    assert runtime["window_counters_unavailable"] == ["hits"]
    assert runtime["window_cache_hit_rate"] is None
    assert runtime["counter_baseline_step"] is None
    assert runtime["cumulative_through_window_end"]["hits"] == 900


@pytest.mark.parametrize("warmup,measured", [(-1, 1), (0, 0), (1.5, 1), (True, 1)])
def test_invalid_window_arguments_fail_before_reading(tmp_path, warmup, measured):
    with pytest.raises(ValueError, match="warmup"):
        summarize(tmp_path, warmup, measured)


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), float("-inf"), 0, -1, True, "1"])
def test_invalid_timing_is_not_performance_evidence(tmp_path, duration):
    _write_observations(tmp_path, [{"step": 1, "step_time_sec": duration}])
    with pytest.raises(ValueError, match="finite and positive"):
        summarize(tmp_path, 0, 1)
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize("steps", [[0, 2, 4], [0, 1, 1], [2, 1, 0], [0, 2, 3]])
def test_sparse_or_duplicate_steps_are_not_contiguous_performance_evidence(tmp_path, steps):
    _write_observations(tmp_path, [{"step": step, "step_time_sec": 1} for step in steps])
    with pytest.raises(ValueError, match="consecutive"):
        summarize(tmp_path, 1, 2)


@pytest.mark.parametrize("counter", [float("nan"), float("inf"), -1, .5])
def test_invalid_runtime_counter_is_rejected(tmp_path, counter):
    _write_observations(tmp_path, [{"step": 1, "step_time_sec": 1, "runtime/hits": counter}])
    with pytest.raises(ValueError, match="runtime metric|Runtime counter"):
        summarize(tmp_path, 0, 1)


def test_counter_reset_is_not_a_negative_window_count(tmp_path):
    _write_observations(tmp_path, [
        {"step": 1, "step_time_sec": 1, "runtime/hits": 10},
        {"step": 2, "step_time_sec": 1, "runtime/hits": 2},
    ])
    with pytest.raises(ValueError, match="counter hits reset"):
        summarize(tmp_path, 1, 1)


def test_paper_student_scripts_consume_training_checkpoint():
    root = Path(__file__).resolve().parents[2]
    for script in ("run_fewstep_t2v_suite.sh", "run_world_model_suite.sh", "run_streaming_longvideo_suite.sh"):
        text = (root / "scripts" / "paper" / script).read_text()
        assert "tools/export_student.py" in text
        assert '--checkpoint "${OUTPUT_ROOT}/train"' in text
        assert 'tools/sample_student.py' in text
        assert '--bundle "${OUTPUT_ROOT}/student_export"' in text


def test_runtime_ablation_holds_tf32_constant_and_runs_three_seeds():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/paper/run_runtime_ablation.sh").read_text()
    assert "--enable_tf32" not in text
    assert 'SEEDS:-42 43 44' in text
    assert 'log_every=1' in text
    assert '--warmup 50 --measured 200' in text
