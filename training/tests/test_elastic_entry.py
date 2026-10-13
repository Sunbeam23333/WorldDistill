from types import SimpleNamespace

import pytest

from training.train_distill import _data_parallel_coordinates, _resolve_elastic_auto_resume


@pytest.mark.parametrize("rank,expected", [(0, (0, 2)), (3, (0, 2)), (4, (1, 2)), (7, (1, 2))])
def test_training_data_coordinates_share_samples_within_pipeline_tensor_mesh(rank, expected):
    assert _data_parallel_coordinates(rank, 8, 1, 2, 2) == expected


def test_auto_resume_rejects_empty_restart_but_allows_explicit_checkpoint(tmp_path, monkeypatch):
    args = SimpleNamespace(elastic_auto_resume=True, resume_from="", output_dir=str(tmp_path))
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")
    with pytest.raises(RuntimeError, match="no fully published"):
        _resolve_elastic_auto_resume(args)
    args.resume_from = "/explicit/checkpoint"
    _resolve_elastic_auto_resume(args)
    assert args.resume_from == "/explicit/checkpoint"


def test_auto_resume_initial_run_retains_explicit_selection(tmp_path, monkeypatch):
    from training.utils.elastic import publish_elastic_checkpoint
    directory = tmp_path / "checkpoint-4"
    directory.mkdir()
    (directory / "trainer_state.pt").write_bytes(b"fixture")
    publish_elastic_checkpoint(directory, 4)
    args = SimpleNamespace(elastic_auto_resume=True, resume_from="initial", output_dir=str(tmp_path))
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    _resolve_elastic_auto_resume(args)
    assert args.resume_from == "initial"
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")
    _resolve_elastic_auto_resume(args)
    assert args.resume_from == str(directory)
