import pytest

from training.parallel.topology import parallel_coordinates, parallel_rank_groups
from training.trainer_args import TrainerArgs


@pytest.mark.parametrize("world,tp,pp", [(1, 1, 1), (2, 2, 1), (4, 2, 2), (8, 2, 2), (16, 4, 2)])
def test_model_parallel_groups_partition_ranks_without_mixing_parameter_owners(world, tp, pp):
    groups = parallel_rank_groups(world, tp, pp)
    for axis_groups in groups.values():
        assert sorted(rank for group in axis_groups for rank in group) == list(range(world))
    for group in groups["dp"]:
        coordinates = [parallel_coordinates(rank, world, tp, pp) for rank in group]
        assert len({(p, t) for _, p, t in coordinates}) == 1
    for group in groups["mp"]:
        assert len({parallel_coordinates(rank, world, tp, pp)[0] for rank in group}) == 1


@pytest.mark.parametrize("arguments", [(0, 3, 2, 1), (0, 4, 0, 1), (4, 4, 1, 1), (False, 4, 1, 1)])
def test_invalid_parallel_coordinates_fail(arguments):
    with pytest.raises(ValueError):
        parallel_coordinates(*arguments)


@pytest.mark.parametrize("kwargs", [dict(tp_size=0), dict(pp_size=True), dict(sp_size=-1),
    dict(elastic_auto_resume=True), dict(tp_size=2, sp_size=2),
    dict(tp_size=2, distill_method="mean_flow_distill"), dict(pp_size=2, enable_torch_compile=True),
    dict(tp_size=2, optimizer="muon")])
def test_invalid_model_parallel_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        TrainerArgs(**kwargs)
