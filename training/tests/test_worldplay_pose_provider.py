from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "inference/lightx2v/models/networks/worldplay/pose_utils.py"
PROVIDER_ENV = "LIGHTX2V_WORLDPLAY_POSE_PROVIDER"


def _load_bridge():
    spec = importlib.util.spec_from_file_location(
        "worlddistill_worldplay_pose_bridge",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worldplay_pose_bridge_is_lazy_and_fails_with_actionable_message(monkeypatch) -> None:
    monkeypatch.delenv(PROVIDER_ENV, raising=False)
    bridge = _load_bridge()

    with pytest.raises(RuntimeError, match=PROVIDER_ENV):
        bridge.pose_to_input({}, 1)


def test_worldplay_pose_bridge_delegates_to_explicit_provider(monkeypatch) -> None:
    provider_name = "test_worldplay_pose_provider_module"
    provider = types.ModuleType(provider_name)
    provider.generate_camera_trajectory_local = lambda motions: ("trajectory", motions)
    provider.parse_pose_string = lambda value: ("parsed", value)
    provider.pose_string_to_json = lambda value: {"pose": value}
    provider.pose_to_input = lambda data, count, tps=False: (data, count, tps)
    monkeypatch.setitem(sys.modules, provider_name, provider)
    monkeypatch.setenv(PROVIDER_ENV, provider_name)
    bridge = _load_bridge()

    assert bridge.parse_pose_string("w-3") == ("parsed", "w-3")
    assert bridge.pose_to_input({"0": {}}, 1, tps=True) == ({"0": {}}, 1, True)


def test_worldplay_pose_bridge_rejects_incomplete_provider(monkeypatch) -> None:
    provider_name = "test_incomplete_worldplay_pose_provider"
    monkeypatch.setitem(sys.modules, provider_name, types.ModuleType(provider_name))
    monkeypatch.setenv(PROVIDER_ENV, provider_name)
    bridge = _load_bridge()

    with pytest.raises(RuntimeError, match="missing callables"):
        bridge.pose_to_input({}, 1)
