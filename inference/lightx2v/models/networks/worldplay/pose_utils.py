"""Optional bridge to a user-supplied HY-WorldPlay pose adapter.

WorldDistill does not redistribute HY-WorldPlay pose-processing source because
that project's community license has territory and use restrictions.  Users
who are eligible under the upstream license may install their own adapter and
name its importable module with ``LIGHTX2V_WORLDPLAY_POSE_PROVIDER``.

The provider must expose the four callables wrapped below.  Keeping imports
lazy means the rest of LightX2V remains usable when this optional integration
is not configured.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any


_PROVIDER_ENV = "LIGHTX2V_WORLDPLAY_POSE_PROVIDER"
_REQUIRED_CALLABLES = (
    "generate_camera_trajectory_local",
    "parse_pose_string",
    "pose_string_to_json",
    "pose_to_input",
)


@lru_cache(maxsize=1)
def _load_provider():
    module_name = os.environ.get(_PROVIDER_ENV, "").strip()
    if not module_name:
        raise RuntimeError(
            "HY-WorldPlay pose processing is an optional, separately licensed "
            "integration. Install an eligible provider module and set "
            f"{_PROVIDER_ENV}=<module.name>; see THIRD_PARTY_NOTICES."
        )
    if module_name == __name__:
        raise RuntimeError(f"{_PROVIDER_ENV} must not point back to {__name__}")

    try:
        provider = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Could not import HY-WorldPlay pose provider {module_name!r}"
        ) from exc

    missing = [
        name for name in _REQUIRED_CALLABLES if not callable(getattr(provider, name, None))
    ]
    if missing:
        raise RuntimeError(
            f"HY-WorldPlay pose provider {module_name!r} is missing callables: "
            f"{', '.join(missing)}"
        )
    return provider


def _provider_callable(name: str) -> Callable[..., Any]:
    return getattr(_load_provider(), name)


def generate_camera_trajectory_local(motions):
    return _provider_callable("generate_camera_trajectory_local")(motions)


def parse_pose_string(pose_string):
    return _provider_callable("parse_pose_string")(pose_string)


def pose_string_to_json(pose_string):
    return _provider_callable("pose_string_to_json")(pose_string)


def pose_to_input(pose_data, latent_num, tps=False):
    return _provider_callable("pose_to_input")(pose_data, latent_num, tps=tps)
