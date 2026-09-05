from __future__ import annotations

from collections.abc import Iterable
from typing import Any

"""Shared capability helpers for distillation method metadata.

This module intentionally stays dependency-free so training argument parsing,
trainer registration, model catalog resolution, and tests can all consume the
same method contract without importing heavyweight model code.
"""

REGISTRY_BACKED_DISTILL_METHODS: tuple[str, ...] = (
    "step_distill",
    "stream_distill",
    "progressive_distill",
    "consistency_distill",
    "context_forcing",
    "adversarial_distill",
    "dmd_distill",
)

REGISTRY_BACKED_DISTILL_METHOD_SET = frozenset(REGISTRY_BACKED_DISTILL_METHODS)
ONLINE_RUNTIME_DISTILL_METHODS = frozenset({"progressive_distill", "stream_distill", "context_forcing"})


def _normalize_method_list(methods: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for method in methods or ():
        name = str(method or "").strip()
        if name and name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def split_distill_methods(methods: Iterable[str] | None) -> tuple[list[str], list[str]]:
    ordered_methods = _normalize_method_list(methods)
    registry_backed = [method for method in ordered_methods if method in REGISTRY_BACKED_DISTILL_METHOD_SET]
    non_registry = [method for method in ordered_methods if method not in REGISTRY_BACKED_DISTILL_METHOD_SET]
    return registry_backed, non_registry


def resolve_distill_capability_fields(spec: dict[str, Any]) -> dict[str, Any]:
    distill_methods = _normalize_method_list(spec.get("distill_methods"))
    registry_backed_distill_methods, non_registry_distill_methods = split_distill_methods(distill_methods)
    catalog_only_distill_methods = _normalize_method_list(spec.get("catalog_only_distill_methods"))

    if not catalog_only_distill_methods and str(spec.get("distill_stage") or "").strip() == "student":
        catalog_only_distill_methods = list(non_registry_distill_methods)

    supports_registry_training_entry = bool(registry_backed_distill_methods)
    supports_opd_like_runtime = any(
        method in ONLINE_RUNTIME_DISTILL_METHODS for method in registry_backed_distill_methods
    )

    return {
        "distill_methods": distill_methods,
        "registry_backed_distill_methods": registry_backed_distill_methods,
        "non_registry_distill_methods": non_registry_distill_methods,
        "catalog_only_distill_methods": catalog_only_distill_methods,
        "supports_registry_training_entry": supports_registry_training_entry,
        "supports_opd_like_runtime": supports_opd_like_runtime,
    }
