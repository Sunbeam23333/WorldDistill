from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "paper"


@dataclass(frozen=True)
class AssetSpec:
    pattern: str
    mode: str
    required: bool
    description: str


@dataclass
class AssetFinding:
    pattern: str
    mode: str
    required: bool
    description: str
    exists: bool
    matches: list[str]


@dataclass
class BlockAudit:
    block: str
    root: str
    findings: list[AssetFinding]

    @property
    def missing_required(self) -> list[str]:
        return [finding.pattern for finding in self.findings if finding.required and not finding.exists]

    @property
    def missing_recommended(self) -> list[str]:
        return [finding.pattern for finding in self.findings if not finding.required and not finding.exists]

    @property
    def has_missing_required(self) -> bool:
        return bool(self.missing_required)

    def to_dict(self) -> dict[str, object]:
        return {
            "block": self.block,
            "root": self.root,
            "missing_required": self.missing_required,
            "missing_recommended": self.missing_recommended,
            "findings": [asdict(finding) for finding in self.findings],
        }


COMMON_RECOMMENDED_BUNDLE = (
    AssetSpec("run_card.md", "recursive_name", False, "Per-run reporting card with hardware, data split, and checkpoint policy."),
    AssetSpec("metrics.json", "recursive_name", False, "Aggregated quantitative metrics."),
    AssetSpec("runtime_stats.json", "recursive_name", False, "Runtime counters and timing summary."),
    AssetSpec("config_snapshot.json", "recursive_name", False, "Frozen config used for the reported run."),
    AssetSpec("sample_sheet.html", "recursive_name", False, "Human-auditable qualitative asset index."),
)

BLOCK_SPECS: dict[str, tuple[AssetSpec, ...]] = {
    "runtime_ablation": (
        AssetSpec("eager", "exact_dir", True, "Baseline eager ablation output directory."),
        AssetSpec("cache_only", "exact_dir", True, "Cache-only ablation output directory."),
        AssetSpec("cache_async", "exact_dir", True, "Cache+async ablation output directory."),
        AssetSpec("cache_async_fused", "exact_dir", True, "Cache+async+fused ablation output directory."),
        AssetSpec(
            "cache_async_fused_compile",
            "exact_dir",
            True,
            "Cache+async+fused+compile ablation output directory.",
        ),
        AssetSpec("trace.json", "recursive_name", False, "Optional trace or profiling snapshot for appendix evidence."),
        *COMMON_RECOMMENDED_BUNDLE,
    ),
    "fewstep_t2v": (
        AssetSpec("train", "exact_dir", True, "Few-step training output directory."),
        AssetSpec("teacher_samples/teacher.mp4", "exact_file", True, "Teacher reference video."),
        AssetSpec("student_samples/student.mp4", "exact_file", True, "Distilled student video."),
        *COMMON_RECOMMENDED_BUNDLE,
    ),
    "streaming_longvideo": (
        AssetSpec("train", "exact_dir", True, "Streaming training output directory."),
        AssetSpec("samples/streaming.mp4", "exact_file", True, "Long-video sample for boundary inspection."),
        *COMMON_RECOMMENDED_BUNDLE,
    ),
    "world_model": (
        AssetSpec("train", "exact_dir", True, "World-model training output directory."),
        AssetSpec("samples/worldplay_distill.mp4", "exact_file", True, "Distilled world-model rollout video."),
        AssetSpec("samples/worldplay_ar.mp4", "exact_file", True, "Autoregressive baseline rollout video."),
        *COMMON_RECOMMENDED_BUNDLE,
    ),
    "camera_control": (
        AssetSpec("poses/orbit.json", "exact_file", True, "Orbit trajectory definition."),
        AssetSpec("poses/pan_left.json", "exact_file", True, "Pan-left trajectory definition."),
        AssetSpec("poses/zoom_in.json", "exact_file", True, "Zoom-in trajectory definition."),
        AssetSpec("samples/orbit.mp4", "exact_file", True, "Orbit camera-control sample."),
        AssetSpec("samples/pan_left.mp4", "exact_file", True, "Pan-left camera-control sample."),
        AssetSpec("samples/zoom_in.mp4", "exact_file", True, "Zoom-in camera-control sample."),
        *COMMON_RECOMMENDED_BUNDLE,
    ),
}


def _resolve_matches(block_root: Path, spec: AssetSpec) -> list[str]:
    if spec.mode == "exact_file":
        target = block_root / spec.pattern
        return [spec.pattern] if target.is_file() else []
    if spec.mode == "exact_dir":
        target = block_root / spec.pattern
        return [spec.pattern] if target.is_dir() else []
    if spec.mode == "recursive_name":
        return sorted(str(path.relative_to(block_root)) for path in block_root.rglob(spec.pattern) if path.is_file())
    raise ValueError(f"Unsupported asset mode: {spec.mode}")


def audit_block(results_root: Path | str, block: str) -> BlockAudit:
    if block not in BLOCK_SPECS:
        available = ", ".join(sorted(BLOCK_SPECS))
        raise ValueError(f"Unknown block '{block}'. Available blocks: {available}")

    results_root = Path(results_root)
    block_root = results_root / block
    findings: list[AssetFinding] = []
    for spec in BLOCK_SPECS[block]:
        matches = _resolve_matches(block_root, spec)
        findings.append(
            AssetFinding(
                pattern=spec.pattern,
                mode=spec.mode,
                required=spec.required,
                description=spec.description,
                exists=bool(matches),
                matches=matches,
            )
        )
    return BlockAudit(block=block, root=str(block_root), findings=findings)


def audit_blocks(results_root: Path | str, blocks: Sequence[str] | None = None) -> list[BlockAudit]:
    selected_blocks = tuple(blocks) if blocks else tuple(BLOCK_SPECS.keys())
    return [audit_block(results_root, block) for block in selected_blocks]


def build_summary(audits: Iterable[BlockAudit]) -> dict[str, object]:
    audits = list(audits)
    return {
        "results_root": str(DEFAULT_RESULTS_ROOT),
        "blocks": [audit.to_dict() for audit in audits],
        "missing_required_blocks": [audit.block for audit in audits if audit.has_missing_required],
    }


def _format_status_line(audit: BlockAudit) -> str:
    status = "MISSING_REQUIRED" if audit.has_missing_required else "OK"
    return (
        f"[{status}] {audit.block}: "
        f"required_missing={len(audit.missing_required)}, recommended_missing={len(audit.missing_recommended)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether paper experiment outputs are complete enough for reporting.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root directory containing paper result blocks (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--block",
        action="append",
        choices=sorted(BLOCK_SPECS.keys()),
        help="Restrict the audit to one or more specific paper blocks.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full audit report as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero code when any required artifact is missing.",
    )
    args = parser.parse_args()

    audits = audit_blocks(args.root, args.block)
    summary = {
        "results_root": str(args.root),
        "blocks": [audit.to_dict() for audit in audits],
        "missing_required_blocks": [audit.block for audit in audits if audit.has_missing_required],
    }

    print(f"Auditing paper artifacts under: {args.root}")
    for audit in audits:
        print(_format_status_line(audit))
        if audit.missing_required:
            print(f"  required: {', '.join(audit.missing_required)}")
        if audit.missing_recommended:
            print(f"  recommended: {', '.join(audit.missing_recommended)}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {args.json_out}")

    if args.strict and any(audit.has_missing_required for audit in audits):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
