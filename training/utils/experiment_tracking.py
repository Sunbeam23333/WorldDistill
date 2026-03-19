"""Experiment tracking utilities for WorldDistill training."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Optional TensorBoard / W&B tracker wrapper."""

    def __init__(self, args: Any):
        self.args = args
        self.output_dir = getattr(args, "output_dir", ".")
        self.report_to = self._parse_backends(getattr(args, "report_to", "console"))
        self._tensorboard_writer = None
        self._wandb_run = None

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self._init_tensorboard()
        self._init_wandb()

        if self.report_to:
            logger.info(f"Experiment trackers enabled: {', '.join(self.report_to)}")

    @staticmethod
    def _parse_backends(report_to: str) -> list[str]:
        raw_items = [item.strip().lower() for item in str(report_to or "console").split(",") if item.strip()]
        if not raw_items or "none" in raw_items:
            return []
        if "all" in raw_items:
            raw_items = ["tensorboard", "wandb"]
        valid = [name for name in ("tensorboard", "wandb") if name in raw_items]
        return valid

    def active_backends(self) -> list[str]:
        active: list[str] = []
        if self._tensorboard_writer is not None:
            active.append("tensorboard")
        if self._wandb_run is not None:
            active.append("wandb")
        return active

    def log_config(self, config: dict[str, Any]) -> None:
        payload = self._to_serializable(config)
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.add_text(
                "config/json",
                json.dumps(payload, indent=2, ensure_ascii=False),
                global_step=0,
            )
        if self._wandb_run is not None:
            self._wandb_run.config.update(payload, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        payload: dict[str, float | int] = {}
        for key, value in metrics.items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                payload[key] = scalar
        if not payload:
            return
        if self._tensorboard_writer is not None:
            for key, value in payload.items():
                self._tensorboard_writer.add_scalar(key, value, global_step=step)
            self._tensorboard_writer.flush()
        if self._wandb_run is not None:
            self._wandb_run.log(payload, step=step)

    def close(self) -> None:
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.flush()
            self._tensorboard_writer.close()
            self._tensorboard_writer = None
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def _init_tensorboard(self) -> None:
        if "tensorboard" not in self.report_to:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as exc:
            logger.warning(f"TensorBoard 未安装，已跳过看板初始化: {exc}")
            return

        log_dir = getattr(self.args, "tensorboard_log_dir", "") or os.path.join(self.output_dir, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        self._tensorboard_writer = SummaryWriter(log_dir=log_dir)

    def _init_wandb(self) -> None:
        if "wandb" not in self.report_to:
            return
        try:
            import wandb
        except ModuleNotFoundError as exc:
            logger.warning(f"wandb 未安装，已跳过在线实验跟踪: {exc}")
            return

        mode = os.environ.get("WANDB_MODE")
        if mode is None and not os.environ.get("WANDB_API_KEY"):
            mode = "offline"
            logger.warning("未检测到 `WANDB_API_KEY`，W&B 将默认以 offline 模式记录。")

        run_name = getattr(self.args, "wandb_run_name", "") or os.path.basename(os.path.abspath(self.output_dir))
        tags = [tag.strip() for tag in getattr(self.args, "wandb_tags", "").split(",") if tag.strip()]
        try:
            self._wandb_run = wandb.init(
                project=getattr(self.args, "wandb_project", "worlddistill"),
                entity=getattr(self.args, "wandb_entity", "") or None,
                name=run_name,
                dir=self.output_dir,
                tags=tags,
                mode=mode,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(f"初始化 W&B 失败，已回退到 console/TensorBoard: {exc}")
            self._wandb_run = None

    @staticmethod
    def _to_scalar(value: Any) -> float | int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        if hasattr(value, "item"):
            try:
                scalar = value.item()
            except Exception:
                return None
            if isinstance(scalar, bool):
                return int(scalar)
            if isinstance(scalar, (int, float)):
                return scalar
        return None

    @classmethod
    def _to_serializable(cls, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if is_dataclass(value):
            return cls._to_serializable(asdict(value))
        if isinstance(value, dict):
            return {str(k): cls._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_serializable(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return str(value)
