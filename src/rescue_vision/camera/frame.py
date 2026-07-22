from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np


MetadataValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """一帧 BGR 图像及其应用侧单调时钟时间。"""

    sequence: int
    timestamp_ns: int
    image_bgr: np.ndarray
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError(f"sequence must be non-negative, got {self.sequence}.")
        if self.timestamp_ns < 0:
            raise ValueError(
                f"timestamp_ns must be non-negative, got {self.timestamp_ns}."
            )
        if self.image_bgr.ndim != 3 or self.image_bgr.shape[2] != 3:
            raise ValueError(
                "image_bgr must have shape (height, width, 3), got "
                f"{self.image_bgr.shape}."
            )
        self.image_bgr.setflags(write=False)

    def age_ns(self, now_ns: int) -> int:
        """计算观测年龄；调用方必须传入同一单调时钟。"""

        if now_ns < self.timestamp_ns:
            raise ValueError(
                f"now_ns {now_ns} is earlier than frame {self.timestamp_ns}."
            )
        return now_ns - self.timestamp_ns

    def is_stale(self, now_ns: int, max_observation_age_ms: float) -> bool:
        if max_observation_age_ms <= 0:
            raise ValueError("max_observation_age_ms must be positive.")
        return self.age_ns(now_ns) > max_observation_age_ms * 1_000_000


@runtime_checkable
class FrameSource(Protocol):
    """真机和离线回放共享的最小帧源接口。"""

    @property
    def image_size(self) -> tuple[int, int]: ...

    def start(self) -> None: ...

    def read(self, timeout: float = 1.0) -> CameraFrame: ...

    def stop(self) -> None: ...
