"""实验性门前清障参数；场地区域继续由 world.static_map 定义。"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True, slots=True)
class GateClearanceConfig:
    enabled: bool = True
    front_depth_mm: float = 200.0
    side_x_mm: float = 350.0
    sweep_y_mm: float = 1115.0
    center_release_y_mm: float = 650.0
    release_reverse_m: float = 0.12
    sweep_speed_m_s: float = 1.0
    transit_speed_m_s: float = 0.5
    reacquire_radius_mm: float = 180.0
    observation_timeout_ms: float = 1400.0
    # 从首次触发到重新进入普通夹取的整个尝试截止时间，不按动作段重置。
    attempt_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"gate_clearance.enabled must be bool, got {self.enabled!r}.")
        for item in fields(self):
            if item.name == "enabled":
                continue
            value = getattr(self, item.name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"gate_clearance.{item.name} must be finite and positive, got {value!r}.")
        if self.center_release_y_mm >= self.sweep_y_mm - self.front_depth_mm:
            raise ValueError(f"gate_clearance release must be toward field center: {self.center_release_y_mm!r}.")
