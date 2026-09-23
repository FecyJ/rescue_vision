"""实验性门前清障参数；场地区域继续由 world.static_map 定义。"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True, slots=True)
class GateClearanceConfig:
    enabled: bool = False
    # 相对静态安全区前沿向场地中心移动触发基准线；0 保持静态地图边界。
    front_edge_inset_mm: float = 0.0
    front_depth_mm: float = 200.0
    # 每个安全区左右边界同时内缩，避免分界线和外边缘附近的投影抖动触发。
    lateral_inset_mm: float = 0.0
    # S1/S2 在 x 方向分别远离对应安全区外缘 150 mm。
    side_x_mm: float = 500.0
    # S1/S2 横扫线相对原 y=±1115 mm 点位向场地中心内移 230 mm。
    sweep_y_mm: float = 885.0
    # 清出的门前物块送至距场地原点该半径内再释放。
    center_stop_radius_mm: float = 200.0
    release_reverse_m: float = 0.12
    sweep_speed_m_s: float = 1.0
    transit_speed_m_s: float = 0.5
    # 横扫必须先精确平行于安全区横线，运行中超过该误差立即刹车重对正。
    sweep_heading_tolerance_rad: float = 0.035
    # 横扫轴心偏离 S1-S2 水平线超过此值时不再继续推进。
    sweep_cross_track_tolerance_mm: float = 20.0
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
                    or not math.isfinite(value)):
                raise ValueError(f"gate_clearance.{item.name} must be finite, got {value!r}.")
            if item.name in {"front_edge_inset_mm", "lateral_inset_mm"}:
                if value < 0:
                    raise ValueError(
                        f"gate_clearance.{item.name} must be non-negative, got {value!r}."
                    )
            elif value <= 0:
                raise ValueError(
                    f"gate_clearance.{item.name} must be positive, got {value!r}."
                )
        if self.center_stop_radius_mm >= math.hypot(self.side_x_mm, self.sweep_y_mm):
            raise ValueError(
                "gate_clearance.center_stop_radius_mm must be inside the S-point radius, "
                f"got {self.center_stop_radius_mm!r}."
            )
