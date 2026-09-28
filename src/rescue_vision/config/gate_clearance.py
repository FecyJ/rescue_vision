"""实验性门前定距路线参数；区域触发几何沿用 ``world.static_map``。"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True, slots=True)
class GateClearanceConfig:
    enabled: bool = False
    # 相对静态安全区前沿向场地中心移动触发基准线。
    front_edge_inset_mm: float = 0.0
    # 从基准线向场地中心延伸的门前触发纵深。
    front_depth_mm: float = 200.0
    # 每个安全区左右边界同时内缩，避免边线投影抖动触发。
    lateral_inset_mm: float = 0.0
    # 红方基准路线：绿/黑取 x=-lane_x_abs_mm，橙色取正值。
    lane_x_abs_mm: float = 450.0
    # 红方基准路线的门前横移线 y；蓝方按场地原点中心对称变换。
    lane_y_abs_mm: float = 1022.5
    lateral_forward_distance_m: float = 0.9
    lateral_reverse_distance_m: float = 0.6
    d2_push_distance_m: float = 0.2
    transit_speed_m_s: float = 0.5
    # 从首次触发起限制门前横移动作，完成倒退并移交正式 D2 投递前的截止时间。
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
