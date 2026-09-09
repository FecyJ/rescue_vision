"""独立近场收拢试验的配置；机械标定仍由 motion 管理。"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

__all__ = ["NearFieldGraspConfig"]


@dataclass(frozen=True, slots=True)
class NearFieldGraspConfig:
    max_targets: int = 3
    max_candidates: int = 12
    max_range_mm: float = 450.0
    # 已锁定目标允许超出进入半径的距离滞回，避免边界观测抖动立即换组。
    range_hysteresis_mm: float = 50.0
    max_forward_distance_mm: float = 450.0
    # 绿/黑组前进结束时，最远目标 K0 希望保留在机器人前方的距离。
    target_final_x_mm: float = 110.0
    # 单个橙色目标按最近端加完整纵向跨度计算，减去该终点参考距离。
    orange_target_final_x_mm: float = 210.0
    # 夹爪前端参考线；危险目标 K0 只有落入从此处开始的扫掠矩形才阻挡动作。
    corridor_start_x_mm: float = 60.0
    corridor_lateral_margin_mm: float = 10.0
    # 侧邻异类门禁使用预测抓取方向下的 K0 中心差，单位 mm。只有纵向
    # 基本齐平且横向紧邻的蓝/橙目标才阻挡；明显前后错开的目标不触发。
    side_neighbor_longitudinal_margin_mm: float = 60.0
    side_neighbor_lateral_margin_mm: float = 90.0
    clearance_mm: float = 4.0
    # 组中心进入此横向允许范围后停止旋转；边界包含，单位 mm。
    center_tolerance_mm: float = 20.0
    # 进入允许范围后的滞回余量，单位 mm。
    alignment_hysteresis_mm: float = 10.0
    # 从近场会话打开到完成对准/确认的总预算，单位 ms。
    alignment_timeout_ms: float = 6_000.0
    # 无静止证据的几何、准备发布和静止遥测的年龄上限，单位 ms。
    # 停车后采集且持续静止的当前计划使用 processing 的观测失联上限。
    grasp_commit_max_observation_age_ms: float = 150.0
    stationary_max_gyro_rad_s: float = 0.03
    # 唯一近场确认窗口需要的不同有效感知帧数量。
    confirmation_frames: int = 3
    # 进入最终精对准区的角度半径，单位 rad。
    fine_alignment_zone_rad: float = 0.08
    # 最终精对准区内的单轮最小速度，允许为 0，单位 m/s。
    fine_alignment_min_wheel_velocity_m_s: float = 0.0
    # 橙色伤员周围拒绝其它目标的地面半径，单位 mm。
    orange_isolation_radius_mm: float = 50.0
    min_mask_pixels: int = 1
    # 可抓目标轨迹出现一次未知/普通质量异常后，需要连续多少个干净观测才恢复可选。
    # 已出现明确危险模型证据的轨迹不使用此恢复路径。
    supply_recovery_frames: int = 3
    clearance_scale_mm: float = 100.0
    # 初赛规则分值；候选先按总分排序，几何权重只处理同分方案。
    green_score_points: float = 5.0
    black_score_points: float = 10.0
    orange_score_points: float = 15.0
    # 单橙同分优先。该权重必须大于其余次级权重之和，保证单橙严格
    # 优先于同为 15 分的三绿或一绿一黑方案。
    orange_priority_weight: float = 0.55
    count_weight: float = 0.10
    clearance_weight: float = 0.15
    distance_weight: float = 0.12
    alignment_weight: float = 0.08

    def __post_init__(self) -> None:
        integer_names = {
            "max_targets",
            "max_candidates",
            "min_mask_pixels",
            "confirmation_frames",
            "supply_recovery_frames",
        }
        nonnegative = {
            "corridor_lateral_margin_mm",
            "clearance_mm",
            "range_hysteresis_mm",
            "side_neighbor_longitudinal_margin_mm",
            "side_neighbor_lateral_margin_mm",
            "fine_alignment_min_wheel_velocity_m_s",
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in integer_names:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"near_field_grasp.{field.name} must be a positive integer, got {value!r}.")
            else:
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"near_field_grasp.{field.name} must be finite, got {value!r}.")
                if value < 0 or (value == 0 and field.name not in nonnegative and not field.name.endswith('_weight')):
                    raise ValueError(f"near_field_grasp.{field.name} has invalid value {value!r}.")
        if self.max_targets > 3:
            raise ValueError(f"near_field_grasp.max_targets must be <= 3, got {self.max_targets}.")
        if self.max_candidates < self.max_targets or self.max_candidates > 20:
            raise ValueError(f"near_field_grasp.max_candidates must be in [max_targets,20], got {self.max_candidates}.")
        if self.corridor_start_x_mm > self.target_final_x_mm:
            raise ValueError(
                "near_field_grasp.corridor_start_x_mm must be <= "
                f"target_final_x_mm, got {self.corridor_start_x_mm} > "
                f"{self.target_final_x_mm}."
            )
        try:
            total = math.fsum(self.weights)
        except OverflowError as exc:
            raise ValueError(f"near_field_grasp weight sum overflow, got {self.weights}.") from exc
        if total <= 0 or not math.isfinite(total):
            raise ValueError(f"near_field_grasp weights must have a finite positive sum, got {self.weights}.")
        other_weight = math.fsum(self.weights[1:])
        if self.orange_priority_weight <= other_weight:
            raise ValueError(
                "near_field_grasp.orange_priority_weight must exceed the sum "
                "of count/clearance/distance/alignment weights so a single "
                "orange target wins equal-score comparisons."
            )

    @property
    def weights(self) -> tuple[float, float, float, float, float]:
        return (
            self.orange_priority_weight,
            self.count_weight,
            self.clearance_weight,
            self.distance_weight,
            self.alignment_weight,
        )
