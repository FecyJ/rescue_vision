"""CC 独立比赛流程：稳健解团、单块通道搜索与正式安全区运输。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

from rescue_vision.app.cluster_breakup import GripperPosture
from rescue_vision.app.match import (
    _GroundClusterMeasurement,
    MatchDecision,
    MatchPreflight,
    MatchSequence,
    MatchStartArea,
    MatchState,
    configure_match_start_area,
)
from rescue_vision.app.match_runtime import _run_hardware
from rescue_vision.app.near_field_grasp import NearFieldGraspPolicy, NearFieldGraspSelector
from rescue_vision.config.match_cc import MatchCCRuntimeConfig
from rescue_vision.geometry.types import GroundPoint
from rescue_vision.mission import SafetySignals
from rescue_vision.perception import TargetClass
from rescue_vision.perception.gripper_width import TargetGroundEnvelope, measure_target_envelope
from rescue_vision.tracking import TrackedTarget

if TYPE_CHECKING:
    from rescue_vision.config import AppConfig
    from rescue_vision.geometry.ground_projector import GroundProjector


_GRASPABLE = frozenset(
    (TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE, TargetClass.ORANGE_INJURED)
)


@dataclass(frozen=True, slots=True)
class CCCluster:
    """由同一锚点 100 mm 邻域形成的确定性目标团。"""

    member_ids: tuple[int, ...]
    points: tuple[GroundPoint, ...]
    classes: tuple[TargetClass, ...]

    @property
    def center(self) -> GroundPoint:
        return GroundPoint(
            math.fsum(point.x for point in self.points) / len(self.points),
            math.fsum(point.y for point in self.points) / len(self.points),
        )


def find_cc_clusters(
    targets: Iterable[TrackedTarget],
    *,
    neighbor_distance_mm: float = 100.0,
) -> tuple[CCCluster, ...]:
    """找出团内每块都至少有两个 100 mm 内邻居的最大连通团。"""

    if (
        isinstance(neighbor_distance_mm, bool)
        or not isinstance(neighbor_distance_mm, (int, float))
        or not math.isfinite(float(neighbor_distance_mm))
        or float(neighbor_distance_mm) <= 0.0
    ):
        raise ValueError("neighbor_distance_mm must be finite and positive.")

    candidates = tuple(
        target
        for target in targets
        if target.ground_point is not None and target.target_class in _GRASPABLE
    )
    neighbours: dict[int, set[int]] = {item.track_id: set() for item in candidates}
    by_id = {item.track_id: item for item in candidates}
    for index, target in enumerate(candidates):
        assert target.ground_point is not None
        for other in candidates[index + 1 :]:
            assert other.ground_point is not None
            if math.hypot(
                target.ground_point.x - other.ground_point.x,
                target.ground_point.y - other.ground_point.y,
            ) <= neighbor_distance_mm + 1e-9:
                neighbours[target.track_id].add(other.track_id)
                neighbours[other.track_id].add(target.track_id)

    # 反复剔除邻居不足两个的点，得到图的 2-core；这样返回团中的每一个
    # 物块都确实仍与团内至少两个物块相邻，而不是只满足最初全集中的度数。
    active = set(by_id)
    while True:
        removed = {
            track_id
            for track_id in active
            if len(neighbours[track_id] & active) < 2
        }
        if not removed:
            break
        active -= removed

    groups: dict[tuple[int, ...], CCCluster] = {}
    unseen = set(active)
    while unseen:
        pending = [min(unseen)]
        component: set[int] = set()
        while pending:
            track_id = pending.pop()
            if track_id in component:
                continue
            component.add(track_id)
            pending.extend(sorted((neighbours[track_id] & active) - component))
        unseen -= component
        if len(component) < 3:
            continue
        members = tuple(by_id[track_id] for track_id in sorted(component))
        ids = tuple(item.track_id for item in members)
        groups[ids] = CCCluster(
            ids,
            tuple(item.ground_point for item in members if item.ground_point is not None),
            tuple(item.target_class for item in members),
        )
    return tuple(
        sorted(
            groups.values(),
            key=lambda item: (min(item.member_ids), -len(item.member_ids), item.member_ids),
        )
    )


def point_to_origin_segment_distance_mm(point: GroundPoint, target: GroundPoint) -> float:
    """返回点到 ``(0,0)→target`` 线段的欧氏距离。"""

    if not isinstance(point, GroundPoint) or not isinstance(target, GroundPoint):
        raise TypeError("point and target must be GroundPoint values.")
    length_squared = target.x * target.x + target.y * target.y
    if length_squared <= 1e-12:
        return math.hypot(point.x, point.y)
    scale = max(0.0, min(1.0, (point.x * target.x + point.y * target.y) / length_squared))
    return math.hypot(point.x - scale * target.x, point.y - scale * target.y)


def isolated_targets(
    targets: Iterable[TrackedTarget],
    *,
    allowed_classes: frozenset[TargetClass],
    clearance_for_target: Callable[[TrackedTarget], float],
) -> tuple[TrackedTarget, ...]:
    """按车辆原点至 K0 的线段检查其它物块是否侵入通道。"""

    if not isinstance(allowed_classes, frozenset) or not allowed_classes or not all(
        isinstance(item, TargetClass) for item in allowed_classes
    ):
        raise ValueError("allowed_classes must be a non-empty TargetClass frozenset.")
    if not callable(clearance_for_target):
        raise TypeError("clearance_for_target must be callable.")

    # 候选类别受 ``allowed_classes`` 限制；通道阻挡证据则必须保留蓝色和
    # 颜色不确定物块，不能把危险目标从“其它物块”中漏掉。
    candidates = tuple(target for target in targets if target.ground_point is not None)
    result = []
    for target in candidates:
        if target.target_class not in allowed_classes:
            continue
        assert target.ground_point is not None
        clearance = float(clearance_for_target(target))
        if not math.isfinite(clearance) or clearance <= 0.0:
            continue
        if any(
            other.track_id != target.track_id
            and other.ground_point is not None
            and point_to_origin_segment_distance_mm(other.ground_point, target.ground_point)
            <= clearance + 1e-9
            for other in candidates
        ):
            continue
        result.append(target)
    return tuple(result)


class MatchCCSequence(MatchSequence):
    """CC 状态机；仅继承正式流程的启动、安全短路、运输和退出实现。"""

    _dynamic_breakup_enabled = False

    def __init__(self, *args, cc_config: MatchCCRuntimeConfig, projector=None, orange_selector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_cc(cc_config, projector, orange_selector)

    def _initialize_cc(self, cc_config: MatchCCRuntimeConfig, projector, orange_selector=None) -> None:
        if not isinstance(cc_config, MatchCCRuntimeConfig) or not cc_config.enabled:
            raise ValueError("MatchCCSequence requires enabled match_cc config.")
        self.cc_config = cc_config
        self._cc_projector: GroundProjector | None = projector
        self._cc_orange_selector: NearFieldGraspSelector | None = orange_selector
        self._cc_mode = "initial_cluster"
        self._cc_last_align_frame: int | None = None
        self._cc_stable_frames = 0
        self._cc_motion_base_m: float | None = None
        self._cc_motion_distance_m = 0.0
        self._cc_command_ns: int | None = None
        self._cc_selected_class: TargetClass | None = None
        self._cc_gripper_angles_deg: tuple[float, float] | None = None
        self._cc_scan_last_heading: float | None = None
        self._cc_scan_progress_rad = 0.0

    @classmethod
    def from_app_config(
        cls,
        config: AppConfig,
        *,
        start_area: MatchStartArea | str | int = MatchStartArea.AREA_2,
    ) -> "MatchCCSequence":
        configured = configure_match_start_area(config, start_area)
        # CC 约定伤员运输终点的场地绝对 x 始终为正；y 和其它路线仍按
        # start-area 使用正式流程的中心对称变换。
        injured = configured.match.safe_zone_injured_target_field
        configured = replace(
            configured,
            match=replace(
                configured.match,
                safe_zone_injured_target_field=type(injured)(abs(injured.x), injured.y),
            ),
        )
        base = MatchSequence.from_app_config(configured)
        geometry = configured.build_geometry()
        projector = None if geometry is None else geometry.ground_projector
        gripper = configured.motion.gripper.build_calibration()
        if projector is None or gripper is None:
            raise RuntimeError("match_cc requires ground projection and gripper calibration.")
        from rescue_vision.motion import GripperKinematics
        orange_selector = NearFieldGraspSelector(
            configured.near_field_grasp,
            projector,
            GripperKinematics(),
            target_geometry=configured.perception.target_ground_geometry,
            open_servo_angles_deg=(gripper.open_left_angle_deg, gripper.open_right_angle_deg),
            closed_servo_angles_deg=(gripper.closed_left_angle_deg, gripper.closed_right_angle_deg),
        )
        sequence = cls.__new__(cls)
        sequence.__dict__.update(base.__dict__)
        sequence._initialize_cc(configured.match_cc, projector, orange_selector)
        return sequence

    _greedy_pickup_enabled = False

    @property
    def near_field_policy(self) -> NearFieldGraspPolicy:
        if self._cc_selected_class is TargetClass.ORANGE_INJURED:
            return NearFieldGraspPolicy(frozenset((TargetClass.ORANGE_INJURED,)), 1)
        return NearFieldGraspPolicy(frozenset((TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE)), 1)

    def _begin_cluster_search(self) -> None:
        super()._begin_cluster_search()
        self._cc_scan_last_heading = None
        self._cc_scan_progress_rad = 0.0
        self._cc_stable_frames = 0
        if self._transport_count == 0 and self._cc_mode != "after_breakup_green":
            self._cc_mode = "initial_cluster"
        elif self._transport_count > 0:
            self._cc_mode = "orange"

    def _fresh_cc_tracks(self, timestamp_ns: int) -> tuple[TrackedTarget, ...]:
        return tuple(
            target
            for target in self._tracker.tracks
            if target.ever_confirmed
            and target.ground_point is not None
            and self._target_is_fresh(target, timestamp_ns)
        )

    def _track(self, track_id: int, timestamp_ns: int) -> TrackedTarget | None:
        return next(
            (target for target in self._fresh_cc_tracks(timestamp_ns) if target.track_id == track_id),
            None,
        )

    def _update_scan_progress(self, heading_rad: float | None, velocity: float) -> bool:
        if heading_rad is None:
            return False
        previous = self._cc_scan_last_heading
        self._cc_scan_last_heading = heading_rad
        if previous is not None:
            self._cc_scan_progress_rad += self._directional_delta(previous, heading_rad, velocity)
        return self._cc_scan_progress_rad >= self.config.spin_angle_rad

    def _start_scan(self, mode: str, state: MatchState) -> None:
        self._cc_mode = mode
        self._cc_scan_last_heading = None
        self._cc_scan_progress_rad = 0.0
        self._selected_track_id = None
        self.state = state

    def _alignment_command(self, y_mm: float) -> float:
        limit = self.cc_config.alignment_max_angular_velocity_rad_s
        return max(-limit, min(limit, self.cc_config.alignment_kp_rad_s_per_mm * y_mm))

    def _orange_envelope(self, target: TrackedTarget) -> TargetGroundEnvelope | None:
        snapshot = self._latest_perception
        projector = self._cc_projector
        if (
            snapshot is None
            or projector is None
            or target.frame_sequence != snapshot.frame_sequence
            or target.missed_count != 0
        ):
            return None
        observations = tuple(
            item
            for item in snapshot.observations
            if item.target_class is TargetClass.ORANGE_INJURED and item.ground_point is not None
        )
        if target.ground_point is None or not observations:
            return None
        observation = min(
            observations,
            key=lambda item: math.hypot(
                item.ground_point.x - target.ground_point.x,
                item.ground_point.y - target.ground_point.y,
            ),
        )
        return measure_target_envelope(
            observation,
            projector,
            min_mask_pixels=self._near_field_grasp_config.min_mask_pixels,
        )

    def _orange_width_mm(self, target: TrackedTarget) -> float | None:
        envelope = self._orange_envelope(target)
        if envelope is None:
            return None
        ys = [point.y for point in envelope.corners]
        return max(ys) - min(ys)

    def _select_isolated(self, timestamp_ns: int, mode: str) -> TrackedTarget | None:
        tracks = self._fresh_cc_tracks(timestamp_ns)
        if mode == "orange":
            widths = {target.track_id: self._orange_width_mm(target) for target in tracks}
            choices = isolated_targets(
                tracks,
                allowed_classes=frozenset((TargetClass.ORANGE_INJURED,)),
                clearance_for_target=lambda target: widths[target.track_id] or -1.0,
            )
            return min(choices, key=lambda item: math.hypot(item.ground_point.x, item.ground_point.y), default=None)
        choices = isolated_targets(
            tracks,
            allowed_classes=frozenset((TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE))
            if mode == "supply"
            else frozenset((TargetClass.GREEN_SUPPLY,)),
            clearance_for_target=lambda _target: self.cc_config.isolated_line_clearance_mm,
        )
        return min(
            choices,
            key=lambda item: (
                0 if mode == "supply" and item.target_class is TargetClass.BLACK_CORE else 1,
                math.hypot(item.ground_point.x, item.ground_point.y),
                item.track_id,
            ),
            default=None,
        )

    def _cluster_ground_measurement(
        self,
        timestamp_ns: int,
    ) -> _GroundClusterMeasurement | None:
        """按 CC 严格团规则每帧重新成团，不绑定首次 tracker ID。"""

        self._cluster_selected_track_ids = ()
        self._last_cluster_rejection_reason = None
        clusters = find_cc_clusters(
            self._fresh_cc_tracks(timestamp_ns),
            neighbor_distance_mm=self.cc_config.cluster_neighbor_distance_mm,
        )
        if self._cc_mode == "mixed_cluster":
            priority = {
                TargetClass.ORANGE_INJURED: 0,
                TargetClass.BLACK_CORE: 1,
                TargetClass.GREEN_SUPPLY: 2,
            }
            clusters = tuple(
                sorted(
                    clusters,
                    key=lambda cluster: (
                        min(priority[item] for item in cluster.classes),
                        math.hypot(cluster.center.x, cluster.center.y),
                        cluster.member_ids,
                    ),
                )
            )
        for cluster in clusters:
            center = cluster.center
            forward_x = [point.x for point in cluster.points if point.x > 0.0]
            if not forward_x:
                continue
            field_center = self._field_point_from_ground(center)
            if not self._breakup_center_within_field_boundary(field_center):
                self._last_cluster_rejection_reason = self._breakup_center_boundary_reason(field_center)
                continue
            if self._candidate_path_blocked(center, breakup=True):
                self._last_cluster_rejection_reason = "breakup_candidate_path_blocked"
                continue
            self._cluster_selected_track_ids = cluster.member_ids
            return _GroundClusterMeasurement(center, min(forward_x))
        return None

    def _begin_original_cluster_reference(
        self,
        timestamp_ns: int,
        heading_rad: float | None,
        measurement: _GroundClusterMeasurement,
    ) -> MatchDecision:
        self._cluster_reference_samples = [measurement.center]
        self._cluster_distance_samples_mm = [measurement.nearest_forward_x_mm]
        self._cluster_capture_heading_rad = heading_rad
        self._cluster_align_hold_center = None
        self._cluster_align_lost_since_ns = None
        self._breakup_only = False
        self.state = MatchState.ALIGN_CLUSTER_ONCE
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "cc_cluster_seen_stop_collect_reference",
        )

    def _step_breakup_forward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        if cumulative_distance_m is None:
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_breakup_forward_waiting_for_odometry")
        if self._breakup_forward_base_distance_m is None:
            self._breakup_forward_base_distance_m = cumulative_distance_m
        if cumulative_distance_m - self._breakup_forward_base_distance_m >= self.config.breakup_forward_distance_m - 1e-9:
            self._cc_command_ns = timestamp_ns
            self.state = MatchState.CC_BREAKUP_OPEN_GAP
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_breakup_forward_complete_open", posture=GripperPosture.OPEN)
        return self._decision(timestamp_ns, self.config.breakup_forward_speed_m_s, 0.0, "cc_breakup_forward_fixed_distance")

    def _step_breakup_backward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        if cumulative_distance_m is None:
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_breakup_backward_waiting_for_odometry", posture=GripperPosture.OPEN)
        if self._breakup_backward_base_distance_m is None:
            self._breakup_backward_base_distance_m = cumulative_distance_m
        if self._breakup_backward_base_distance_m - cumulative_distance_m >= self.config.breakup_backward_distance_m - 1e-9:
            self._cc_command_ns = timestamp_ns
            self.state = MatchState.CC_BREAKUP_CLOSE_GAP
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_breakup_backward_complete_close", posture=GripperPosture.CLOSED)
        return self._decision(timestamp_ns, -self.config.breakup_backward_speed_m_s, 0.0, "cc_breakup_backward_fixed_distance", posture=GripperPosture.OPEN)

    def _step_actions(self, timestamp_ns: int, **kwargs) -> MatchDecision:
        shared_states = {
            MatchState.GATE_CLEARANCE,
            MatchState.STARTUP_TURN_RIGHT,
            MatchState.STARTUP_TURN_SETTLE,
            MatchState.STARTUP_FORWARD,
            MatchState.STARTUP_FORWARD_SETTLE,
            MatchState.ALIGN_CLUSTER_ONCE,
            MatchState.APPROACH_CLUSTER,
            MatchState.BREAKUP_SETTLE,
            MatchState.BREAKUP_FORWARD,
            MatchState.BREAKUP_BACKWARD,
            MatchState.TRANSPORT_NEAR_FIELD_GRASP,
            MatchState.TRANSPORT_ALIGN_RED_ZONE,
            MatchState.TRANSPORT_FORWARD,
            MatchState.TRANSPORT_RELEASE,
            MatchState.RETURN_BACKUP,
            MatchState.FINISH_STOP,
            MatchState.TERMINAL_STOP,
        }
        if self.state in shared_states:
            return super()._step_actions(timestamp_ns, **kwargs)
        if self.state is MatchState.SEARCH_CLUSTER and self._cc_mode != "orange":
            return super()._step_actions(timestamp_ns, **kwargs)

        self._validate_timestamp(timestamp_ns)
        heading_rad = kwargs.get("heading_rad")
        cumulative_distance_m = kwargs.get("cumulative_distance_m")
        self._update_fallback_field_position(heading_rad, cumulative_distance_m)
        safety = kwargs.get("safety") or SafetySignals.nominal(timestamp_ns)
        direct_reason = self._direct_safety_reason(safety)
        if direct_reason is not None:
            self.state = MatchState.TERMINAL_STOP
            return self._decision(timestamp_ns, 0.0, 0.0, direct_reason)
        self._update_tracker(timestamp_ns, kwargs.get("perception"))

        if self.state is MatchState.SEARCH_CLUSTER:
            self._start_scan("orange", MatchState.CC_SCAN_ORANGE)
            return self._decision(
                timestamp_ns,
                0.0,
                self.config.close_gripper_spin_angular_velocity_rad_s,
                "cc_safe_zone_exit_start_orange_scan",
                posture=GripperPosture.OPEN,
            )

        interval_ns = round(self.cc_config.command_interval_ms * 1e6)
        if self.state is MatchState.CC_BREAKUP_OPEN_GAP:
            command_ns = timestamp_ns if self._cc_command_ns is None else self._cc_command_ns
            if timestamp_ns - command_ns < interval_ns:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_open_command_gap", posture=GripperPosture.OPEN)
            self.state = MatchState.BREAKUP_BACKWARD
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_start_backward", posture=GripperPosture.OPEN)
        if self.state is MatchState.CC_BREAKUP_CLOSE_GAP:
            command_ns = timestamp_ns if self._cc_command_ns is None else self._cc_command_ns
            if timestamp_ns - command_ns < interval_ns:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_close_command_gap", posture=GripperPosture.CLOSED)
            if self._transport_count == 0:
                self._start_scan("green", MatchState.CC_SCAN_GREEN)
                reason = "cc_breakup_complete_search_green"
            else:
                self._start_scan("orange", MatchState.CC_SCAN_ORANGE)
                reason = "cc_breakup_complete_search_orange"
            return self._decision(timestamp_ns, 0.0, self.cc_config.target_search_angular_velocity_rad_s, reason)

        if self.state in {MatchState.CC_SCAN_GREEN, MatchState.CC_SCAN_ORANGE, MatchState.CC_SCAN_SUPPLY}:
            mode = {MatchState.CC_SCAN_GREEN: "green", MatchState.CC_SCAN_ORANGE: "orange", MatchState.CC_SCAN_SUPPLY: "supply"}[self.state]
            target = self._select_isolated(timestamp_ns, mode)
            if target is not None:
                self._selected_track_id = target.track_id
                self._cc_selected_class = target.target_class
                self._cc_stable_frames = 0
                self._cc_last_align_frame = None
                self.state = MatchState.CC_ALIGN_TARGET
                return self._decision(timestamp_ns, 0.0, 0.0, f"cc_{mode}_target_found:{target.track_id}")
            scan_velocity = (
                self.cc_config.target_search_angular_velocity_rad_s
                if mode == "green"
                else self.config.close_gripper_spin_angular_velocity_rad_s
            )
            completed = self._update_scan_progress(heading_rad, scan_velocity)
            if completed and mode == "orange":
                self._start_scan("supply", MatchState.CC_SCAN_SUPPLY)
                return self._decision(timestamp_ns, 0.0, self.config.close_gripper_spin_angular_velocity_rad_s, "cc_no_orange_search_supply")
            if completed and mode == "supply":
                self._start_scan("mixed_cluster", MatchState.CC_SCAN_MIXED_CLUSTER)
                return self._decision(timestamp_ns, 0.0, self.config.close_gripper_spin_angular_velocity_rad_s, "cc_no_supply_search_mixed_cluster")
            return self._decision(timestamp_ns, 0.0, scan_velocity, f"cc_searching_{mode}")

        if self.state is MatchState.CC_SCAN_MIXED_CLUSTER:
            measurement = self._cluster_ground_measurement(timestamp_ns)
            if measurement is not None:
                return self._begin_original_cluster_reference(
                    timestamp_ns,
                    heading_rad,
                    measurement,
                )
            self._update_scan_progress(heading_rad, self.config.close_gripper_spin_angular_velocity_rad_s)
            return self._decision(timestamp_ns, 0.0, self.config.close_gripper_spin_angular_velocity_rad_s, "cc_searching_mixed_cluster")

        if self.state is MatchState.CC_ALIGN_TARGET:
            target = None if self._selected_track_id is None else self._track(self._selected_track_id, timestamp_ns)
            if target is None or target.ground_point is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_target_temporarily_lost")
            point = target.ground_point
            frame = self._last_tracker_frame_sequence
            if abs(point.y) <= self.cc_config.block_alignment_tolerance_mm:
                if frame is not None and frame != self._cc_last_align_frame:
                    self._cc_stable_frames += 1
                    self._cc_last_align_frame = frame
                if self._cc_stable_frames >= self.cc_config.alignment_stable_frames:
                    if point.x < self.cc_config.near_field_threshold_mm:
                        return self._begin_cc_pickup(timestamp_ns, point, cumulative_distance_m)
                    self._cc_motion_base_m = cumulative_distance_m
                    self.state = MatchState.CC_APPROACH_TARGET
                    return self._decision(timestamp_ns, 0.0, 0.0, "cc_target_aligned_start_live_approach")
            else:
                self._cc_stable_frames = 0
            return self._decision(timestamp_ns, 0.0, self._alignment_command(point.y), f"cc_align_target_y_mm={point.y:.1f}")

        if self.state is MatchState.CC_APPROACH_TARGET:
            if self._cc_mode == "green_forward":
                return self._step_green_forward(timestamp_ns, heading_rad, cumulative_distance_m)
            target = None if self._selected_track_id is None else self._track(self._selected_track_id, timestamp_ns)
            if target is None or target.ground_point is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_live_approach_target_lost")
            if target.ground_point.x < self.cc_config.near_field_threshold_mm:
                return self._begin_cc_pickup(timestamp_ns, target.ground_point, cumulative_distance_m)
            return self._decision(timestamp_ns, self.cc_config.target_approach_speed_m_s, 0.0, f"cc_live_approach_x_mm={target.ground_point.x:.1f}")

        if self.state is MatchState.CC_GREEN_CLOSE_GAP:
            command_ns = timestamp_ns if self._cc_command_ns is None else self._cc_command_ns
            if timestamp_ns - command_ns < interval_ns:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_green_close_command_gap", posture=GripperPosture.CLOSED)
            self._transport_target_classes = (self._cc_selected_class or TargetClass.GREEN_SUPPLY,)
            return self._start_safe_zone_transport(timestamp_ns, transport_opened=False, posture=GripperPosture.CLOSED, reason="cc_green_closed_start_safe_zone_transport")

        if self.state is MatchState.CC_ORANGE_OPEN_GAP:
            command_ns = timestamp_ns if self._cc_command_ns is None else self._cc_command_ns
            if timestamp_ns - command_ns < interval_ns:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_orange_open_command_gap", gripper_angles_deg=self._cc_gripper_angles_deg)
            self._cc_motion_base_m = cumulative_distance_m
            self.state = MatchState.CC_ORANGE_FORWARD
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_orange_start_forward", gripper_angles_deg=self._cc_gripper_angles_deg)

        if self.state is MatchState.CC_ORANGE_FORWARD:
            if cumulative_distance_m is None or self._cc_motion_base_m is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_orange_waiting_for_odometry", gripper_angles_deg=self._cc_gripper_angles_deg)
            if abs(cumulative_distance_m - self._cc_motion_base_m) >= self._cc_motion_distance_m:
                self._cc_command_ns = timestamp_ns
                self.state = MatchState.CC_ORANGE_CLOSE_GAP
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_orange_forward_complete_close", posture=GripperPosture.CLOSED)
            return self._decision(timestamp_ns, self.cc_config.target_approach_speed_m_s, 0.0, "cc_orange_fixed_forward", gripper_angles_deg=self._cc_gripper_angles_deg)

        if self.state is MatchState.CC_ORANGE_CLOSE_GAP:
            command_ns = timestamp_ns if self._cc_command_ns is None else self._cc_command_ns
            if timestamp_ns - command_ns < interval_ns:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_orange_close_command_gap", posture=GripperPosture.CLOSED)
            self._transport_target_classes = (TargetClass.ORANGE_INJURED,)
            return self._start_safe_zone_transport(timestamp_ns, transport_opened=False, posture=GripperPosture.CLOSED, reason="cc_orange_closed_start_safe_zone_transport")

        return self._decision(timestamp_ns, 0.0, 0.0, f"cc_unhandled_state:{self.state.value}")

    def _begin_cc_pickup(self, timestamp_ns: int, point: GroundPoint, cumulative_distance_m: float | None) -> MatchDecision:
        if self._cc_selected_class is TargetClass.ORANGE_INJURED:
            target = None if self._selected_track_id is None else self._track(self._selected_track_id, timestamp_ns)
            envelope = None if target is None else self._orange_envelope(target)
            selector = self._cc_orange_selector
            if envelope is None or selector is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_orange_waiting_for_width_measurement")
            xs = [corner.x for corner in envelope.corners]
            ys = [corner.y for corner in envelope.corners]
            clearance_half = self._near_field_grasp_config.clearance_mm / 2.0
            try:
                self._cc_gripper_angles_deg = selector.servo_angles_for_edges(
                    max(ys) + clearance_half,
                    min(ys) - clearance_half,
                )
            except ValueError:
                return self._decision(timestamp_ns, 0.0, 0.0, "cc_orange_width_exceeds_gripper")
            # 与 gripper_width 的定义一致：center_x + d，即颜色包络最前端 x。
            self._cc_motion_distance_m = max(0.0, max(xs) / 1000.0)
            self._cc_motion_base_m = cumulative_distance_m
            self._cc_command_ns = timestamp_ns
            self.state = MatchState.CC_ORANGE_OPEN_GAP
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_orange_asymmetric_gripper_once", gripper_angles_deg=self._cc_gripper_angles_deg)
        self._cc_mode = "green_forward"
        self._cc_motion_distance_m = max(0.0, (point.x - self.cc_config.green_target_final_x_mm) / 1000.0)
        self._cc_motion_base_m = cumulative_distance_m
        self._cc_command_ns = timestamp_ns
        self.state = MatchState.CC_APPROACH_TARGET
        return self._decision(timestamp_ns, 0.0, 0.0, "cc_green_transport_posture", posture=GripperPosture.TRANSPORT)

    def _step_green_forward(self, timestamp_ns: int, heading_rad: float | None, cumulative_distance_m: float | None) -> MatchDecision:
        interval_ns = round(self.cc_config.command_interval_ms * 1e6)
        command_ns = timestamp_ns if self._cc_command_ns is None else self._cc_command_ns
        if timestamp_ns - command_ns < interval_ns:
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_green_transport_command_gap", posture=GripperPosture.TRANSPORT)
        if cumulative_distance_m is None or self._cc_motion_base_m is None:
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_green_waiting_for_odometry", posture=GripperPosture.TRANSPORT)
        if abs(cumulative_distance_m - self._cc_motion_base_m) >= self._cc_motion_distance_m:
            self._cc_command_ns = timestamp_ns
            self._transport_target_classes = (self._cc_selected_class or TargetClass.GREEN_SUPPLY,)
            self.state = MatchState.CC_GREEN_CLOSE_GAP
            return self._decision(timestamp_ns, 0.0, 0.0, "cc_green_forward_complete_close", posture=GripperPosture.CLOSED)
        return self._decision(timestamp_ns, self.cc_config.target_approach_speed_m_s, 0.0, "cc_green_fixed_forward", posture=GripperPosture.TRANSPORT)


def _build_sequence(config: AppConfig, *, start_area=MatchStartArea.AREA_2) -> MatchCCSequence:
    """按父类权威装配后，以相同依赖构造 CC 序列。"""

    return MatchCCSequence.from_app_config(config, start_area=start_area)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent CC rescue match flow.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-area", choices=("2", "3"), default="2")
    parser.add_argument("--supervised-physical-stop-ready", action="store_true")
    parser.add_argument("--local-preview", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--observer-image-interval-seconds", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if not math.isfinite(args.observer_image_interval_seconds) or args.observer_image_interval_seconds <= 0.0:
        parser.error("--observer-image-interval-seconds must be positive")
    _run_hardware(
        args.config,
        supervised_stop_ready=args.supervised_physical_stop_ready,
        local_preview=args.local_preview,
        jpeg_quality=args.jpeg_quality,
        observer_image_interval_s=args.observer_image_interval_seconds,
        log_dir=args.log_dir,
        start_area=MatchStartArea.parse(args.start_area),
        sequence_factory=lambda config: _build_sequence(config),
        mode_name="match_cc",
        log_file_prefix="match_cc_",
        preview_title="Match CC perception",
    )


__all__ = [
    "CCCluster",
    "MatchCCSequence",
    "find_cc_clusters",
    "isolated_targets",
    "point_to_origin_segment_distance_mm",
]


if __name__ == "__main__":
    main()
