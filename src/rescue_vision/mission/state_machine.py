"""集中实现智能救援初赛规则和保守降级策略。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from rescue_vision.perception import TargetClass
from rescue_vision.tracking import TrackStatus
from rescue_vision.world import (
    HazardState,
    RegionKind,
    WorldSnapshot,
    WorldTarget,
    WorldUncertainty,
)


class MissionPhase(str, Enum):
    WAIT_START = "wait_start"
    FIRST_NORMAL_REQUIRED = "first_normal_required"
    GENERAL_RESCUE = "general_rescue"
    ENDED = "ended"


class ActivityState(str, Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    APPROACHING = "approaching"
    PUSHING = "pushing"
    VERIFYING_DELIVERY = "verifying_delivery"
    AVOIDING = "avoiding"
    SAFETY_HOLD = "safety_hold"
    STOPPED = "stopped"


class AbstractAction(str, Enum):
    SEARCH = "search"
    APPROACH = "approach"
    PUSH = "push"
    AVOID = "avoid"
    DELIVER = "deliver"
    STOP = "stop"


class DeliveryDestination(str, Enum):
    OWN_MATERIAL = "own_material"
    OWN_INJURED = "own_injured"
    OPPONENT_SAFE = "opponent_safe"
    OUT_OF_FIELD = "out_of_field"


class TerminationReason(str, Enum):
    EXTERNAL_STOP = "external_stop"
    SAFETY_ACCIDENT = "safety_accident"
    LOST_CONTROL = "lost_control"
    HUMAN_TOUCHED_AFTER_START = "human_touched_after_start"
    ILLEGAL_CARRY = "illegal_carry"
    ACTIVE_ATTACK = "active_attack"
    DANGER_ENGAGED = "danger_engaged"
    DANGER_IN_SAFE_ZONE = "danger_in_safe_zone"
    DANGER_OUT_OF_FIELD = "danger_out_of_field"
    INVALID_FIRST_DELIVERY = "invalid_first_delivery"
    TOO_MANY_TARGETS = "too_many_targets"
    INJURED_MIXED_TRANSPORT = "injured_mixed_transport"
    NO_MOTION_TIMEOUT = "no_motion_timeout"
    MATCH_TIMEOUT = "match_timeout"


def _positive_finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return converted


@dataclass(frozen=True, slots=True)
class MissionConfig:
    match_duration_s: float
    no_motion_timeout_s: float
    opponent_contact_timeout_s: float
    target_priority: tuple[TargetClass, ...]

    def __post_init__(self) -> None:
        for name in (
            "match_duration_s",
            "no_motion_timeout_s",
            "opponent_contact_timeout_s",
        ):
            _positive_finite(getattr(self, name), name)
        allowed = {
            TargetClass.GREEN_SUPPLY,
            TargetClass.BLACK_CORE,
            TargetClass.ORANGE_INJURED,
        }
        if (
            not self.target_priority
            or len(set(self.target_priority)) != len(self.target_priority)
            or set(self.target_priority) != allowed
        ):
            raise ValueError(
                "target_priority must contain green_supply, black_core and "
                "orange_injured exactly once."
            )

    def build_state_machine(self) -> MissionStateMachine:
        return MissionStateMachine(self)


@dataclass(frozen=True, slots=True)
class TransportStatus:
    engaged_track_ids: tuple[int, ...] = ()
    contact_started_ns: int | None = None

    def __post_init__(self) -> None:
        if (
            len(set(self.engaged_track_ids)) != len(self.engaged_track_ids)
            or any(
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id <= 0
                for track_id in self.engaged_track_ids
            )
        ):
            raise ValueError(
                "engaged_track_ids must contain unique positive integers."
            )
        if bool(self.engaged_track_ids) != (self.contact_started_ns is not None):
            raise ValueError(
                "contact_started_ns is required exactly when targets are engaged."
            )
        if (
            self.contact_started_ns is not None
            and (
                isinstance(self.contact_started_ns, bool)
                or not isinstance(self.contact_started_ns, int)
                or self.contact_started_ns < 0
            )
        ):
            raise ValueError("contact_started_ns must be non-negative.")


@dataclass(frozen=True, slots=True)
class DeliveryEvidence:
    delivery_id: str
    track_ids: tuple[int, ...]
    destination: DeliveryDestination
    fully_entered: bool

    def __post_init__(self) -> None:
        if not isinstance(self.delivery_id, str) or not self.delivery_id.strip():
            raise ValueError("delivery_id must be a non-empty string.")
        if (
            not self.track_ids
            or len(set(self.track_ids)) != len(self.track_ids)
            or any(
                isinstance(track_id, bool)
                or not isinstance(track_id, int)
                or track_id <= 0
                for track_id in self.track_ids
            )
        ):
            raise ValueError("track_ids must contain unique positive integers.")
        if not isinstance(self.destination, DeliveryDestination):
            raise ValueError("destination must be a DeliveryDestination.")
        if not isinstance(self.fully_entered, bool):
            raise ValueError("fully_entered must be a boolean.")


@dataclass(frozen=True, slots=True)
class SafetySignals:
    last_motion_timestamp_ns: int
    opponent_contact_since_ns: int | None = None
    active_attack: bool = False
    lost_control: bool = False
    safety_accident: bool = False
    human_touched_after_start: bool = False
    target_carried_on_robot: bool = False
    external_stop_requested: bool = False

    def __post_init__(self) -> None:
        for name in ("last_motion_timestamp_ns", "opponent_contact_since_ns"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        for name in (
            "active_attack",
            "lost_control",
            "safety_accident",
            "human_touched_after_start",
            "target_carried_on_robot",
            "external_stop_requested",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

    @classmethod
    def nominal(cls, timestamp_ns: int) -> SafetySignals:
        return cls(last_motion_timestamp_ns=timestamp_ns)


@dataclass(frozen=True, slots=True)
class MissionProgress:
    delivered_green_supply: int = 0
    delivered_black_core: int = 0
    delivered_orange_injured: int = 0
    wrong_zone_targets: int = 0
    opponent_zone_targets: int = 0
    out_of_field_targets: int = 0


@dataclass(frozen=True, slots=True)
class MissionDecision:
    timestamp_ns: int
    phase: MissionPhase
    activity: ActivityState
    action: AbstractAction
    target_track_id: int | None
    reason: str
    terminal: bool
    termination_reason: TerminationReason | None


@dataclass(frozen=True, slots=True)
class MissionReplayStep:
    snapshot: WorldSnapshot
    transport: TransportStatus
    safety: SafetySignals
    delivery: DeliveryEvidence | None = None


def replay_mission(
    config: MissionConfig,
    *,
    start_timestamp_ns: int,
    steps: tuple[MissionReplayStep, ...] | list[MissionReplayStep],
) -> tuple[MissionDecision, ...]:
    """从空状态重放一轮合成或记录事件，包括启动决策。"""

    machine = config.build_state_machine()
    decisions = [machine.start(start_timestamp_ns)]
    decisions.extend(
        machine.step(
            step.snapshot,
            transport=step.transport,
            safety=step.safety,
            delivery=step.delivery,
        )
        for step in steps
    )
    return tuple(decisions)


@dataclass(frozen=True, slots=True)
class _DeliveryRecord:
    signature: tuple[tuple[int, ...], DeliveryDestination, bool]
    activity: ActivityState
    action: AbstractAction
    reason: str
    target_track_id: int | None


class MissionStateMachine:
    """事件时间驱动且可重放的规则状态机。"""

    def __init__(self, config: MissionConfig) -> None:
        self._config = config
        self._phase = MissionPhase.WAIT_START
        self._activity = ActivityState.IDLE
        self._termination_reason: TerminationReason | None = None
        self._start_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None
        self._progress = MissionProgress()
        self._processed_deliveries: dict[str, _DeliveryRecord] = {}

    @property
    def phase(self) -> MissionPhase:
        return self._phase

    @property
    def activity(self) -> ActivityState:
        return self._activity

    @property
    def progress(self) -> MissionProgress:
        return self._progress

    @property
    def termination_reason(self) -> TerminationReason | None:
        return self._termination_reason

    def start(self, timestamp_ns: int) -> MissionDecision:
        self._validate_timestamp(timestamp_ns)
        if self._phase is not MissionPhase.WAIT_START:
            raise RuntimeError("Mission can only start from WAIT_START.")
        self._start_timestamp_ns = timestamp_ns
        self._last_timestamp_ns = timestamp_ns
        self._phase = MissionPhase.FIRST_NORMAL_REQUIRED
        self._activity = ActivityState.SEARCHING
        return self._decision(
            timestamp_ns,
            AbstractAction.SEARCH,
            "mission_started",
        )

    def step(
        self,
        snapshot: WorldSnapshot,
        *,
        transport: TransportStatus,
        safety: SafetySignals,
        delivery: DeliveryEvidence | None = None,
    ) -> MissionDecision:
        timestamp_ns = snapshot.timestamp_ns
        self._validate_timestamp(timestamp_ns)
        if self._phase is MissionPhase.WAIT_START:
            self._last_timestamp_ns = timestamp_ns
            return self._set_activity(
                timestamp_ns,
                ActivityState.IDLE,
                AbstractAction.STOP,
                "waiting_start",
            )
        if self._phase is MissionPhase.ENDED:
            self._last_timestamp_ns = timestamp_ns
            return self._decision(
                timestamp_ns,
                AbstractAction.STOP,
                "mission_already_ended",
            )
        assert self._start_timestamp_ns is not None
        self._validate_inputs(timestamp_ns, transport, safety)
        self._last_timestamp_ns = timestamp_ns

        direct_reason = self._direct_termination_reason(safety)
        if direct_reason is not None:
            return self._terminate(timestamp_ns, direct_reason)

        elapsed_s = (
            timestamp_ns - self._start_timestamp_ns
        ) / 1_000_000_000.0
        if elapsed_s >= self._config.match_duration_s:
            return self._terminate(
                timestamp_ns,
                TerminationReason.MATCH_TIMEOUT,
            )
        still_s = (
            timestamp_ns - safety.last_motion_timestamp_ns
        ) / 1_000_000_000.0
        if still_s >= self._config.no_motion_timeout_s:
            return self._terminate(
                timestamp_ns,
                TerminationReason.NO_MOTION_TIMEOUT,
            )

        delivery_signature = None
        if delivery is not None:
            delivery_signature = (
                delivery.track_ids,
                delivery.destination,
                delivery.fully_entered,
            )
            previous = self._processed_deliveries.get(delivery.delivery_id)
            if previous is not None:
                if previous.signature != delivery_signature:
                    raise ValueError(
                        f"delivery_id {delivery.delivery_id!r} was reused with "
                        "different evidence."
                    )
                return self._set_activity(
                    timestamp_ns,
                    previous.activity,
                    previous.action,
                    previous.reason,
                    previous.target_track_id,
                )

        delivery_targets: tuple[WorldTarget, ...] | None = None
        if delivery is not None:
            delivery_targets = self._resolve_targets(
                snapshot,
                delivery.track_ids,
            )
            if delivery_targets is not None:
                danger = next(
                    (
                        target
                        for target in delivery_targets
                        if target.hazard_state is HazardState.CONFIRMED
                    ),
                    None,
                )
                if danger is not None:
                    reason = (
                        TerminationReason.DANGER_OUT_OF_FIELD
                        if delivery.destination
                        is DeliveryDestination.OUT_OF_FIELD
                        else TerminationReason.DANGER_IN_SAFE_ZONE
                    )
                    return self._terminate(timestamp_ns, reason)
                delivery_violation = self._transport_violation(
                    delivery_targets
                )
                if delivery_violation is not None:
                    return self._terminate(
                        timestamp_ns,
                        delivery_violation,
                    )

        transport_targets = self._resolve_targets(
            snapshot,
            transport.engaged_track_ids,
        )
        if len(transport.engaged_track_ids) > 3:
            return self._terminate(
                timestamp_ns,
                TerminationReason.TOO_MANY_TARGETS,
            )
        if transport.engaged_track_ids and transport_targets is None:
            return self._hold(
                timestamp_ns,
                "transport_target_missing",
            )
        assert transport_targets is not None
        transport_violation = self._transport_violation(transport_targets)
        if transport_violation is not None:
            return self._terminate(timestamp_ns, transport_violation)
        if any(
            target.hazard_state is HazardState.SUSPECTED
            for target in transport_targets
        ):
            return self._hold(timestamp_ns, "suspected_danger_engaged")

        if WorldUncertainty.STALE_VISION in snapshot.uncertainties:
            return self._hold(timestamp_ns, "stale_vision")
        if (
            WorldUncertainty.MISSING_ROBOT_FIELD_POSITION
            in snapshot.uncertainties
            or snapshot.robot_field_point is None
        ):
            return self._hold(timestamp_ns, "robot_field_position_missing")

        if safety.opponent_contact_since_ns is not None:
            contact_s = (
                timestamp_ns - safety.opponent_contact_since_ns
            ) / 1_000_000_000.0
            if contact_s >= self._config.opponent_contact_timeout_s:
                return self._set_activity(
                    timestamp_ns,
                    ActivityState.AVOIDING,
                    AbstractAction.STOP,
                    "opponent_contact_timeout",
                )
        if snapshot.robot_in_region(RegionKind.OPPONENT_SAFE):
            return self._avoid(timestamp_ns, "robot_in_opponent_safe")
        if (
            snapshot.robot_field_point is not None
            and any(
                occupancy.contains(snapshot.robot_field_point)
                for occupancy in snapshot.opponent_occupancies
            )
        ):
            return self._avoid(timestamp_ns, "opponent_occupancy")
        if delivery is not None:
            if delivery_targets is None:
                return self._hold(timestamp_ns, "delivery_target_missing")
            if any(
                target.hazard_state is HazardState.SUSPECTED
                for target in delivery_targets
            ):
                return self._hold(
                    timestamp_ns,
                    "suspected_danger_delivery",
                )
            if (
                transport.engaged_track_ids
                and set(transport.engaged_track_ids) != set(delivery.track_ids)
            ):
                return self._hold(timestamp_ns, "delivery_transport_mismatch")
            result = self._process_delivery(
                timestamp_ns,
                delivery,
                delivery_targets,
            )
            if result is not None:
                assert delivery_signature is not None
                self._processed_deliveries[delivery.delivery_id] = (
                    _DeliveryRecord(
                        signature=delivery_signature,
                        activity=result.activity,
                        action=result.action,
                        reason=result.reason,
                        target_track_id=result.target_track_id,
                    )
                )
                return result

        if transport_targets:
            target = transport_targets[0]
            destination = self._destination_for(transport_targets)
            if (
                destination is DeliveryDestination.OWN_MATERIAL
                and snapshot.robot_in_region(RegionKind.OWN_MATERIAL)
            ) or (
                destination is DeliveryDestination.OWN_INJURED
                and snapshot.robot_in_region(RegionKind.OWN_INJURED)
            ):
                return self._set_activity(
                    timestamp_ns,
                    ActivityState.VERIFYING_DELIVERY,
                    AbstractAction.DELIVER,
                    "delivery_region_reached",
                    target.track_id,
                )
            return self._set_activity(
                timestamp_ns,
                ActivityState.PUSHING,
                AbstractAction.PUSH,
                "legal_transport",
                target.track_id,
            )

        candidate = self._select_candidate(snapshot)
        if candidate is None:
            return self._set_activity(
                timestamp_ns,
                ActivityState.SEARCHING,
                AbstractAction.SEARCH,
                "no_legal_candidate",
            )
        return self._set_activity(
            timestamp_ns,
            ActivityState.APPROACHING,
            AbstractAction.APPROACH,
            "target_selected",
            candidate.track_id,
        )

    def reset(self) -> None:
        if self._phase not in {MissionPhase.WAIT_START, MissionPhase.ENDED}:
            raise RuntimeError("An active mission cannot be reset.")
        self._phase = MissionPhase.WAIT_START
        self._activity = ActivityState.IDLE
        self._termination_reason = None
        self._start_timestamp_ns = None
        self._last_timestamp_ns = None
        self._progress = MissionProgress()
        self._processed_deliveries.clear()

    def _validate_timestamp(self, timestamp_ns: int) -> None:
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if (
            self._last_timestamp_ns is not None
            and timestamp_ns < self._last_timestamp_ns
        ):
            raise ValueError(
                f"timestamp_ns moved backwards from {self._last_timestamp_ns} "
                f"to {timestamp_ns}."
            )

    @staticmethod
    def _validate_inputs(
        timestamp_ns: int,
        transport: TransportStatus,
        safety: SafetySignals,
    ) -> None:
        if (
            transport.contact_started_ns is not None
            and transport.contact_started_ns > timestamp_ns
        ):
            raise ValueError("contact_started_ns must not be in the future.")
        if safety.last_motion_timestamp_ns > timestamp_ns:
            raise ValueError(
                "last_motion_timestamp_ns must not be in the future."
            )
        if (
            safety.opponent_contact_since_ns is not None
            and safety.opponent_contact_since_ns > timestamp_ns
        ):
            raise ValueError(
                "opponent_contact_since_ns must not be in the future."
            )

    @staticmethod
    def _direct_termination_reason(
        safety: SafetySignals,
    ) -> TerminationReason | None:
        ordered = (
            (safety.external_stop_requested, TerminationReason.EXTERNAL_STOP),
            (safety.safety_accident, TerminationReason.SAFETY_ACCIDENT),
            (safety.lost_control, TerminationReason.LOST_CONTROL),
            (
                safety.human_touched_after_start,
                TerminationReason.HUMAN_TOUCHED_AFTER_START,
            ),
            (
                safety.target_carried_on_robot,
                TerminationReason.ILLEGAL_CARRY,
            ),
            (safety.active_attack, TerminationReason.ACTIVE_ATTACK),
        )
        return next((reason for active, reason in ordered if active), None)

    @staticmethod
    def _resolve_targets(
        snapshot: WorldSnapshot,
        track_ids: tuple[int, ...],
    ) -> tuple[WorldTarget, ...] | None:
        targets = tuple(snapshot.target(track_id) for track_id in track_ids)
        if any(target is None for target in targets):
            return None
        return tuple(target for target in targets if target is not None)

    @staticmethod
    def _transport_violation(
        targets: tuple[WorldTarget, ...],
    ) -> TerminationReason | None:
        if len(targets) > 3:
            return TerminationReason.TOO_MANY_TARGETS
        if any(
            target.hazard_state is HazardState.CONFIRMED
            for target in targets
        ):
            return TerminationReason.DANGER_ENGAGED
        injured_count = sum(
            target.target_class is TargetClass.ORANGE_INJURED
            for target in targets
        )
        if injured_count and len(targets) != 1:
            return TerminationReason.INJURED_MIXED_TRANSPORT
        return None

    def _process_delivery(
        self,
        timestamp_ns: int,
        delivery: DeliveryEvidence,
        targets: tuple[WorldTarget, ...],
    ) -> MissionDecision | None:
        if delivery.destination is DeliveryDestination.OPPONENT_SAFE:
            self._progress = MissionProgress(
                delivered_green_supply=self._progress.delivered_green_supply,
                delivered_black_core=self._progress.delivered_black_core,
                delivered_orange_injured=self._progress.delivered_orange_injured,
                wrong_zone_targets=self._progress.wrong_zone_targets,
                opponent_zone_targets=(
                    self._progress.opponent_zone_targets + len(targets)
                ),
                out_of_field_targets=self._progress.out_of_field_targets,
            )
            return self._avoid(timestamp_ns, "targets_entered_opponent_safe")
        if delivery.destination is DeliveryDestination.OUT_OF_FIELD:
            self._progress = MissionProgress(
                delivered_green_supply=self._progress.delivered_green_supply,
                delivered_black_core=self._progress.delivered_black_core,
                delivered_orange_injured=self._progress.delivered_orange_injured,
                wrong_zone_targets=self._progress.wrong_zone_targets,
                opponent_zone_targets=self._progress.opponent_zone_targets,
                out_of_field_targets=(
                    self._progress.out_of_field_targets + len(targets)
                ),
            )
            return self._hold(timestamp_ns, "targets_out_of_field")

        classes = tuple(target.target_class for target in targets)
        correct_material = (
            delivery.destination is DeliveryDestination.OWN_MATERIAL
            and all(
                target_class
                in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE}
                for target_class in classes
            )
        )
        correct_injured = (
            delivery.destination is DeliveryDestination.OWN_INJURED
            and classes == (TargetClass.ORANGE_INJURED,)
        )
        if self._phase is MissionPhase.FIRST_NORMAL_REQUIRED:
            if not (
                correct_material
                and classes == (TargetClass.GREEN_SUPPLY,)
            ):
                return self._terminate(
                    timestamp_ns,
                    TerminationReason.INVALID_FIRST_DELIVERY,
                )
            if not delivery.fully_entered:
                return self._set_activity(
                    timestamp_ns,
                    ActivityState.VERIFYING_DELIVERY,
                    AbstractAction.DELIVER,
                    "delivery_not_fully_entered",
                    targets[0].track_id,
                )
            self._phase = MissionPhase.GENERAL_RESCUE
        elif not (correct_material or correct_injured):
            self._progress = MissionProgress(
                delivered_green_supply=self._progress.delivered_green_supply,
                delivered_black_core=self._progress.delivered_black_core,
                delivered_orange_injured=self._progress.delivered_orange_injured,
                wrong_zone_targets=(
                    self._progress.wrong_zone_targets + len(targets)
                ),
                opponent_zone_targets=self._progress.opponent_zone_targets,
                out_of_field_targets=self._progress.out_of_field_targets,
            )
            return self._set_activity(
                timestamp_ns,
                ActivityState.SEARCHING,
                AbstractAction.SEARCH,
                "wrong_own_zone",
            )
        elif not delivery.fully_entered:
            return self._set_activity(
                timestamp_ns,
                ActivityState.VERIFYING_DELIVERY,
                AbstractAction.DELIVER,
                "delivery_not_fully_entered",
                targets[0].track_id,
            )

        self._progress = MissionProgress(
            delivered_green_supply=(
                self._progress.delivered_green_supply
                + classes.count(TargetClass.GREEN_SUPPLY)
            ),
            delivered_black_core=(
                self._progress.delivered_black_core
                + classes.count(TargetClass.BLACK_CORE)
            ),
            delivered_orange_injured=(
                self._progress.delivered_orange_injured
                + classes.count(TargetClass.ORANGE_INJURED)
            ),
            wrong_zone_targets=self._progress.wrong_zone_targets,
            opponent_zone_targets=self._progress.opponent_zone_targets,
            out_of_field_targets=self._progress.out_of_field_targets,
        )
        return self._set_activity(
            timestamp_ns,
            ActivityState.SEARCHING,
            AbstractAction.SEARCH,
            "delivery_accepted",
        )

    def _select_candidate(
        self,
        snapshot: WorldSnapshot,
    ) -> WorldTarget | None:
        allowed = (
            {TargetClass.GREEN_SUPPLY}
            if self._phase is MissionPhase.FIRST_NORMAL_REQUIRED
            else {
                TargetClass.GREEN_SUPPLY,
                TargetClass.BLACK_CORE,
                TargetClass.ORANGE_INJURED,
            }
        )
        priority = {
            target_class: index
            for index, target_class in enumerate(self._config.target_priority)
        }
        candidates = [
            target
            for target in snapshot.targets
            if target.track_status is TrackStatus.CONFIRMED
            and target.hazard_state is HazardState.CLEAR
            and target.target_class in allowed
            and target.ground_point is not None
            and not self._target_is_in_excluded_area(snapshot, target)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (
                priority[target.target_class],
                math.hypot(target.ground_point.x, target.ground_point.y),  # type: ignore[union-attr]
                target.track_id,
            ),
        )

    @staticmethod
    def _target_is_in_excluded_area(
        snapshot: WorldSnapshot,
        target: WorldTarget,
    ) -> bool:
        region_kinds = snapshot.target_region_kinds(target.track_id)
        return (
            region_kinds is not None
            and bool(
                region_kinds
                & {
                    RegionKind.OWN_MATERIAL,
                    RegionKind.OWN_INJURED,
                    RegionKind.OPPONENT_SAFE,
                }
            )
        ) or snapshot.target_in_opponent_occupancy(target.track_id) is True

    @staticmethod
    def _destination_for(
        targets: tuple[WorldTarget, ...],
    ) -> DeliveryDestination:
        return (
            DeliveryDestination.OWN_INJURED
            if targets[0].target_class is TargetClass.ORANGE_INJURED
            else DeliveryDestination.OWN_MATERIAL
        )

    def _terminate(
        self,
        timestamp_ns: int,
        reason: TerminationReason,
    ) -> MissionDecision:
        self._phase = MissionPhase.ENDED
        self._activity = ActivityState.STOPPED
        self._termination_reason = reason
        return self._decision(
            timestamp_ns,
            AbstractAction.STOP,
            reason.value,
        )

    def _hold(self, timestamp_ns: int, reason: str) -> MissionDecision:
        return self._set_activity(
            timestamp_ns,
            ActivityState.SAFETY_HOLD,
            AbstractAction.STOP,
            reason,
        )

    def _avoid(self, timestamp_ns: int, reason: str) -> MissionDecision:
        return self._set_activity(
            timestamp_ns,
            ActivityState.AVOIDING,
            AbstractAction.AVOID,
            reason,
        )

    def _set_activity(
        self,
        timestamp_ns: int,
        activity: ActivityState,
        action: AbstractAction,
        reason: str,
        target_track_id: int | None = None,
    ) -> MissionDecision:
        self._activity = activity
        return self._decision(
            timestamp_ns,
            action,
            reason,
            target_track_id,
        )

    def _decision(
        self,
        timestamp_ns: int,
        action: AbstractAction,
        reason: str,
        target_track_id: int | None = None,
    ) -> MissionDecision:
        return MissionDecision(
            timestamp_ns=timestamp_ns,
            phase=self._phase,
            activity=self._activity,
            action=action,
            target_track_id=target_track_id,
            reason=reason,
            terminal=self._phase is MissionPhase.ENDED,
            termination_reason=self._termination_reason,
        )
