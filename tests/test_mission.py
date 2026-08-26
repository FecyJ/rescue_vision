from __future__ import annotations

from dataclasses import replace

import pytest

from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.mission import (
    AbstractAction,
    ActivityState,
    DeliveryDestination,
    DeliveryEvidence,
    MissionConfig,
    MissionPhase,
    MissionReplayStep,
    MissionStateMachine,
    SafetySignals,
    TerminationReason,
    TransportStatus,
    replay_mission,
)
from rescue_vision.perception import ClassProbabilities, TargetClass
from rescue_vision.tracking import TrackStatus
from rescue_vision.world import (
    HazardState,
    OpponentOccupancy,
    RegionKind,
    StaticRegion,
    WorldSnapshot,
    WorldTarget,
    WorldUncertainty,
)


def config(**overrides: object) -> MissionConfig:
    values = {
        "match_duration_s": 180.0,
        "no_motion_timeout_s": 15.0,
        "opponent_contact_timeout_s": 10.0,
        "target_priority": (
            TargetClass.ORANGE_INJURED,
            TargetClass.BLACK_CORE,
            TargetClass.GREEN_SUPPLY,
        ),
    }
    values.update(overrides)
    return MissionConfig(**values)  # type: ignore[arg-type]


def target(
    track_id: int,
    target_class: TargetClass,
    *,
    hazard_state: HazardState = HazardState.CLEAR,
    distance_mm: float = 500.0,
    status: TrackStatus = TrackStatus.CONFIRMED,
) -> WorldTarget:
    return WorldTarget(
        track_id=track_id,
        track_status=status,
        ever_confirmed=status is TrackStatus.CONFIRMED,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(
            target_class,
            0.9 if target_class is not TargetClass.UNKNOWN else 1.0,
        ),
        confidence=0.9,
        hazard_state=hazard_state,
        ground_point=GroundPoint(distance_mm, 0.0),
        field_point=None,
        last_seen_timestamp_ns=0,
    )


def snapshot(
    timestamp_ns: int,
    targets: tuple[WorldTarget, ...] = (),
    *,
    regions: frozenset[RegionKind] = frozenset(),
    uncertainties: frozenset[WorldUncertainty] = frozenset(),
    robot_field_point: FieldPoint | None = FieldPoint(-1000.0, -1000.0),
    static_regions: tuple[StaticRegion, ...] = (),
    opponent_occupancies: tuple[OpponentOccupancy, ...] = (),
) -> WorldSnapshot:
    return WorldSnapshot(
        timestamp_ns=timestamp_ns,
        visual_timestamp_ns=timestamp_ns,
        regions=static_regions,
        targets=targets,
        opponent_occupancies=opponent_occupancies,
        robot_field_point=robot_field_point,
        robot_region_kinds=regions,
        uncertainties=uncertainties,
    )


def step(
    machine: MissionStateMachine,
    timestamp_ns: int,
    targets: tuple[WorldTarget, ...] = (),
    *,
    transport_ids: tuple[int, ...] = (),
    delivery: DeliveryEvidence | None = None,
    safety: SafetySignals | None = None,
    regions: frozenset[RegionKind] = frozenset(),
    uncertainties: frozenset[WorldUncertainty] = frozenset(),
) :
    transport = TransportStatus(
        transport_ids,
        timestamp_ns if transport_ids else None,
    )
    return machine.step(
        snapshot(
            timestamp_ns,
            targets,
            regions=regions,
            uncertainties=uncertainties,
        ),
        transport=transport,
        delivery=delivery,
        safety=safety or SafetySignals.nominal(timestamp_ns),
    )


def started_machine() -> MissionStateMachine:
    machine = MissionStateMachine(config())
    machine.start(0)
    return machine


def delivery(
    delivery_id: str,
    track_ids: tuple[int, ...],
    destination: DeliveryDestination,
    *,
    fully_entered: bool = True,
) -> DeliveryEvidence:
    return DeliveryEvidence(
        delivery_id,
        track_ids,
        destination,
        fully_entered,
    )


def complete_first_delivery(
    machine: MissionStateMachine,
    timestamp_ns: int = 1,
) -> WorldTarget:
    green = target(1, TargetClass.GREEN_SUPPLY)
    decision = step(
        machine,
        timestamp_ns,
        (green,),
        delivery=delivery(
            "first",
            (1,),
            DeliveryDestination.OWN_MATERIAL,
        ),
    )
    assert decision.phase is MissionPhase.GENERAL_RESCUE
    return green


def test_normal_replay_opens_general_phase_and_selects_by_priority() -> None:
    machine = started_machine()
    green = target(1, TargetClass.GREEN_SUPPLY)
    black = target(2, TargetClass.BLACK_CORE)
    injured = target(3, TargetClass.ORANGE_INJURED)

    first_choice = step(machine, 1, (green, black, injured))
    accepted = step(
        machine,
        2,
        (green, black, injured),
        delivery=delivery(
            "first",
            (1,),
            DeliveryDestination.OWN_MATERIAL,
        ),
    )
    general_choice = step(machine, 3, (green, black, injured))

    assert first_choice.action is AbstractAction.APPROACH
    assert first_choice.target_track_id == 1
    assert accepted.phase is MissionPhase.GENERAL_RESCUE
    assert machine.progress.delivered_green_supply == 1
    assert general_choice.target_track_id == 3


@pytest.mark.parametrize(
    ("targets", "destination"),
    [
        ((target(2, TargetClass.BLACK_CORE),), DeliveryDestination.OWN_MATERIAL),
        ((target(3, TargetClass.ORANGE_INJURED),), DeliveryDestination.OWN_INJURED),
        (
            (
                target(1, TargetClass.GREEN_SUPPLY),
                target(4, TargetClass.GREEN_SUPPLY),
            ),
            DeliveryDestination.OWN_MATERIAL,
        ),
        ((target(1, TargetClass.GREEN_SUPPLY),), DeliveryDestination.OWN_INJURED),
    ],
)
def test_invalid_first_delivery_terminates(
    targets: tuple[WorldTarget, ...],
    destination: DeliveryDestination,
) -> None:
    machine = started_machine()
    decision = step(
        machine,
        1,
        targets,
        delivery=delivery(
            "invalid-first",
            tuple(item.track_id for item in targets),
            destination,
        ),
    )
    assert decision.terminal
    assert decision.termination_reason is TerminationReason.INVALID_FIRST_DELIVERY


def test_general_material_and_single_injured_deliveries_are_counted() -> None:
    machine = started_machine()
    complete_first_delivery(machine)
    green = target(2, TargetClass.GREEN_SUPPLY)
    black = target(3, TargetClass.BLACK_CORE)
    injured = target(4, TargetClass.ORANGE_INJURED)
    step(
        machine,
        2,
        (green, black),
        delivery=delivery(
            "materials",
            (2, 3),
            DeliveryDestination.OWN_MATERIAL,
        ),
    )
    step(
        machine,
        3,
        (injured,),
        delivery=delivery(
            "injured",
            (4,),
            DeliveryDestination.OWN_INJURED,
        ),
    )
    assert machine.progress.delivered_green_supply == 2
    assert machine.progress.delivered_black_core == 1
    assert machine.progress.delivered_orange_injured == 1


def test_injured_mixing_and_over_capacity_terminate() -> None:
    machine = started_machine()
    complete_first_delivery(machine)
    mixed = (
        target(2, TargetClass.ORANGE_INJURED),
        target(3, TargetClass.GREEN_SUPPLY),
    )
    decision = step(machine, 2, mixed, transport_ids=(2, 3))
    assert decision.termination_reason is TerminationReason.INJURED_MIXED_TRANSPORT

    machine = started_machine()
    complete_first_delivery(machine)
    four = tuple(
        target(index, TargetClass.GREEN_SUPPLY)
        for index in range(2, 6)
    )
    decision = step(
        machine,
        2,
        four,
        transport_ids=tuple(item.track_id for item in four),
    )
    assert decision.termination_reason is TerminationReason.TOO_MANY_TARGETS


def test_delivery_capacity_violation_precedes_stale_visual_hold() -> None:
    machine = started_machine()
    four = tuple(
        target(index, TargetClass.GREEN_SUPPLY)
        for index in range(1, 5)
    )
    decision = step(
        machine,
        1,
        four,
        delivery=delivery(
            "too-many-at-boundary",
            tuple(item.track_id for item in four),
            DeliveryDestination.OWN_MATERIAL,
        ),
        uncertainties=frozenset({WorldUncertainty.STALE_VISION}),
    )
    assert decision.terminal
    assert decision.termination_reason is TerminationReason.TOO_MANY_TARGETS


def test_danger_engagement_safe_zone_and_field_exit_terminate() -> None:
    danger = target(
        9,
        TargetClass.BLUE_DANGER,
        hazard_state=HazardState.CONFIRMED,
    )
    machine = started_machine()
    assert step(
        machine,
        1,
        (danger,),
        transport_ids=(9,),
    ).termination_reason is TerminationReason.DANGER_ENGAGED

    machine = started_machine()
    assert step(
        machine,
        1,
        (danger,),
        delivery=delivery(
            "danger-zone",
            (9,),
            DeliveryDestination.OPPONENT_SAFE,
        ),
    ).termination_reason is TerminationReason.DANGER_IN_SAFE_ZONE

    machine = started_machine()
    assert step(
        machine,
        1,
        (danger,),
        delivery=delivery(
            "danger-out",
            (9,),
            DeliveryDestination.OUT_OF_FIELD,
        ),
    ).termination_reason is TerminationReason.DANGER_OUT_OF_FIELD


def test_stale_or_missing_transport_holds_and_recovers() -> None:
    machine = started_machine()
    green = target(1, TargetClass.GREEN_SUPPLY)
    stale = step(
        machine,
        1,
        (green,),
        uncertainties=frozenset({WorldUncertainty.STALE_VISION}),
    )
    fresh = step(machine, 2, (green,))
    missing = step(machine, 3, (), transport_ids=(1,))
    recovered = step(machine, 4, (green,), transport_ids=(1,))

    assert stale.activity is ActivityState.SAFETY_HOLD
    assert stale.action is AbstractAction.STOP
    assert fresh.action is AbstractAction.APPROACH
    assert missing.activity is ActivityState.SAFETY_HOLD
    assert recovered.action is AbstractAction.PUSH


def test_suspected_danger_is_never_selected_or_pushed() -> None:
    machine = started_machine()
    suspected = target(
        1,
        TargetClass.UNKNOWN,
        hazard_state=HazardState.SUSPECTED,
        distance_mm=100.0,
    )
    nearby = step(machine, 1, (suspected,))
    engaged = step(machine, 2, (suspected,), transport_ids=(1,))
    assert nearby.action is AbstractAction.SEARCH
    assert nearby.activity is ActivityState.SEARCHING
    assert engaged.activity is ActivityState.SAFETY_HOLD
    assert engaged.action is AbstractAction.STOP


def test_nearby_confirmed_danger_without_transport_is_not_global_avoidance() -> None:
    machine = started_machine()
    danger = target(
        1,
        TargetClass.BLUE_DANGER,
        hazard_state=HazardState.CONFIRMED,
        distance_mm=50.0,
    )
    decision = step(machine, 1, (danger,))
    assert decision.activity is ActivityState.SEARCHING
    assert decision.action is AbstractAction.SEARCH


def test_candidate_in_known_safe_or_opponent_area_is_not_selected() -> None:
    machine = started_machine()
    green = replace(
        target(1, TargetClass.GREEN_SUPPLY),
        field_point=FieldPoint(50.0, 50.0),
    )
    own_material = StaticRegion(
        "own-material",
        RegionKind.OWN_MATERIAL,
        (
            FieldPoint(0.0, 0.0),
            FieldPoint(100.0, 0.0),
            FieldPoint(100.0, 100.0),
        ),
    )
    decision = machine.step(
        snapshot(
            1,
            (green,),
            static_regions=(own_material,),
        ),
        transport=TransportStatus(),
        safety=SafetySignals.nominal(1),
    )
    assert decision.action is AbstractAction.SEARCH

    opponent_area = OpponentOccupancy(
        "opponent",
        own_material.polygon_field,
        confidence=0.9,
        timestamp_ns=2,
    )
    decision = machine.step(
        snapshot(
            2,
            (green,),
            opponent_occupancies=(opponent_area,),
        ),
        transport=TransportStatus(),
        safety=SafetySignals.nominal(2),
    )
    assert decision.action is AbstractAction.SEARCH


def test_opponent_zone_and_contact_timeout_are_recoverable() -> None:
    machine = started_machine()
    in_zone = step(
        machine,
        1,
        regions=frozenset({RegionKind.OPPONENT_SAFE}),
    )
    contact = step(
        machine,
        10_000_000_001,
        safety=SafetySignals(
            last_motion_timestamp_ns=10_000_000_001,
            opponent_contact_since_ns=1,
        ),
    )
    recovered = step(machine, 10_000_000_002)
    assert in_zone.action is AbstractAction.AVOID
    assert contact.action is AbstractAction.STOP
    assert contact.activity is ActivityState.AVOIDING
    assert recovered.action is AbstractAction.SEARCH


@pytest.mark.parametrize(
    ("safety", "reason"),
    [
        (
            SafetySignals(1, active_attack=True),
            TerminationReason.ACTIVE_ATTACK,
        ),
        (
            SafetySignals(1, lost_control=True),
            TerminationReason.LOST_CONTROL,
        ),
        (
            SafetySignals(1, safety_accident=True),
            TerminationReason.SAFETY_ACCIDENT,
        ),
        (
            SafetySignals(1, human_touched_after_start=True),
            TerminationReason.HUMAN_TOUCHED_AFTER_START,
        ),
        (
            SafetySignals(1, target_carried_on_robot=True),
            TerminationReason.ILLEGAL_CARRY,
        ),
        (
            SafetySignals(1, external_stop_requested=True),
            TerminationReason.EXTERNAL_STOP,
        ),
    ],
)
def test_direct_safety_signals_terminate(
    safety: SafetySignals,
    reason: TerminationReason,
) -> None:
    machine = started_machine()
    decision = step(machine, 1, safety=safety)
    assert decision.termination_reason is reason


def test_motion_and_match_timeouts_terminate() -> None:
    machine = started_machine()
    no_motion = step(
        machine,
        15_000_000_000,
        safety=SafetySignals(last_motion_timestamp_ns=0),
    )
    assert no_motion.termination_reason is TerminationReason.NO_MOTION_TIMEOUT

    machine = started_machine()
    timed_out = step(machine, 180_000_000_000)
    assert timed_out.termination_reason is TerminationReason.MATCH_TIMEOUT


@pytest.mark.parametrize(
    "transport_ids",
    [(999,), (1,)],
)
def test_timeouts_preempt_transport_holds(
    transport_ids: tuple[int, ...],
) -> None:
    machine = started_machine()
    targets = (
        (target(1, TargetClass.UNKNOWN, hazard_state=HazardState.SUSPECTED),)
        if transport_ids == (1,)
        else ()
    )
    decision = step(
        machine,
        180_000_000_000,
        targets,
        transport_ids=transport_ids,
    )
    assert decision.termination_reason is TerminationReason.MATCH_TIMEOUT


def test_missing_robot_field_position_holds_but_does_not_mask_timeout() -> None:
    machine = started_machine()
    current = machine.step(
        snapshot(
            1,
            (target(1, TargetClass.GREEN_SUPPLY),),
            robot_field_point=None,
            uncertainties=frozenset(
                {WorldUncertainty.MISSING_ROBOT_FIELD_POSITION}
            ),
        ),
        transport=TransportStatus(),
        safety=SafetySignals.nominal(1),
    )
    assert current.activity is ActivityState.SAFETY_HOLD
    assert current.reason == "robot_field_position_missing"

    timeout = machine.step(
        snapshot(
            180_000_000_000,
            robot_field_point=None,
            uncertainties=frozenset(
                {WorldUncertainty.MISSING_ROBOT_FIELD_POSITION}
            ),
        ),
        transport=TransportStatus(),
        safety=SafetySignals.nominal(180_000_000_000),
    )
    assert timeout.termination_reason is TerminationReason.MATCH_TIMEOUT


def test_wrong_opponent_and_outside_deliveries_do_not_advance_phase() -> None:
    machine = started_machine()
    green = target(1, TargetClass.GREEN_SUPPLY)
    opponent = step(
        machine,
        1,
        (green,),
        delivery=delivery(
            "opponent",
            (1,),
            DeliveryDestination.OPPONENT_SAFE,
        ),
    )
    outside = step(
        machine,
        2,
        (green,),
        delivery=delivery(
            "outside",
            (1,),
            DeliveryDestination.OUT_OF_FIELD,
        ),
    )
    assert opponent.action is AbstractAction.AVOID
    assert outside.action is AbstractAction.STOP
    assert machine.phase is MissionPhase.FIRST_NORMAL_REQUIRED
    assert machine.progress.opponent_zone_targets == 1
    assert machine.progress.out_of_field_targets == 1


def test_wrong_zone_after_first_is_counted_without_termination() -> None:
    machine = started_machine()
    complete_first_delivery(machine)
    black = target(2, TargetClass.BLACK_CORE)
    decision = step(
        machine,
        2,
        (black,),
        delivery=delivery(
            "wrong-zone",
            (2,),
            DeliveryDestination.OWN_INJURED,
        ),
    )
    assert not decision.terminal
    assert machine.progress.wrong_zone_targets == 1


def test_partial_wrong_zone_is_not_treated_as_valid_delivery_progress() -> None:
    machine = started_machine()
    green = target(1, TargetClass.GREEN_SUPPLY)
    invalid_first = step(
        machine,
        1,
        (green,),
        delivery=delivery(
            "partial-wrong-first",
            (1,),
            DeliveryDestination.OWN_INJURED,
            fully_entered=False,
        ),
    )
    assert (
        invalid_first.termination_reason
        is TerminationReason.INVALID_FIRST_DELIVERY
    )

    machine = started_machine()
    complete_first_delivery(machine)
    black = target(2, TargetClass.BLACK_CORE)
    wrong_general = step(
        machine,
        2,
        (black,),
        delivery=delivery(
            "partial-wrong-general",
            (2,),
            DeliveryDestination.OWN_INJURED,
            fully_entered=False,
        ),
    )
    assert wrong_general.action is AbstractAction.SEARCH
    assert machine.progress.wrong_zone_targets == 1


def test_partial_and_duplicate_delivery_are_idempotent() -> None:
    machine = started_machine()
    green = target(1, TargetClass.GREEN_SUPPLY)
    partial = step(
        machine,
        1,
        (green,),
        delivery=delivery(
            "partial",
            (1,),
            DeliveryDestination.OWN_MATERIAL,
            fully_entered=False,
        ),
    )
    assert partial.action is AbstractAction.DELIVER
    assert machine.progress.delivered_green_supply == 0
    partial_replay = step(
        machine,
        2,
        (green,),
        delivery=delivery(
            "partial",
            (1,),
            DeliveryDestination.OWN_MATERIAL,
            fully_entered=False,
        ),
    )
    assert partial_replay.action is AbstractAction.DELIVER
    assert partial_replay.activity is ActivityState.VERIFYING_DELIVERY

    evidence = delivery(
        "complete",
        (1,),
        DeliveryDestination.OWN_MATERIAL,
    )
    step(machine, 3, (green,), delivery=evidence)
    step(machine, 4, (green,), delivery=evidence)
    assert machine.progress.delivered_green_supply == 1
    with pytest.raises(ValueError, match="reused"):
        step(
            machine,
            5,
            (green,),
            delivery=replace(
                evidence,
                destination=DeliveryDestination.OWN_INJURED,
            ),
        )


def test_terminal_state_absorbs_events_until_reset() -> None:
    machine = started_machine()
    terminal = step(
        machine,
        1,
        safety=SafetySignals(1, lost_control=True),
    )
    absorbed = step(
        machine,
        2,
        (target(1, TargetClass.GREEN_SUPPLY),),
    )
    assert terminal.terminal and absorbed.terminal
    assert absorbed.action is AbstractAction.STOP
    with pytest.raises(RuntimeError, match="only start"):
        machine.start(3)
    machine.reset()
    assert machine.start(4).phase is MissionPhase.FIRST_NORMAL_REQUIRED


def test_complete_event_sequence_can_be_replayed_without_hardware() -> None:
    green = target(1, TargetClass.GREEN_SUPPLY)
    black = target(2, TargetClass.BLACK_CORE)
    decisions = replay_mission(
        config(),
        start_timestamp_ns=0,
        steps=(
            MissionReplayStep(
                snapshot(1, (green, black)),
                TransportStatus(),
                SafetySignals.nominal(1),
            ),
            MissionReplayStep(
                snapshot(2, (green, black)),
                TransportStatus((1,), 2),
                SafetySignals.nominal(2),
            ),
            MissionReplayStep(
                snapshot(3, (green, black)),
                TransportStatus(),
                SafetySignals.nominal(3),
                delivery(
                    "replay-first",
                    (1,),
                    DeliveryDestination.OWN_MATERIAL,
                ),
            ),
            MissionReplayStep(
                snapshot(4, (black,)),
                TransportStatus(),
                SafetySignals(4, lost_control=True),
            ),
        ),
    )
    assert [decision.action for decision in decisions] == [
        AbstractAction.SEARCH,
        AbstractAction.APPROACH,
        AbstractAction.PUSH,
        AbstractAction.SEARCH,
        AbstractAction.STOP,
    ]
    assert decisions[3].phase is MissionPhase.GENERAL_RESCUE
    assert decisions[-1].termination_reason is TerminationReason.LOST_CONTROL


def test_time_and_input_validation() -> None:
    machine = started_machine()
    step(machine, 2)
    with pytest.raises(ValueError, match="backwards"):
        step(machine, 1)
    with pytest.raises(ValueError, match="exactly"):
        TransportStatus((1,), None)
    with pytest.raises(ValueError, match="future"):
        machine = started_machine()
        step(
            machine,
            1,
            safety=SafetySignals(last_motion_timestamp_ns=2),
        )
    with pytest.raises(ValueError, match="active_attack must be a boolean"):
        SafetySignals(1, active_attack=1)  # type: ignore[arg-type]
