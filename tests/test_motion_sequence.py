from __future__ import annotations

from dataclasses import dataclass

import pytest

from rescue_vision.app.motion_sequence import (
    _build_sequence_controller,
    MotionSequencePhase,
    MotionSequencePlan,
    MotionSequenceRunner,
)
from rescue_vision.config import load_runtime_config
from rescue_vision.motion import MotionController, MotionLimits


class FakeCarChannel:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def send_frame(self, payload: bytes) -> None:
        self.sent.append(payload)

    def receive_frame(self, timeout: float | None = None) -> bytes:
        del timeout
        raise TimeoutError


@dataclass
class FakeClock:
    timestamp_ns: int = 0

    def __call__(self) -> int:
        return self.timestamp_ns

    def sleep(self, seconds: float) -> None:
        self.timestamp_ns += round(seconds * 1_000_000_000)


def limits() -> MotionLimits:
    return MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.80,
        max_angular_velocity_rad_s=1.0,
        max_wheel_velocity_m_s=0.80,
        max_linear_acceleration_m_s2=1.0,
        max_linear_deceleration_m_s2=1.0,
        max_angular_acceleration_rad_s2=10,
        max_angular_deceleration_rad_s2=10,
        min_wheel_velocity_m_s=0.001,
        max_remote_command_valid_for_ms=500,
    )


def test_motion_sequence_plan_derives_distance_profile() -> None:
    plan = MotionSequencePlan(0.20, 0.40)

    assert plan.distance_m == pytest.approx(1.5)
    assert plan.peak_linear_velocity_m_s == pytest.approx(0.632455532)
    assert plan.acceleration_duration_s == pytest.approx(3.16227766)
    assert plan.deceleration_duration_s == pytest.approx(1.58113883)
    assert plan.nominal_total_duration_s == pytest.approx(4.74341649)
    assert plan.target_linear_velocity_m_s(1.0) == pytest.approx(0.2)
    assert plan.target_linear_velocity_m_s(3.16227766) == pytest.approx(
        plan.peak_linear_velocity_m_s
    )
    assert plan.target_linear_velocity_m_s(3.66227766) == pytest.approx(
        plan.peak_linear_velocity_m_s - 0.2
    )
    assert plan.target_linear_velocity_m_s(4.8) == pytest.approx(0.0)
    assert plan.planned_distance_m(3.16227766) == pytest.approx(1.0)
    assert plan.planned_distance_m(4.8) == pytest.approx(1.5)
    plan.validate_for_protocol(limits())


def test_motion_sequence_plan_accepts_configured_distance() -> None:
    plan = MotionSequencePlan(0.20, 0.40, distance_m=2.0)

    assert plan.distance_m == pytest.approx(2.0)
    assert plan.peak_linear_velocity_m_s == pytest.approx(0.730296743)
    assert plan.planned_distance_m(plan.nominal_total_duration_s) == pytest.approx(
        2.0
    )


@pytest.mark.parametrize(
    ("a1", "a2"),
    [
        (0.0, 0.2),
        (0.2, 0.0),
        (float("nan"), 0.2),
    ],
)
def test_motion_sequence_plan_rejects_invalid_inputs(
    a1: float,
    a2: float,
) -> None:
    with pytest.raises(ValueError):
        MotionSequencePlan(a1, a2)


def test_motion_sequence_plan_rejects_protocol_overflow() -> None:
    with pytest.raises(ValueError, match="forward peak velocity"):
        MotionSequencePlan(100_000.0, 0.5).validate_for_protocol(limits())


def test_motion_sequence_protocol_check_ignores_configured_speed_and_acceleration() -> None:
    configured_limits = MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.10,
        max_angular_velocity_rad_s=0.10,
        max_wheel_velocity_m_s=0.80,
        max_linear_acceleration_m_s2=0.10,
        max_linear_deceleration_m_s2=0.10,
        max_angular_acceleration_rad_s2=1,
        max_angular_deceleration_rad_s2=1,
        min_wheel_velocity_m_s=0.02,
        max_remote_command_valid_for_ms=500,
    )

    MotionSequencePlan(0.20, 0.40).validate_for_protocol(configured_limits)


def test_sequence_controller_does_not_copy_configured_speed_or_acceleration() -> None:
    config = load_runtime_config("configs/runtime.match.yaml")
    controller = _build_sequence_controller(config, FakeCarChannel())

    assert controller.limits.max_wheel_velocity_m_s == pytest.approx(32.767)
    assert controller.limits.max_linear_acceleration_m_s2 == pytest.approx(
        1_000_000.0
    )
    assert controller.limits.max_linear_deceleration_m_s2 == pytest.approx(
        1_000_000.0
    )
    assert controller.limits.min_wheel_velocity_m_s == pytest.approx(1e-9)


def test_motion_sequence_runner_executes_all_phases_and_stops() -> None:
    clock = FakeClock()
    channel = FakeCarChannel()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    phases: list[MotionSequencePhase] = []

    result = MotionSequenceRunner(
        controller,
        MotionSequencePlan(0.20, 0.40),
        monotonic_ns=clock,
        sleep=clock.sleep,
    ).run(on_status=lambda status: phases.append(status.phase))

    assert result.completed
    assert not result.stopped_by_operator
    assert result.final_phase is MotionSequencePhase.COMPLETE
    assert phases[0] is MotionSequencePhase.ACCELERATING
    assert MotionSequencePhase.DECELERATING in phases
    assert set(phases).issubset(
        {
            MotionSequencePhase.ACCELERATING,
            MotionSequencePhase.DECELERATING,
            MotionSequencePhase.COMPLETE,
        }
    )
    assert result.elapsed_s >= 4.743
    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)
    assert controller.acceleration_limits.linear_acceleration_m_s2 == pytest.approx(
        1.0
    )
    assert controller.acceleration_limits.linear_deceleration_m_s2 == pytest.approx(
        1.0
    )
    assert channel.sent


def test_motion_sequence_runner_handles_controller_clock_after_runner_timestamp() -> None:
    runner_clock = FakeClock()
    controller_clock = FakeClock(1_000_000)
    channel = FakeCarChannel()
    controller = MotionController(
        channel,
        limits(),
        monotonic_ns=controller_clock,
    )

    def sleep(seconds: float) -> None:
        runner_clock.sleep(seconds)
        controller_clock.sleep(seconds)

    result = MotionSequenceRunner(
        controller,
        MotionSequencePlan(0.20, 0.40),
        monotonic_ns=runner_clock,
        sleep=sleep,
    ).run()

    assert result.completed


def test_motion_sequence_runner_soft_brakes_when_operator_cancels() -> None:
    clock = FakeClock()
    controller = MotionController(
        FakeCarChannel(),
        limits(),
        monotonic_ns=clock,
    )

    result = MotionSequenceRunner(
        controller,
        MotionSequencePlan(0.20, 0.40),
        monotonic_ns=clock,
        sleep=clock.sleep,
    ).run(stop_requested=lambda: clock.timestamp_ns >= 500_000_000)

    assert not result.completed
    assert result.stopped_by_operator
    assert result.final_phase is MotionSequencePhase.ABORTED
    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)
