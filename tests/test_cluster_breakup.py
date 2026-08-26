from __future__ import annotations

import numpy as np
import pytest
import time
from threading import Event
from types import SimpleNamespace

from rescue_vision.app.cluster_breakup import (
    BreakupState,
    ClusterBreakupSequence,
    CameraPerceptionPump,
    EncoderTravelTracker,
    GripperPosture,
    OdometryFusionPump,
    RemotePerceptionPublisher,
    RemotePerceptionTransport,
    RemoteLocalizationPublisher,
    _submit_breakup_odometry,
    build_cluster_observation_status,
)
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.config import ClusterBreakupRuntimeConfig
from rescue_vision.communication import (
    ImageCoordinateSystem,
    MapStateObservation,
    RemoteAccessMode,
    RemoteConnectionOptions,
    RemoteRole,
    RemoteTcpServer,
    RemoteTopic,
    TeamColor as RemoteTeamColor,
    VideoFrameMode,
)
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.localization import OdometryCalibration
from rescue_vision.localization import FieldPose2D, FusedPoseEstimate
from rescue_vision.motion import OdometryImu, SensorFlags
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    PerceptionSnapshot,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)


def breakup_config(**changes: object) -> ClusterBreakupRuntimeConfig:
    values: dict[str, object] = {
        "enabled": True,
        "departure_distance_m": 0.5,
        "departure_speed_m_s": 0.15,
        "search_angular_velocity_rad_s": 0.4,
        "search_timeout_s": 10.0,
        "cluster_min_detections": 2,
        "center_tolerance_ratio": 0.05,
        "center_confirm_frames": 2,
        "center_kp_rad_s": 1.0,
        "center_max_angular_velocity_rad_s": 0.5,
        "approach_speed_m_s": 0.1,
        "gripper_open_distance_mm": 250.0,
        "breakup_speed_m_s": 0.25,
        "breakup_distance_m": 0.2,
        "gripper_open_retreat_distance_m": 0.1,
        "retreat_speed_m_s": 0.1,
        "retreat_distance_m": 0.1,
        "scan_green_angular_velocity_rad_s": 0.3,
        "green_confirm_frames": 2,
        "target_loss_timeout_ms": 400.0,
        "motion_phase_timeout_s": 8.0,
    }
    values.update(changes)
    return ClusterBreakupRuntimeConfig(**values)  # type: ignore[arg-type]


def observation(
    *,
    frame_sequence: int,
    x_min: float,
    x_max: float,
    distance_mm: float,
    target_class: TargetClass = TargetClass.GREEN_SUPPLY,
) -> TargetObservation:
    box = UndistortedBoundingBox(x_min, 30.0, x_max, 60.0)
    roi_box = UndistortedBoundingBox(
        int(x_min),
        30,
        int(x_max),
        60,
    )
    segmentation = RoiColorSegmentation(
        candidate_class=target_class,
        status=ColorSegmentationStatus.ACCEPTED,
        roi_box=roi_box,
        mask=np.full((30, int(x_max) - int(x_min)), 255, dtype=np.uint8),
        color_fraction=1.0,
        dominance=1.0,
    )
    timestamp_ns = frame_sequence * 100_000_000
    return TargetObservation(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns + 1,
        image_size=(100, 100),
        model_target_class=target_class,
        target_class=target_class,
        class_probabilities=ClassProbabilities.from_top_class(target_class, 0.9),
        detection_confidence=0.9,
        box=box,
        color_segmentation=segmentation,
        k0=UndistortedPixel((x_min + x_max) * 0.5, 55.0),
        k0_confidence=0.9,
        ground_point=GroundPoint(distance_mm, 0.0),
        quality=frozenset(),
    )


def snapshot(
    frame_sequence: int,
    *observations: TargetObservation,
) -> PerceptionSnapshot:
    timestamp_ns = frame_sequence * 100_000_000
    return PerceptionSnapshot(
        frame_sequence=frame_sequence,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns + 1,
        observations=tuple(observations),
    )


def cluster_snapshot(
    frame_sequence: int,
    *,
    distance_mm: float,
    offset: float = 0.0,
) -> PerceptionSnapshot:
    return snapshot(
        frame_sequence,
        observation(
            frame_sequence=frame_sequence,
            x_min=35.0 + offset,
            x_max=48.0 + offset,
            distance_mm=distance_mm,
        ),
        observation(
            frame_sequence=frame_sequence,
            x_min=52.0 + offset,
            x_max=65.0 + offset,
            distance_mm=distance_mm + 20.0,
            target_class=TargetClass.BLACK_CORE,
        ),
    )


def test_breakup_sequence_reaches_scan_green_then_stops_on_green() -> None:
    sequence = ClusterBreakupSequence(
        breakup_config(),
        gripper_full_travel_time_s=1.0,
    )

    waiting = sequence.step(
        timestamp_ns=0,
        cumulative_distance_m=None,
        perception=None,
    )
    assert waiting.state is BreakupState.WAIT_ODOMETRY

    leaving = sequence.step(
        timestamp_ns=10_000_000,
        cumulative_distance_m=0.0,
        perception=None,
    )
    assert leaving.state is BreakupState.LEAVE_START
    assert leaving.linear_velocity_m_s == 0.15

    searching = sequence.step(
        timestamp_ns=1_000_000_000,
        cumulative_distance_m=0.5,
        perception=None,
    )
    assert searching.state is BreakupState.SEARCH_CLUSTER
    assert searching.angular_velocity_rad_s > 0.0

    centering = sequence.step(
        timestamp_ns=1_100_000_000,
        cumulative_distance_m=0.5,
        perception=cluster_snapshot(11, distance_mm=500.0, offset=15.0),
    )
    assert centering.state is BreakupState.CENTER_CLUSTER
    assert centering.angular_velocity_rad_s < 0.0

    centered_1 = sequence.step(
        timestamp_ns=1_200_000_000,
        cumulative_distance_m=0.5,
        perception=cluster_snapshot(12, distance_mm=500.0),
    )
    assert centered_1.state is BreakupState.CENTER_CLUSTER
    centered_2 = sequence.step(
        timestamp_ns=1_300_000_000,
        cumulative_distance_m=0.5,
        perception=cluster_snapshot(13, distance_mm=500.0),
    )
    assert centered_2.state is BreakupState.APPROACH_CLUSTER
    assert centered_2.linear_velocity_m_s == 0.1

    pushing = sequence.step(
        timestamp_ns=1_400_000_000,
        cumulative_distance_m=0.55,
        perception=cluster_snapshot(14, distance_mm=240.0),
    )
    assert pushing.state is BreakupState.BREAKUP_PUSH
    assert pushing.gripper_posture is GripperPosture.CLOSED
    assert pushing.linear_velocity_m_s == 0.25

    releasing = sequence.step(
        timestamp_ns=2_000_000_000,
        cumulative_distance_m=0.75,
        perception=None,
    )
    assert releasing.state is BreakupState.BREAKUP_RELEASE
    assert releasing.gripper_posture is GripperPosture.OPEN
    assert releasing.linear_velocity_m_s == 0.0

    holding_open = sequence.step(
        timestamp_ns=2_500_000_000,
        cumulative_distance_m=0.75,
        perception=None,
    )
    assert holding_open.state is BreakupState.BREAKUP_RELEASE
    assert holding_open.gripper_posture is GripperPosture.OPEN
    assert holding_open.linear_velocity_m_s == 0.0

    open_retreating = sequence.step(
        timestamp_ns=3_000_000_000,
        cumulative_distance_m=0.75,
        perception=None,
    )
    assert open_retreating.state is BreakupState.BREAKUP_OPEN_RETREAT
    assert open_retreating.gripper_posture is GripperPosture.OPEN
    assert open_retreating.linear_velocity_m_s < 0.0

    open_retreating_done = sequence.step(
        timestamp_ns=3_500_000_000,
        cumulative_distance_m=0.65,
        perception=None,
    )
    assert open_retreating_done.state is BreakupState.BREAKUP_CLOSE
    assert open_retreating_done.gripper_posture is GripperPosture.CLOSED
    assert open_retreating_done.linear_velocity_m_s == 0.0

    closing = sequence.step(
        timestamp_ns=4_000_000_000,
        cumulative_distance_m=0.65,
        perception=None,
    )
    assert closing.state is BreakupState.BREAKUP_CLOSE
    assert closing.gripper_posture is GripperPosture.CLOSED
    assert closing.linear_velocity_m_s == 0.0

    retreating = sequence.step(
        timestamp_ns=4_500_000_000,
        cumulative_distance_m=0.65,
        perception=None,
    )
    assert retreating.state is BreakupState.RETREAT
    assert retreating.gripper_posture is GripperPosture.CLOSED
    assert retreating.linear_velocity_m_s < 0.0

    scanning = sequence.step(
        timestamp_ns=5_000_000_000,
        cumulative_distance_m=0.55,
        perception=None,
    )
    assert scanning.state is BreakupState.SCAN_GREEN
    assert scanning.gripper_posture is GripperPosture.CLOSED
    assert scanning.angular_velocity_rad_s > 0.0

    green_1 = snapshot(
        26,
        observation(
            frame_sequence=26,
            x_min=40.0,
            x_max=55.0,
            distance_mm=600.0,
        ),
    )
    assert sequence.step(
        timestamp_ns=5_100_000_000,
        cumulative_distance_m=0.65,
        perception=green_1,
    ).state is BreakupState.SCAN_GREEN
    green_2 = snapshot(
        27,
        observation(
            frame_sequence=27,
            x_min=40.0,
            x_max=55.0,
            distance_mm=600.0,
        ),
    )
    found = sequence.step(
        timestamp_ns=5_200_000_000,
        cumulative_distance_m=0.65,
        perception=green_2,
    )
    assert found.state is BreakupState.GREEN_FOUND
    assert found.linear_velocity_m_s == 0.0


def odometry(
    timestamp_ns: int,
    count: int,
    *,
    sensor_flags: SensorFlags | None = None,
) -> OdometryImu:
    return OdometryImu(
        uart_sequence=count,
        received_timestamp_ns=timestamp_ns,
        telemetry_sequence=count,
        sample_timestamp_us=timestamp_ns // 1000,
        left_encoder_count=count,
        right_encoder_count=count,
        gyro_x_urad_s=0,
        gyro_y_urad_s=0,
        gyro_z_urad_s=0,
        accel_x_mm_s2=0,
        accel_y_mm_s2=0,
        accel_z_mm_s2=9800,
        imu_temperature_cdeg=2500,
        sensor_flags=(
            sensor_flags
            if sensor_flags is not None
            else SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
        ),
    )


def test_encoder_travel_tracker_uses_effective_wheel_calibration() -> None:
    radius_mm = 1000.0 / (2.0 * np.pi)
    tracker = EncoderTravelTracker(
        OdometryCalibration(1000, radius_mm, radius_mm),
        max_wheel_velocity_m_s=0.3,
    )
    assert tracker.submit(odometry(1_000_000_000, 0)) == 0.0
    assert np.isclose(tracker.submit(odometry(2_000_000_000, 100)), 0.1)
    assert np.isclose(tracker.left_distance_m, 0.1)
    assert np.isclose(tracker.right_distance_m, 0.1)
    assert not tracker.forward_sign_mismatch()
    assert "encoder_counts=(100,100)" in tracker.diagnostic()


def test_encoder_travel_tracker_accepts_one_overrun_and_rebases_after_clean_sample() -> None:
    radius_mm = 1000.0 / (2.0 * np.pi)
    tracker = EncoderTravelTracker(
        OdometryCalibration(1000, radius_mm, radius_mm),
        max_wheel_velocity_m_s=0.3,
    )
    valid = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
    tracker.submit(odometry(1_000_000_000, 0, sensor_flags=valid))
    overrun = odometry(
        1_010_000_000,
        5,
        sensor_flags=valid | SensorFlags.SAMPLE_OVERRUN,
    )
    clean = odometry(1_020_000_000, 10, sensor_flags=valid)

    assert tracker.submit(overrun) == pytest.approx(0.005)
    assert tracker.consecutive_overrun_samples == 1
    assert tracker.submit(clean) == pytest.approx(0.01)
    assert tracker.consecutive_overrun_samples == 0


def test_encoder_travel_tracker_rejects_a_second_consecutive_overrun() -> None:
    radius_mm = 1000.0 / (2.0 * np.pi)
    tracker = EncoderTravelTracker(
        OdometryCalibration(1000, radius_mm, radius_mm),
        max_wheel_velocity_m_s=0.3,
    )
    valid = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
    tracker.submit(odometry(1_000_000_000, 0, sensor_flags=valid))
    tracker.submit(
        odometry(
            1_010_000_000,
            5,
            sensor_flags=valid | SensorFlags.SAMPLE_OVERRUN,
        )
    )

    with pytest.raises(RuntimeError, match="Consecutive odometry sample overruns"):
        tracker.submit(
            odometry(
                1_020_000_000,
                10,
                sensor_flags=valid | SensorFlags.SAMPLE_OVERRUN,
            )
        )


def test_encoder_travel_tracker_honors_larger_overrun_budget() -> None:
    radius_mm = 1000.0 / (2.0 * np.pi)
    tracker = EncoderTravelTracker(
        OdometryCalibration(1000, radius_mm, radius_mm),
        max_wheel_velocity_m_s=0.3,
        max_consecutive_overrun_samples=2,
    )
    valid = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
    tracker.submit(odometry(1_000_000_000, 0, sensor_flags=valid))

    assert (
        tracker.submit(
            odometry(
                1_010_000_000,
                5,
                sensor_flags=valid | SensorFlags.SAMPLE_OVERRUN,
            )
        )
        == pytest.approx(0.005)
    )
    assert tracker.consecutive_overrun_samples == 1
    assert (
        tracker.submit(
            odometry(
                1_020_000_000,
                10,
                sensor_flags=valid | SensorFlags.SAMPLE_OVERRUN,
            )
        )
        == pytest.approx(0.01)
    )
    assert tracker.consecutive_overrun_samples == 2

    with pytest.raises(RuntimeError, match="Consecutive odometry sample overruns"):
        tracker.submit(
            odometry(
                1_030_000_000,
                15,
                sensor_flags=valid | SensorFlags.SAMPLE_OVERRUN,
            )
        )


def test_encoder_travel_tracker_without_overrun_limit_keeps_integrating() -> None:
    radius_mm = 1000.0 / (2.0 * np.pi)
    tracker = EncoderTravelTracker(
        OdometryCalibration(1000, radius_mm, radius_mm),
        max_wheel_velocity_m_s=0.3,
        max_consecutive_overrun_samples=None,
    )
    valid = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
    tracker.submit(odometry(1_000_000_000, 0, sensor_flags=valid))

    distance = 0.0
    for index in range(1, 6):
        distance = tracker.submit(
            odometry(
                1_000_000_000 + index * 10_000_000,
                index * 5,
                sensor_flags=valid | SensorFlags.SAMPLE_OVERRUN,
            )
        )

    assert distance == pytest.approx(0.025)
    assert tracker.consecutive_overrun_samples == 5


def test_encoder_travel_tracker_rejects_invalid_overrun_budgets() -> None:
    radius_mm = 1000.0 / (2.0 * np.pi)
    for bad_value in (-1, True, 1.5, "2"):
        with pytest.raises(
            ValueError,
            match="max_consecutive_overrun_samples",
        ):
            EncoderTravelTracker(
                OdometryCalibration(1000, radius_mm, radius_mm),
                max_wheel_velocity_m_s=0.3,
                max_consecutive_overrun_samples=bad_value,
            )


def test_encoder_travel_tracker_detects_opposed_forward_signs() -> None:
    radius_mm = 1000.0 / (2.0 * np.pi)
    tracker = EncoderTravelTracker(
        OdometryCalibration(1000, radius_mm, radius_mm),
        max_wheel_velocity_m_s=0.3,
    )
    tracker.submit(odometry(1_000_000_000, 0))
    opposed = odometry(2_000_000_000, 100)
    opposed = OdometryImu(
        uart_sequence=opposed.uart_sequence,
        received_timestamp_ns=opposed.received_timestamp_ns,
        telemetry_sequence=opposed.telemetry_sequence,
        sample_timestamp_us=opposed.sample_timestamp_us,
        left_encoder_count=100,
        right_encoder_count=-100,
        gyro_x_urad_s=opposed.gyro_x_urad_s,
        gyro_y_urad_s=opposed.gyro_y_urad_s,
        gyro_z_urad_s=opposed.gyro_z_urad_s,
        accel_x_mm_s2=opposed.accel_x_mm_s2,
        accel_y_mm_s2=opposed.accel_y_mm_s2,
        accel_z_mm_s2=opposed.accel_z_mm_s2,
        imu_temperature_cdeg=opposed.imu_temperature_cdeg,
        sensor_flags=opposed.sensor_flags,
    )
    tracker.submit(opposed)
    assert tracker.forward_sign_mismatch()
    assert np.isclose(tracker.distance_m, 0.0)


def test_breakup_odometry_is_dispatched_to_distance_and_pose_consumers() -> None:
    class _RecordingFusion:
        def __init__(self) -> None:
            self.messages: list[OdometryImu] = []

        def submit_odometry(self, message: OdometryImu) -> str:
            self.messages.append(message)
            return "estimate"

    radius_mm = 1000.0 / (2.0 * np.pi)
    tracker = EncoderTravelTracker(
        OdometryCalibration(1000, radius_mm, radius_mm),
        max_wheel_velocity_m_s=0.3,
    )
    fusion = _RecordingFusion()
    first = odometry(1_000_000_000, 0)
    second = odometry(2_000_000_000, 100)

    _submit_breakup_odometry(tracker, fusion, first)  # type: ignore[arg-type]
    result = _submit_breakup_odometry(  # type: ignore[arg-type]
        tracker,
        fusion,
        second,
    )

    assert result == "estimate"
    assert fusion.messages == [first, second]
    assert tracker.distance_m == pytest.approx(0.1)


def test_odometry_fusion_pump_submit_does_not_wait_for_slow_fusion() -> None:
    class _SlowFusion:
        def __init__(self) -> None:
            self.messages: list[OdometryImu] = []

        def submit_odometry(self, message: OdometryImu) -> None:
            time.sleep(0.08)
            self.messages.append(message)

        def latest_estimate(self, _timestamp_ns: int) -> object:
            return None

    fusion = _SlowFusion()
    pump = OdometryFusionPump(fusion, queue_capacity=2)  # type: ignore[arg-type]
    pump.start()
    try:
        started_at = time.monotonic()
        pump.submit_odometry(odometry(1_000_000_000, 0))
        assert time.monotonic() - started_at < 0.03
        time.sleep(0.12)
        pump.check_health()
    finally:
        pump.stop()
    assert fusion.messages[0].telemetry_sequence == 0


def test_odometry_fusion_startup_wait_services_callback() -> None:
    class _DelayedReadyFusion:
        def __init__(self) -> None:
            self.latest_calls = 0

        def submit_odometry(self, _message: OdometryImu) -> None:
            pass

        def latest_estimate(self, _timestamp_ns: int) -> SimpleNamespace:
            self.latest_calls += 1
            return SimpleNamespace(
                pose=None if self.latest_calls == 1 else object()
            )

    fusion = _DelayedReadyFusion()
    pump = OdometryFusionPump(fusion)  # type: ignore[arg-type]
    pump.start()
    callbacks = 0

    def service_uart() -> None:
        nonlocal callbacks
        callbacks += 1

    try:
        pump.wait_until_ready(timeout_s=0.2, on_wait=service_uart)
        assert callbacks == 1
    finally:
        pump.stop()


def test_breakup_search_accepts_negative_rightward_velocity() -> None:
    sequence = ClusterBreakupSequence(
        breakup_config(search_angular_velocity_rad_s=-0.4),
        gripper_full_travel_time_s=1.0,
    )
    sequence.step(timestamp_ns=0, cumulative_distance_m=0.0, perception=None)
    decision = sequence.step(
        timestamp_ns=1_000_000_000,
        cumulative_distance_m=0.5,
        perception=None,
    )
    assert decision.state is BreakupState.SEARCH_CLUSTER
    assert decision.angular_velocity_rad_s == pytest.approx(-0.4)
    assert decision.reason == "departure_complete_search_right"


def test_breakup_scan_green_follows_signed_velocity() -> None:
    sequence = ClusterBreakupSequence(
        breakup_config(
            search_angular_velocity_rad_s=-0.4,
            scan_green_angular_velocity_rad_s=-0.3,
        ),
        gripper_full_travel_time_s=1.0,
    )
    sequence.step(timestamp_ns=0, cumulative_distance_m=None, perception=None)
    sequence.step(
        timestamp_ns=10_000_000, cumulative_distance_m=0.0, perception=None
    )
    searching = sequence.step(
        timestamp_ns=1_000_000_000,
        cumulative_distance_m=0.5,
        perception=None,
    )
    assert searching.state is BreakupState.SEARCH_CLUSTER
    assert searching.angular_velocity_rad_s == pytest.approx(-0.4)
    assert searching.reason == "departure_complete_search_right"

    still_searching = sequence.step(
        timestamp_ns=1_050_000_000,
        cumulative_distance_m=0.5,
        perception=None,
    )
    assert still_searching.reason == "search_cluster_right"

    sequence.step(
        timestamp_ns=1_100_000_000,
        cumulative_distance_m=0.5,
        perception=cluster_snapshot(11, distance_mm=500.0, offset=15.0),
    )
    sequence.step(
        timestamp_ns=1_200_000_000,
        cumulative_distance_m=0.5,
        perception=cluster_snapshot(12, distance_mm=500.0),
    )
    sequence.step(
        timestamp_ns=1_300_000_000,
        cumulative_distance_m=0.5,
        perception=cluster_snapshot(13, distance_mm=500.0),
    )
    sequence.step(
        timestamp_ns=1_400_000_000,
        cumulative_distance_m=0.55,
        perception=cluster_snapshot(14, distance_mm=240.0),
    )
    sequence.step(
        timestamp_ns=2_000_000_000, cumulative_distance_m=0.75, perception=None
    )
    sequence.step(
        timestamp_ns=2_500_000_000, cumulative_distance_m=0.75, perception=None
    )
    sequence.step(
        timestamp_ns=3_000_000_000, cumulative_distance_m=0.75, perception=None
    )
    sequence.step(
        timestamp_ns=3_500_000_000, cumulative_distance_m=0.65, perception=None
    )
    sequence.step(
        timestamp_ns=4_000_000_000, cumulative_distance_m=0.65, perception=None
    )
    sequence.step(
        timestamp_ns=4_500_000_000, cumulative_distance_m=0.65, perception=None
    )

    scanning = sequence.step(
        timestamp_ns=5_000_000_000,
        cumulative_distance_m=0.55,
        perception=None,
    )
    assert scanning.state is BreakupState.SCAN_GREEN
    assert scanning.angular_velocity_rad_s == pytest.approx(-0.3)
    assert scanning.reason == "retreat_complete_close_gripper_scan_green"

    still_scanning = sequence.step(
        timestamp_ns=5_050_000_000,
        cumulative_distance_m=0.55,
        perception=None,
    )
    assert still_scanning.state is BreakupState.SCAN_GREEN
    assert still_scanning.angular_velocity_rad_s == pytest.approx(-0.3)
    assert still_scanning.reason == "scan_green_right"


def test_breakup_rejects_zero_or_tiny_scan_velocities() -> None:
    with pytest.raises(ValueError, match="search_angular_velocity_rad_s"):
        breakup_config(search_angular_velocity_rad_s=0.0)
    with pytest.raises(ValueError, match="search_angular_velocity_rad_s"):
        breakup_config(search_angular_velocity_rad_s=0.0005)
    with pytest.raises(ValueError, match="scan_green_angular_velocity_rad_s"):
        breakup_config(scan_green_angular_velocity_rad_s=0.0)


class _OneFrameSource:
    image_size = (2, 2)

    def __init__(self) -> None:
        self.started = False
        self.delivered = False

    def start(self) -> None:
        self.started = True

    def read(self, timeout: float = 1.0) -> CameraFrame:
        assert self.started
        if not self.delivered:
            self.delivered = True
            return CameraFrame(
                1,
                1,
                np.zeros((2, 2, 3), dtype=np.uint8),
            )
        time.sleep(min(timeout, 0.005))
        raise TimeoutError

    def stop(self) -> None:
        self.started = False


class _RecordingPerception:
    def __init__(self) -> None:
        self.submitted = Event()

    def start(self) -> None:
        pass

    def submit(self, frame: CameraFrame) -> None:
        assert frame.sequence == 1
        self.submitted.set()

    def check_health(self) -> None:
        pass

    def stop(self) -> None:
        pass


class _SlowStartingPerception(_RecordingPerception):
    def start(self) -> None:
        time.sleep(0.12)


def test_camera_prepare_runs_outside_control_caller() -> None:
    source = _OneFrameSource()
    perception = _RecordingPerception()

    def slow_prepare(frame: CameraFrame) -> CameraFrame:
        time.sleep(0.12)
        return frame

    pump = CameraPerceptionPump(source, slow_prepare, perception)
    started_at = time.monotonic()
    pump.start()
    try:
        assert time.monotonic() - started_at < 0.05
        assert perception.submitted.wait(timeout=1.0)
        pump.check_health()
    finally:
        pump.stop()


def test_camera_startup_wait_can_service_uart_callback() -> None:
    source = _OneFrameSource()
    perception = _SlowStartingPerception()
    pump = CameraPerceptionPump(source, lambda frame: frame, perception)
    ticks = 0

    def service_uart() -> None:
        nonlocal ticks
        ticks += 1

    started_at = time.monotonic()
    startup_thread = pump.start_in_background()
    try:
        assert time.monotonic() - started_at < 0.05
        pump.wait_until_started(startup_thread, on_wait=service_uart)
        assert ticks > 0
        assert perception.submitted.wait(timeout=1.0)
    finally:
        pump.stop()


class _ObservationConnection:
    def __init__(self) -> None:
        self.reliable: list[tuple[str, bytes]] = []
        self.observations: list[tuple[str, bytes, dict[str, object]]] = []

    def send_reliable_observation(
        self,
        topic: str,
        payload: bytes,
        **_kwargs: object,
    ) -> None:
        self.reliable.append((topic, payload))

    def send_observation(
        self,
        topic: str,
        payload: bytes,
        **kwargs: object,
    ) -> None:
        attributes = kwargs.get("attributes")
        self.observations.append(
            (
                topic,
                payload,
                dict(attributes) if isinstance(attributes, dict) else {},
            )
        )

    def check_health(self) -> None:
        pass

    def receive_control(self, *, timeout: float) -> object:
        del timeout
        raise TimeoutError


def _observer_config() -> SimpleNamespace:
    return SimpleNamespace(
        remote=SimpleNamespace(
            enabled=True,
            role=RemoteRole.SERVER,
            access_mode=RemoteAccessMode.OBSERVE_ONLY,
        ),
        motion=SimpleNamespace(max_remote_command_valid_for_ms=500),
    )


def _observer_pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        coordinate_system=ImageCoordinateSystem.UNDISTORTED_PIXEL,
        calibration_id="test-calibration",
        ground_projector=None,
    )


def test_remote_perception_publisher_only_sends_new_visualization_frames() -> None:
    status = build_cluster_observation_status(
        _observer_config(),
        server_instance_id="cluster-test-server",
    )
    connection = _ObservationConnection()
    publisher = RemotePerceptionPublisher(
        connection,
        _observer_pipeline(),
        status,
        min_publish_interval_s=0.01,
    )
    publisher.start()
    frame = CameraFrame(7, 100, np.zeros((4, 5, 3), dtype=np.uint8))
    started_at = time.monotonic()
    publisher.submit(frame)
    assert time.monotonic() - started_at < 0.05
    deadline = time.monotonic() + 1.0
    while not connection.observations and time.monotonic() < deadline:
        time.sleep(0.005)
    publisher.submit(frame)
    publisher.stop()

    assert [topic for topic, _payload in connection.reliable] == [
        RemoteTopic.SESSION_STATUS.value
    ]
    assert len(connection.observations) == 1
    topic, _payload, attributes = connection.observations[0]
    assert topic == RemoteTopic.VIDEO_FRAME.value
    assert attributes["mode"] == VideoFrameMode.PERCEPTION.value
    assert attributes["coordinate_system"] == ImageCoordinateSystem.UNDISTORTED_PIXEL.value


def test_remote_localization_publisher_sends_latest_map_state_without_blocking() -> None:
    status = build_cluster_observation_status(
        _observer_config(),
        server_instance_id="cluster-test-server",
        map_state_available=True,
    )
    assert status.map_state_available
    assert status.map_state_period_ms == 200
    connection = _ObservationConnection()
    publisher = RemoteLocalizationPublisher(
        connection,
        RemoteTeamColor.UNKNOWN,
        publish_interval_s=0.01,
    )
    publisher.start()
    estimate = FusedPoseEstimate(
        pose=FieldPose2D(FieldPoint(-1350.0, -1200.0), 1.5),
        estimate_timestamp_ns=100,
        position_uncertainty_mm=25.0,
        heading_uncertainty_rad=0.08,
        confidence=0.5,
        anchor_source="configured_start",
        quality=frozenset(),
    )
    started_at = time.monotonic()
    publisher.submit(estimate, 200)
    assert time.monotonic() - started_at < 0.05
    deadline = time.monotonic() + 1.0
    localized_state: MapStateObservation | None = None
    while localized_state is None and time.monotonic() < deadline:
        for topic, payload, _attributes in connection.observations:
            if topic == RemoteTopic.MAP_STATE.value:
                candidate = MapStateObservation.from_payload(payload)
                if candidate.robot_localized:
                    localized_state = candidate
                    break
        time.sleep(0.005)
    publisher.stop()

    assert localized_state is not None
    state = localized_state
    assert state.robot_localized
    assert state.robot_x_mm == pytest.approx(-1350.0)
    assert state.robot_y_mm == pytest.approx(-1200.0)
    assert state.robot_heading_rad == pytest.approx(1.5)
    assert state.localization_source == "odometry_imu"


def test_remote_perception_transport_does_not_wait_for_observer() -> None:
    class _NoClientServer(RemoteTcpServer):
        def start(self) -> None:
            self.started_for_test = True

        def stop(self) -> None:
            self.started_for_test = False

        def accept(self, timeout: float | None = None):
            del timeout
            time.sleep(0.01)
            raise TimeoutError

    server = _NoClientServer(
        host="127.0.0.1",
        port=8765,
        access_mode=RemoteAccessMode.OBSERVE_ONLY,
        connection_options=RemoteConnectionOptions(
            io_timeout_s=0.05,
            control_queue_capacity=4,
            observation_queue_capacity=2,
            max_header_bytes=4096,
            max_payload_bytes=1_000_000,
            non_actuating_control_topics=(RemoteTopic.VIDEO_MODE.value,),
        ),
    )
    transport = RemotePerceptionTransport(
        server,
        _observer_pipeline(),
        _observer_config(),
    )
    started_at = time.monotonic()
    transport.start()
    try:
        assert time.monotonic() - started_at < 0.2
        assert not transport.connected
    finally:
        transport.stop()
