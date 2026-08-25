from __future__ import annotations

import json

import pytest

from rescue_vision.communication import (
    CaptureAction,
    CaptureRecordingState,
    CaptureRequestResult,
    CaptureStatusObservation,
    ImageCoordinateSystem,
    MapStateObservation,
    MapTargetState,
    RemoteAccessMode,
    RemoteSessionStatus,
    RemoteTopic,
    TeamColor,
    VehicleMotionState,
    VehicleSafetyMode,
    VehicleStateObservation,
    VehicleStopReason,
    VideoFrameAttributes,
    VideoFrameMode,
)


def session_status(**overrides: object) -> RemoteSessionStatus:
    values = {
        "session_id": "session-001",
        "server_instance_id": "server-001",
        "timestamp_ns": 100,
        "access_mode": RemoteAccessMode.DEBUG_CONTROL,
        "motion_control_available": True,
        "gripper_control_available": True,
        "capture_control_available": True,
        "video_stream_available": True,
        "video_modes": (VideoFrameMode.RAW, VideoFrameMode.PERCEPTION),
        "map_state_available": False,
        "vehicle_state_available": True,
        "capture_status_available": True,
        "target_heading_control_available": False,
        "session_status_period_ms": 1000,
        "vehicle_state_period_ms": 100,
        "map_state_period_ms": None,
        "capture_status_period_ms": 500,
        "video_nominal_fps": 15.0,
        "max_linear_velocity_m_s": 0.25,
        "max_angular_velocity_rad_s": 1.0,
        "max_control_command_valid_for_ms": 500,
    }
    values.update(overrides)
    return RemoteSessionStatus(**values)  # type: ignore[arg-type]


def test_session_status_round_trip_and_topic() -> None:
    status = session_status()
    observe_only = session_status(
        access_mode=RemoteAccessMode.OBSERVE_ONLY,
        motion_control_available=False,
        gripper_control_available=False,
        capture_control_available=False,
        max_linear_velocity_m_s=None,
        max_angular_velocity_rad_s=None,
    )

    assert RemoteSessionStatus.from_payload(status.to_payload()) == status
    assert (
        RemoteSessionStatus.from_payload(observe_only.to_payload())
        == observe_only
    )
    assert (
        RemoteTopic.SESSION_STATUS.value == "observation/session/status"
    )
    unexpected = json.loads(status.to_payload())
    unexpected["legacy_field"] = 1
    with pytest.raises(ValueError, match="keys must be exactly"):
        RemoteSessionStatus.from_payload(
            json.dumps(unexpected).encode("utf-8")
        )


def test_session_status_rejects_unsafe_or_inconsistent_capabilities() -> None:
    with pytest.raises(ValueError, match="observe_only"):
        session_status(access_mode=RemoteAccessMode.OBSERVE_ONLY)
    with pytest.raises(ValueError, match="video_nominal_fps"):
        session_status(video_nominal_fps=None)
    with pytest.raises(ValueError, match="motion limits"):
        session_status(max_linear_velocity_m_s=None)
    with pytest.raises(ValueError, match="requires video"):
        session_status(
            video_stream_available=False,
            video_nominal_fps=None,
        )
    with pytest.raises(ValueError, match="gripper control requires"):
        session_status(
            motion_control_available=False,
            max_linear_velocity_m_s=None,
            max_angular_velocity_rad_s=None,
            video_stream_available=False,
            video_nominal_fps=None,
        )


def test_video_attributes_round_trip_and_coordinate_binding() -> None:
    raw = VideoFrameAttributes(
        frame_sequence=3,
        timestamp_ns=200,
        width=1280,
        height=720,
        coordinate_system=ImageCoordinateSystem.RAW_PIXEL,
        calibration_id=None,
    )
    undistorted = VideoFrameAttributes(
        frame_sequence=4,
        timestamp_ns=300,
        width=1280,
        height=720,
        coordinate_system=ImageCoordinateSystem.UNDISTORTED_PIXEL,
        calibration_id="camera-front-20260729",
        mode=VideoFrameMode.PERCEPTION,
    )
    bev = VideoFrameAttributes(
        frame_sequence=5,
        timestamp_ns=400,
        width=40,
        height=30,
        coordinate_system=ImageCoordinateSystem.BEV_PIXEL,
        calibration_id="camera-front-20260729",
        mode=VideoFrameMode.BEV,
        bev_x_min_mm=0.0,
        bev_x_max_mm=300.0,
        bev_y_min_mm=-200.0,
        bev_y_max_mm=200.0,
        bev_mm_per_pixel=10.0,
    )

    assert VideoFrameAttributes.from_attributes(raw.to_attributes()) == raw
    assert (
        VideoFrameAttributes.from_attributes(undistorted.to_attributes())
        == undistorted
    )
    assert VideoFrameAttributes.from_attributes(bev.to_attributes()) == bev
    with pytest.raises(ValueError, match="requires"):
        VideoFrameAttributes(
            frame_sequence=0,
            timestamp_ns=0,
            width=640,
            height=480,
            coordinate_system=ImageCoordinateSystem.UNDISTORTED_PIXEL,
            calibration_id=None,
        )
    with pytest.raises(ValueError, match="dimensions"):
        VideoFrameAttributes(
            frame_sequence=0,
            timestamp_ns=0,
            width=41,
            height=30,
            coordinate_system=ImageCoordinateSystem.BEV_PIXEL,
            calibration_id="test",
            mode=VideoFrameMode.BEV,
            bev_x_min_mm=0.0,
            bev_x_max_mm=300.0,
            bev_y_min_mm=-200.0,
            bev_y_max_mm=200.0,
            bev_mm_per_pixel=10.0,
        )
    unexpected = raw.to_attributes()
    unexpected["legacy_field"] = True
    with pytest.raises(ValueError, match="keys must be exactly"):
        VideoFrameAttributes.from_attributes(unexpected)


def test_map_state_round_trip_with_robot_and_confirmed_target() -> None:
    state = MapStateObservation(
        state_sequence=3,
        timestamp_ns=1_000,
        team_color=TeamColor.BLUE,
        robot_localized=True,
        robot_x_mm=100.0,
        robot_y_mm=-200.0,
        robot_heading_rad=0.5,
        localization_timestamp_ns=900,
        localization_confidence=0.75,
        localization_position_uncertainty_mm=30.0,
        localization_heading_uncertainty_rad=0.1,
        localization_source="blue_safe_zone",
        targets=(
            MapTargetState(
                track_id=7,
                target_class="green_supply",
                x_mm=-200.0,
                y_mm=300.0,
                observation_timestamp_ns=850,
                confidence=0.8,
                position_uncertainty_mm=40.0,
            ),
        ),
    )

    assert MapStateObservation.from_payload(state.to_payload()) == state
    assert RemoteTopic.MAP_STATE.value == "observation/map/state"


def test_map_state_rejects_partial_robot_and_duplicate_targets() -> None:
    with pytest.raises(ValueError, match="all be present"):
        MapStateObservation(
            state_sequence=0,
            timestamp_ns=100,
            team_color=TeamColor.UNKNOWN,
            robot_localized=True,
            robot_x_mm=0.0,
            robot_y_mm=None,
            robot_heading_rad=0.0,
            localization_timestamp_ns=90,
            localization_confidence=0.5,
            localization_position_uncertainty_mm=10.0,
            localization_heading_uncertainty_rad=0.1,
            localization_source="prior",
        )
    target = MapTargetState(1, "green_supply", 0.0, 0.0, 90, 0.5, 10.0)
    with pytest.raises(ValueError, match="unique track_id"):
        MapStateObservation(
            state_sequence=0,
            timestamp_ns=100,
            team_color=TeamColor.UNKNOWN,
            robot_localized=False,
            robot_x_mm=None,
            robot_y_mm=None,
            robot_heading_rad=None,
            localization_timestamp_ns=None,
            localization_confidence=None,
            localization_position_uncertainty_mm=None,
            localization_heading_uncertainty_rad=None,
            localization_source=None,
            targets=(target, target),
        )


def test_vehicle_state_round_trip_and_safety_invariants() -> None:
    state = VehicleStateObservation(
        state_sequence=5,
        timestamp_ns=500,
        control_ready=True,
        safety_mode=VehicleSafetyMode.FIRMWARE_WATCHDOG,
        uart_connected=True,
        watchdog_armed=True,
        emergency_stop_latched=False,
        motion_state=VehicleMotionState.MOVING,
        stop_reason=VehicleStopReason.NONE,
        controller_uptime_ms=1200,
        target_left_velocity_m_s=0.2,
        target_right_velocity_m_s=0.15,
        measured_left_velocity_m_s=0.19,
        measured_right_velocity_m_s=0.14,
        gripper_left_angle_deg=27.0,
        gripper_right_angle_deg=167.0,
        heading_rad=None,
        heading_reference=None,
        last_received_motion_command_id="drive-005",
        last_applied_motion_command_id="drive-005",
        last_received_gripper_command_id="grip-005",
        last_applied_gripper_command_id="grip-005",
    )

    assert VehicleStateObservation.from_payload(state.to_payload()) == state
    unexpected = json.loads(state.to_payload())
    unexpected["legacy_field"] = 2
    with pytest.raises(ValueError, match="keys must be exactly"):
        VehicleStateObservation.from_payload(
            json.dumps(unexpected).encode("utf-8")
        )
    with pytest.raises(ValueError, match="control_ready requires"):
        VehicleStateObservation(
            state_sequence=0,
            timestamp_ns=0,
            control_ready=True,
            safety_mode=VehicleSafetyMode.FIRMWARE_WATCHDOG,
            uart_connected=False,
            watchdog_armed=True,
            emergency_stop_latched=False,
            motion_state=VehicleMotionState.UNKNOWN,
            stop_reason=VehicleStopReason.UNKNOWN,
            controller_uptime_ms=None,
            target_left_velocity_m_s=None,
            target_right_velocity_m_s=None,
            measured_left_velocity_m_s=None,
            measured_right_velocity_m_s=None,
            gripper_left_angle_deg=None,
            gripper_right_angle_deg=None,
            heading_rad=None,
            heading_reference=None,
            last_received_motion_command_id=None,
            last_applied_motion_command_id=None,
            last_received_gripper_command_id=None,
            last_applied_gripper_command_id=None,
        )

    supervised = VehicleStateObservation(
        state_sequence=6,
        timestamp_ns=600,
        control_ready=True,
        safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        uart_connected=True,
        watchdog_armed=False,
        emergency_stop_latched=False,
        motion_state=VehicleMotionState.UNKNOWN,
        stop_reason=VehicleStopReason.UNKNOWN,
        controller_uptime_ms=None,
        target_left_velocity_m_s=None,
        target_right_velocity_m_s=None,
        measured_left_velocity_m_s=None,
        measured_right_velocity_m_s=None,
        gripper_left_angle_deg=None,
        gripper_right_angle_deg=None,
        heading_rad=None,
        heading_reference=None,
        last_received_motion_command_id=None,
        last_applied_motion_command_id=None,
        last_received_gripper_command_id=None,
        last_applied_gripper_command_id=None,
    )
    assert (
        VehicleStateObservation.from_payload(supervised.to_payload())
        == supervised
    )
    with pytest.raises(ValueError, match="watchdog_armed=true"):
        VehicleStateObservation(
            state_sequence=7,
            timestamp_ns=700,
            control_ready=True,
            safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
            uart_connected=True,
            watchdog_armed=True,
            emergency_stop_latched=False,
            motion_state=VehicleMotionState.UNKNOWN,
            stop_reason=VehicleStopReason.UNKNOWN,
            controller_uptime_ms=None,
            target_left_velocity_m_s=None,
            target_right_velocity_m_s=None,
            measured_left_velocity_m_s=None,
            measured_right_velocity_m_s=None,
            gripper_left_angle_deg=None,
            gripper_right_angle_deg=None,
            heading_rad=None,
            heading_reference=None,
            last_received_motion_command_id=None,
            last_applied_motion_command_id=None,
            last_received_gripper_command_id=None,
            last_applied_gripper_command_id=None,
        )


def test_vehicle_state_gripper_angles_must_be_paired_and_bounded() -> None:
    state = VehicleStateObservation(
        state_sequence=0,
        timestamp_ns=0,
        control_ready=True,
        safety_mode=VehicleSafetyMode.SUPERVISED_PHYSICAL_STOP,
        uart_connected=True,
        watchdog_armed=False,
        emergency_stop_latched=False,
        motion_state=VehicleMotionState.UNKNOWN,
        stop_reason=VehicleStopReason.UNKNOWN,
        controller_uptime_ms=None,
        target_left_velocity_m_s=None,
        target_right_velocity_m_s=None,
        measured_left_velocity_m_s=None,
        measured_right_velocity_m_s=None,
        gripper_left_angle_deg=None,
        gripper_right_angle_deg=None,
        heading_rad=None,
        heading_reference=None,
        last_received_motion_command_id=None,
        last_applied_motion_command_id=None,
        last_received_gripper_command_id=None,
        last_applied_gripper_command_id=None,
    )
    values = {
        field: getattr(state, field)
        for field in state.__dataclass_fields__
    }
    values["gripper_left_angle_deg"] = 27.0
    with pytest.raises(ValueError, match="both be present"):
        VehicleStateObservation(**values)
    values["gripper_right_angle_deg"] = 181.0
    with pytest.raises(ValueError, match=r"\[0, 180\]"):
        VehicleStateObservation(**values)


def test_capture_status_round_trip_and_strict_keys() -> None:
    status = CaptureStatusObservation(
        status_sequence=6,
        timestamp_ns=600,
        recording_state=CaptureRecordingState.RECORDING,
        recording_id="recording-001",
        accepted_frames=10,
        written_frames=9,
        dropped_frames=1,
        available_disk_bytes=1_000_000,
        last_request_id="capture-001",
        last_request_action=CaptureAction.START,
        last_request_result=CaptureRequestResult.COMPLETED,
        last_request_artifact_id="recording-001",
        stop_reason=None,
        error_code=None,
        error_message=None,
    )

    assert CaptureStatusObservation.from_payload(status.to_payload()) == status
    document = json.loads(status.to_payload())
    document["output_path"] = "/tmp/not-allowed"
    with pytest.raises(ValueError, match="keys must be exactly"):
        CaptureStatusObservation.from_payload(
            json.dumps(document).encode("utf-8")
        )
    rejected = CaptureStatusObservation(
        status_sequence=7,
        timestamp_ns=700,
        recording_state=CaptureRecordingState.IDLE,
        recording_id=None,
        accepted_frames=0,
        written_frames=0,
        dropped_frames=0,
        available_disk_bytes=None,
        last_request_id="capture-002",
        last_request_action=CaptureAction.STOP,
        last_request_result=CaptureRequestResult.REJECTED,
        last_request_artifact_id=None,
        stop_reason=None,
        error_code="state_conflict",
        error_message="No recording is active",
    )
    assert CaptureStatusObservation.from_payload(rejected.to_payload()) == rejected
