from __future__ import annotations

import math

import pytest

from rescue_vision.communication import (
    CaptureAction,
    DebugCaptureCommand,
    DebugMotionCommand,
    HeadingReference,
    MotionControlMode,
    RemoteTopic,
)


def test_debug_motion_twist_round_trip() -> None:
    command = DebugMotionCommand(
        command_id="drive-001",
        issued_timestamp_ns=123,
        valid_for_ms=250,
        deadman_enabled=True,
        control_mode=MotionControlMode.TWIST,
        linear_velocity_m_s=-0.2,
        angular_velocity_rad_s=0.5,
    )

    restored = DebugMotionCommand.from_payload(command.to_payload())

    assert restored == command
    assert RemoteTopic.DEBUG_MOTION.value == "control/debug/motion"


def test_debug_motion_reserves_target_heading_but_requires_explicit_mode() -> None:
    command = DebugMotionCommand(
        command_id="heading-001",
        issued_timestamp_ns=456,
        valid_for_ms=200,
        deadman_enabled=True,
        control_mode=MotionControlMode.TARGET_HEADING,
        linear_velocity_m_s=0.1,
        angular_velocity_rad_s=0.0,
        target_heading_rad=math.pi / 2,
        heading_reference=HeadingReference.SESSION_START,
    )

    assert DebugMotionCommand.from_payload(command.to_payload()) == command
    with pytest.raises(ValueError, match="requires target_heading"):
        DebugMotionCommand(
            command_id="bad",
            issued_timestamp_ns=0,
            valid_for_ms=200,
            deadman_enabled=True,
            control_mode=MotionControlMode.TARGET_HEADING,
            linear_velocity_m_s=0.0,
            angular_velocity_rad_s=0.0,
        )
    with pytest.raises(ValueError, match="owns turn rate"):
        DebugMotionCommand(
            command_id="bad-rate",
            issued_timestamp_ns=0,
            valid_for_ms=200,
            deadman_enabled=True,
            control_mode=MotionControlMode.TARGET_HEADING,
            linear_velocity_m_s=0.0,
            angular_velocity_rad_s=0.5,
            target_heading_rad=0.0,
            heading_reference=HeadingReference.FIELD,
        )


def test_deadman_disabled_can_only_request_stop() -> None:
    stop = DebugMotionCommand(
        command_id="stop-001",
        issued_timestamp_ns=0,
        valid_for_ms=100,
        deadman_enabled=False,
        control_mode=MotionControlMode.TWIST,
        linear_velocity_m_s=0.0,
        angular_velocity_rad_s=0.0,
    )

    assert DebugMotionCommand.from_payload(stop.to_payload()) == stop
    with pytest.raises(ValueError, match="deadman disabled"):
        DebugMotionCommand(
            command_id="unsafe",
            issued_timestamp_ns=0,
            valid_for_ms=100,
            deadman_enabled=False,
            control_mode=MotionControlMode.TWIST,
            linear_velocity_m_s=0.1,
            angular_velocity_rad_s=0.0,
        )


def test_capture_commands_do_not_accept_remote_output_paths() -> None:
    start = DebugCaptureCommand(
        request_id="capture-001",
        issued_timestamp_ns=123,
        action=CaptureAction.START,
        session_tags={"lighting": "indoor_bright"},
    )
    mark = DebugCaptureCommand(
        request_id="mark-001",
        issued_timestamp_ns=456,
        action=CaptureAction.MARK_EVENT,
        label="turn-left",
    )

    assert DebugCaptureCommand.from_payload(start.to_payload()) == start
    assert DebugCaptureCommand.from_payload(mark.to_payload()) == mark
    assert b"output" not in start.to_payload()
    with pytest.raises(ValueError, match="requires label"):
        DebugCaptureCommand(
            request_id="bad-mark",
            issued_timestamp_ns=0,
            action=CaptureAction.MARK_EVENT,
        )


def test_control_payload_rejects_unknown_schema_fields() -> None:
    payload = (
        b'{"schema_version":1,"request_id":"x","issued_timestamp_ns":0,'
        b'"action":"stop","label":null,"session_tags":{},"path":"/tmp/x"}'
    )

    with pytest.raises(ValueError, match="keys must be exactly"):
        DebugCaptureCommand.from_payload(payload)

    boolean_version = (
        b'{"schema_version":true,"request_id":"x","issued_timestamp_ns":0,'
        b'"action":"stop","label":null,"session_tags":{}}'
    )
    with pytest.raises(ValueError, match="schema_version"):
        DebugCaptureCommand.from_payload(boolean_version)
