from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from rescue_vision.app.manual_capture import CameraPipeline, CaptureSession
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.communication import (
    CaptureAction,
    CaptureRecordingState,
    CaptureRequestResult,
    DebugCaptureCommand,
    ImageCoordinateSystem,
)


def command(
    request_id: str,
    action: CaptureAction,
    *,
    label: str | None = None,
) -> DebugCaptureCommand:
    return DebugCaptureCommand(
        request_id=request_id,
        issued_timestamp_ns=1,
        action=action,
        label=label,
    )


def test_capture_session_executes_artifacts_and_deduplicates_requests(
    tmp_path,
) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("schema_version: 10\n", encoding="utf-8")
    config = SimpleNamespace(
        schema_version=10,
        camera=SimpleNamespace(image_size=(4, 3)),
        recording=SimpleNamespace(queue_capacity=2, image_format="jpg"),
    )
    pipeline = CameraPipeline(
        source=SimpleNamespace(),
        camera_model=None,
        coordinate_system=ImageCoordinateSystem.RAW_PIXEL,
        intrinsics_fingerprint_sha256=None,
    )
    session = CaptureSession(
        output_root=tmp_path,
        config=config,
        config_snapshot={"schema_version": 10},
        pipeline=pipeline,
    )
    frame = CameraFrame(
        sequence=4,
        timestamp_ns=123,
        image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
    )

    start = command("start-1", CaptureAction.START)
    session.accepted(start)
    assert session.status().last_request_result is CaptureRequestResult.ACCEPTED
    start_outcome = session.execute(start, frame)
    assert start_outcome.result is CaptureRequestResult.COMPLETED
    assert start_outcome.artifact_id is not None
    session.record(frame)

    mark = command("mark-1", CaptureAction.MARK_EVENT, label="turn")
    first_mark = session.execute(mark, frame)
    second_mark = session.execute(mark, frame)
    assert second_mark == first_mark
    recording_directory = (
        tmp_path / "recordings" / str(start_outcome.artifact_id)
    )
    event_lines = (
        recording_directory / "events.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 1

    snapshot = command(
        "snapshot-1",
        CaptureAction.SNAPSHOT,
        label="front",
    )
    first_snapshot = session.execute(snapshot, frame)
    second_snapshot = session.execute(snapshot, frame)
    assert second_snapshot == first_snapshot
    assert len(list((tmp_path / "snapshots").glob("*.jpg"))) == 1
    metadata_path = next((tmp_path / "snapshots").glob("*.json"))
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["label"] == "front"

    stop = command("stop-1", CaptureAction.STOP)
    stop_outcome = session.execute(stop, frame)
    assert stop_outcome.result is CaptureRequestResult.COMPLETED
    final_status = session.status()
    assert final_status.recording_state is CaptureRecordingState.IDLE
    assert final_status.accepted_frames == 1
    assert final_status.written_frames == 1
    assert (
        recording_directory / "session.json"
    ).read_text(encoding="utf-8").find('"completed": true') >= 0


def test_capture_session_rejects_state_conflicts(tmp_path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("schema_version: 10\n", encoding="utf-8")
    session = CaptureSession(
        output_root=tmp_path,
        config=SimpleNamespace(
            schema_version=10,
            camera=SimpleNamespace(image_size=(2, 2)),
            recording=SimpleNamespace(queue_capacity=1, image_format="png"),
        ),
        config_snapshot={"schema_version": 10},
        pipeline=CameraPipeline(
            source=SimpleNamespace(),
            camera_model=None,
            coordinate_system=ImageCoordinateSystem.RAW_PIXEL,
            intrinsics_fingerprint_sha256=None,
        ),
    )
    frame = CameraFrame(
        sequence=0,
        timestamp_ns=0,
        image_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
    )

    stop = session.execute(command("stop", CaptureAction.STOP), frame)
    mark = session.execute(
        command("mark", CaptureAction.MARK_EVENT, label="event"),
        frame,
    )
    assert stop.result is CaptureRequestResult.REJECTED
    assert mark.result is CaptureRequestResult.REJECTED
    assert session.status().error_code == "state_conflict"
