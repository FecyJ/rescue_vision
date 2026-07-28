"""检查一次相机记录的完整性、采集速率和逐帧元数据覆盖。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.replay import RecordingSource
from rescue_vision.camera.viewer import OpenCvFrameViewer, playback_delay_ms
from rescue_vision.geometry.camera_model import IMAGE_BORDER_FILL_VALUE
from rescue_vision.motion.recording import (
    MANUAL_MOTION_LOG_FILENAME,
    MANUAL_MOTION_LOG_SCHEMA_VERSION,
    MANUAL_MOTION_STREAM_NAME,
    inspect_manual_motion_log,
)


PICAMERA2_METADATA = (
    "source",
    "sensor_timestamp_ns",
    "exposure_time_us",
    "analogue_gain",
    "digital_gain",
    "colour_temperature_k",
    "colour_gain_red",
    "colour_gain_blue",
    "lens_position",
    "frame_duration_us",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _non_negative_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer.")
    return value


def inspect_recording(
    session_directory: str | Path,
    *,
    frame_observer: Callable[[CameraFrame], bool] | None = None,
) -> dict[str, Any]:
    """完整回放记录并返回适合保存为 JSON 的诊断报告。"""

    directory = Path(session_directory).expanduser().resolve()
    session = _load_object(directory / "session.json")
    session_schema_version = session.get("schema_version")
    if session_schema_version not in {3, 4}:
        raise ValueError("Recording session schema_version must be 3 or 4.")
    auxiliary_streams = (
        session.get("auxiliary_streams")
        if session_schema_version == 4
        else {}
    )
    if not isinstance(auxiliary_streams, dict):
        raise ValueError("Recording auxiliary_streams must be a mapping.")
    recording_kind = (
        session.get("recording_kind")
        if session_schema_version == 4
        else "camera"
    )
    if recording_kind not in {"camera", "supervised_manual_motion"}:
        raise ValueError("Recording recording_kind is invalid.")
    if recording_kind == "supervised_manual_motion" and set(
        auxiliary_streams
    ) != {MANUAL_MOTION_STREAM_NAME}:
        raise ValueError(
            "supervised_manual_motion recording requires exactly the "
            "manual_motion auxiliary stream."
        )
    auxiliary_reports = _inspect_auxiliary_streams(
        directory,
        auxiliary_streams,
    )
    if session.get("completed") is not True:
        raise ValueError(f"Recording is not marked completed: {directory}.")

    statistics = session.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("Recording statistics must be a mapping.")
    accepted = _non_negative_integer(
        statistics.get("accepted_frames"),
        "statistics.accepted_frames",
    )
    written = _non_negative_integer(
        statistics.get("written_frames"),
        "statistics.written_frames",
    )
    dropped = _non_negative_integer(
        statistics.get("dropped_frames"),
        "statistics.dropped_frames",
    )

    source = RecordingSource(directory)
    sequences: list[int] = []
    timestamps_ns: list[int] = []
    luma_means: list[float] = []
    luma_standard_deviations: list[float] = []
    metadata_counts: dict[str, int] = {}
    active_observer = frame_observer
    with source:
        while True:
            try:
                frame = source.read()
            except EOFError:
                break
            sequences.append(frame.sequence)
            timestamps_ns.append(frame.timestamp_ns)
            gray = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
            luma_means.append(float(np.mean(gray)))
            luma_standard_deviations.append(float(np.std(gray)))
            for name, value in frame.metadata.items():
                if value is not None:
                    metadata_counts[name] = metadata_counts.get(name, 0) + 1
            if active_observer is not None and not active_observer(frame):
                active_observer = None

    frame_count = len(sequences)
    if frame_count == 0:
        raise ValueError("Recording contains no frames.")
    if written != frame_count:
        raise ValueError(
            "statistics.written_frames does not match replayed frames "
            f"({written} != {frame_count})."
        )
    if accepted != written:
        raise ValueError(
            "statistics.accepted_frames does not match written frames "
            f"({accepted} != {written})."
        )
    if any(
        current <= previous
        for previous, current in zip(sequences, sequences[1:])
    ):
        raise ValueError("Recorded frame sequences must be strictly increasing.")
    if any(
        current <= previous
        for previous, current in zip(timestamps_ns, timestamps_ns[1:])
    ):
        raise ValueError("Recorded frame timestamps must be strictly increasing.")
    manual_motion_report = auxiliary_reports.get(
        MANUAL_MOTION_STREAM_NAME
    )
    if isinstance(manual_motion_report, dict):
        motion_first_ns = int(manual_motion_report["first_timestamp_ns"])
        motion_last_ns = int(manual_motion_report["last_timestamp_ns"])
        if (
            motion_first_ns > timestamps_ns[0]
            or motion_last_ns < timestamps_ns[-1]
        ):
            raise ValueError(
                "Manual motion log time range must cover the recorded frame "
                "time range."
            )
        manual_motion_report["covers_frame_time_range"] = True

    elapsed_seconds = (
        (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000
        if frame_count > 1
        else 0.0
    )
    effective_fps = (
        (frame_count - 1) / elapsed_seconds if elapsed_seconds > 0 else None
    )
    attempts = accepted + dropped
    config = session.get("config")
    camera_config = config.get("camera") if isinstance(config, dict) else None
    configured_fps_value = (
        camera_config.get("fps") if isinstance(camera_config, dict) else None
    )
    configured_fps = (
        float(configured_fps_value)
        if isinstance(configured_fps_value, (int, float))
        and not isinstance(configured_fps_value, bool)
        and configured_fps_value > 0
        else None
    )

    return {
        "schema_version": 1,
        "recording_id": session.get("recording_id"),
        "recording_kind": recording_kind,
        "session_directory": str(directory),
        "image_size": session.get("image_size"),
        "image_format": session.get("image_format"),
        "image_coordinate_system": session.get("image_coordinate_system"),
        "intrinsics_fingerprint_sha256": session.get(
            "intrinsics_fingerprint_sha256"
        ),
        "valid_pixel_ratio": session.get("valid_pixel_ratio"),
        "undistort_fill_value": session.get("undistort_fill_value"),
        "auxiliary_streams": auxiliary_streams,
        "auxiliary_stream_reports": auxiliary_reports,
        "configured_fps": configured_fps,
        "frame_count": frame_count,
        "first_sequence": sequences[0],
        "last_sequence": sequences[-1],
        "sequence_gap_count": sum(
            current - previous - 1
            for previous, current in zip(sequences, sequences[1:])
        ),
        "elapsed_seconds": elapsed_seconds,
        "effective_fps": effective_fps,
        "accepted_frames": accepted,
        "written_frames": written,
        "dropped_frames": dropped,
        "drop_ratio": dropped / attempts if attempts else 0.0,
        "metadata_coverage": {
            name: {
                "present_frames": count,
                "ratio": count / frame_count,
            }
            for name, count in sorted(metadata_counts.items())
        },
        "luma": {
            "mean_min": min(luma_means),
            "mean_median": float(np.median(luma_means)),
            "mean_max": max(luma_means),
            "stddev_min": min(luma_standard_deviations),
            "stddev_median": float(np.median(luma_standard_deviations)),
            "stddev_max": max(luma_standard_deviations),
        },
    }


def _inspect_auxiliary_streams(
    directory: Path,
    streams: dict[str, Any],
) -> dict[str, object]:
    reports: dict[str, object] = {}
    for name, descriptor in streams.items():
        if name != MANUAL_MOTION_STREAM_NAME:
            raise ValueError(f"Unknown recording auxiliary stream {name!r}.")
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "schema_version",
            "path",
            "time_base",
        }:
            raise ValueError(
                f"Auxiliary stream {name!r} descriptor has invalid keys."
            )
        if descriptor["schema_version"] != MANUAL_MOTION_LOG_SCHEMA_VERSION:
            raise ValueError(
                f"Auxiliary stream {name!r} schema_version must be "
                f"{MANUAL_MOTION_LOG_SCHEMA_VERSION}."
            )
        if descriptor["path"] != MANUAL_MOTION_LOG_FILENAME:
            raise ValueError(
                f"Auxiliary stream {name!r} path must be "
                f"{MANUAL_MOTION_LOG_FILENAME!r}."
            )
        if descriptor["time_base"] != "application_monotonic_ns":
            raise ValueError(
                f"Auxiliary stream {name!r} has unsupported time_base."
            )
        reports[name] = inspect_manual_motion_log(
            directory / MANUAL_MOTION_LOG_FILENAME
        )
    return reports


def check_requirements(
    report: dict[str, Any],
    *,
    minimum_frames: int,
    maximum_drop_ratio: float,
    minimum_fps_ratio: float,
    required_metadata: tuple[str, ...] = (),
    require_undistorted: bool = False,
) -> list[str]:
    """根据显式门限返回失败原因；空列表表示通过。"""

    if minimum_frames <= 0:
        raise ValueError("minimum_frames must be positive.")
    if not 0 <= maximum_drop_ratio <= 1:
        raise ValueError("maximum_drop_ratio must be in [0, 1].")
    if not 0 < minimum_fps_ratio <= 1:
        raise ValueError("minimum_fps_ratio must be in (0, 1].")

    failures: list[str] = []
    if (
        require_undistorted
        and report.get("image_coordinate_system") != "undistorted_pixel"
    ):
        failures.append(
            "image_coordinate_system must be 'undistorted_pixel'"
        )
    if (
        require_undistorted
        and report.get("undistort_fill_value") != IMAGE_BORDER_FILL_VALUE
    ):
        failures.append(
            f"undistort_fill_value must be {IMAGE_BORDER_FILL_VALUE}"
        )
    frame_count = int(report["frame_count"])
    if frame_count < minimum_frames:
        failures.append(
            f"frame_count {frame_count} is below minimum {minimum_frames}"
        )
    drop_ratio = float(report["drop_ratio"])
    if drop_ratio > maximum_drop_ratio:
        failures.append(
            f"drop_ratio {drop_ratio:.3%} exceeds {maximum_drop_ratio:.3%}"
        )
    configured_fps = report.get("configured_fps")
    effective_fps = report.get("effective_fps")
    if configured_fps is None or effective_fps is None:
        failures.append("configured/effective FPS is unavailable")
    elif float(effective_fps) < float(configured_fps) * minimum_fps_ratio:
        failures.append(
            f"effective_fps {float(effective_fps):.3f} is below "
            f"{minimum_fps_ratio:.1%} of configured {float(configured_fps):.3f}"
        )
    metadata_coverage = report.get("metadata_coverage", {})
    for name in required_metadata:
        coverage = metadata_coverage.get(name, {})
        if coverage.get("present_frames") != frame_count:
            failures.append(f"metadata {name!r} is not present on every frame")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay and check one completed recording session."
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--minimum-frames", type=int, default=20)
    parser.add_argument("--maximum-drop-ratio", type=float, default=0.05)
    parser.add_argument("--minimum-fps-ratio", type=float, default=0.80)
    parser.add_argument(
        "--require-picamera2-metadata",
        action="store_true",
        help="Require sensor/exposure/gain/focus metadata on every frame.",
    )
    parser.add_argument(
        "--allow-raw",
        action="store_true",
        help=(
            "Allow raw_pixel recordings for calibration/diagnostics. "
            "Target-data checks require undistorted_pixel by default."
        ),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Replay frames in a window using capture timing. Q/Esc closes "
            "the window while verification continues headlessly."
        ),
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Display playback speed multiplier, default: 1.0.",
    )
    args = parser.parse_args()
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive and finite")

    viewer = (
        OpenCvFrameViewer(
            "rescue-vision-check-recording — Q/Esc closes preview"
        )
        if args.display
        else None
    )
    previous_timestamp_ns: int | None = None

    def observe_frame(frame: CameraFrame) -> bool:
        nonlocal previous_timestamp_ns
        assert viewer is not None
        delay_ms = playback_delay_ms(
            previous_timestamp_ns,
            frame.timestamp_ns,
            speed=args.playback_speed,
        )
        previous_timestamp_ns = frame.timestamp_ns
        keep_displaying = viewer.show(frame.image_bgr, delay_ms=delay_ms)
        if not keep_displaying:
            viewer.close()
        return keep_displaying

    try:
        report = inspect_recording(
            args.recording,
            frame_observer=observe_frame if viewer is not None else None,
        )
        failures = check_requirements(
            report,
            minimum_frames=args.minimum_frames,
            maximum_drop_ratio=args.maximum_drop_ratio,
            minimum_fps_ratio=args.minimum_fps_ratio,
            required_metadata=(
                PICAMERA2_METADATA if args.require_picamera2_metadata else ()
            ),
            require_undistorted=not args.allow_raw,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        cv2.error,
        RuntimeError,
        TypeError,
    ) as error:
        parser.error(str(error))
    finally:
        if viewer is not None:
            viewer.close()

    report["requirements"] = {
        "minimum_frames": args.minimum_frames,
        "maximum_drop_ratio": args.maximum_drop_ratio,
        "minimum_fps_ratio": args.minimum_fps_ratio,
        "required_metadata": (
            list(PICAMERA2_METADATA)
            if args.require_picamera2_metadata
            else []
        ),
        "require_undistorted": not args.allow_raw,
        "display": args.display,
        "playback_speed": args.playback_speed,
    }
    report["passed"] = not failures
    report["failures"] = failures
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
