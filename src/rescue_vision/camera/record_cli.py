"""使用真实相机创建可确定性回放的记录目录。"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import cv2
import yaml

from rescue_vision.camera.frame import CameraFrame, FrameSource
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.recording import FrameRecorder
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.camera.viewer import OpenCvFrameViewer
from rescue_vision.config.runtime import load_runtime_config
from rescue_vision.data.split_manifest import REQUIRED_TAGS
from rescue_vision.geometry.camera_model import (
    IMAGE_BORDER_FILL_VALUE,
    CameraModel,
)
def parse_tags(values: list[str]) -> dict[str, str]:
    tags = {name: "unknown" for name in REQUIRED_TAGS}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Tag must use name=value, got {value!r}.")
        name, tag_value = value.split("=", 1)
        if name not in REQUIRED_TAGS:
            raise ValueError(
                f"Unknown tag {name!r}; expected one of {sorted(REQUIRED_TAGS)}."
            )
        if not tag_value:
            raise ValueError(f"Tag {name!r} cannot be empty.")
        tags[name] = tag_value
    return tags


def record_session(
    camera: FrameSource,
    recorder: FrameRecorder,
    *,
    frame_limit: int | None = None,
    duration_seconds: float | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    frame_transform: Callable[[CameraFrame], CameraFrame] | None = None,
    frame_observer: Callable[[CameraFrame], bool] | None = None,
) -> int:
    """启动帧源与记录器并采集，始终按相机、记录器顺序释放资源。"""

    if frame_limit is not None and frame_limit <= 0:
        raise ValueError("frame_limit must be positive.")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if frame_limit is not None and duration_seconds is not None:
        raise ValueError("frame_limit and duration_seconds are mutually exclusive.")

    delivered = 0
    primary_error: BaseException | None = None
    interrupted = False
    try:
        camera.start()
        recorder.start()
        started_ns = monotonic_ns()
        while True:
            frame = camera.read()
            if frame_transform is not None:
                transformed = frame_transform(frame)
                if (
                    transformed.sequence != frame.sequence
                    or transformed.timestamp_ns != frame.timestamp_ns
                ):
                    raise ValueError(
                        "frame_transform must preserve sequence and timestamp_ns."
                    )
                frame = transformed
            recorder.record(frame)
            delivered += 1
            if frame_observer is not None and not frame_observer(frame):
                break
            if frame_limit is not None and delivered >= frame_limit:
                break
            if duration_seconds is not None:
                elapsed = (monotonic_ns() - started_ns) / 1_000_000_000
                if elapsed >= duration_seconds:
                    break
    except KeyboardInterrupt:
        interrupted = True
    except BaseException as error:
        primary_error = error

    cleanup_errors: list[tuple[str, BaseException]] = []
    try:
        camera.stop()
    except BaseException as error:
        cleanup_errors.append(("camera.stop", error))
    try:
        recorder.stop()
    except BaseException as error:
        cleanup_errors.append(("recorder.stop", error))

    if primary_error is not None:
        for location, error in cleanup_errors:
            primary_error.add_note(f"{location} also failed: {error!r}")
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_errors:
        location, error = cleanup_errors[0]
        for later_location, later_error in cleanup_errors[1:]:
            error.add_note(f"{later_location} also failed: {later_error!r}")
        raise RuntimeError(f"Capture cleanup failed in {location}.") from error
    if interrupted:
        return delivered
    return delivered


def undistort_camera_frame(
    frame: CameraFrame,
    *,
    camera_model: CameraModel,
) -> CameraFrame:
    """把相机原始帧转换为感知统一使用的去畸变帧。"""

    metadata = dict(frame.metadata)
    metadata["image_coordinate_system"] = "undistorted_pixel"
    metadata["calibration_id"] = camera_model.calibration.calibration_id
    metadata["undistort_fill_value"] = IMAGE_BORDER_FILL_VALUE
    return CameraFrame(
        sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        image_bgr=camera_model.undistort_image(frame.image_bgr),
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record camera frames without blocking the live source; apply the "
            "configured intrinsics undistortion when enabled."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Session stratification tag. May be repeated; omitted required "
            "tags are stored explicitly as 'unknown'."
        ),
    )
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--frames", type=int, default=None)
    limit.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Show the post-undistortion frame being recorded; Q/Esc stops "
            "the recording cleanly."
        ),
    )
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")
    try:
        session_tags = parse_tags(args.tag)
    except ValueError as error:
        parser.error(str(error))

    config_path = args.config.expanduser().resolve()
    config = load_runtime_config(config_path)
    camera_model = config.build_camera_model()
    config_snapshot = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    camera: FrameSource
    if config.camera.backend == "picamera2":
        camera = Picamera2Source(
            image_size=config.camera.image_size,
            fps=config.camera.fps,
            lens_position=config.camera.lens_position,
        )
    else:
        camera = RpicamSource(
            image_size=config.camera.image_size,
            fps=config.camera.fps,
            lens_position=config.camera.lens_position,
        )
    recorder = FrameRecorder(
        args.output,
        image_size=config.camera.image_size,
        config_snapshot=config_snapshot,
        session_tags=session_tags,
        queue_capacity=config.recording.queue_capacity,
        image_format=config.recording.image_format,
        image_coordinate_system=(
            "undistorted_pixel" if camera_model is not None else "raw_pixel"
        ),
        calibration_id=(
            camera_model.calibration.calibration_id
            if camera_model is not None
            else None
        ),
        valid_pixel_ratio=(
            cv2.countNonZero(camera_model.valid_mask)
            / camera_model.valid_mask.size
            if camera_model is not None
            else None
        ),
        undistort_fill_value=(
            IMAGE_BORDER_FILL_VALUE if camera_model is not None else None
        ),
    )

    viewer = (
        OpenCvFrameViewer("rescue-vision-record — Q/Esc to stop")
        if args.display
        else None
    )
    try:
        delivered = record_session(
            camera,
            recorder,
            frame_limit=args.frames,
            duration_seconds=args.duration_seconds,
            frame_transform=(
                partial(undistort_camera_frame, camera_model=camera_model)
                if camera_model is not None
                else None
            ),
            frame_observer=(
                (lambda frame: viewer.show(frame.image_bgr))
                if viewer is not None
                else None
            ),
        )
    finally:
        if viewer is not None:
            viewer.close()

    if recorder.written_frames == 0:
        raise RuntimeError(
            f"Recording produced no frames; output is incomplete: "
            f"{args.output.resolve()}"
        )
    print(
        f"Delivered={delivered}, written={recorder.written_frames}, "
        f"dropped={recorder.dropped_frames}, output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
