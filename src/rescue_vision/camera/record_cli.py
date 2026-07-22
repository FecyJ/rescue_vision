"""使用真实相机创建可确定性回放的记录目录。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from rescue_vision.camera.frame import FrameSource
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.recording import FrameRecorder
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.config.runtime import load_runtime_config
from rescue_vision.data.split_manifest import REQUIRED_TAGS
from rescue_vision.versioning import git_version


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record raw camera frames without blocking the live source."
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
        versions={
            "code": git_version(),
            "config_schema": str(config.schema_version),
            "opencv": cv2.__version__,
        },
        session_tags=session_tags,
        queue_capacity=config.recording.queue_capacity,
        image_format=config.recording.image_format,
    )

    delivered = 0
    try:
        camera.start()
        recorder.start()
        started_ns = time.monotonic_ns()
        while True:
            frame = camera.read()
            recorder.record(frame)
            delivered += 1
            if args.frames is not None and delivered >= args.frames:
                break
            if args.duration_seconds is not None:
                elapsed = (time.monotonic_ns() - started_ns) / 1_000_000_000
                if elapsed >= args.duration_seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            camera.stop()
        finally:
            recorder.stop()

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
