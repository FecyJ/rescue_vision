"""等待电脑端连接，并发送真实相机的最新 JPEG 画面。"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

if __package__:
    from manual_tests.remote_camera_support import (
        SESSION_STATUS_PERIOD_MS,
        build_camera_session_status,
        build_perception_renderer,
        build_remote_camera_pipeline,
        refresh_session_status,
        send_video_frame,
    )
else:
    from remote_camera_support import (
        SESSION_STATUS_PERIOD_MS,
        build_camera_session_status,
        build_perception_renderer,
        build_remote_camera_pipeline,
        refresh_session_status,
        send_video_frame,
    )
from rescue_vision.communication import (
    RemoteDisconnectedError,
    RemoteRole,
    RemoteTopic,
    VideoFrameMode,
    VideoModeCommand,
)
from rescue_vision.config import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Accept one desktop client and publish latest camera frames as JPEG."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")

    config = load_runtime_config(args.config)
    if not config.remote.enabled:
        raise RuntimeError("remote.enabled must be true")
    if config.remote.role is not RemoteRole.SERVER:
        raise RuntimeError(
            "remote_video.py accepts desktop clients; set remote.role: server"
        )
    if args.video_fps > config.camera.fps:
        raise RuntimeError("--video-fps must not exceed camera.fps")

    server = config.remote.build_server()
    assert server is not None
    pipeline = build_remote_camera_pipeline(config)
    perception_renderer = build_perception_renderer(config)
    video_modes = (
        (VideoFrameMode.RAW, VideoFrameMode.PERCEPTION)
        if perception_renderer is not None
        else (VideoFrameMode.RAW,)
    )
    session_status = build_camera_session_status(
        config,
        server_instance_id=f"remote-video-{uuid.uuid4()}",
        video_nominal_fps=args.video_fps,
        capture_control_available=False,
        video_modes=video_modes,
    )
    video_period_ns = int(1_000_000_000 / args.video_fps)
    status_period_ns = SESSION_STATUS_PERIOD_MS * 1_000_000

    with server:
        print(
            f"listening={config.remote.host}:{server.bound_port} "
            f"access_mode={config.remote.access_mode.value}",
            flush=True,
        )
        try:
            if perception_renderer is not None:
                perception_renderer.start()
            pipeline.source.start()
            try:
                connection = server.accept(timeout=args.timeout_seconds)
                with connection:
                    connection.send_reliable_observation(
                        RemoteTopic.SESSION_STATUS.value,
                        session_status.to_payload(),
                        content_type="application/json",
                    )
                    print("client_connected=true", flush=True)
                    next_video_ns = 0
                    next_status_ns = time.monotonic_ns() + status_period_ns
                    sent_frames = 0
                    active_mode = VideoFrameMode.RAW
                    last_sent_sequence: int | None = None
                    minimum_perception_sequence: int | None = None
                    try:
                        while True:
                            frame = pipeline.prepare(
                                pipeline.source.read(timeout=1.0)
                            )
                            if (
                                active_mode is VideoFrameMode.PERCEPTION
                                and perception_renderer is not None
                            ):
                                perception_renderer.submit(frame)
                            try:
                                while True:
                                    message = connection.receive_control(
                                        timeout=0
                                    )
                                    if (
                                        message.topic != RemoteTopic.VIDEO_MODE.value
                                        or message.content_type != "application/json"
                                        or message.attributes
                                    ):
                                        raise ValueError(
                                            "remote_video.py only accepts "
                                            "control/video/mode requests."
                                        )
                                    command = VideoModeCommand.from_payload(
                                        message.payload
                                    )
                                    if command.mode not in video_modes:
                                        raise ValueError(
                                            f"Video mode {command.mode.value!r} "
                                            "is not available."
                                        )
                                    active_mode = command.mode
                                    last_sent_sequence = None
                                    if active_mode is VideoFrameMode.PERCEPTION:
                                        if perception_renderer is None:
                                            raise RuntimeError(
                                                "Perception video mode is not configured."
                                            )
                                        perception_renderer.clear_latest()
                                        minimum_perception_sequence = frame.sequence
                                    else:
                                        minimum_perception_sequence = None
                                    print(
                                        f"video_mode={active_mode.value} "
                                        f"request_id={command.request_id}",
                                        flush=True,
                                    )
                            except TimeoutError:
                                pass
                            now_ns = time.monotonic_ns()
                            if now_ns >= next_status_ns:
                                session_status = refresh_session_status(
                                    connection,
                                    session_status,
                                    timestamp_ns=now_ns,
                                )
                                next_status_ns = now_ns + status_period_ns
                            if now_ns < next_video_ns:
                                continue
                            output_frame = frame
                            if active_mode is VideoFrameMode.PERCEPTION:
                                if perception_renderer is None:
                                    raise RuntimeError(
                                        "Perception video mode is not configured."
                                    )
                                output_frame = perception_renderer.latest()
                                if output_frame is None:
                                    next_video_ns = now_ns + video_period_ns
                                    continue
                                if (
                                    minimum_perception_sequence is not None
                                    and output_frame.sequence
                                    < minimum_perception_sequence
                                ):
                                    next_video_ns = now_ns + video_period_ns
                                    continue
                            if output_frame.sequence == last_sent_sequence:
                                next_video_ns = now_ns + video_period_ns
                                continue
                            payload_bytes = send_video_frame(
                                connection,
                                output_frame,
                                pipeline,
                                jpeg_quality=args.jpeg_quality,
                                mode=active_mode,
                            )
                            last_sent_sequence = output_frame.sequence
                            sent_frames += 1
                            next_video_ns = now_ns + video_period_ns
                            if sent_frames == 1 or sent_frames % 30 == 0:
                                print(
                                    f"video_frames={sent_frames} "
                                    f"source_sequence={output_frame.sequence} "
                                    f"mode={active_mode.value} "
                                    f"jpeg_bytes={payload_bytes}",
                                    flush=True,
                                )
                    except RemoteDisconnectedError:
                        print("client_connected=false", flush=True)
                    except KeyboardInterrupt:
                        print("stopped_by_user=true", flush=True)
            finally:
                pipeline.source.stop()
        finally:
            if perception_renderer is not None:
                perception_renderer.stop()


if __name__ == "__main__":
    main()
