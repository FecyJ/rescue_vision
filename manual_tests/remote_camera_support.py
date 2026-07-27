"""远程相机人工测试脚本共享的装配与 JPEG 发布逻辑。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, replace

import cv2

from rescue_vision.camera.frame import CameraFrame, FrameSource
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.record_cli import undistort_camera_frame
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.communication import (
    ImageCoordinateSystem,
    RemoteMessageConnection,
    RemoteSessionStatus,
    RemoteTopic,
    VideoFrameAttributes,
)
from rescue_vision.config.runtime import AppConfig
from rescue_vision.geometry.camera_model import CameraModel


SESSION_STATUS_PERIOD_MS = 1000
CAPTURE_STATUS_PERIOD_MS = 500


@dataclass(frozen=True, slots=True)
class RemoteCameraPipeline:
    """配置驱动的真机源，以及其对外图像坐标身份。"""

    source: FrameSource
    camera_model: CameraModel | None
    coordinate_system: ImageCoordinateSystem
    intrinsics_fingerprint_sha256: str | None

    def prepare(self, frame: CameraFrame) -> CameraFrame:
        if self.camera_model is None:
            return frame
        return undistort_camera_frame(
            frame,
            camera_model=self.camera_model,
        )


def build_remote_camera_pipeline(config: AppConfig) -> RemoteCameraPipeline:
    camera_model = config.build_camera_model()
    source_class = (
        Picamera2Source
        if config.camera.backend == "picamera2"
        else RpicamSource
    )
    source = source_class(
        image_size=config.camera.image_size,
        fps=config.camera.fps,
        lens_position=config.camera.lens_position,
    )
    return RemoteCameraPipeline(
        source=source,
        camera_model=camera_model,
        coordinate_system=(
            ImageCoordinateSystem.UNDISTORTED_PIXEL
            if camera_model is not None
            else ImageCoordinateSystem.RAW_PIXEL
        ),
        intrinsics_fingerprint_sha256=(
            camera_model.calibration.fingerprint()
            if camera_model is not None
            else None
        ),
    )


def build_camera_session_status(
    config: AppConfig,
    *,
    server_instance_id: str,
    video_nominal_fps: float,
    capture_control_available: bool,
) -> RemoteSessionStatus:
    return RemoteSessionStatus(
        session_id=f"session-{uuid.uuid4()}",
        server_instance_id=server_instance_id,
        timestamp_ns=time.monotonic_ns(),
        access_mode=config.remote.access_mode,
        motion_control_available=False,
        capture_control_available=capture_control_available,
        video_stream_available=True,
        map_snapshot_available=False,
        vehicle_state_available=False,
        capture_status_available=capture_control_available,
        target_heading_control_available=False,
        session_status_period_ms=SESSION_STATUS_PERIOD_MS,
        vehicle_state_period_ms=None,
        map_snapshot_period_ms=None,
        capture_status_period_ms=(
            CAPTURE_STATUS_PERIOD_MS
            if capture_control_available
            else None
        ),
        video_nominal_fps=video_nominal_fps,
        max_linear_velocity_m_s=None,
        max_angular_velocity_rad_s=None,
        max_motion_command_valid_for_ms=500,
    )


def refresh_session_status(
    connection: RemoteMessageConnection,
    status: RemoteSessionStatus,
    *,
    timestamp_ns: int | None = None,
) -> RemoteSessionStatus:
    refreshed = replace(
        status,
        timestamp_ns=(
            time.monotonic_ns()
            if timestamp_ns is None
            else timestamp_ns
        ),
    )
    connection.send_reliable_observation(
        RemoteTopic.SESSION_STATUS.value,
        refreshed.to_payload(),
        content_type="application/json",
    )
    return refreshed


def send_video_frame(
    connection: RemoteMessageConnection,
    frame: CameraFrame,
    pipeline: RemoteCameraPipeline,
    *,
    jpeg_quality: int,
) -> int:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame.image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode the remote JPEG frame.")
    payload = encoded.tobytes()
    height, width = frame.image_bgr.shape[:2]
    attributes = VideoFrameAttributes(
        frame_sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        width=width,
        height=height,
        coordinate_system=pipeline.coordinate_system,
        intrinsics_fingerprint_sha256=(
            pipeline.intrinsics_fingerprint_sha256
        ),
    )
    connection.send_observation(
        RemoteTopic.VIDEO_FRAME.value,
        payload,
        content_type="image/jpeg",
        attributes=attributes.to_attributes(),
        sender_timestamp_ns=frame.timestamp_ns,
    )
    return len(payload)
