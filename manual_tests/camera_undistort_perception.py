"""按 runtime.yaml 实时预览去畸变后的任务目标 Pose 观测。"""

from __future__ import annotations

import argparse
from time import monotonic_ns

import cv2
import numpy as np

from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.camera.viewer import OpenCvFrameViewer
from rescue_vision.config import load_runtime_config
from rescue_vision.perception import TargetPoseDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument(
        "--frames",
        type=int,
        help="处理指定帧数后退出；默认持续运行到按 Q/Esc。",
    )
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")

    config = load_runtime_config(args.config)
    geometry = config.build_geometry()
    if geometry is None:
        raise RuntimeError(
            "geometry.intrinsics_enabled must be true for live perception."
        )

    backend = config.hailo.build_backend()
    if backend is None:
        raise RuntimeError(
            "hailo.enabled must be true for live perception."
        )

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

    # 检测器从构造开始接管 backend；参数校验失败时也会先关闭设备。
    detector = TargetPoseDetector(
        backend=backend,
        class_mapping=config.hailo.model_class_mapping(),
        detection_threshold=config.perception.detection_threshold,
        k0_threshold=config.perception.k0_threshold,
        color_classifier=config.perception.color_classifier,
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=geometry.ground_projector,
    )

    with (
        source,
        detector,
        OpenCvFrameViewer(
            "undistorted perception — Q/Esc to stop"
        ) as viewer,
    ):
        processed_frames = 0
        while args.frames is None or processed_frames < args.frames:
            frame = source.read(timeout=1.0)
            undistort_started_ns = monotonic_ns()
            undistorted_bgr = geometry.camera_model.undistort_image(
                frame.image_bgr
            )
            undistort_finished_ns = monotonic_ns()
            detection_result = detector.detect_realtime(
                frame,
                undistorted_bgr,
            )
            result_timestamp_ns = monotonic_ns()
            processed_frames += 1

            preview = undistorted_bgr.copy()
            for observation in detection_result.observations:
                box = observation.box
                segmentation = observation.color_segmentation
                roi_box = segmentation.roi_box
                roi = preview[
                    int(roi_box.y_min) : int(roi_box.y_max),
                    int(roi_box.x_min) : int(roi_box.x_max),
                ]
                selected = segmentation.mask != 0
                overlay_colors = {
                    "green_supply": (0, 255, 0),
                    "black_core": (80, 80, 80),
                    "orange_injured": (0, 128, 255),
                    "blue_danger": (255, 255, 0),
                    "unknown": (255, 0, 255),
                }
                overlay_color = overlay_colors[
                    segmentation.candidate_class.value
                ]
                if selected.any():
                    roi[selected] = (
                        roi[selected].astype("float32") * 0.55
                        + np.asarray(overlay_color, dtype=np.float32) * 0.45
                    ).astype("uint8")
                cv2.rectangle(
                    preview,
                    (round(box.x_min), round(box.y_min)),
                    (round(box.x_max), round(box.y_max)),
                    (0, 255, 0),
                    2,
                )
                if observation.k0 is not None:
                    cv2.circle(
                        preview,
                        (
                            round(observation.k0.u),
                            round(observation.k0.v),
                        ),
                        6,
                        (0, 0, 255),
                        -1,
                    )
                ground_text = (
                    ""
                    if observation.ground_point is None
                    else (
                        f" ({observation.ground_point.x:.0f},"
                        f"{observation.ground_point.y:.0f})mm"
                    )
                )
                cv2.putText(
                    preview,
                    (
                        f"{observation.target_class.value}"
                        f" conf={observation.detection_confidence:.2f}"
                        f" hsv={segmentation.candidate_class.value}"
                        f"/{segmentation.status.value}"
                        f" cov={segmentation.color_fraction:.2f}"
                        f" dom={segmentation.dominance:.2f}"
                        f"{ground_text}"
                    ),
                    (round(box.x_min), max(20, round(box.y_min) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            age_ms = (result_timestamp_ns - frame.timestamp_ns) / 1_000_000.0
            undistort_ms = (
                undistort_finished_ns - undistort_started_ns
            ) / 1_000_000.0
            status = (
                "STALE dropped: "
                f"{detection_result.dropped_stale_age_ms:.1f} ms"
                if detection_result.stale_dropped
                else f"age {age_ms:.1f} ms"
            )
            color = (
                (0, 0, 255)
                if detection_result.stale_dropped
                else (0, 255, 0)
            )
            cv2.putText(
                preview,
                f"{status} | undistort {undistort_ms:.1f} ms",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            if not viewer.show(preview):
                break


if __name__ == "__main__":
    main()
