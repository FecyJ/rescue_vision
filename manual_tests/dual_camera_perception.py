"""主相机正式感知 + 无标定近场模型像素预览，共享一个 Hailo 后端。"""
from __future__ import annotations

from collections.abc import Sequence
from time import monotonic_ns

import cv2

from dual_camera_preview import preview_parser, run_preview, validate_args
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.record_cli import undistort_camera_frame
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.types import RawPixel
from rescue_vision.perception.detector import TargetPoseDetector
from rescue_vision.perception.types import ModelDetection, POSE_MODEL_CLASSES, TargetClass
from rescue_vision.perception.visualization import render_target_observations


def render_near_model(frame: CameraFrame, detections: Sequence[ModelDetection], *, detection_threshold: float, k0_threshold: float) -> CameraFrame:
    """后端反 letterbox 数值按实际输入解释为 RawPixel，仅供本地可视化。

    不产生 TargetObservation/地面点，不将原图坐标交给正式去畸变感知链。
    """
    preview = frame.image_bgr.copy()
    count = 0
    danger_count = 0
    class_counts = {kind.value: 0 for kind in POSE_MODEL_CLASSES}
    for detection in detections:
        if detection.confidence < detection_threshold:
            continue
        count += 1
        class_counts[detection.model_class.value] += 1
        danger = detection.model_class.value == "blue_danger"
        danger_count += int(danger)
        color = (0, 0, 255) if danger else (0, 255, 255)
        box = detection.box
        top_left = RawPixel(box.x_min, box.y_min)
        bottom_right = RawPixel(box.x_max, box.y_max)
        cv2.rectangle(preview, (round(top_left.u), round(top_left.v)),
                      (round(bottom_right.u), round(bottom_right.v)), color, 2)
        cv2.putText(preview, f"{detection.model_class.value} {detection.confidence:.2f}",
                    (round(top_left.u), max(20, round(top_left.v)-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        for index, keypoint in enumerate(detection.keypoints):
            if keypoint.point is None or keypoint.confidence < k0_threshold:
                continue
            point = RawPixel(keypoint.point.u, keypoint.point.v)
            pixel = (round(point.u), round(point.v))
            cv2.circle(preview, pixel, 5, (0, 0, 255), -1)
            cv2.putText(preview, f"K{index}", pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(preview, "NEAR raw_pixel | NO calibration / ground projection", (12, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return CameraFrame(frame.sequence, frame.timestamp_ns, preview, {
        **frame.metadata, "image_coordinate_system": "raw_pixel",
        "result_timestamp_ns": monotonic_ns(), "model_detections": count, "blue_danger_detections": danger_count,
        **{f"{name}_detections": count for name, count in class_counts.items()},
    })


def main() -> None:
    parser = preview_parser("双相机原图与 perception 预览：主相机去畸变/HSV/地面点，近场无标定模型框/K0。")
    args = parser.parse_args()
    validate_args(parser, args)
    config = load_runtime_config(args.config)
    geometry = config.build_geometry()
    if geometry is None:
        raise RuntimeError("主相机 perception 需要配置有效内参；近场相机不加载标定。")
    backend = config.hailo.build_backend()
    if backend is None:
        raise RuntimeError("perception 预览需要 hailo.enabled=true")
    # detector 接管后端资源；近场直接 infer，worker 串行访问同一个后端。
    with TargetPoseDetector(
        backend, detection_threshold=config.perception.detection_threshold,
        k0_threshold=config.perception.k0_threshold, color_classifier=config.perception.color_classifier,
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=geometry.ground_projector,
        center_cross_refinement=config.perception.center_cross_refinement,
        safe_zone_color=config.perception.safe_zone_color, gripper_color=config.perception.gripper_color,
    ) as detector:
        def process(stream: int, frame: CameraFrame) -> CameraFrame:
            if stream == 1:
                return render_near_model(
                    frame, backend.infer(frame.image_bgr),
                    detection_threshold=config.perception.detection_threshold,
                    k0_threshold=config.perception.k0_threshold,
                )
            undistorted = undistort_camera_frame(frame, camera_model=geometry.camera_model)
            result = detector.detect_realtime(undistorted, undistorted.image_bgr)
            preview = render_target_observations(
                undistorted.image_bgr, result.observations,
                dropped_stale_age_ms=result.dropped_stale_age_ms, field_features=result.field_features,
            )
            return CameraFrame(frame.sequence, frame.timestamp_ns, preview, {
                **undistorted.metadata, "result_timestamp_ns": monotonic_ns(),
                "targets": len(result.observations), "stale_dropped": result.stale_dropped,
                **{f"{kind.value}_detections": sum(item.target_class is kind for item in result.observations)
                   for kind in TargetClass},
            })
        try:
            run_preview(args, config, process)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
