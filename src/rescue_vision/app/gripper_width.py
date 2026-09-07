"""每秒按视觉宽度调整双舵机夹爪的独立测试程序。"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import cv2
import numpy as np

from rescue_vision.app.cluster_breakup import CameraPerceptionPump
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    CommandResult,
    GripperKinematics,
    MotionSynchronizationError,
)
from rescue_vision.perception import (
    GripperWidthEstimatorConfig,
    GripperWidthMeasurement,
    PerceptionFrameRenderer,
    PerceptionSnapshot,
    TargetObservation,
    average_gripper_width_measurements,
    estimate_gripper_width,
)


_MEASUREMENT_INTERVAL_NS = 1_000_000_000
_PREFLIGHT_RETRY_WINDOW_NS = 5_000_000_000
_PREVIEW_WINDOW_NAME = "gripper-width"
_MAX_PREVIEW_WIDTH = 1280
_MAX_PREVIEW_HEIGHT = 720


def _front_measurement(
    snapshot: PerceptionSnapshot,
    *,
    ground_projector: GroundProjector,
    config: GripperWidthEstimatorConfig,
) -> GripperWidthMeasurement | None:
    candidates: list[tuple[float, GripperWidthMeasurement]] = []
    for observation in snapshot.observations:
        measurement = estimate_gripper_width(
            observation,
            ground_projector,
            config,
        )
        if measurement is not None:
            ground_point = observation.ground_point
            assert ground_point is not None
            candidates.append((ground_point.x, measurement))
    if not candidates:
        return None
    # x 向车辆前方增大，最小 x 是当前横向居中候选中离车最近的物块。
    return min(candidates, key=lambda item: item[0])[1]


def _front_observation(
    snapshot: PerceptionSnapshot,
) -> TargetObservation | None:
    """返回快照中最前方的目标，不应用横向中心门限。"""

    candidates = [
        observation
        for observation in snapshot.observations
        if observation.ground_point is not None
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda observation: observation.ground_point.x
        if observation.ground_point is not None
        else float("inf"),
    )


def _front_object_text(
    observation: TargetObservation | None,
    *,
    center_y_half_range_mm: float,
) -> str:
    if observation is None or observation.ground_point is None:
        return "front_object=none"
    center_y = observation.ground_point.y
    center_gate = (
        "inside"
        if -center_y_half_range_mm < center_y < center_y_half_range_mm
        else "outside"
    )
    return (
        "front_object=("
        f"class={observation.target_class.value},"
        f"frame={observation.frame_sequence},"
        f"x_mm={observation.ground_point.x:+.2f},"
        f"y_mm={center_y:+.2f},"
        f"center_gate={center_gate})"
    )


def _print_cycle_result(
    *,
    samples_valid: int,
    sample_frames: int,
    measurement: GripperWidthMeasurement | None,
    servo_angles_deg: tuple[float, float] | None,
    front_observation: TargetObservation | None,
    center_y_half_range_mm: float,
    command: str,
    reason: str | None = None,
) -> None:
    def value_or_none(value: float | None) -> str:
        return "none" if value is None else f"{value:.2f}"

    line = (
        f"samples_valid={samples_valid}/{sample_frames} "
        f"center_y_mm={value_or_none(None if measurement is None else measurement.center_y_mm)} "
        f"width_mm={value_or_none(None if measurement is None else measurement.width_mm)} "
        f"opening_target_mm={value_or_none(None if measurement is None else measurement.opening_width_mm)} "
        f"left_angle_deg={value_or_none(None if servo_angles_deg is None else servo_angles_deg[0])} "
        f"right_angle_deg={value_or_none(None if servo_angles_deg is None else servo_angles_deg[1])} "
        f"{_front_object_text(front_observation, center_y_half_range_mm=center_y_half_range_mm)} "
        f"command={command}"
    )
    if reason is not None:
        line += f" reason={reason}"
    print(line, flush=True)


def _draw_measurements(
    image_bgr: np.ndarray,
    measurement: GripperWidthMeasurement | None,
    *,
    servo_angles_deg: tuple[float, float] | None = None,
) -> np.ndarray:
    preview = np.ascontiguousarray(image_bgr).copy()
    if measurement is None:
        cv2.putText(
            preview,
            "measurement=none",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview
    left = (
        round(measurement.left_edge_pixel.u),
        round(measurement.left_edge_pixel.v),
    )
    right = (
        round(measurement.right_edge_pixel.u),
        round(measurement.right_edge_pixel.v),
    )
    cv2.drawMarker(
        preview,
        left,
        (0, 255, 255),
        cv2.MARKER_TRIANGLE_UP,
        18,
        2,
    )
    cv2.drawMarker(
        preview,
        right,
        (255, 255, 0),
        cv2.MARKER_TRIANGLE_DOWN,
        18,
        2,
    )
    cv2.line(preview, left, right, (255, 255, 255), 2, cv2.LINE_AA)
    lines = [
        f"{measurement.target_class.value} frame={measurement.frame_sequence}",
        f"center_y={measurement.center_y_mm:+.1f} mm",
        f"left_y={measurement.left_y_mm:+.1f} mm  "
        f"right_y={measurement.right_y_mm:+.1f} mm",
        f"width={measurement.width_mm:.1f} mm  "
        f"open_target={measurement.opening_width_mm:.1f} mm",
    ]
    if servo_angles_deg is not None:
        lines.append(
            f"servo_left={servo_angles_deg[0]:.1f} deg  "
            f"servo_right={servo_angles_deg[1]:.1f} deg"
        )
    for index, text in enumerate(lines):
        cv2.putText(
            preview,
            text,
            (16, 34 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return preview


def _fit_preview_image(image_bgr: np.ndarray) -> np.ndarray:
    """把高分辨率相机帧缩放到适合本地窗口的尺寸，保持宽高比。"""

    height, width = image_bgr.shape[:2]
    scale = min(
        1.0,
        _MAX_PREVIEW_WIDTH / width,
        _MAX_PREVIEW_HEIGHT / height,
    )
    if scale >= 1.0:
        return image_bgr
    return cv2.resize(
        image_bgr,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _run(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
    sample_frames: int,
    center_y_half_range_mm: float,
    clearance_mm: float,
    min_mask_pixels: int,
    local_preview: bool,
    once: bool,
) -> None:
    # 仅这个独立测试入口创建运动/UART；正式流程不调用本模块。
    from rescue_vision.app.manual_capture import build_camera_pipeline

    runtime_config = load_runtime_config(config_path)
    if (
        isinstance(sample_frames, bool)
        or not isinstance(sample_frames, int)
        or sample_frames <= 0
    ):
        raise ValueError(
            "sample_frames must be a positive integer, "
            f"got {sample_frames!r}."
        )
    if not supervised_stop_ready:
        raise RuntimeError(
            "A physical emergency stop and continuous supervision are required "
            "for gripper-width control."
        )
    if not runtime_config.hailo.enabled:
        raise RuntimeError("gripper-width control requires hailo.enabled=true.")
    if not runtime_config.motion.enabled:
        raise RuntimeError("gripper-width control requires motion.enabled=true.")
    if not runtime_config.motion.gripper.enabled:
        raise RuntimeError(
            "gripper-width control requires motion.gripper.enabled=true."
        )
    if not runtime_config.uart.enabled:
        raise RuntimeError("gripper-width control requires uart.enabled=true.")

    pipeline = build_camera_pipeline(runtime_config)
    if pipeline.ground_projector is None:
        raise RuntimeError(
            "gripper-width requires a usable ground mapping with matching "
            "camera calibration."
        )
    gripper = runtime_config.motion.gripper.build_calibration()
    if gripper is None:
        raise RuntimeError("gripper-width control requires gripper calibration.")
    channel = runtime_config.uart.build_channel()
    if channel is None:
        raise RuntimeError("gripper-width control requires an enabled UART channel.")
    controller = runtime_config.motion.build_controller(channel)
    if controller is None:
        raise RuntimeError("gripper-width control requires an enabled controller.")

    estimator_config = GripperWidthEstimatorConfig(
        center_y_half_range_mm=center_y_half_range_mm,
        clearance_mm=clearance_mm,
        min_mask_pixels=min_mask_pixels,
    )
    renderer = PerceptionFrameRenderer(
        lambda: runtime_config.build_target_pose_detector(
            ground_projector=pipeline.ground_projector,
        )
    )
    camera_pump = CameraPerceptionPump(
        pipeline.source,
        pipeline.prepare,
        renderer,
    )
    kinematics = GripperKinematics()

    stop_requested = False
    latest_status: CarSystemStatus | None = None

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    camera_start_thread = None
    latest_measurement: GripperWidthMeasurement | None = None
    latest_servo_angles: tuple[float, float] | None = None
    preview_window_created = False
    try:
        with channel:
            try:
                # 相机/Hailo 预热与 UART 同步并行，避免启动期间不处理回传。
                camera_start_thread = camera_pump.start_in_background()

                def handle_sync_message(message: object) -> None:
                    nonlocal latest_status
                    if isinstance(message, CarSystemStatus):
                        latest_status = message
                        if message.emergency_stop_latched:
                            raise RuntimeError(
                                "STM32 emergency stop is latched during startup."
                            )

                sync_deadline_ns = (
                    time.monotonic_ns() + _PREFLIGHT_RETRY_WINDOW_NS
                )
                while True:
                    try:
                        controller.synchronize(
                            timeout_s=runtime_config.motion.synchronization_timeout_s,
                            on_message=handle_sync_message,
                        )
                        break
                    except MotionSynchronizationError:
                        if (
                            latest_status is not None
                            and latest_status.emergency_stop_latched
                        ):
                            raise RuntimeError(
                                "STM32 emergency stop is latched during startup."
                            )
                        if time.monotonic_ns() >= sync_deadline_ns:
                            raise
                        time.sleep(0.02)

                def service_uart_during_startup() -> None:
                    nonlocal latest_status
                    controller.update(now_ns=time.monotonic_ns())
                    for message in controller.drain_messages():
                        if isinstance(message, CarSystemStatus):
                            latest_status = message
                            if message.emergency_stop_latched:
                                raise RuntimeError(
                                    "STM32 emergency stop is latched during camera startup."
                                )

                camera_pump.wait_until_started(
                    camera_start_thread,
                    on_wait=service_uart_during_startup,
                )
                controller.query_state()
                if local_preview:
                    cv2.namedWindow(_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(
                        _PREVIEW_WINDOW_NAME,
                        _MAX_PREVIEW_WIDTH,
                        _MAX_PREVIEW_HEIGHT,
                    )
                    preview_window_created = True
                next_measurement_ns = time.monotonic_ns()
                sampling_active = False
                sample_frame_count = 0
                sample_measurements: list[GripperWidthMeasurement] = []
                sample_last_sequence: int | None = None
                last_snapshot_sequence: int | None = None
                sample_front_observation: TargetObservation | None = None
                while not stop_requested:
                    now_ns = time.monotonic_ns()
                    controller.update(now_ns=now_ns)
                    for message in controller.drain_messages():
                        if isinstance(message, CarSystemStatus):
                            latest_status = message
                            if message.emergency_stop_latched:
                                raise RuntimeError("STM32 emergency stop is latched.")
                            if (
                                not message.protocol_ready
                                or message.reply_queue_full
                                or message.tx_degraded
                            ):
                                raise RuntimeError(
                                    "STM32 UART health is degraded; "
                                    f"protocol_ready={message.protocol_ready},"
                                    f"reply_queue_full={message.reply_queue_full},"
                                    f"tx_degraded={message.tx_degraded}."
                                )
                            if not message.gripper_output_available:
                                raise RuntimeError(
                                    "STM32 gripper output is unavailable."
                                )
                        elif isinstance(message, CarCommandReply):
                            if message.command_type.name == "SET_GRIPPER":
                                print(
                                    "command_reply=set_gripper "
                                    f"sequence={message.command_sequence} "
                                    f"result={message.result.name.lower()}",
                                    flush=True,
                                )
                            if message.result is not CommandResult.ACCEPTED:
                                raise RuntimeError(
                                    "STM32 rejected command "
                                    f"{message.command_type.name.lower()}: "
                                    f"{message.result.name.lower()}."
                                )

                    camera_pump.check_health()
                    if not sampling_active and now_ns >= next_measurement_ns:
                        sampling_active = True
                        sample_frame_count = 0
                        sample_measurements = []
                        sample_last_sequence = None
                        sample_front_observation = None
                        print(
                            f"sampling=start target_frames={sample_frames}",
                            flush=True,
                        )

                    snapshot = renderer.latest_snapshot()
                    if (
                        sampling_active
                        and snapshot is not None
                        and snapshot.frame_sequence != last_snapshot_sequence
                    ):
                        if (
                            sample_last_sequence is not None
                            and snapshot.frame_sequence != sample_last_sequence + 1
                        ):
                            print(
                                "sampling=sequence_gap "
                                f"previous={sample_last_sequence} "
                                f"current={snapshot.frame_sequence} "
                                "action=keep_window",
                                flush=True,
                            )
                        sample_last_sequence = snapshot.frame_sequence
                        last_snapshot_sequence = snapshot.frame_sequence
                        sample_frame_count += 1
                        front_observation = _front_observation(snapshot)
                        if front_observation is not None:
                            sample_front_observation = front_observation
                        frame_measurement: GripperWidthMeasurement | None = None
                        age_ms = (
                            now_ns - snapshot.capture_timestamp_ns
                        ) / 1_000_000.0
                        if (
                            snapshot.dropped_stale_age_ms is None
                            and 0.0 <= age_ms
                            <= runtime_config.processing.max_observation_age_ms
                        ):
                            try:
                                frame_measurement = _front_measurement(
                                    snapshot,
                                    ground_projector=pipeline.ground_projector,
                                    config=estimator_config,
                                )
                            except ValueError as exc:
                                print(
                                    "sample=ignored "
                                    f"frame={snapshot.frame_sequence} "
                                    f"reason=invalid_ground_width_measurement:{exc}",
                                    flush=True,
                                )
                        if frame_measurement is not None:
                            sample_measurements.append(frame_measurement)
                            latest_measurement = frame_measurement
                        print(
                            f"sample_frame={sample_frame_count}/{sample_frames} "
                            f"frame={snapshot.frame_sequence} "
                            f"measurement={'valid' if frame_measurement is not None else 'lost_or_invalid'} "
                            f"{_front_object_text(front_observation, center_y_half_range_mm=center_y_half_range_mm)}",
                            flush=True,
                        )

                        if sample_frame_count >= sample_frames:
                            measurement = (
                                average_gripper_width_measurements(
                                    sample_measurements,
                                    clearance_mm=estimator_config.clearance_mm,
                                )
                                if sample_measurements
                                else None
                            )
                            latest_measurement = measurement
                            servo_angles: tuple[float, float] | None = None
                            command = "skipped"
                            reason: str | None = None
                            if measurement is None:
                                reason = "no_fresh_centered_measurable_target"
                            else:
                                try:
                                    servo_angles = kinematics.servo_angles_for_opening(
                                        measurement.opening_width_mm,
                                        open_left_angle_deg=gripper.open_left_angle_deg,
                                        open_right_angle_deg=gripper.open_right_angle_deg,
                                        closed_left_angle_deg=gripper.closed_left_angle_deg,
                                        closed_right_angle_deg=gripper.closed_right_angle_deg,
                                    )
                                except ValueError as exc:
                                    reason = f"angle_conversion:{exc}"
                                else:
                                    if (
                                        latest_status is None
                                        or not latest_status.gripper_output_available
                                    ):
                                        reason = "gripper_status_unavailable"
                                    else:
                                        assert servo_angles is not None
                                        controller.set_gripper_angles(*servo_angles)
                                        latest_servo_angles = servo_angles
                                        command = "set_gripper"
                            _print_cycle_result(
                                samples_valid=len(sample_measurements),
                                sample_frames=sample_frames,
                                measurement=measurement,
                                servo_angles_deg=servo_angles,
                                front_observation=sample_front_observation,
                                center_y_half_range_mm=center_y_half_range_mm,
                                command=command,
                                reason=reason,
                            )
                            sampling_active = False
                            while next_measurement_ns <= now_ns:
                                next_measurement_ns += _MEASUREMENT_INTERVAL_NS
                            if once and command == "set_gripper":
                                break

                    if local_preview:
                        rendered = renderer.latest()
                        if rendered is not None:
                            preview = _draw_measurements(
                                rendered.image_bgr,
                                latest_measurement,
                                servo_angles_deg=latest_servo_angles,
                            )
                            cv2.imshow(
                                _PREVIEW_WINDOW_NAME,
                                _fit_preview_image(preview),
                            )
                            key = cv2.waitKey(1) & 0xFF
                            if key in (ord("q"), 27):
                                stop_requested = True
                    time.sleep(0.005)
            finally:
                # 此程序不驱动车轮，但发送软刹车确保离开时没有残留运动目标。
                controller.soft_brake()
    finally:
        try:
            if camera_start_thread is not None:
                camera_pump.stop()
        finally:
            if local_preview and preview_window_created:
                cv2.destroyAllWindows()
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the nearest centered object's ground-projected width "
            "every second and set symmetric gripper servo angles."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help="Confirm physical emergency stop and continuous supervision.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=5,
        help="Number of consecutive snapshots per measurement (default: 5).",
    )
    parser.add_argument(
        "--center-y-half-range-mm",
        type=float,
        default=5.0,
        help="Strict center gate is -N < ground y < N (default: 5).",
    )
    parser.add_argument(
        "--clearance-mm",
        type=float,
        default=4.0,
        help="Opening margin added to measured width (default: 4).",
    )
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=1,
        help="Minimum accepted color-mask pixels (default: 1).",
    )
    parser.add_argument(
        "--local-preview",
        action="store_true",
        help="Show the latest perception image and measured edge markers.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after the first valid centered measurement.",
    )
    args = parser.parse_args()
    _run(
        args.config.expanduser().resolve(),
        supervised_stop_ready=args.supervised_physical_stop_ready,
        sample_frames=args.sample_frames,
        center_y_half_range_mm=args.center_y_half_range_mm,
        clearance_mm=args.clearance_mm,
        min_mask_pixels=args.min_mask_pixels,
        local_preview=args.local_preview,
        once=args.once,
    )


if __name__ == "__main__":
    main()
