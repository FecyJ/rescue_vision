"""受监督近场多物资收拢入口；相机、规划与显示均为有界旁路。"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import signal
from threading import Event, Thread
import time

import cv2
import numpy as np

from rescue_vision.app.cluster_breakup import CameraPerceptionPump, EncoderTravelTracker
from rescue_vision.app.gripper_width_sequence import (
    GraspPreparationWorker,
    GraspPreparation,
    GraspPreparationSession,
    GripperWidthPickupSequence,
    GripperWidthPickupState,
)
from rescue_vision.app.near_field_grasp import (
    GraspTargetTracker,
    NearFieldGraspSelector,
)
from rescue_vision.config import load_runtime_config
from rescue_vision.app.session_log import _begin_time_named_log, _end_time_named_log
from rescue_vision.motion import (
    CarCommandReply, CarSystemStatus, CommandResult, GripperKinematics,
    MotionSynchronizationError, OdometryImu,
)
from rescue_vision.perception import PerceptionFrameRenderer

_PREFLIGHT_RETRY_WINDOW_NS = 5_000_000_000
_PREVIEW_WINDOW_NAME = "gripper-width"
_MAX_PREVIEW_WIDTH = 1280
_MAX_PREVIEW_HEIGHT = 720


def _age_ms_text(age_ns: int | None, *, digits: int = 3) -> str:
    if age_ns is None:
        return "none"
    return f"{age_ns / 1_000_000.0:.{digits}f}"


class _PreparationWorker:
    """正式与独立入口共用近场准备 worker，独立入口额外提供预览。"""

    def __init__(self, session, selector, renderer, local_preview):
        self._worker = GraspPreparationWorker(
            session,
            selector,
            diagnostics_callback=self._emit_diagnostics,
        )
        self.renderer = renderer
        self.selector = selector
        self.local_preview = local_preview
        self.exit_requested = False
        self._preview_error: str | None = None
        self._preview_stop = Event()
        self._preview_thread: Thread | None = None

    def _emit_diagnostics(self, diagnostics) -> None:
        for diagnostic in diagnostics:
            self.log(diagnostic.as_log_line())

    @property
    def error(self):
        return self._worker.error or self._preview_error

    def __enter__(self):
        self._worker.__enter__()
        if self.local_preview:
            self._preview_thread = Thread(
                target=self._preview_loop, name="near-field-preview", daemon=True
            )
            self._preview_thread.start()
        return self

    def __exit__(self, *_args):
        self._preview_stop.set()
        if self._preview_thread is not None:
            self._preview_thread.join()
        self._worker.__exit__()

    def submit(self, snapshot, locked_ids):
        self._worker.submit(
            snapshot,
            session_id=0,
            policy=self.selector.default_policy,
            locked_ids=locked_ids,
        )

    def log(self, text: str, *, flush: bool = True) -> None:
        print(text, flush=flush)

    def latest(self) -> GraspPreparation | None:
        return self._worker.latest(0)

    def _preview_loop(self) -> None:
        window_created = False
        try:
            try:
                cv2.namedWindow(_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    _PREVIEW_WINDOW_NAME, _MAX_PREVIEW_WIDTH, _MAX_PREVIEW_HEIGHT
                )
                window_created = True
            except Exception as exc:
                self._preview_error = f"preview:{type(exc).__name__}:{exc}"
                return
            while not self._preview_stop.is_set():
                rendered = self.renderer.latest()
                if rendered is not None:
                    cv2.imshow(
                        _PREVIEW_WINDOW_NAME,
                        _fit_preview_image(
                            _draw_plan(rendered.image_bgr, self.latest(), self.selector)
                        ),
                    )
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    self.exit_requested = True
                self._preview_stop.wait(0.03)
        except Exception as exc:
            self._preview_error = f"preview:{type(exc).__name__}:{exc}"
        finally:
            if window_created:
                cv2.destroyAllWindows()


def _draw_plan(image, prep, selector):
    preview = image.copy()
    if prep is None:
        return preview
    plan = prep.selection.plan
    ids = () if plan is None else plan.member_ids
    for target in prep.targets:
        box = target.observation.box
        color = (0, 255, 0) if target.track_id in ids else (0, 0, 255) if not target.selectable else (0, 220, 255)
        cv2.rectangle(preview, (round(box.x_min), round(box.y_min)), (round(box.x_max), round(box.y_max)), color, 2)
        cv2.putText(preview, f"id={target.track_id}", (round(box.x_min), max(15, round(box.y_min)-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    lines = [f"capture_ns={prep.capture_timestamp_ns} capture=unconfirmed"]
    if plan is not None:
        for region in plan.regions:
            try:
                pixels = selector.region_pixels(region)
                if all(np.isfinite(v) and abs(v) < 1e7 for p in pixels for v in (p.u, p.v)):
                    cv2.polylines(preview, [np.asarray([(round(p.u), round(p.v)) for p in pixels], np.int32)], True, (255, 255, 0), 1)
            except ValueError:
                pass
        score = plan.score
        lines += [f"ids={ids} width={plan.width_mm:.1f} open={plan.opening_width_mm:.1f}/{plan.maximum_opening_mm:.1f} mm",
                  f"forward={plan.forward_distance_mm:.1f} mm score={score.total:.3f}",
                  f"points={score.rule_points:.0f} orange={score.orange_priority:.0f} count={score.count:.2f}",
                  f"clearance={score.clearance:.2f} distance={score.distance:.2f} alignment={score.alignment:.2f}"]
    if prep.ready:
        lines.append(",".join(prep.selection.rejections)[:110] or "ready")
    else:
        lines.append(
            f"confirmation={prep.confirmation_count}/{prep.confirmation_required} "
            f"{','.join(prep.selection.rejections)[:90]}"
        )
    for index, line in enumerate(lines):
        cv2.putText(preview, line, (16, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
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


def _run_session(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
    local_preview: bool,
    once: bool,
) -> None:
    # 仅这个独立测试入口创建运动/UART；正式流程不调用本模块。
    from rescue_vision.app.manual_capture import build_camera_pipeline

    runtime_config = load_runtime_config(config_path)
    grasp_config = runtime_config.near_field_grasp
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
    odometry_calibration = runtime_config.motion.odometry.build_calibration()
    if odometry_calibration is None:
        raise RuntimeError(
            "gripper-width pickup requires motion.odometry calibration."
        )
    if runtime_config.motion.wheel_track_m is None:
        raise RuntimeError("gripper-width pickup requires motion.wheel_track_m.")
    encoder_tracker = EncoderTravelTracker(
        odometry_calibration,
        max_wheel_velocity_m_s=runtime_config.motion.max_wheel_velocity_m_s,
        max_consecutive_overrun_samples=(
            runtime_config.motion.odometry.max_consecutive_overrun_samples
        ),
    )
    renderer = PerceptionFrameRenderer(
        lambda: runtime_config.build_target_pose_detector(
            ground_projector=pipeline.ground_projector,
        ),
        render_enabled=local_preview,
    )
    camera_pump = CameraPerceptionPump(
        pipeline.source,
        pipeline.prepare,
        renderer,
    )
    kinematics = GripperKinematics()
    selector = NearFieldGraspSelector(grasp_config, pipeline.ground_projector, kinematics,
        open_servo_angles_deg=(gripper.open_left_angle_deg, gripper.open_right_angle_deg),
        closed_servo_angles_deg=(gripper.closed_left_angle_deg, gripper.closed_right_angle_deg))
    target_tracker = GraspTargetTracker(runtime_config.tracking.build_tracker(), pipeline.ground_projector, grasp_config,
        max_relative_speed_mm_s=runtime_config.motion.max_wheel_velocity_m_s * 1000
        + runtime_config.match.green_alignment_max_angular_velocity_rad_s * grasp_config.max_range_mm)
    session = GraspPreparationSession(target_tracker, selector)
    pickup_sequence = GripperWidthPickupSequence(
        gripper_full_travel_time_s=gripper.full_travel_time_s,
        forward_speed_m_s=runtime_config.match.green_approach_speed_m_s,
        closed_servo_angles_deg=(gripper.closed_left_angle_deg, gripper.closed_right_angle_deg),
        max_observation_age_ms=runtime_config.processing.max_observation_age_ms,
        alignment_kp_rad_s=runtime_config.match.green_alignment_kp_rad_s,
        alignment_max_angular_velocity_rad_s=runtime_config.match.green_alignment_max_angular_velocity_rad_s,
            alignment_min_wheel_velocity_m_s=(
                runtime_config.match.green_alignment_min_wheel_velocity_m_s
            ),
            alignment_timeout_ms=grasp_config.alignment_timeout_ms,
            grasp_commit_max_observation_age_ms=(
                grasp_config.grasp_commit_max_observation_age_ms
            ),
            stationary_max_gyro_rad_s=grasp_config.stationary_max_gyro_rad_s,
            fine_alignment_zone_rad=grasp_config.fine_alignment_zone_rad,
            fine_alignment_min_wheel_velocity_m_s=(
                grasp_config.fine_alignment_min_wheel_velocity_m_s
            ),
        )

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
    worker = None
    last_odometry_ns = -1
    odometry_history: deque[tuple[int, float | None]] = deque(maxlen=1024)

    def record_odometry(message: OdometryImu) -> None:
        """保存编码器累计距离，并使用 UART 接收时刻做同钟对齐。"""
        nonlocal last_odometry_ns
        pickup_sequence.observe_motion(message)
        encoder_tracker.submit(message)
        received_timestamp_ns = getattr(message, "received_timestamp_ns", None)
        if isinstance(received_timestamp_ns, bool) or not isinstance(received_timestamp_ns, int) or received_timestamp_ns < 0:
            received_timestamp_ns = time.monotonic_ns()
        last_odometry_ns = received_timestamp_ns
        odometry_history.append((received_timestamp_ns, encoder_tracker.distance_m))

    try:
        with channel:
            try:
                # 相机/Hailo 预热与 UART 同步并行，避免启动期间不处理回传。
                camera_start_thread = camera_pump.start_in_background()

                def handle_sync_message(message: object) -> None:
                    nonlocal latest_status, last_odometry_ns
                    if isinstance(message, OdometryImu):
                        record_odometry(message)
                    elif isinstance(message, CarSystemStatus):
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
                    nonlocal latest_status, last_odometry_ns
                    controller.update(now_ns=time.monotonic_ns())
                    for message in controller.drain_messages():
                        if isinstance(message, OdometryImu):
                            record_odometry(message)
                        elif isinstance(message, CarSystemStatus):
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
                worker = _PreparationWorker(session, selector, renderer, local_preview)
                worker.__enter__()
                last_pickup_state = None
                last_pickup_reason = None
                last_diagnostic = None
                last_report_ns = 0
                while not stop_requested:
                    now_ns = time.monotonic_ns()
                    controller.update(now_ns=now_ns)
                    for message in controller.drain_messages():
                        if isinstance(message, OdometryImu):
                            record_odometry(message)
                        elif isinstance(message, CarSystemStatus):
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
                                worker.log(
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

                    try:
                        camera_pump.check_health()
                    except Exception as exc:
                        diagnostic = f"perception:{type(exc).__name__}:{exc}"
                        if diagnostic != last_diagnostic:
                            worker.log(diagnostic, flush=True)
                            last_diagnostic = diagnostic
                    if worker.error is not None and worker.error != last_diagnostic:
                        worker.log(f"sidecar_error={worker.error}", flush=True)
                        last_diagnostic = worker.error
                    snapshot = renderer.latest_fresh_snapshot(
                        now_ns,
                        runtime_config.processing.max_observation_age_ms,
                    )
                    # 进入动作后冻结静止阶段形成的计划；避免运动模糊触发
                    # 走廊重检，也避免规划线程继续消耗实时链路预算。
                    if (
                        snapshot is not None
                        and pickup_sequence.active_plan is None
                    ):
                        capture_distance = next((distance for stamp, distance in reversed(odometry_history)
                                                 if stamp <= snapshot.capture_timestamp_ns), None)
                        if capture_distance is not None:
                            worker.submit(
                                snapshot,
                                pickup_sequence.locked_ids,
                            )
                    preparation = worker.latest()
                    distance = encoder_tracker.distance_m if latest_status is not None and latest_status.gripper_output_available and now_ns - last_odometry_ns <= runtime_config.processing.max_observation_age_ms * 1e6 else None
                    now_ns = time.monotonic_ns()
                    decision = pickup_sequence.step(now_ns, preparation, cumulative_distance_m=distance)
                    if decision.soft_brake:
                        # 对准中丢失目标时会在多个控制周期保持等待；只在
                        # 首次进入该停车原因时发送软刹车，避免反复刷 UART。
                        if (
                            decision.state is not last_pickup_state
                            or decision.reason != last_pickup_reason
                        ):
                            controller.soft_brake()
                    else:
                        controller.drive_wheel_limited(
                            decision.linear_velocity_m_s,
                            decision.angular_velocity_rad_s,
                            min_wheel_velocity_m_s=(
                                decision.min_wheel_velocity_m_s
                            ),
                        )
                    if decision.gripper_angles_deg is not None:
                        controller.set_gripper_angles(*decision.gripper_angles_deg)
                        worker.log(
                            f"command=set_gripper reason={decision.reason} "
                            f"left_angle_deg={decision.gripper_angles_deg[0]:.2f} "
                            f"right_angle_deg={decision.gripper_angles_deg[1]:.2f}",
                            flush=True,
                        )
                    if decision.state is not last_pickup_state or decision.reason != last_pickup_reason or now_ns - last_report_ns >= 1_000_000_000:
                        plan_age = (
                            None
                            if preparation is None
                            else preparation.plan_age_ns(now_ns)
                        )
                        preparation_age = (
                            None
                            if preparation is None
                            else preparation.preparation_age_ns(now_ns)
                        )
                        worker.log(f"pickup_state={decision.state.value} reason={decision.reason} distance_m={distance} rx_degraded={None if latest_status is None else getattr(latest_status, 'rx_degraded', None)} "
                              f"plan_age_ms={_age_ms_text(plan_age)} "
                              f"preparation_age_ms={_age_ms_text(preparation_age)} "
                              f"confirmation={None if preparation is None else preparation.confirmation_progress} "
                              f"{pickup_sequence.motion_diagnostic(now_ns)}" if preparation is not None else
                              f"pickup_state={decision.state.value} reason={decision.reason} observation=none", flush=True)
                        if preparation is not None:
                            plan = preparation.selection.plan
                            worker.log(f"selection={None if plan is None else plan.member_ids} "
                                  f"score={None if plan is None else plan.score} rejections={preparation.selection.rejections}", flush=True)
                        last_report_ns = now_ns
                    if decision.state is GripperWidthPickupState.OPENING and last_pickup_state is not GripperWidthPickupState.OPENING:
                        plan_age = (
                            None
                            if preparation is None
                            else preparation.plan_age_ns(now_ns)
                        )
                        preparation_age = (
                            None
                            if preparation is None
                            else preparation.preparation_age_ns(now_ns)
                        )
                        worker.log(
                            "near_field_grasp_commit="
                            f"session=0 plan_age_ms={_age_ms_text(plan_age)} "
                            f"preparation_age_ms={_age_ms_text(preparation_age)} "
                            f"confirmation={None if preparation is None else preparation.confirmation_progress} "
                            f"ids={None if pickup_sequence.active_plan is None else pickup_sequence.active_plan.member_ids}",
                            flush=True,
                        )
                    if decision.state is GripperWidthPickupState.COMPLETE and last_pickup_state is not GripperWidthPickupState.COMPLETE:
                        worker.log(f"pickup_result={pickup_sequence.result}", flush=True)
                    last_pickup_state, last_pickup_reason = decision.state, decision.reason
                    if worker.exit_requested or once and decision.state is GripperWidthPickupState.COMPLETE:
                        break
                    time.sleep(0.005)
            finally:
                # 确保退出时底盘没有残留运动目标；夹爪角度保持最后一次显式命令。
                controller.soft_brake()
    finally:
        try:
            if worker is not None:
                worker.__exit__()
        finally:
            try:
                if camera_start_thread is not None:
                    camera_pump.stop()
            finally:
                signal.signal(signal.SIGINT, previous_sigint)
                signal.signal(signal.SIGTERM, previous_sigterm)


def _run(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
    local_preview: bool,
    once: bool,
    log_dir: Path | None = None,
) -> None:
    """运行近场收拢，并可选地把完整终端输出 tee 到日志目录。"""

    log_stream, original_stdout, original_stderr = _begin_time_named_log(
        log_dir,
        file_prefix="gripper_width_",
    )
    try:
        _run_session(
            config_path,
            supervised_stop_ready=supervised_stop_ready,
            local_preview=local_preview,
            once=once,
        )
    finally:
        _end_time_named_log(log_stream, original_stdout, original_stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select 1-3 green/black supplies or one orange target, align, gather and close; no turn or transport.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--supervised-physical-stop-ready", action="store_true", help="Confirm physical emergency stop and continuous supervision.")
    parser.add_argument("--local-preview", action="store_true")
    parser.add_argument("--once", action="store_true", help="Exit after one gathering action; capture remains unconfirmed.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for the time-named gripper-width log (default: logs).",
    )
    args = parser.parse_args()
    _run(args.config.expanduser().resolve(), supervised_stop_ready=args.supervised_physical_stop_ready,
         local_preview=args.local_preview, once=args.once,
         log_dir=args.log_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
