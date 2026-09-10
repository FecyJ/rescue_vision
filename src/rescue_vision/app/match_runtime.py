"""正式流程和夹取—运送联调共用的硬件生命周期与控制循环。"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Callable, TYPE_CHECKING
from rescue_vision.app.match import (
    MatchPreflight,
    MatchStartArea,
    MatchSequence,
    MatchState,
    _print_state_banner,
    configure_match_start_area,
)
from rescue_vision.app.match_observers import _LocalPreview, _publish_remote_match_state
from rescue_vision.app.session_log import _begin_time_named_log, _end_time_named_log
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import normalize_angle
from rescue_vision.perception.types import UndistortedBoundingBox
if TYPE_CHECKING:
    from rescue_vision.config import AppConfig
    from rescue_vision.camera.frame import CameraFrame
    from rescue_vision.app.gripper_width_sequence import GraspPreparation
    from rescue_vision.app.near_field_grasp import NearFieldGraspSelector
_PREFLIGHT_RETRY_WINDOW_NS = 5_000_000_000


def _age_ms_text(age_ns: int | None, *, digits: int = 1) -> str:
    if age_ns is None:
        return "none"
    return f"{age_ns / 1_000_000.0:.{digits}f}"


def _overlay_near_field_corridor(
    frame: CameraFrame | None,
    preparation: GraspPreparation | None,
    selector: NearFieldGraspSelector | None,
) -> CameraFrame | None:
    """在同一采集帧上显示近场对准预览或真实规划走廊。"""

    if frame is None or preparation is None or selector is None:
        return frame
    selection = preparation.selection
    plan = selection.plan or selection.preview_plan
    if (
        plan is None
        or preparation.capture_timestamp_ns != frame.timestamp_ns
        or plan.frame_sequence != frame.sequence
        or plan.capture_timestamp_ns != frame.timestamp_ns
    ):
        return frame

    import cv2
    import numpy as np

    from rescue_vision.camera.frame import CameraFrame
    from rescue_vision.geometry.types import GroundPoint
    from rescue_vision.perception.detector import MODEL_GROUND_FORWARD_BIAS_MM

    image = frame.image_bgr.copy()
    overlay = image.copy()
    if plan.alignment_angle_rad != 0.0:
        color = (0, 220, 255)
        label = "ALIGN PREVIEW"
    elif selection.plan is None:
        color = (0, 0, 255)
        label = "CORRIDOR BLOCKED"
    elif preparation.ready:
        color = (0, 200, 0)
        label = "CORRIDOR READY"
    else:
        color = (255, 220, 0)
        label = "CORRIDOR VERIFY"
    label_origin: tuple[int, int] | None = None
    for region in plan.regions:
        try:
            pixels = selector.region_pixels(region)
        except ValueError:
            continue
        polygon = np.asarray(
            [(round(point.u), round(point.v)) for point in pixels],
            dtype=np.int32,
        )
        if polygon.shape != (len(region), 2) or len(polygon) < 3:
            continue
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)
        if label_origin is None:
            label_origin = tuple(int(value) for value in polygon[0])
    # 计划成员的观测框在同一采集帧内仍保留 track_id；用粗线高亮，避免只看
    # 走廊时无法判断究竟是哪些物块被纳入本次动作。
    selected_color = (0, 165, 255)
    for member in plan.members:
        box = member.observation.box
        cv2.rectangle(
            image,
            (round(box.x_min), round(box.y_min)),
            (round(box.x_max), round(box.y_max)),
            selected_color,
            5,
        )
        cv2.putText(
            image,
            f"SEL#{member.track_id}",
            (max(0, round(box.x_min)), max(20, round(box.y_min) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            selected_color,
            2,
            cv2.LINE_AA,
        )
    try:
        center_pixel = selector.projector.ground_to_pixels((
            GroundPoint(
                plan.alignment_point.x - MODEL_GROUND_FORWARD_BIAS_MM,
                plan.alignment_point.y,
            ),
        ))[0]
        center_xy = (round(center_pixel.u), round(center_pixel.v))
        cv2.drawMarker(
            image,
            center_xy,
            selected_color,
            cv2.MARKER_CROSS,
            22,
            3,
        )
        angle_length_mm = 180.0
        heading_pixel = selector.projector.ground_to_pixels((
            GroundPoint(
                plan.alignment_point.x - MODEL_GROUND_FORWARD_BIAS_MM
                + angle_length_mm * math.cos(plan.alignment_angle_rad),
                plan.alignment_point.y
                + angle_length_mm * math.sin(plan.alignment_angle_rad),
            ),
        ))[0]
        cv2.arrowedLine(
            image,
            center_xy,
            (round(heading_pixel.u), round(heading_pixel.v)),
            selected_color,
            3,
            cv2.LINE_AA,
            tipLength=0.2,
        )
    except (IndexError, ValueError):
        pass
    cv2.addWeighted(overlay, 0.22, image, 0.78, 0.0, image)
    if label_origin is not None:
        cv2.putText(
            image,
            f"{label} ids={plan.member_ids} width={plan.opening_width_mm:.0f}mm",
            (max(0, label_origin[0]), max(24, label_origin[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return CameraFrame(
        sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        image_bgr=image,
        metadata={
            **frame.metadata,
            "near_field_corridor": label,
            "near_field_selected_ids": ",".join(
                str(track_id) for track_id in plan.member_ids
            ),
        },
    )


def _overlay_match_selected_targets(
    frame: CameraFrame | None,
    sequence: MatchSequence,
    preparation: GraspPreparation | None = None,
) -> CameraFrame | None:
    """用统一样式高亮 match 当前选中的远场目标或近场目标组。

    ``MatchSequence`` 只返回与当前显示帧同一 ``frame_sequence`` 的轨迹；近场
    尚未锁定到 sequence 的候选则直接使用同帧准备计划中的成员框。这样不会把
    选中状态从旧帧错误地投影到最新图像。
    """

    if frame is None:
        return frame
    if not isinstance(sequence, MatchSequence):
        raise TypeError("sequence must be a MatchSequence.")

    plan = None
    if preparation is not None:
        plan = preparation.selection.plan or preparation.selection.preview_plan
    if plan is None:
        plan = sequence.near_field_active_plan
    plan_ids = frozenset(() if plan is None else plan.member_ids)
    boxes: dict[int, UndistortedBoundingBox] = {
        target.track_id: target.box
        for target in sequence.preview_selected_targets(
            frame.sequence,
            frame.timestamp_ns,
        )
        if target.track_id not in plan_ids
    }
    if preparation is not None and plan is not None:
        for target in preparation.targets:
            observation = target.observation
            if (
                target.track_id in plan_ids
                and observation.frame_sequence == frame.sequence
                and observation.capture_timestamp_ns == frame.timestamp_ns
            ):
                boxes[target.track_id] = observation.box
    if (
        plan is not None
        and preparation is not None
        and preparation.capture_timestamp_ns == frame.timestamp_ns
        and plan.frame_sequence == frame.sequence
        and plan.capture_timestamp_ns == frame.timestamp_ns
    ):
        for member in plan.members:
            boxes[member.track_id] = member.observation.box
    if not boxes:
        return frame

    import cv2
    from rescue_vision.camera.frame import CameraFrame

    selected_color = (0, 165, 255)
    image = frame.image_bgr.copy()
    for track_id, box in sorted(boxes.items()):
        cv2.rectangle(
            image,
            (round(box.x_min), round(box.y_min)),
            (round(box.x_max), round(box.y_max)),
            selected_color,
            5,
        )
        cv2.putText(
            image,
            f"SEL#{track_id}",
            (max(0, round(box.x_min)), max(20, round(box.y_min) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            selected_color,
            2,
            cv2.LINE_AA,
        )
    return CameraFrame(
        sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        image_bgr=image,
        metadata={
            **frame.metadata,
            "match_selected_ids": ",".join(str(track_id) for track_id in sorted(boxes)),
        },
    )


def _format_match_preview_localization(
    sequence: MatchSequence,
    heading_rad: float | None,
) -> tuple[str, ...]:
    """格式化 正式入口可用的受限航位信息。"""

    position = sequence.estimated_field_position
    visual_calibration = getattr(sequence, "safe_zone_calibration_pose", None)
    position_text = (
        "position=unavailable"
        if position is None
        else f"position=({position.x:+.0f},{position.y:+.0f})mm"
    )
    heading_text = (
        "heading=unavailable"
        if heading_rad is None
        else f"heading={math.degrees(normalize_angle(heading_rad)):+.1f}deg"
    )
    return (
        position_text,
        heading_text,
        "error=not_estimated",
        (
            "source=odometry+gyro+safe_zone_visual"
            if visual_calibration is not None
            else "source=odometry+gyro"
        ),
    )


def _run_hardware(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
    local_preview: bool = False,
    jpeg_quality: int = 80,
    observer_image_interval_s: float = 1.0,
    log_dir: Path | None = Path("logs"),
    start_area: MatchStartArea | str | int | None = None,
    sequence_factory: Callable[[AppConfig], MatchSequence]
    | None = None,
    initial_field_position: FieldPoint | None = None,
    initial_heading_rad: float | None = None,
    mode_name: str = "match",
    log_file_prefix: str = "match_",
    preview_title: str = "Match perception",
) -> None:
    """装配正式流程的相对视觉、编码器和陀螺仪控制循环。

    ``sequence_factory`` 和初始位姿参数供同一套绿色夹取—运输链路的受限测试
    入口复用；默认值保持正式流程入口的行为不变。
    """

    import signal
    import time
    from threading import Thread

    log_stream, original_stdout, original_stderr = _begin_time_named_log(
        log_dir,
        file_prefix=log_file_prefix,
        line_prefix="",
    )
    camera_pump = None
    camera_start_thread: Thread | None = None
    camera_started = False
    remote_transport = None
    near_field_worker = None
    near_field_selector = None
    preview = None
    d2_telemetry_logger = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    stop_requested = False

    try:
        from rescue_vision.app.cluster_breakup import (
            CameraPerceptionPump,
            EncoderTravelTracker,
            GripperPosture,
            RemotePerceptionTransport,
        )
        from rescue_vision.app.manual_capture import build_camera_pipeline
        from rescue_vision.app.gripper_width_sequence import (
            GraspPreparationSession,
            GraspPreparationWorker,
        )
        from rescue_vision.app.near_field_grasp import (
            GraspTargetTracker,
            NearFieldGraspSelector,
        )
        from rescue_vision.geometry.ground_projector import GroundProjector
        from rescue_vision.config import load_runtime_config
        from rescue_vision.communication import (
            RemoteAccessMode,
            RemoteRole,
            TeamColor as RemoteTeamColor,
        )
        from rescue_vision.motion import (
            CarCommandReply,
            CarSystemStatus,
            CommandResult,
            MotionSynchronizationError,
            OdometryImu,
            D2TelemetryLogger,
            GripperKinematics,
        )
        from rescue_vision.perception import PerceptionFrameRenderer, PerceptionSnapshot
        from rescue_vision.mission import SafetySignals

        config = load_runtime_config(config_path)
        selected_start_area = (
            None
            if start_area is None
            else MatchStartArea.parse(start_area)
        )
        if selected_start_area is not None:
            config = configure_match_start_area(config, selected_start_area)
        if sequence_factory is None:
            sequence_factory = MatchSequence.from_app_config
        if initial_field_position is not None and not isinstance(
            initial_field_position,
            FieldPoint,
        ):
            raise TypeError("initial_field_position must be a FieldPoint or None.")
        effective_initial_heading_rad = (
            config.localization.fusion.initial_pose.heading_rad
            if initial_heading_rad is None
            else float(initial_heading_rad)
        )
        if not math.isfinite(effective_initial_heading_rad):
            raise ValueError("initial_heading_rad must be finite when present.")
        effective_initial_position = (
            config.localization.fusion.initial_pose.position
            if initial_field_position is None
            else initial_field_position
        )
        initial_position_text = (
            f"({effective_initial_position.x:g},{effective_initial_position.y:g})mm"
        )
        start_area_text = (
            "config"
            if selected_start_area is None
            else selected_start_area.value
        )
        if not config.match.enabled:
            raise RuntimeError("match.enabled must be true.")
        if not supervised_stop_ready:
            raise RuntimeError(
                "A physical emergency stop and continuous supervision are "
                "required until the firmware watchdog has been verified."
            )
        if config.remote.enabled and (
            config.remote.role is not RemoteRole.SERVER
            or config.remote.access_mode is not RemoteAccessMode.OBSERVE_ONLY
        ):
            raise RuntimeError(
                "Match remote access must be server/observe_only."
            )

        channel = config.uart.build_channel()
        assert channel is not None
        controller = config.motion.build_controller(channel)
        assert controller is not None
        gripper = config.motion.gripper.build_calibration()
        if gripper is None or gripper.transport_angles_deg is None:
            raise RuntimeError(
                "Match strategy requires gripper calibration."
            )
        sequence = sequence_factory(config)
        print(
            "config="
            f"path={config_path.resolve()} mode={mode_name} "
            f"start_area={start_area_text} "
            f"start_position_field={initial_position_text} "
            f"start_heading_rad={effective_initial_heading_rad:g} "
            f"team_color={config.world.team_color.value} "
            f"required_transports={config.match.required_transports} "
            "motion_min_wheel_velocity_m_s="
            f"{config.motion.min_wheel_velocity_m_s:g} "
            "green_alignment_min_wheel_velocity_m_s="
            f"{config.match.green_alignment_min_wheel_velocity_m_s:g} "
            "green_alignment_tolerance_mm="
            f"{config.match.green_alignment_tolerance_mm:g} "
            "green_alignment_hysteresis_mm="
            f"{config.match.green_alignment_hysteresis_mm:g} "
            "green_alignment_timeout_ms="
            f"{config.match.green_alignment_timeout_ms:g} "
            "green_alignment_stable_frames="
            f"{config.match.green_alignment_stable_frames} "
            f"green_path_half_width_mm={config.match.green_path_half_width_mm:g} "
            "opportunistic_single_green="
            f"{config.match.opportunistic_single_green_enabled} "
            "opportunistic_single_green_clearance_mm="
            f"{config.match.opportunistic_single_green_clearance_mm:g} "
            "near_field_handoff_range_mm="
            f"{config.near_field_grasp.max_range_mm:g} "
            "near_field_max_targets="
            f"{config.near_field_grasp.max_targets} "
            "near_field_center_tolerance_mm="
            f"{config.near_field_grasp.center_tolerance_mm:g} "
            "near_field_alignment_hysteresis_mm="
            f"{config.near_field_grasp.alignment_hysteresis_mm:g} "
            "near_field_alignment_timeout_ms="
            f"{config.near_field_grasp.alignment_timeout_ms:g} "
            "near_field_grasp_commit_max_observation_age_ms="
            f"{config.near_field_grasp.grasp_commit_max_observation_age_ms:g} "
            "near_field_confirmation_frames="
            f"{config.near_field_grasp.confirmation_frames} "
            "near_field_fine_alignment_zone_rad="
            f"{config.near_field_grasp.fine_alignment_zone_rad:g} "
            "near_field_fine_alignment_min_wheel_velocity_m_s="
            f"{config.near_field_grasp.fine_alignment_min_wheel_velocity_m_s:g} "
            "orange_isolation_radius_mm="
            f"{config.near_field_grasp.orange_isolation_radius_mm:g} "
            "transport_corridor_half_width_mm="
            f"{sequence.transport_corridor_half_width_mm:g} "
            "transport_corridor_effective_half_width_mm="
            f"{sequence.transport_corridor_effective_half_width_mm:g} "
            f"safe_zone_d1_mm={config.match.safe_zone_calibration_start_offset_mm:g} "
            f"safe_zone_d2_mm={config.match.safe_zone_open_offset_mm:g} "
            "safe_zone_d2_to_final_max_wheel_acceleration_m_s2="
            f"{config.match.safe_zone_d2_to_final_max_wheel_acceleration_m_s2} "
            f"safe_zone_braking_overrun_mm=({config.match.safe_zone_d2_braking_overrun_x_mm:g},"
            f"{config.match.safe_zone_d2_braking_overrun_y_mm:g}) "
            f"action_settle_time_s={config.match.action_settle_time_s:g} "
            f"close_gripper_spin_angle_rad={config.match.spin_angle_rad:g} "
            f"breakup_field_half_extent_mm={config.match.breakup_field_half_extent_mm:g} "
            f"breakup_gripper_offset_mm={config.match.breakup_gripper_offset_mm:g}",
            flush=True,
        )
        print(
            "gripper_config="
            f"open=({gripper.open_left_angle_deg:g},{gripper.open_right_angle_deg:g}) "
            f"transport=({gripper.transport_angles_deg[0]:g},"
            f"{gripper.transport_angles_deg[1]:g}) "
            f"closed=({gripper.closed_left_angle_deg:g},"
            f"{gripper.closed_right_angle_deg:g}) "
            f"full_travel_s={gripper.full_travel_time_s:g}",
            flush=True,
        )

        odometry_calibration = config.motion.odometry.build_calibration()
        if odometry_calibration is None:
            raise RuntimeError("Match strategy requires odometry calibration.")
        encoder_tracker = EncoderTravelTracker(
            odometry_calibration,
            max_wheel_velocity_m_s=config.motion.max_wheel_velocity_m_s,
            max_consecutive_overrun_samples=(
                config.motion.odometry.max_consecutive_overrun_samples
            ),
        )
        pipeline = build_camera_pipeline(config)
        if (
            pipeline.ground_projector is None
            or not pipeline.ground_projector.supports_robot_projection
        ):
            raise RuntimeError(
                "Match visual calibration requires a ground mapping "
                "with full physical camera extrinsics."
            )
        renderer = PerceptionFrameRenderer(
            lambda: config.build_target_pose_detector(
                ground_projector=pipeline.ground_projector
            ),
            render_enabled=local_preview or config.remote.enabled,
        )
        camera_pump = CameraPerceptionPump(pipeline.source, pipeline.prepare, renderer)
        if local_preview:
            preview = _LocalPreview(title=preview_title)
        if config.remote.enabled:
            server = config.remote.build_server()
            assert server is not None
            remote_transport = RemotePerceptionTransport(
                server,
                pipeline,
                config,
                jpeg_quality=jpeg_quality,
                min_publish_interval_s=observer_image_interval_s,
                map_team_color=RemoteTeamColor(config.world.team_color.value),
            )
        latest_status: CarSystemStatus | None = None
        latest_snapshot: PerceptionSnapshot | None = None
        previous_odometry: OdometryImu | None = None
        latest_speed_feedback: tuple[float | None, float | None] = (None, None)
        gyro_heading_rad = effective_initial_heading_rad
        d2_telemetry_active = False
        process_started_timestamp_ns: int | None = None

        def ensure_near_field_worker() -> None:
            nonlocal near_field_worker, near_field_selector
            if near_field_worker is not None:
                return
            if not sequence.near_field_enabled:
                return
            if not isinstance(pipeline.ground_projector, GroundProjector):
                raise RuntimeError(
                    "Near-field match requires a GroundProjector instance."
                )
            selector = NearFieldGraspSelector(
                config.near_field_grasp,
                pipeline.ground_projector,
                GripperKinematics(),
                open_servo_angles_deg=(
                    gripper.open_left_angle_deg,
                    gripper.open_right_angle_deg,
                ),
                closed_servo_angles_deg=(
                    gripper.closed_left_angle_deg,
                    gripper.closed_right_angle_deg,
                ),
            )
            target_tracker = GraspTargetTracker(
                config.tracking.build_tracker(),
                pipeline.ground_projector,
                config.near_field_grasp,
                max_relative_speed_mm_s=(
                    config.motion.max_wheel_velocity_m_s * 1000.0
                    + config.match.green_alignment_max_angular_velocity_rad_s
                    * config.near_field_grasp.max_range_mm
                ),
            )

            def log_near_field_diagnostics(diagnostics) -> None:
                for item in diagnostics:
                    print(item.as_log_line(), flush=True)

            near_field_worker = GraspPreparationWorker(
                GraspPreparationSession(target_tracker, selector),
                selector,
                diagnostics_callback=log_near_field_diagnostics,
            )
            near_field_selector = selector
            near_field_worker.__enter__()

        def preview_heading_rad() -> float:
            calibrated = sequence.estimated_field_heading_rad
            return gyro_heading_rad if calibrated is None else calibrated

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True

        def process_timestamp_ms(timestamp_ns: int) -> float | None:
            started = process_started_timestamp_ns
            if started is None:
                return None
            return max(0.0, (timestamp_ns - started) / 1_000_000.0)

        def frame_process_timestamp_ms(frame: CameraFrame | None) -> float | None:
            if frame is None:
                return None
            return process_timestamp_ms(frame.timestamp_ns)

        def update_d2_telemetry_phase(
            decision: MatchDecision,
            timestamp_ns: int,
        ) -> None:
            nonlocal d2_telemetry_active
            logger = d2_telemetry_logger
            if logger is None:
                return
            if (
                not d2_telemetry_active
                and decision.state is MatchState.TRANSPORT_RELEASE
                and decision.reason.startswith("safe_zone_d2_")
            ):
                logger.begin_phase(
                    timestamp_ns=timestamp_ns,
                    state=decision.state.value,
                    route_phase=sequence.safe_zone_route_phase,
                    reason=decision.reason,
                )
                d2_telemetry_active = True
                print(
                    "d2_telemetry_phase=started "
                    f"state={decision.state.value} "
                    f"route_phase={sequence.safe_zone_route_phase} "
                    f"reason={decision.reason}",
                    flush=True,
                )
                return
            if not d2_telemetry_active:
                return
            if (
                decision.reason.startswith(
                    "safe_zone_reached_transport_endpoint_"
                )
                or decision.state is MatchState.TERMINAL_STOP
            ):
                logger.end_phase(
                    timestamp_ns=timestamp_ns,
                    state=decision.state.value,
                    route_phase=sequence.safe_zone_route_phase,
                    reason=decision.reason,
                )
                d2_telemetry_active = False
                print(
                    "d2_telemetry_phase=finished "
                    f"state={decision.state.value} "
                    f"route_phase={sequence.safe_zone_route_phase} "
                    f"reason={decision.reason}",
                    flush=True,
                )

        def consume(message: object) -> None:
            nonlocal latest_status, previous_odometry
            nonlocal latest_speed_feedback, gyro_heading_rad
            if isinstance(message, OdometryImu):
                if d2_telemetry_logger is not None:
                    d2_telemetry_logger.record_odometry(
                        message,
                        active=d2_telemetry_active,
                        state=sequence.state.value,
                        route_phase=sequence.safe_zone_route_phase,
                        target_wheel_speeds_m_s=controller.target_wheel_speeds_m_s,
                        commanded_wheel_speeds_m_s=(
                            controller.commanded_wheel_speeds_m_s
                        ),
                    )
                sequence.observe_grasp_motion(message)
                encoder_tracker.submit(message)
                previous = previous_odometry
                if previous is not None:
                    elapsed_s = (
                        message.sample_timestamp_us - previous.sample_timestamp_us
                    ) / 1_000_000.0
                    if elapsed_s > 0.0:
                        gyro_heading_rad += (
                            odometry_calibration.gyro_z_sign
                            * message.gyro_z_rad_s
                            * elapsed_s
                        )
                        left_revolutions = (
                            message.left_encoder_count
                            - previous.left_encoder_count
                        ) / odometry_calibration.encoder_counts_per_revolution
                        right_revolutions = (
                            message.right_encoder_count
                            - previous.right_encoder_count
                        ) / odometry_calibration.encoder_counts_per_revolution
                        latest_speed_feedback = (
                            left_revolutions
                            * 2.0 * math.pi
                            * odometry_calibration.left_wheel_radius_mm
                            / 1000.0 / elapsed_s,
                            right_revolutions
                            * 2.0 * math.pi
                            * odometry_calibration.right_wheel_radius_mm
                            / 1000.0 / elapsed_s,
                        )
                    else:
                        latest_speed_feedback = (None, None)
                previous_odometry = message
            elif isinstance(message, CarSystemStatus):
                latest_status = message

        def service_uart_during_startup() -> None:
            controller.update(now_ns=time.monotonic_ns())
            for message in controller.drain_messages():
                consume(message)
            if latest_status is not None and latest_status.emergency_stop_latched:
                raise RuntimeError("STM32 emergency stop is latched during startup.")

        def apply_motion_acceleration_limit() -> None:
            controller.set_wheel_acceleration_limit_m_s2(
                sequence.motion_acceleration_limit_m_s2
            )

        class _MotionChannelContext:
            def __enter__(self):
                channel.start()
                return channel

            def __exit__(self, exc_type, exc_value, traceback):
                try:
                    controller.soft_brake()
                finally:
                    channel.stop()
                return False

        signal.signal(signal.SIGTERM, request_stop)
        if log_dir is not None:
            telemetry_path = log_dir / (
                f"{log_file_prefix}d2_safe_zone_telemetry_"
                f"{time.strftime('%Y%m%d_%H%M%S')}_{time.monotonic_ns()}.jsonl"
            )
            d2_telemetry_logger = D2TelemetryLogger(
                telemetry_path,
                odometry_calibration,
            )
            d2_telemetry_logger.start(timestamp_ns=time.monotonic_ns())
            print(
                "d2_telemetry_log="
                f"{d2_telemetry_logger.path} sampling=every_odometry_imu "
                "target_period_ms=10 writer=background_bounded_queue",
                flush=True,
            )
        else:
            print("d2_telemetry_log=disabled", flush=True)
        try:
            if remote_transport is not None:
                remote_transport.start()
            with _MotionChannelContext():
                camera_start_thread = camera_pump.start_in_background()
                if preview is not None:
                    preview.start()
                sync_deadline_ns = time.monotonic_ns() + 5_000_000_000
                while True:
                    try:
                        controller.synchronize(
                            timeout_s=config.motion.synchronization_timeout_s,
                            on_message=consume,
                        )
                        break
                    except MotionSynchronizationError:
                        if latest_status is not None and latest_status.emergency_stop_latched:
                            raise RuntimeError(
                                "STM32 emergency stop is latched during startup."
                            )
                        if time.monotonic_ns() >= sync_deadline_ns:
                            raise
                        time.sleep(0.02)
                camera_pump.wait_until_started(
                    camera_start_thread,
                    on_wait=service_uart_during_startup,
                )
                camera_started = True
                controller.query_state()
                while latest_snapshot is None and not stop_requested:
                    controller.update(now_ns=time.monotonic_ns())
                    for message in controller.drain_messages():
                        consume(message)
                    camera_pump.check_health()
                    latest_snapshot = renderer.latest_fresh_snapshot(
                        time.monotonic_ns(),
                        config.processing.max_observation_age_ms,
                    )
                    time.sleep(0.005)
                if latest_snapshot is None:
                    raise RuntimeError("No fresh perception snapshot before start.")
                preflight_deadline_ns = time.monotonic_ns() + _PREFLIGHT_RETRY_WINDOW_NS
                while True:
                    checks = MatchPreflight(
                        telemetry_fresh=encoder_tracker.distance_m is not None,
                        watchdog_armed=(latest_status is not None and latest_status.watchdog_armed),
                        emergency_stop_clear=not (
                            latest_status is not None
                            and latest_status.emergency_stop_latched
                        ),
                        zero_speed_command_accepted=controller.motion_synchronized,
                        camera_observation_fresh=True,
                        ground_mapping_enabled=config.geometry.ground_mapping_enabled,
                    )
                    if checks.ready or not checks.emergency_stop_clear:
                        break
                    if time.monotonic_ns() >= preflight_deadline_ns:
                        break
                    service_uart_during_startup()
                    time.sleep(0.005)
                preflight = sequence.preflight(time.monotonic_ns(), checks)
                if preflight.state is MatchState.TERMINAL_STOP:
                    raise RuntimeError(preflight.reason)
                process_started_timestamp_ns = time.monotonic_ns()
                sequence.start(process_started_timestamp_ns)
                if d2_telemetry_logger is not None:
                    d2_telemetry_logger.set_process_start_timestamp_ns(
                        process_started_timestamp_ns
                    )
                last_gripper_angles = None
                last_soft_brake_key = None
                near_field_worker_session_id: int | None = None
                near_field_last_error: str | None = None
                near_field_last_failure_reported: str | None = None
                last_state: MatchState | None = None
                preview_state_text = sequence.state.value
                preview_reason_text = "started"
                next_progress_ns = 0
                recent_rendered_frames = {}
                while not stop_requested:
                    now_ns = time.monotonic_ns()
                    apply_motion_acceleration_limit()
                    controller.update(now_ns=now_ns)
                    branch_error: str | None = None
                    try:
                        for message in controller.drain_messages():
                            if (
                                isinstance(message, CarCommandReply)
                                and message.result is not CommandResult.ACCEPTED
                            ):
                                print(
                                    f"stm32_rejected={message.command_type.name.lower()}:"
                                    f"{message.result.name.lower()}",
                                    flush=True,
                                )
                                continue
                            consume(message)
                    except Exception as exc:
                        branch_error = f"uart_consumer:{exc}"
                    try:
                        camera_pump.check_health()
                    except Exception as exc:
                        branch_error = branch_error or f"side_path:{exc}"
                    if preview is not None:
                        try:
                            preview.check_health()
                        except Exception as exc:
                            branch_error = branch_error or f"local_preview:{exc}"
                    fresh_snapshot = None
                    try:
                        fresh_snapshot = renderer.latest_fresh_snapshot(
                            now_ns,
                            config.processing.max_observation_age_ms,
                        )
                    except Exception as exc:
                        branch_error = branch_error or f"perception_renderer:{exc}"
                    latest_snapshot = fresh_snapshot
                    rendered = renderer.latest()
                    if rendered is not None:
                        recent_rendered_frames[rendered.timestamp_ns] = rendered
                        while len(recent_rendered_frames) > 8:
                            recent_rendered_frames.pop(next(iter(recent_rendered_frames)))
                    if preview is not None:
                        preview.submit(
                            rendered,
                            state_text=preview_state_text,
                            reason_text=preview_reason_text,
                            localization_lines=_format_match_preview_localization(
                                sequence, preview_heading_rad()
                            ),
                            process_timestamp_ms=frame_process_timestamp_ms(rendered),
                        )
                        stop_requested = stop_requested or preview.user_requested_stop
                    if branch_error is not None:
                        preview_reason_text = branch_error
                        if preview is not None:
                            preview.submit(
                                rendered,
                                state_text=preview_state_text,
                                reason_text=preview_reason_text,
                                localization_lines=_format_match_preview_localization(
                                    sequence, preview_heading_rad()
                                ),
                                process_timestamp_ms=frame_process_timestamp_ms(
                                    rendered
                                ),
                            )
                            stop_requested = stop_requested or preview.user_requested_stop
                        print(f"side_path_hold={branch_error}", flush=True)
                        controller.drive_wheel_limited(0.0, 0.0)
                        time.sleep(0.005)
                        continue
                    safety = SafetySignals.nominal(now_ns)
                    if stop_requested or (
                        latest_status is not None and latest_status.emergency_stop_latched
                    ):
                        safety = SafetySignals(
                            last_motion_timestamp_ns=now_ns,
                            safety_accident=bool(
                                latest_status is not None
                                and latest_status.emergency_stop_latched
                            ),
                            external_stop_requested=stop_requested,
                        )
                    near_field_preparation = None
                    near_field_path_clear = None
                    if (
                        sequence.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
                        and sequence.near_field_enabled
                    ):
                        ensure_near_field_worker()
                    if (
                        near_field_worker is not None
                        and sequence.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
                    ):
                        session_id = sequence.near_field_session_id
                        if near_field_worker_session_id != session_id:
                            near_field_worker.begin(session_id)
                            near_field_worker_session_id = session_id
                        observation_window_open = (
                            sequence.near_field_observation_window_open(now_ns)
                        )
                        if (
                            latest_snapshot is not None
                            and observation_window_open
                            and sequence.near_field_active_plan is None
                        ):
                            near_field_worker.submit(
                                latest_snapshot,
                                session_id=session_id,
                                policy=sequence.near_field_policy,
                                locked_ids=sequence.near_field_locked_ids,
                                handoff_prior=sequence.near_field_handoff_prior,
                            )
                        near_field_preparation = near_field_worker.latest(session_id)
                        near_field_selection = (
                            None
                            if near_field_preparation is None
                            else near_field_preparation.selection
                        )
                        if (
                            near_field_preparation is not None
                            and near_field_selection is not None
                            and near_field_selection.plan is not None
                            and near_field_selection.plan.alignment_angle_rad == 0.0
                        ):
                            near_field_path_clear = sequence.near_field_plan_path_clear(
                                near_field_selection.plan,
                                sequence.estimated_field_heading_rad
                                if sequence.estimated_field_heading_rad is not None
                                else gyro_heading_rad,
                            )
                        if near_field_worker.error is not None and (
                            near_field_worker.error != near_field_last_error
                        ):
                            print(
                                f"near_field_worker_hold={near_field_worker.error}",
                                flush=True,
                            )
                        near_field_last_error = near_field_worker.error
                    # 消费UART和准备结果后再取控制时刻，避免新结果看起来来自未来。
                    now_ns = time.monotonic_ns()
                    decision = sequence.step(
                        now_ns,
                        perception=latest_snapshot,
                        heading_rad=gyro_heading_rad,
                        cumulative_distance_m=encoder_tracker.distance_m,
                        left_speed_feedback_m_s=latest_speed_feedback[0],
                        right_speed_feedback_m_s=latest_speed_feedback[1],
                        safety=safety,
                        near_field_preparation=near_field_preparation,
                        near_field_path_clear=near_field_path_clear,
                    )
                    if (
                        decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
                        and sequence.near_field_active_plan is None
                    ):
                        corridor_frame = rendered
                        if near_field_preparation is not None:
                            corridor_frame = recent_rendered_frames.get(
                                near_field_preparation.capture_timestamp_ns,
                                rendered,
                            )
                        rendered = _overlay_near_field_corridor(
                            corridor_frame,
                            near_field_preparation,
                            near_field_selector,
                        )
                    rendered = _overlay_match_selected_targets(
                        rendered,
                        sequence,
                        near_field_preparation,
                    )
                    if decision.reason == "near_field_grasp_complete_start_safe_zone_d1_line":
                        print(
                            f"near_field_result={sequence.near_field_result}",
                            flush=True,
                        )
                    failure_diagnostic = sequence.near_field_last_failure_diagnostic
                    if (
                        failure_diagnostic is not None
                        and failure_diagnostic != near_field_last_failure_reported
                    ):
                        print(failure_diagnostic, flush=True)
                        near_field_last_failure_reported = failure_diagnostic
                    update_d2_telemetry_phase(decision, now_ns)
                    apply_motion_acceleration_limit()
                    preview_state_text = decision.state.value
                    preview_reason_text = decision.reason
                    if decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP:
                        near_plan = (
                            None
                            if near_field_preparation is None
                            else (
                                near_field_preparation.selection.plan
                                or near_field_preparation.selection.preview_plan
                            )
                        )
                        if near_plan is not None:
                            preview_reason_text = (
                                f"{decision.reason} ids={near_plan.member_ids} "
                                f"classes={[item.observation.target_class.value for item in near_plan.members]} "
                                f"open={near_plan.opening_width_mm:.1f}mm "
                                f"forward={near_plan.forward_distance_mm:.1f}mm"
                            )
                    if preview is not None:
                        preview.submit(
                            rendered,
                            state_text=preview_state_text,
                            reason_text=preview_reason_text,
                            localization_lines=_format_match_preview_localization(
                                sequence, preview_heading_rad()
                            ),
                            process_timestamp_ms=frame_process_timestamp_ms(rendered),
                        )
                        stop_requested = stop_requested or preview.user_requested_stop
                    state_changed = decision.state is not last_state
                    if state_changed:
                        _print_state_banner(decision.state, decision.reason)
                    if decision.reason == "near_field_opening:open_group_width":
                        plan_age = (
                            None
                            if near_field_preparation is None
                            else near_field_preparation.plan_age_ns(now_ns)
                        )
                        preparation_age = (
                            None
                            if near_field_preparation is None
                            else near_field_preparation.preparation_age_ns(now_ns)
                        )
                        print(
                            "near_field_grasp_commit="
                            f"session={sequence.near_field_session_id} "
                            f"ids={None if sequence.near_field_active_plan is None else sequence.near_field_active_plan.member_ids} "
                            f"plan_age_ms={_age_ms_text(plan_age)} "
                            f"preparation_age_ms={_age_ms_text(preparation_age)} "
                            f"confirmation={None if near_field_preparation is None else near_field_preparation.confirmation_progress}",
                            flush=True,
                        )
                    if decision.gripper_angles_deg is not None:
                        angles = decision.gripper_angles_deg
                    elif decision.gripper_posture is GripperPosture.OPEN:
                        angles = (gripper.open_left_angle_deg, gripper.open_right_angle_deg)
                    elif decision.gripper_posture is GripperPosture.TRANSPORT:
                        angles = gripper.transport_angles_deg
                    else:
                        angles = (gripper.closed_left_angle_deg, gripper.closed_right_angle_deg)
                    assert angles is not None
                    if angles != last_gripper_angles:
                        controller.set_gripper_angles(*angles)
                        print(
                            "gripper_command="
                            f"posture={decision.gripper_posture.value} "
                            f"left={angles[0]:g} right={angles[1]:g}",
                            flush=True,
                        )
                        last_gripper_angles = angles
                    if remote_transport is not None:
                        try:
                            _publish_remote_match_state(
                                remote_transport,
                                pose=None,
                                timestamp_ns=now_ns,
                                rendered=rendered,
                            )
                        except Exception as exc:
                            print(f"remote_image_error={exc}", flush=True)
                    if decision.state in {
                        MatchState.FINISH_STOP,
                        MatchState.TERMINAL_STOP,
                    }:
                        if not state_changed:
                            _print_state_banner(decision.state, decision.reason)
                        controller.soft_brake()
                        print(
                            f"terminal_state={decision.state.value} "
                            f"reason={decision.reason}",
                            flush=True,
                        )
                        break
                    if decision.soft_brake:
                        brake_key = (decision.state, decision.reason)
                        if brake_key != last_soft_brake_key:
                            controller.soft_brake()
                            last_soft_brake_key = brake_key
                    else:
                        last_soft_brake_key = None
                        controller.drive_wheel_limited(
                            decision.linear_velocity_m_s,
                            decision.angular_velocity_rad_s,
                            min_wheel_velocity_m_s=(
                                decision.min_wheel_velocity_m_s
                            ),
                        )
                    if decision.state is not last_state or now_ns >= next_progress_ns:
                        if not state_changed:
                            _print_state_banner(decision.state, decision.reason)
                        path_text = (
                            sequence.green_isolation_diagnostic(now_ns)
                            if decision.state
                            in {
                                MatchState.CHECK_ISOLATED_GREEN,
                                MatchState.TRANSPORT_ALIGN_GREEN,
                                MatchState.TRANSPORT_APPROACH_GREEN,
                                MatchState.TRANSPORT_NEAR_FIELD_GRASP,
                                MatchState.TRANSPORT_PRE_CLOSE_RECHECK,
                                MatchState.TRANSPORT_CLOSE_GRIPPER,
                            }
                            or decision.reason.startswith("no_isolated_green")
                            or decision.reason.startswith("green_path")
                            else "not_checked"
                        )
                        safe_zone_text = (
                            sequence.safe_zone_diagnostic(latest_snapshot)
                            if decision.state
                            in {
                                MatchState.TRANSPORT_ALIGN_RED_ZONE,
                                MatchState.TRANSPORT_FORWARD,
                                MatchState.TRANSPORT_RELEASE,
                            }
                            else "not_checked"
                        )
                        near_field_text = "not_checked"
                        if decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP:
                            near_log_plan = sequence.near_field_active_plan
                            if near_log_plan is None and near_field_preparation is not None:
                                near_log_plan = (
                                    near_field_preparation.selection.plan
                                    or near_field_preparation.selection.preview_plan
                                )
                            near_field_text = (
                                f"{sequence.near_field_route_diagnostic},"
                                f"session={sequence.near_field_session_id},"
                                f"policy={sorted(item.value for item in sequence.near_field_policy.allowed_classes)},"
                                f"max_targets={sequence.near_field_policy.max_targets},"
                                f"locked_ids={sequence.near_field_locked_ids},"
                                f"active_ids={None if near_log_plan is None else near_log_plan.member_ids},"
                                f"progress_mm={sequence.near_field_progress_mm(encoder_tracker.distance_m):.1f},"
                                f"opening_angles={None if near_log_plan is None else near_log_plan.opening_servo_angles_deg},"
                                f"prep={None if near_field_preparation is None else near_field_preparation.selection.rejections},"
                                f"{sequence.near_field_confirmation_diagnostic(now_ns, near_field_preparation)}"
                            )
                        print(
                            f"state={decision.state.value} "
                            f"reason={decision.reason} "
                            f"process_timestamp_ms={process_timestamp_ms(now_ns)} "
                            f"distance_m={encoder_tracker.distance_m} "
                            f"heading_rad={preview_heading_rad()} "
                            f"target_wheel_speeds={controller.target_wheel_speeds_m_s} "
                            f"commanded_wheel_speeds={controller.commanded_wheel_speeds_m_s} "
                            f"wheel_acceleration_limit_m_s2={controller.wheel_acceleration_limit_m_s2} "
                            f"isolation={path_text} "
                            f"safe_zone={safe_zone_text} "
                            f"near_field={near_field_text} "
                            f"green_target={sequence.green_target_diagnostic(now_ns)} "
                            f"cluster_target={sequence.cluster_diagnostic(now_ns)} "
                            f"green_angular_velocity_rad_s={decision.angular_velocity_rad_s} "
                            f"d2_telemetry_active={d2_telemetry_active} "
                            f"d2_telemetry_dropped_records="
                            f"{0 if d2_telemetry_logger is None else d2_telemetry_logger.dropped_records}",
                            flush=True,
                        )
                        last_state = decision.state
                        next_progress_ns = now_ns + 1_000_000_000
                    time.sleep(0.005)
        finally:
            try:
                if preview is not None:
                    preview.stop()
            finally:
                try:
                    if near_field_worker is not None:
                        near_field_worker.__exit__()
                finally:
                    try:
                        if camera_pump is not None and (
                            camera_start_thread is not None or camera_started
                        ):
                            camera_pump.stop()
                    finally:
                        try:
                            controller.soft_brake()
                        except Exception:
                            pass
                        if remote_transport is not None:
                            remote_transport.stop()
                        if d2_telemetry_logger is not None:
                            worker_error = d2_telemetry_logger.worker_error
                            try:
                                d2_telemetry_logger.stop(timestamp_ns=time.monotonic_ns())
                            except Exception as exc:
                                print(f"d2_telemetry_log_error={exc}", flush=True)
                            if worker_error is not None:
                                print(
                                    f"d2_telemetry_worker_error={worker_error}",
                                    flush=True,
                                )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        _end_time_named_log(log_stream, original_stdout, original_stderr)
