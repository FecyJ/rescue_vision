"""正式流程和夹取—运送联调共用的硬件生命周期与控制循环。"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Callable, TYPE_CHECKING
from rescue_vision.app.match import (
    MatchPreflight,
    MatchSequence,
    MatchState,
    _print_state_banner,
)
from rescue_vision.app.match_observers import _LocalPreview, _publish_remote_match_state
from rescue_vision.app.session_log import _begin_time_named_log, _end_time_named_log
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import normalize_angle
if TYPE_CHECKING:
    from rescue_vision.config import AppConfig
    from rescue_vision.camera.frame import CameraFrame
_PREFLIGHT_RETRY_WINDOW_NS = 5_000_000_000

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
        )
        from rescue_vision.perception import PerceptionFrameRenderer, PerceptionSnapshot
        from rescue_vision.mission import SafetySignals

        config = load_runtime_config(config_path)
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
        initial_position_text = "configured"
        if initial_field_position is not None:
            initial_position_text = (
                f"({initial_field_position.x:g},{initial_field_position.y:g})mm"
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
        print(
            "config="
            f"path={config_path.resolve()} mode={mode_name} "
            f"start_position_field={initial_position_text} "
            f"start_heading_rad={effective_initial_heading_rad:g} "
            f"team_color={config.world.team_color.value} "
            f"required_transports={config.match.required_transports} "
            f"green_path_half_width_mm={config.match.green_path_half_width_mm:g} "
            "opportunistic_single_green="
            f"{config.match.opportunistic_single_green_enabled} "
            "opportunistic_single_green_clearance_mm="
            f"{config.match.opportunistic_single_green_clearance_mm:g} "
            "opportunistic_single_green_realign_standoff_mm="
            f"{config.match.opportunistic_single_green_realign_standoff_mm:g} "
            f"green_grab_offset_mm={config.match.green_grab_offset_mm:g} "
            "green_preclose_recheck_range_mm="
            f"{config.match.green_preclose_recheck_range_mm:g} "
            "green_preclose_recheck_hold_ms="
            f"{config.match.green_preclose_recheck_hold_ms:g} "
            "green_preclose_max_carried_blocks="
            f"{config.match.green_preclose_max_carried_blocks} "
            f"safe_zone_d1_mm={config.match.safe_zone_calibration_start_offset_mm:g} "
            f"safe_zone_d2_mm={config.match.safe_zone_open_offset_mm:g} "
            "safe_zone_d2_to_final_max_wheel_acceleration_m_s2="
            f"{config.match.safe_zone_d2_to_final_max_wheel_acceleration_m_s2} "
            f"safe_zone_braking_overrun_mm=({config.match.safe_zone_d2_braking_overrun_x_mm:g},"
            f"{config.match.safe_zone_d2_braking_overrun_y_mm:g}) "
            f"safe_zone_exit_turn_angle_rad={config.match.safe_zone_exit_turn_angle_rad:g} "
            f"action_settle_time_s={config.match.action_settle_time_s:g} "
            f"green_first_scan_angle_rad={config.match.spin_angle_rad:g} "
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
            )
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
        sequence = sequence_factory(config)
        latest_status: CarSystemStatus | None = None
        latest_snapshot: PerceptionSnapshot | None = None
        previous_odometry: OdometryImu | None = None
        latest_speed_feedback: tuple[float | None, float | None] = (None, None)
        gyro_heading_rad = effective_initial_heading_rad
        d2_telemetry_active = False
        process_started_timestamp_ns: int | None = None

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
                sequence.safe_zone_motion_acceleration_limit_m_s2
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
                    latest_snapshot = renderer.latest_snapshot()
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
                last_posture = None
                last_state: MatchState | None = None
                preview_state_text = sequence.state.value
                preview_reason_text = "started"
                next_progress_ns = 0
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
                        fresh_snapshot = renderer.latest_snapshot()
                    except Exception as exc:
                        branch_error = branch_error or f"perception_renderer:{exc}"
                    if fresh_snapshot is not None:
                        latest_snapshot = fresh_snapshot
                    rendered = renderer.latest()
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
                    decision = sequence.step(
                        now_ns,
                        perception=latest_snapshot,
                        heading_rad=gyro_heading_rad,
                        cumulative_distance_m=encoder_tracker.distance_m,
                        left_speed_feedback_m_s=latest_speed_feedback[0],
                        right_speed_feedback_m_s=latest_speed_feedback[1],
                        safety=safety,
                    )
                    update_d2_telemetry_phase(decision, now_ns)
                    apply_motion_acceleration_limit()
                    preview_state_text = decision.state.value
                    preview_reason_text = decision.reason
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
                    if decision.gripper_posture is not last_posture:
                        if decision.gripper_posture is GripperPosture.OPEN:
                            angles = (gripper.open_left_angle_deg, gripper.open_right_angle_deg)
                        elif decision.gripper_posture is GripperPosture.TRANSPORT:
                            angles = gripper.transport_angles_deg
                        else:
                            angles = (gripper.closed_left_angle_deg, gripper.closed_right_angle_deg)
                        assert angles is not None
                        controller.set_gripper_angles(*angles)
                        print(
                            "gripper_command="
                            f"posture={decision.gripper_posture.value} "
                            f"left={angles[0]:g} right={angles[1]:g}",
                            flush=True,
                        )
                        last_posture = decision.gripper_posture
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
                    controller.drive_wheel_limited(
                        decision.linear_velocity_m_s,
                        decision.angular_velocity_rad_s,
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
