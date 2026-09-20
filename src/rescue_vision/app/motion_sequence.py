"""受监督的 1.5 m 定距动作及其终端交互入口。"""

from __future__ import annotations

import curses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Event, Thread
from typing import ClassVar

import cv2

from rescue_vision.app.cluster_breakup import CameraPerceptionPump
from rescue_vision.app.manual_capture import build_camera_pipeline
from rescue_vision.communication import UartFrameChannel
from rescue_vision.config import AppConfig, load_runtime_config
from rescue_vision.motion import MotionController, MotionLimits
from rescue_vision.perception import PerceptionFrameRenderer


# SET_WHEEL_SPEED uses signed int16 millimetres per second. The sequence TUI
# does not use the configured motion speed/acceleration ceilings, but it still
# must not emit a value the wire protocol cannot encode.
_PROTOCOL_MAX_WHEEL_SPEED_M_S = 32.767
_UNBOUNDED_ACCELERATION_M_S2 = 1_000_000.0
_EFFECTIVELY_ZERO_MIN_WHEEL_SPEED_M_S = 1e-9
_LOCAL_PREVIEW_WINDOW_NAME = "motion-sequence perception"
_LOCAL_PREVIEW_WIDTH = 1280
_LOCAL_PREVIEW_HEIGHT = 720


def _finite_float(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return float(value)


def _positive_float(value: object, location: str) -> float:
    converted = _finite_float(value, location)
    if converted <= 0.0:
        raise ValueError(f"{location} must be > 0, got {value!r}.")
    return converted


class MotionSequencePhase(str, Enum):
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class MotionSequencePlan:
    """可配置定距动作参数；``a1``、``a2`` 用 m/s²。

    速度曲线采用无恒速段的三角速度剖面：先按 ``a1`` 加速到自动计算的
    峰值速度，再按 ``a2`` 减速到零。峰值速度和两个阶段时长由目标路程
    约束自动求出。
    """

    a1_m_s2: float
    a2_m_s2: float
    distance_m: float = 1.5

    DEFAULT_DISTANCE_M: ClassVar[float] = 1.5

    def __post_init__(self) -> None:
        a1 = _positive_float(self.a1_m_s2, "a1_m_s2")
        a2 = _positive_float(self.a2_m_s2, "a2_m_s2")
        distance = _positive_float(self.distance_m, "distance_m")
        object.__setattr__(self, "a1_m_s2", a1)
        object.__setattr__(self, "a2_m_s2", a2)
        object.__setattr__(self, "distance_m", distance)
        # Reject only arithmetic overflow here; no configured speed or
        # acceleration ceiling is applied to the user-entered values.
        self.peak_linear_velocity_m_s

    @property
    def peak_linear_velocity_m_s(self) -> float:
        """满足目标定距约束的自动计算峰值线速度。"""

        peak = math.sqrt(
            2.0
            * self.distance_m
            * self.a1_m_s2
            * self.a2_m_s2
            / (self.a1_m_s2 + self.a2_m_s2)
        )
        if not math.isfinite(peak):
            raise ValueError("a1_m_s2 produces a non-finite peak velocity.")
        return peak

    @property
    def acceleration_duration_s(self) -> float:
        """自动计算的加速时长。"""

        return self.peak_linear_velocity_m_s / self.a1_m_s2

    @property
    def deceleration_duration_s(self) -> float:
        """自动计算的减速时长。"""

        return self.peak_linear_velocity_m_s / self.a2_m_s2

    @property
    def nominal_total_duration_s(self) -> float:
        """不计控制循环量化和最终停车斜坡的理论总时长。"""

        return self.acceleration_duration_s + self.deceleration_duration_s

    def target_linear_velocity_m_s(self, elapsed_s: float) -> float:
        """返回给定动作时间的规划线速度。"""

        elapsed = _finite_float(elapsed_s, "elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        acceleration_duration = self.acceleration_duration_s
        if elapsed < acceleration_duration:
            return self.a1_m_s2 * elapsed
        deceleration_elapsed = elapsed - acceleration_duration
        if deceleration_elapsed < self.deceleration_duration_s:
            return max(
                0.0,
                self.peak_linear_velocity_m_s
                - self.a2_m_s2 * deceleration_elapsed,
            )
        return 0.0

    def planned_distance_m(self, elapsed_s: float) -> float:
        """返回给定动作时间按运动学积分得到的计划路程。"""

        elapsed = _finite_float(elapsed_s, "elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        acceleration_duration = self.acceleration_duration_s
        if elapsed < acceleration_duration:
            return min(self.distance_m, 0.5 * self.a1_m_s2 * elapsed**2)
        deceleration_elapsed = min(
            self.deceleration_duration_s,
            elapsed - acceleration_duration,
        )
        distance = (
            0.5 * self.a1_m_s2 * acceleration_duration**2
            + self.peak_linear_velocity_m_s * deceleration_elapsed
            - 0.5 * self.a2_m_s2 * deceleration_elapsed**2
        )
        return min(self.distance_m, max(0.0, distance))

    def validate_for_protocol(self, limits: MotionLimits) -> None:
        """检查派生轮速仍能编码为 STM32 的有符号 int16。"""

        if not isinstance(limits, MotionLimits):
            raise TypeError("limits must be MotionLimits.")
        peak = self.peak_linear_velocity_m_s
        forward_wheels = (
            peak * limits.left_wheel_speed_weight,
            peak * limits.right_wheel_speed_weight,
        )
        self._validate_wheel_targets(
            forward_wheels,
            limits,
            label="forward peak velocity/protocol",
        )

    @staticmethod
    def _validate_wheel_targets(
        wheel_speeds: tuple[float, float],
        limits: MotionLimits,
        *,
        label: str,
    ) -> None:
        maximum = max(abs(speed) for speed in wheel_speeds)
        if maximum > limits.max_wheel_velocity_m_s:
            raise ValueError(
                f"{label} requires {maximum:g} m/s on a wheel, above "
                f"the protocol limit {limits.max_wheel_velocity_m_s:g} m/s."
            )


def _build_sequence_controller(
    config: AppConfig,
    channel: UartFrameChannel,
) -> MotionController:
    """创建仅受线路编码范围约束的动作控制器。

    该入口明确不采用 ``motion`` 中的车体速度和加速度上限。轮距、左右轮
    权重、堵转监测和远程有效期仍沿用配置；轮速上限取协议有符号 int16 的
    可编码范围，避免把“取消配置限制”误解为可以发送无法编码的帧。
    """

    motion_config = config.motion
    if motion_config.wheel_track_m is None:
        raise RuntimeError("Motion sequence requires motion.wheel_track_m.")
    maximum_weight = max(
        motion_config.left_wheel_speed_weight,
        motion_config.right_wheel_speed_weight,
    )
    half_track = motion_config.wheel_track_m / 2.0
    return MotionController(
        channel,
        MotionLimits(
            wheel_track_m=motion_config.wheel_track_m,
            max_linear_velocity_m_s=(
                _PROTOCOL_MAX_WHEEL_SPEED_M_S / maximum_weight
            ),
            max_angular_velocity_rad_s=(
                _PROTOCOL_MAX_WHEEL_SPEED_M_S
                / (half_track * maximum_weight)
            ),
            max_wheel_velocity_m_s=_PROTOCOL_MAX_WHEEL_SPEED_M_S,
            max_linear_acceleration_m_s2=_UNBOUNDED_ACCELERATION_M_S2,
            max_linear_deceleration_m_s2=_UNBOUNDED_ACCELERATION_M_S2,
            max_angular_acceleration_rad_s2=_UNBOUNDED_ACCELERATION_M_S2,
            max_angular_deceleration_rad_s2=_UNBOUNDED_ACCELERATION_M_S2,
            min_wheel_velocity_m_s=_EFFECTIVELY_ZERO_MIN_WHEEL_SPEED_M_S,
            left_wheel_speed_weight=motion_config.left_wheel_speed_weight,
            right_wheel_speed_weight=motion_config.right_wheel_speed_weight,
            max_remote_command_valid_for_ms=(
                motion_config.max_remote_command_valid_for_ms
            ),
            stall_guard_enabled=motion_config.stall_guard_enabled,
            stall_guard_timeout_ms=motion_config.stall_guard_timeout_ms,
            stall_guard_min_command_speed_m_s=(
                min(
                    motion_config.stall_guard_min_command_speed_m_s,
                    _PROTOCOL_MAX_WHEEL_SPEED_M_S,
                )
            ),
            stall_guard_stationary_encoder_delta_count=(
                motion_config.stall_guard_stationary_encoder_delta_count
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class MotionSequenceStatus:
    phase: MotionSequencePhase
    elapsed_s: float
    phase_elapsed_s: float
    target_linear_velocity_m_s: float
    planned_distance_m: float
    commanded_wheel_speeds_m_s: tuple[float, float]


@dataclass(frozen=True, slots=True)
class MotionSequenceResult:
    completed: bool
    stopped_by_operator: bool
    final_phase: MotionSequencePhase
    elapsed_s: float


class LocalPerceptionPreview:
    """在独立线程显示最新 perception 叠加帧，不阻塞运动循环。"""

    def __init__(self, renderer: PerceptionFrameRenderer) -> None:
        if not isinstance(renderer, PerceptionFrameRenderer):
            raise TypeError("renderer must be PerceptionFrameRenderer.")
        self._renderer = renderer
        self._stop_event = Event()
        self._quit_event = Event()
        self._thread: Thread | None = None
        self._window_created = False
        self.error: BaseException | None = None

    @property
    def stop_requested(self) -> bool:
        return self._quit_event.is_set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("LocalPerceptionPreview is already started.")
        self._stop_event.clear()
        self._quit_event.clear()
        self.error = None
        self._thread = Thread(
            target=self._run,
            name="motion-sequence-perception-preview",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("Local perception preview did not stop.")
        self._thread = None

    def _run(self) -> None:
        try:
            cv2.namedWindow(_LOCAL_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                _LOCAL_PREVIEW_WINDOW_NAME,
                _LOCAL_PREVIEW_WIDTH,
                _LOCAL_PREVIEW_HEIGHT,
            )
            self._window_created = True
            while not self._stop_event.is_set():
                rendered = self._renderer.latest()
                if rendered is not None:
                    cv2.imshow(_LOCAL_PREVIEW_WINDOW_NAME, rendered.image_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self._quit_event.set()
                    break
                try:
                    visible = cv2.getWindowProperty(
                        _LOCAL_PREVIEW_WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    )
                except cv2.error:
                    visible = 1.0
                if visible < 1.0:
                    self._quit_event.set()
                    break
                self._stop_event.wait(0.03)
        except BaseException as exc:
            self.error = exc
        finally:
            if self._window_created:
                try:
                    cv2.destroyWindow(_LOCAL_PREVIEW_WINDOW_NAME)
                except cv2.error:
                    pass
                self._window_created = False


class MotionSequenceRunner:
    """按计划驱动一个已同步的 :class:`MotionController`。

    控制循环使用最新控制状态并持续排空 UART 回传。无论正常结束、操作员
    取消还是异常，``run()`` 都会先发送一次柔和停车；调用方仍应在更外层
    保留最终清理停车路径。
    """

    _POLL_INTERVAL_S = 0.02
    _ZERO_SPEED_EPSILON_M_S = 1e-6

    def __init__(
        self,
        controller: MotionController,
        plan: MotionSequencePlan,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(controller, MotionController):
            raise TypeError("controller must be MotionController.")
        if not isinstance(plan, MotionSequencePlan):
            raise TypeError("plan must be MotionSequencePlan.")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable.")
        if not callable(sleep):
            raise TypeError("sleep must be callable.")
        self.controller = controller
        self.plan = plan
        self._monotonic_ns = monotonic_ns
        self._sleep = sleep

    def run(
        self,
        *,
        stop_requested: Callable[[], bool] = lambda: False,
        on_status: Callable[[MotionSequenceStatus], None] | None = None,
    ) -> MotionSequenceResult:
        """执行一次动作；返回是否完整完成或被操作员取消。"""

        if not callable(stop_requested):
            raise TypeError("stop_requested must be callable.")
        if on_status is not None and not callable(on_status):
            raise TypeError("on_status must be callable or None.")
        self.plan.validate_for_protocol(self.controller.limits)
        if self.controller.needs_synchronization or not self.controller.motion_synchronized:
            raise RuntimeError("Motion controller must be synchronized before running.")
        if self.controller.emergency_stop_latched:
            raise RuntimeError("Motion controller emergency stop is latched.")

        start_ns = self._now()
        phase = MotionSequencePhase.ACCELERATING
        phase_start_ns = start_ns
        completed = False
        stopped_by_operator = False

        try:
            while True:
                now_ns = self._now()
                if stop_requested():
                    stopped_by_operator = True
                    phase = MotionSequencePhase.ABORTED
                    break

                elapsed_s = (now_ns - start_ns) / 1_000_000_000.0
                if elapsed_s < self.plan.acceleration_duration_s:
                    phase = MotionSequencePhase.ACCELERATING
                else:
                    if phase is MotionSequencePhase.ACCELERATING:
                        phase = MotionSequencePhase.DECELERATING
                        phase_start_ns = now_ns
                target_velocity = self.plan.target_linear_velocity_m_s(elapsed_s)
                self.controller.set_acceleration_limits(
                    linear_acceleration_m_s2=self.plan.a1_m_s2,
                    linear_deceleration_m_s2=self.plan.a2_m_s2,
                )
                self.controller.forward(target_velocity)
                # forward() internally settles the previous target using the
                # controller's own monotonic clock. Do not pass the earlier
                # runner timestamp back here, otherwise real-time scheduling
                # can make MotionController observe time going backwards.
                self.controller.update()
                self.controller.drain_messages()
                if self.controller.emergency_stop_latched:
                    raise RuntimeError("STM32 emergency stop became latched.")
                if self.controller.needs_synchronization or self.controller.link_degraded:
                    raise RuntimeError(
                        "Motion controller lost synchronization or link health "
                        "during the sequence."
                    )
                phase_elapsed_s = (now_ns - phase_start_ns) / 1_000_000_000.0

                if (
                    phase is MotionSequencePhase.DECELERATING
                    and target_velocity == 0.0
                    and self._is_stopped()
                ):
                    phase = MotionSequencePhase.COMPLETE
                    completed = True

                if on_status is not None:
                    on_status(
                        MotionSequenceStatus(
                            phase=phase,
                            elapsed_s=elapsed_s,
                            phase_elapsed_s=(
                                0.0
                                if phase is MotionSequencePhase.COMPLETE
                                else (now_ns - phase_start_ns)
                                / 1_000_000_000.0
                            ),
                            target_linear_velocity_m_s=target_velocity,
                            planned_distance_m=self.plan.planned_distance_m(
                                elapsed_s
                            ),
                            commanded_wheel_speeds_m_s=(
                                self.controller.commanded_wheel_speeds_m_s
                            ),
                        )
                    )
                if completed:
                    break
                self._sleep(self._POLL_INTERVAL_S)
        finally:
            try:
                self.controller.soft_brake()
            finally:
                self.controller.set_acceleration_limits()

        return MotionSequenceResult(
            completed=completed,
            stopped_by_operator=stopped_by_operator,
            final_phase=phase,
            elapsed_s=(self._now() - start_ns) / 1_000_000_000.0,
        )

    def _is_stopped(self) -> bool:
        # MotionController exposes the slew-limited command, not a measured
        # encoder velocity. Physical stopping still needs hardware validation.
        return all(
            abs(speed) <= self._ZERO_SPEED_EPSILON_M_S
            for speed in self.controller.commanded_wheel_speeds_m_s
        )

    def _now(self) -> int:
        now_ns = self._monotonic_ns()
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError(
                f"monotonic_ns must return a non-negative integer, got {now_ns!r}."
            )
        return now_ns


def _safe_add(screen: curses.window, row: int, text: str) -> None:
    """在小终端中尽可能绘制一行，不让状态显示中断停车逻辑。"""

    try:
        height, width = screen.getmaxyx()
        if 0 <= row < height and width > 1:
            screen.addnstr(row, 0, text, width - 1)
    except curses.error:
        pass


def _read_float(
    screen: curses.window,
    *,
    row: int,
    label: str,
) -> float | None:
    while True:
        screen.nodelay(False)
        curses.echo()
        try:
            screen.move(row, 0)
            screen.clrtoeol()
            _safe_add(screen, row, f"{label}: ")
            screen.refresh()
            raw = screen.getstr()
        finally:
            curses.noecho()
        value_text = raw.decode("utf-8", errors="replace").strip()
        if value_text.lower() in {"q", "quit", "esc"}:
            return None
        try:
            return float(value_text)
        except ValueError:
            _safe_add(screen, row + 1, "请输入有限数字；输入 q 取消。按任意键重试。")
            screen.refresh()
            screen.getch()
            _safe_add(screen, row + 1, "")


def _draw_plan_prompt(
    screen: curses.window,
    distance_m: float,
) -> MotionSequencePlan | None:
    screen.clear()
    _safe_add(screen, 0, "Rescue Vision 定距动作 TUI")
    _safe_add(screen, 1, "单位：a1/a2 = m/s²；a1 为加速度，a2 为减速度幅值")
    _safe_add(screen, 2, f"动作：自动计算峰值速度 → a1 加速 → a2 减速 → 前进 {distance_m:g} m 后停止")
    screen.refresh()

    values: list[float] = []
    for row, label in (
        (4, "a1 加速度 (m/s²)"),
        (6, "a2 减速度幅值 (m/s²)"),
    ):
        value = _read_float(screen, row=row, label=label)
        if value is None:
            return None
        values.append(value)

    try:
        return MotionSequencePlan(*values, distance_m=distance_m)
    except ValueError as exc:
        _safe_add(screen, 10, f"参数无效：{exc}")
        _safe_add(screen, 11, "按任意键重新输入，按 q 退出。")
        screen.refresh()
        screen.nodelay(False)
        key = screen.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        return _draw_plan_prompt(screen, distance_m)


def _prepare_plan_tui(
    screen: curses.window,
    controller: MotionController,
    distance_m: float,
) -> MotionSequencePlan | None:
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    screen.keypad(True)
    plan = _draw_plan_prompt(screen, distance_m)
    if plan is None:
        return None

    try:
        plan.validate_for_protocol(controller.limits)
    except ValueError as exc:
        _safe_add(screen, 10, f"参数超出协议可编码范围：{exc}")
        _safe_add(screen, 11, "按任意键重新输入，按 q 退出。")
        screen.refresh()
        screen.nodelay(False)
        key = screen.getch()
        if key in (ord("q"), ord("Q"), 27):
            return None
        return _prepare_plan_tui(screen, controller, distance_m)

    screen.clear()
    _safe_add(screen, 0, "动作参数已通过协议编码范围检查")
    _safe_add(screen, 2, f"a1={plan.a1_m_s2:g} m/s², a2={plan.a2_m_s2:g} m/s²")
    _safe_add(screen, 3, f"目标距离：{plan.distance_m:g} m")
    _safe_add(screen, 4, f"自动峰值速度：{plan.peak_linear_velocity_m_s:g} m/s")
    _safe_add(screen, 5, f"加速/减速时间：{plan.acceleration_duration_s:.3f} / {plan.deceleration_duration_s:.3f} s")
    _safe_add(screen, 7, "按 Enter 开始；按 q/Esc 取消。开始后按 q/Esc 软刹车退出。")
    screen.refresh()
    screen.nodelay(False)
    curses.noecho()
    key = screen.getch()
    if key in (ord("q"), ord("Q"), 27):
        return None

    return plan


def _run_tui(
    screen: curses.window,
    controller: MotionController,
    plan: MotionSequencePlan,
    preview: LocalPerceptionPreview | None = None,
) -> MotionSequenceResult:
    """在 UART 已同步后运行已确认的动作，不再阻塞等待输入。"""

    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.nodelay(True)

    def stop_requested() -> bool:
        key = screen.getch()
        return (
            key in (ord("q"), ord("Q"), 27)
            or (preview is not None and preview.stop_requested)
        )

    def draw_status(status: MotionSequenceStatus) -> None:
        left, right = status.commanded_wheel_speeds_m_s
        screen.erase()
        _safe_add(screen, 0, "动作执行中（q/Esc：软刹车退出）")
        _safe_add(screen, 2, f"阶段：{status.phase.value}")
        _safe_add(screen, 3, f"总用时：{status.elapsed_s:6.2f} s")
        _safe_add(screen, 4, f"阶段用时：{status.phase_elapsed_s:6.2f} s")
        _safe_add(screen, 5, f"目标速度：{status.target_linear_velocity_m_s:+.3f} m/s")
        _safe_add(screen, 6, f"计划距离：{status.planned_distance_m:.3f} / {plan.distance_m:g} m")
        _safe_add(screen, 7, f"当前轮速：left={left:+.3f} m/s, right={right:+.3f} m/s")
        screen.refresh()

    return MotionSequenceRunner(controller, plan).run(
        stop_requested=stop_requested,
        on_status=draw_status,
    )


def main() -> None:
    """运行一次受监督 TUI 动作序列。"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Run a supervised fixed-acceleration motion sequence in a TUI."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--distance-m",
        type=float,
        default=MotionSequencePlan.DEFAULT_DISTANCE_M,
        help="Target forward distance in metres (default: 1.5).",
    )
    parser.add_argument(
        "--local-preview",
        action="store_true",
        help="Show the latest local Hailo perception overlay in an OpenCV window.",
    )
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help=(
            "Acknowledge that a physical emergency stop is ready and an "
            "operator will supervise the complete motion sequence."
        ),
    )
    args = parser.parse_args()
    if not args.supervised_physical_stop_ready:
        parser.error(
            "pass --supervised-physical-stop-ready only after a physical "
            "emergency stop is ready and an operator is supervising."
        )
    if not math.isfinite(args.distance_m) or args.distance_m <= 0.0:
        parser.error("--distance-m must be finite and > 0.")

    config = load_runtime_config(args.config.expanduser().resolve())
    if not config.uart.enabled or not config.motion.enabled:
        raise RuntimeError(
            "Motion sequence requires uart.enabled=true and motion.enabled=true."
        )
    channel = config.uart.build_channel()
    controller = _build_sequence_controller(config, channel)
    if channel is None or controller is None:
        raise RuntimeError("Runtime config did not build a UART motion controller.")

    # Do not start the UART reader while the operator is typing. STM32 sends
    # high-rate telemetry even when no motion command is active.
    plan = curses.wrapper(_prepare_plan_tui, controller, args.distance_m)
    if plan is None:
        print("动作未开始。")
        return

    camera_pump: CameraPerceptionPump | None = None
    preview: LocalPerceptionPreview | None = None
    if args.local_preview:
        if not config.hailo.enabled:
            raise RuntimeError(
                "--local-preview requires hailo.enabled=true and a deployed v3 model."
            )
        pipeline = build_camera_pipeline(config)
        renderer = PerceptionFrameRenderer(
            lambda: config.build_target_pose_detector(
                ground_projector=pipeline.ground_projector,
            ),
            render_enabled=True,
            report_timing=False,
        )
        camera_pump = CameraPerceptionPump(
            pipeline.source,
            pipeline.prepare,
            renderer,
            report_timing=False,
        )
        preview = LocalPerceptionPreview(renderer)

    print("正在同步 STM32 运动序号，请确认车辆已架空或处于安全测试环境……")
    with channel:
        try:
            startup_thread = (
                None
                if camera_pump is None
                else camera_pump.start_in_background()
            )
            controller.synchronize(
                timeout_s=config.motion.synchronization_timeout_s,
            )
            if controller.emergency_stop_latched:
                raise RuntimeError("STM32 emergency stop is latched; reset it before running.")
            if camera_pump is not None:
                def service_uart_during_camera_startup() -> None:
                    controller.update(now_ns=time.monotonic_ns())
                    controller.drain_messages()

                camera_pump.wait_until_started(
                    startup_thread,
                    on_wait=service_uart_during_camera_startup,
                )
                assert preview is not None
                preview.start()
            result = curses.wrapper(_run_tui, controller, plan, preview)
        finally:
            try:
                # 该停车路径覆盖同步后进入 TUI 前、TUI 异常和正常结束等情况。
                controller.soft_brake()
            finally:
                if preview is not None:
                    try:
                        preview.stop()
                    except BaseException as exc:
                        print(f"本地 perception 预览停止失败：{exc}", flush=True)
                    if preview.error is not None:
                        print(
                            f"本地 perception 预览异常：{preview.error}",
                            flush=True,
                        )
                if camera_pump is not None:
                    try:
                        camera_pump.stop()
                    except BaseException as exc:
                        print(f"相机/perception 旁路停止失败：{exc}", flush=True)

    if result.completed:
        print(f"动作完成，用时 {result.elapsed_s:.2f} s。")
    else:
        print("动作已由操作员取消并软刹车。")


if __name__ == "__main__":
    main()
