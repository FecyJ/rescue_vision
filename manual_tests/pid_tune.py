"""调 pid 脚本：标定左右轮比例系数，使「相同轮速指令」走直线。

当前底盘固件走 UART 文本协议（115200、``\\r\\n`` 结尾），支持运行时调
速度环 PID（``p<Kp>,<Ki>,<Kd>``）并周期回传实测左右轮速。但该 PID 命令只
设置一组共享参数，无法补偿左右轮机械不对称；要让「给左右轮发送相同速度即
走直线」，需要每轮一个与速度无关的比例系数（scale）。

本脚本把比例系数放在树莓派侧：下发轮速时按 ``m<v*scaleL>,<v*scaleR>`` 缩放，
每轮先直行 ``--distance-m``（默认 1.5 m）并持续读取遥测实测轮速，积分出左右
轮实际路程，据此更新左右 scale，再原地转 90°、等待 1 s 进入下一速度轮次。
scale 是乘法系数，理论上与速度无关，脚本会按 ``--speeds-m-s`` 在不同速度
（0.05–0.20 m/s）下各跑一轮验证/收敛。

安全：需要 ``--supervised-physical-stop-ready`` 且全程有人监督、物理急停
随时可用；每轮结束、异常、Ctrl+C 都会先发 ``b0,0`` 柔和刹车，这不替代
物理急停。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
import time

MIN_SPEED_M_S = 0.05
MAX_SPEED_M_S = 0.20
_DEFAULT_DISTANCE_M = 1.5
_DEFAULT_TURN_SPEED_M_S = 0.10
_DEFAULT_TURN_ANGLE_RAD = math.pi / 2.0
_DEFAULT_SETTLE_S = 1.0
_DEFAULT_TIMEOUT_S = 60.0
_MIN_MEASURED_SPEED_M_S = 0.01
_MIN_SCALE_CORRECTION = 0.8
_MAX_SCALE_CORRECTION = 1.25
_MAX_INTEGRATION_DT_S = 0.5
_PID_LIMITS = (("kp", 20.0), ("ki", 20.0), ("kd", 5.0))


@dataclass(frozen=True, slots=True)
class WheelSpeedSample:
    """一帧实测轮速遥测；速度单位为 m/s，正值表示前进。"""

    timestamp_ms: int
    left_speed_m_s: float
    right_speed_m_s: float


@dataclass(frozen=True, slots=True)
class WheelScales:
    """左右轮速度比例系数（下发 ``m<v*left>,<v*right>`` 时套用）。"""

    left: float
    right: float

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} scale must be finite and positive, got {value!r}."
                )


def parse_telemetry_line(line: str) -> WheelSpeedSample | None:
    """解析固件 ``t<ms>,<实测左速>,<实测右速>,...`` 遥测行。

    非遥测行（命令回复、空行）或数值非法时返回 ``None``，由调用方忽略。
    """

    stripped = line.strip()
    if len(stripped) < 2 or stripped[0] not in ("t", "T"):
        return None
    parts = stripped.split(",")
    if len(parts) < 3:
        return None
    timestamp_field = parts[0][1:]
    try:
        timestamp_ms = int(float(timestamp_field)) if timestamp_field else 0
        left = float(parts[1])
        right = float(parts[2])
    except ValueError:
        return None
    if not (math.isfinite(left) and math.isfinite(right)):
        return None
    return WheelSpeedSample(
        timestamp_ms=timestamp_ms,
        left_speed_m_s=left,
        right_speed_m_s=right,
    )


def parse_speeds(text: str) -> tuple[float, ...]:
    """解析逗号分隔的速度列表，并校验落在 [0.05, 0.20] m/s。"""

    raw = [part.strip() for part in text.split(",")]
    if not raw or any(not part for part in raw):
        raise ValueError("speeds must be a non-empty comma-separated list.")
    speeds: list[float] = []
    for part in raw:
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"invalid speed {part!r}.") from exc
        if not math.isfinite(value) or not (MIN_SPEED_M_S <= value <= MAX_SPEED_M_S):
            raise ValueError(
                f"speed {value} must be within "
                f"[{MIN_SPEED_M_S}, {MAX_SPEED_M_S}] m/s."
            )
        speeds.append(value)
    return tuple(speeds)


def turn_duration_s(
    wheel_track_m: float,
    turn_speed_m_s: float,
    angle_rad: float,
) -> float:
    """按差速几何估算原地转过 ``angle_rad`` 所需时长（秒）。"""

    for name, value in (
        ("wheel_track_m", wheel_track_m),
        ("turn_speed_m_s", turn_speed_m_s),
        ("angle_rad", angle_rad),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    angular_rad_s = 2.0 * turn_speed_m_s / wheel_track_m
    return angle_rad / angular_rad_s


class StraightRunAccumulator:
    """按实测轮速积分左右轮路程，供定距停止、偏差和 scale 更新使用。

    ``submit()`` 使用树莓派本机时间差做梯形积分；相邻样本间隔超过
    ``max_dt_s`` 时视为遥测缺口，重新锚定不跨缺口积分，避免虚增路程。
    """

    def __init__(self, wheel_track_m: float, *, max_dt_s: float = _MAX_INTEGRATION_DT_S) -> None:
        if not math.isfinite(wheel_track_m) or wheel_track_m <= 0.0:
            raise ValueError(
                f"wheel_track_m must be finite and positive, got {wheel_track_m!r}."
            )
        if not math.isfinite(max_dt_s) or max_dt_s <= 0.0:
            raise ValueError(
                f"max_dt_s must be finite and positive, got {max_dt_s!r}."
            )
        self.wheel_track_m = wheel_track_m
        self.max_dt_s = max_dt_s
        self.distance_left_m = 0.0
        self.distance_right_m = 0.0
        self.duration_s = 0.0
        self.sample_count = 0
        self._previous_sample: WheelSpeedSample | None = None
        self._previous_now_s: float | None = None

    def submit(self, sample: WheelSpeedSample, now_s: float) -> None:
        if not math.isfinite(now_s) or now_s < 0.0:
            raise ValueError(f"now_s must be finite and non-negative, got {now_s!r}.")
        self.sample_count += 1
        if self._previous_sample is None or self._previous_now_s is None:
            self._previous_sample = sample
            self._previous_now_s = now_s
            return
        dt = now_s - self._previous_now_s
        if dt <= 0.0:
            return
        if dt > self.max_dt_s:
            # 遥测缺口：重新锚定，不跨缺口积分。
            self._previous_sample = sample
            self._previous_now_s = now_s
            return
        left = 0.5 * (sample.left_speed_m_s + self._previous_sample.left_speed_m_s)
        right = 0.5 * (sample.right_speed_m_s + self._previous_sample.right_speed_m_s)
        self.distance_left_m += left * dt
        self.distance_right_m += right * dt
        self.duration_s += dt
        self._previous_sample = sample
        self._previous_now_s = now_s

    @property
    def distance_m(self) -> float:
        """两轮中心行进距离（m）。"""

        return (self.distance_left_m + self.distance_right_m) / 2.0

    @property
    def avg_left_speed_m_s(self) -> float:
        return self.distance_left_m / self.duration_s if self.duration_s > 0.0 else 0.0

    @property
    def avg_right_speed_m_s(self) -> float:
        return self.distance_right_m / self.duration_s if self.duration_s > 0.0 else 0.0

    @property
    def heading_deviation_rad(self) -> float:
        """左轮比右轮多走导致的车体偏航角（rad）；正值表示向左偏。"""

        return (self.distance_left_m - self.distance_right_m) / self.wheel_track_m


def update_wheel_scales(
    scale_left: float,
    scale_right: float,
    *,
    commanded_speed_m_s: float,
    avg_left_speed_m_s: float,
    avg_right_speed_m_s: float,
) -> tuple[float, float]:
    """根据一轮直行的实测平均轮速更新左右比例系数。

    每轮有效增益定义为 ``实测 / (指令 × scale)``；增益低于 1 表示该轮
    欠速，把 scale 放大 ``1/增益``（并限幅）以逼近指令速度。乘法系数与
    速度无关，因此同一组 scale 可在不同速度下保持走直线。
    """

    if not math.isfinite(commanded_speed_m_s) or commanded_speed_m_s <= 0.0:
        raise ValueError(
            f"commanded_speed_m_s must be finite and positive, got "
            f"{commanded_speed_m_s!r}."
        )
    left = _correct_scale(scale_left, commanded_speed_m_s, avg_left_speed_m_s)
    right = _correct_scale(scale_right, commanded_speed_m_s, avg_right_speed_m_s)
    return left, right


def _correct_scale(scale: float, commanded_speed_m_s: float, avg_speed_m_s: float) -> float:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}.")
    commanded = commanded_speed_m_s * scale
    if abs(avg_speed_m_s) < _MIN_MEASURED_SPEED_M_S or abs(commanded) < 1e-9:
        return scale
    gain = avg_speed_m_s / commanded
    correction = min(max(1.0 / gain, _MIN_SCALE_CORRECTION), _MAX_SCALE_CORRECTION)
    return scale * correction


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite value, got {value!r}"
        )
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative finite value, got {value!r}"
        )
    return parsed


def _parse_pid(text: str, parser: argparse.ArgumentParser) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        parser.error("--set-pid must be Kp,Ki,Kd, e.g. 3.5,0.25,0")
    values: list[float] = []
    for part, (name, maximum) in zip(parts, _PID_LIMITS):
        try:
            value = float(part)
        except ValueError:
            parser.error(f"--set-pid {name} must be a number, got {part!r}")
        if not math.isfinite(value) or not (0.0 <= value <= maximum):
            parser.error(f"--set-pid {name} must be in [0, {maximum}], got {value!r}")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune per-wheel speed scale factors over UART text protocol so that "
            "equal wheel-speed commands drive straight, verified across speeds."
        )
    )
    parser.add_argument("--device", required=True, help="Serial device, e.g. /dev/ttyAMA10.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate (default: 115200).")
    parser.add_argument(
        "--wheel-track-m",
        type=_positive_float,
        required=True,
        help="Measured wheel track in meters (needed for heading deviation and turn timing).",
    )
    parser.add_argument(
        "--distance-m",
        type=_positive_float,
        default=_DEFAULT_DISTANCE_M,
        help=f"Straight distance per round in meters (default: {_DEFAULT_DISTANCE_M}).",
    )
    parser.add_argument(
        "--speeds-m-s",
        default="0.05,0.10,0.15,0.20",
        help=(
            f"Comma-separated speeds to test, each within "
            f"[{MIN_SPEED_M_S}, {MAX_SPEED_M_S}] m/s (default: 0.05,0.10,0.15,0.20)."
        ),
    )
    parser.add_argument(
        "--passes",
        type=_positive_int,
        default=1,
        help="Number of full passes over the speed schedule (default: 1).",
    )
    parser.add_argument(
        "--turn-speed-m-s",
        type=_positive_float,
        default=_DEFAULT_TURN_SPEED_M_S,
        help=f"Wheel speed magnitude during the in-place turn (default: {_DEFAULT_TURN_SPEED_M_S}).",
    )
    parser.add_argument(
        "--turn-angle-deg",
        type=_positive_float,
        default=90.0,
        help="In-place turn angle in degrees (default: 90).",
    )
    parser.add_argument(
        "--turn-direction",
        choices=("left", "right"),
        default="left",
        help="Turn direction between rounds (default: left).",
    )
    parser.add_argument(
        "--settle-seconds",
        type=_nonnegative_float,
        default=_DEFAULT_SETTLE_S,
        help=f"Pause after each turn before the next round (default: {_DEFAULT_SETTLE_S}).",
    )
    parser.add_argument(
        "--timeout-s",
        type=_positive_float,
        default=_DEFAULT_TIMEOUT_S,
        help=f"Timeout for a single straight run (default: {_DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--set-pid",
        metavar="KP,KI,KD",
        help="Optional: send p<Kp>,<Ki>,<Kd> once before tuning (shared speed-loop PID).",
    )
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help=(
            "Confirm a physical emergency stop is ready and an operator will "
            "supervise the whole test."
        ),
    )
    args = parser.parse_args()
    if not args.supervised_physical_stop_ready:
        parser.error("--supervised-physical-stop-ready is required for a live motion test")
    if args.baudrate <= 0:
        parser.error("--baudrate must be positive")
    if args.turn_angle_deg > 360.0:
        parser.error("--turn-angle-deg must be <= 360")
    if args.set_pid is not None:
        args.set_pid = _parse_pid(args.set_pid, parser)
    return args


def _send(port: object, text: str) -> None:
    port.write((text + "\r\n").encode("ascii"))
    port.flush()


def _readline(port: object) -> str:
    raw = port.readline()
    if not raw:
        return ""
    return raw.decode("ascii", errors="replace").strip()


def _read_replies(port: object, duration_s: float) -> None:
    """读取并打印命令回复（跳过遥测行），用于 ``v``/``p`` 查询结果。"""

    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        line = _readline(port)
        if line and parse_telemetry_line(line) is None:
            print(f"REPLY {line}", flush=True)


def _turn(
    port: object,
    direction: str,
    wheel_track_m: float,
    turn_speed_m_s: float,
    angle_rad: float,
) -> None:
    duration = turn_duration_s(wheel_track_m, turn_speed_m_s, angle_rad)
    if direction == "left":
        left, right = -turn_speed_m_s, turn_speed_m_s
    else:
        left, right = turn_speed_m_s, -turn_speed_m_s
    _send(port, f"m{left:.4f},{right:.4f}")
    time.sleep(duration)
    _send(port, "b0,0")
    port.reset_input_buffer()


def _run_straight(
    port: object,
    speed_m_s: float,
    scales: WheelScales,
    wheel_track_m: float,
    distance_m: float,
    timeout_s: float,
) -> StraightRunAccumulator:
    accumulator = StraightRunAccumulator(wheel_track_m)
    _send(port, f"m{speed_m_s * scales.left:.4f},{speed_m_s * scales.right:.4f}")
    started = time.monotonic()
    next_progress = started + 1.0
    while True:
        line = _readline(port)
        sample = parse_telemetry_line(line) if line else None
        now = time.monotonic()
        if sample is not None:
            accumulator.submit(sample, now)
        if accumulator.distance_m >= distance_m:
            break
        if now - started >= timeout_s:
            raise RuntimeError(
                f"straight run timed out after {timeout_s:.1f}s: "
                f"distance_m={accumulator.distance_m:.3f} "
                f"samples={accumulator.sample_count}"
            )
        if now >= next_progress:
            print(
                f"PROGRESS speed={speed_m_s:.3f} "
                f"distance_m={accumulator.distance_m:.3f} target_m={distance_m:.3f} "
                f"avg_l={accumulator.avg_left_speed_m_s:.3f} "
                f"avg_r={accumulator.avg_right_speed_m_s:.3f}",
                flush=True,
            )
            next_progress = now + 1.0
    _send(port, "b0,0")
    return accumulator


def main() -> int:
    args = _parse_args()
    speeds = parse_speeds(args.speeds_m_s)
    track = args.wheel_track_m
    turn_angle_rad = math.radians(args.turn_angle_deg)
    scales = WheelScales(1.0, 1.0)

    import serial  # 硬件依赖延迟导入，测试收集时不访问串口

    print(
        "PID_TUNE_START "
        f"device={args.device} baudrate={args.baudrate} "
        f"track_m={track:.4f} distance_m={args.distance_m:.3f} "
        f"speeds_m_s={speeds} passes={args.passes} "
        f"turn={args.turn_direction} {args.turn_angle_deg:.1f}deg "
        f"turn_speed={args.turn_speed_m_s:.3f}",
        flush=True,
    )
    try:
        with serial.Serial(args.device, args.baudrate, timeout=0.05) as port:
            try:
                port.reset_input_buffer()
                if args.set_pid is not None:
                    kp, ki, kd = args.set_pid
                    _send(port, f"p{kp:g},{ki:g},{kd:g}")
                    _read_replies(port, 0.3)
                    print(f"PID_SET kp={kp:g} ki={ki:g} kd={kd:g}", flush=True)
                _send(port, "v")
                _read_replies(port, 0.5)

                for pass_index in range(1, args.passes + 1):
                    for speed in speeds:
                        result = _run_straight(
                            port,
                            speed,
                            scales,
                            track,
                            args.distance_m,
                            args.timeout_s,
                        )
                        new_left, new_right = update_wheel_scales(
                            scales.left,
                            scales.right,
                            commanded_speed_m_s=speed,
                            avg_left_speed_m_s=result.avg_left_speed_m_s,
                            avg_right_speed_m_s=result.avg_right_speed_m_s,
                        )
                        print(
                            "ROUND "
                            f"pass={pass_index} speed={speed:.3f} "
                            f"distance_m={result.distance_m:.3f} "
                            f"dist_l={result.distance_left_m:.3f} "
                            f"dist_r={result.distance_right_m:.3f} "
                            f"avg_l={result.avg_left_speed_m_s:.3f} "
                            f"avg_r={result.avg_right_speed_m_s:.3f} "
                            f"heading_dev_deg={math.degrees(result.heading_deviation_rad):+.2f} "
                            f"scales=({scales.left:.4f},{scales.right:.4f})->"
                            f"({new_left:.4f},{new_right:.4f})",
                            flush=True,
                        )
                        scales = WheelScales(new_left, new_right)
                        _turn(
                            port,
                            args.turn_direction,
                            track,
                            args.turn_speed_m_s,
                            turn_angle_rad,
                        )
                        time.sleep(args.settle_seconds)
                print(
                    "PID_TUNE_DONE "
                    f"left_scale={scales.left:.5f} right_scale={scales.right:.5f} "
                    f"right_over_left={scales.right / scales.left:.5f}",
                    flush=True,
                )
            finally:
                _send(port, "b0,0")
    except KeyboardInterrupt:
        print("INTERRUPTED; soft brake sent.", flush=True)
        return 130
    except BaseException as error:
        print(f"PID_TUNE_ERROR={type(error).__name__}: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
