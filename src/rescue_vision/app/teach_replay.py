"""Record hand-pushed wheel/IMU telemetry and replay the wheel trajectory."""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from rescue_vision.config import load_runtime_config
from rescue_vision.localization import OdometryCalibration
from rescue_vision.motion import MotionController, OdometryImu, SensorFlags


LOG_FORMAT = "rescue_vision.teach_replay"
_REPLAY_INTERVAL_S = 0.05
_CONTROL_POLL_S = 0.01
_GYRO_INITIAL_SAMPLE_TIMEOUT_S = 1.0
_GYRO_MAX_UNAVAILABLE_US = 200_000
_GYRO_MAX_UNAVAILABLE_NS = _GYRO_MAX_UNAVAILABLE_US * 1000
_GYRO_HEADING_KP_S_INV = 2.0
_GYRO_MAX_CORRECTION_RAD_S = 0.6
_REQUIRED_ENCODER_FLAGS = (
    SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
)


@dataclass(frozen=True, slots=True)
class TeachSample:
    received_timestamp_ns: int
    telemetry_sequence: int
    sample_timestamp_us: int
    left_encoder_count: int
    right_encoder_count: int
    gyro_x_urad_s: int
    gyro_y_urad_s: int
    gyro_z_urad_s: int
    accel_x_mm_s2: int
    accel_y_mm_s2: int
    accel_z_mm_s2: int
    imu_temperature_cdeg: int
    sensor_flags: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}.")
        if self.received_timestamp_ns < 0 or self.sample_timestamp_us < 0:
            raise ValueError("Telemetry timestamps must be non-negative.")
        if not 0 <= self.telemetry_sequence <= 0xFFFF:
            raise ValueError("telemetry_sequence must be in [0, 65535].")
        known_flags = 0
        for flag in SensorFlags:
            known_flags |= int(flag)
        if self.sensor_flags < 0 or self.sensor_flags & ~known_flags:
            raise ValueError(
                f"sensor_flags contains unknown bits: {self.sensor_flags:#x}."
            )

    @classmethod
    def from_message(cls, message: OdometryImu) -> TeachSample:
        return cls(
            received_timestamp_ns=message.received_timestamp_ns,
            telemetry_sequence=message.telemetry_sequence,
            sample_timestamp_us=message.sample_timestamp_us,
            left_encoder_count=message.left_encoder_count,
            right_encoder_count=message.right_encoder_count,
            gyro_x_urad_s=message.gyro_x_urad_s,
            gyro_y_urad_s=message.gyro_y_urad_s,
            gyro_z_urad_s=message.gyro_z_urad_s,
            accel_x_mm_s2=message.accel_x_mm_s2,
            accel_y_mm_s2=message.accel_y_mm_s2,
            accel_z_mm_s2=message.accel_z_mm_s2,
            imu_temperature_cdeg=message.imu_temperature_cdeg,
            sensor_flags=int(message.sensor_flags),
        )


@dataclass(frozen=True, slots=True)
class ReplaySegment:
    duration_s: float
    left_m_s: float
    right_m_s: float
    heading_delta_rad: float | None


class GyroHeadingTracker:
    """Integrate replay IMU samples into a relative robot heading."""

    def __init__(self, gyro_z_sign: int) -> None:
        if gyro_z_sign not in {-1, 1}:
            raise ValueError("gyro_z_sign must be exactly -1 or 1.")
        self._gyro_z_sign = gyro_z_sign
        self._previous: OdometryImu | None = None
        self._unavailable_since_us: int | None = None
        self._last_timestamp_us: int | None = None
        self._last_received_timestamp_ns: int | None = None
        self.heading_rad = 0.0

    @property
    def ready(self) -> bool:
        return self._previous is not None

    @property
    def available(self) -> bool:
        return self._previous is not None

    def observe(self, message: OdometryImu) -> None:
        if (
            self._last_timestamp_us is not None
            and message.sample_timestamp_us <= self._last_timestamp_us
        ):
            raise RuntimeError(
                "Replay gyro device timestamps did not increase: "
                f"{self._last_timestamp_us}->{message.sample_timestamp_us}."
            )
        self._last_timestamp_us = message.sample_timestamp_us
        self._last_received_timestamp_ns = message.received_timestamp_ns
        flags = message.sensor_flags
        unavailable = (
            not flags & SensorFlags.IMU_VALID
            or bool(flags & SensorFlags.GYRO_SATURATED)
            or bool(flags & SensorFlags.SAMPLE_OVERRUN)
        )
        if unavailable:
            if self._unavailable_since_us is None:
                self._unavailable_since_us = message.sample_timestamp_us
            elif (
                message.sample_timestamp_us - self._unavailable_since_us
                > _GYRO_MAX_UNAVAILABLE_US
            ):
                raise RuntimeError(
                    "Replay gyro remained unavailable for more than "
                    f"{_GYRO_MAX_UNAVAILABLE_US / 1000:.0f} ms."
                )
            self._previous = None
            return
        previous = self._previous
        self._previous = message
        self._unavailable_since_us = None
        if previous is None:
            return
        dt_s = (
            message.sample_timestamp_us - previous.sample_timestamp_us
        ) / 1_000_000.0
        if dt_s <= 0.0:
            raise RuntimeError(
                "Replay gyro device timestamps did not increase: "
                f"{previous.sample_timestamp_us}->{message.sample_timestamp_us}."
            )
        previous_rate = previous.gyro_z_rad_s * self._gyro_z_sign
        current_rate = message.gyro_z_rad_s * self._gyro_z_sign
        self.heading_rad += 0.5 * (previous_rate + current_rate) * dt_s

    def require_recent(self, now_ns: int) -> None:
        if self._last_received_timestamp_ns is None:
            raise RuntimeError("Replay gyro has not received an initial sample.")
        age_ns = now_ns - self._last_received_timestamp_ns
        if age_ns < 0:
            raise RuntimeError(
                "Replay gyro receive timestamp is later than the local clock: "
                f"age_ns={age_ns}."
            )
        if age_ns > _GYRO_MAX_UNAVAILABLE_NS:
            raise RuntimeError(
                "Replay gyro telemetry has been missing for more than "
                f"{_GYRO_MAX_UNAVAILABLE_US / 1000:.0f} ms."
            )


def _calibration_record(calibration: OdometryCalibration) -> dict[str, object]:
    return {
        "encoder_counts_per_revolution": calibration.encoder_counts_per_revolution,
        "left_wheel_radius_mm": calibration.left_wheel_radius_mm,
        "right_wheel_radius_mm": calibration.right_wheel_radius_mm,
        "gyro_z_sign": calibration.gyro_z_sign,
    }


def write_teach_log(
    path: Path,
    calibration: OdometryCalibration,
    samples: list[TeachSample],
) -> None:
    if len(samples) < 2:
        raise ValueError("A teach log requires at least two odometry/IMU samples.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        header = {
            "record_type": "header",
            "format": LOG_FORMAT,
            "created_at": datetime.now().astimezone().isoformat(),
            "calibration": _calibration_record(calibration),
        }
        stream.write(json.dumps(header, ensure_ascii=False) + "\n")
        for sample in samples:
            record = {"record_type": "sample", **asdict(sample)}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_teach_log(
    path: Path,
    calibration: OdometryCalibration,
) -> list[TeachSample]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read teach log {path}: {exc}") from exc
    if not records or not isinstance(records[0], dict):
        raise ValueError(f"Teach log {path} has no header.")
    header = records[0]
    if header.get("record_type") != "header" or header.get("format") != LOG_FORMAT:
        raise ValueError(f"Teach log {path} has an unsupported header.")
    if header.get("calibration") != _calibration_record(calibration):
        raise ValueError(
            "Teach log odometry calibration does not match the runtime config: "
            f"log={header.get('calibration')!r}, "
            f"runtime={_calibration_record(calibration)!r}."
        )
    field_names = set(TeachSample.__dataclass_fields__)
    samples: list[TeachSample] = []
    for line_number, record in enumerate(records[1:], start=2):
        if not isinstance(record, dict) or record.get("record_type") != "sample":
            raise ValueError(f"Teach log line {line_number} is not a sample.")
        values = {key: value for key, value in record.items() if key != "record_type"}
        if set(values) != field_names:
            raise ValueError(
                f"Teach log line {line_number} has fields {sorted(values)}, "
                f"expected {sorted(field_names)}."
            )
        try:
            samples.append(TeachSample(**values))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid sample on teach log line {line_number}.") from exc
    if len(samples) < 2:
        raise ValueError("Teach log requires at least two samples for replay.")
    return samples


def build_replay_segments(
    samples: list[TeachSample],
    calibration: OdometryCalibration,
    *,
    interval_s: float = _REPLAY_INTERVAL_S,
    speed_scale: float = 1.0,
    use_gyro: bool = False,
) -> list[ReplaySegment]:
    if len(samples) < 2:
        raise ValueError("Replay requires at least two samples.")
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError(f"interval_s must be finite and > 0, got {interval_s!r}.")
    if not math.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale!r}.")
    if not isinstance(use_gyro, bool):
        raise ValueError(f"use_gyro must be a boolean, got {use_gyro!r}.")

    previous_timestamp_us: int | None = None
    for index, sample in enumerate(samples):
        flags = SensorFlags(sample.sensor_flags)
        if flags & _REQUIRED_ENCODER_FLAGS != _REQUIRED_ENCODER_FLAGS:
            raise ValueError(f"Encoder invalid at sample {index}.")
        if flags & SensorFlags.SAMPLE_OVERRUN:
            raise ValueError(f"Sample overrun at sample {index}.")
        if (
            previous_timestamp_us is not None
            and sample.sample_timestamp_us <= previous_timestamp_us
        ):
            raise ValueError(
                "Device sample timestamps must increase; "
                f"sample {index} has {sample.sample_timestamp_us} after "
                f"{previous_timestamp_us}."
            )
        previous_timestamp_us = sample.sample_timestamp_us

    boundaries = [0]
    minimum_us = round(interval_s * 1_000_000)
    for index in range(1, len(samples) - 1):
        if (
            samples[index].sample_timestamp_us
            - samples[boundaries[-1]].sample_timestamp_us
            >= minimum_us
        ):
            boundaries.append(index)
    if boundaries[-1] != len(samples) - 1:
        final_index = len(samples) - 1
        final_remainder_us = (
            samples[final_index].sample_timestamp_us
            - samples[boundaries[-1]].sample_timestamp_us
        )
        if len(boundaries) > 1 and final_remainder_us < minimum_us / 2:
            boundaries[-1] = final_index
        else:
            boundaries.append(final_index)

    metres_per_count_left = (
        2.0
        * math.pi
        * calibration.left_wheel_radius_mm
        / calibration.encoder_counts_per_revolution
        / 1000.0
    )
    metres_per_count_right = (
        2.0
        * math.pi
        * calibration.right_wheel_radius_mm
        / calibration.encoder_counts_per_revolution
        / 1000.0
    )
    segments: list[ReplaySegment] = []
    for start_index, end_index in zip(boundaries, boundaries[1:]):
        start = samples[start_index]
        end = samples[end_index]
        duration_s = (
            end.sample_timestamp_us - start.sample_timestamp_us
        ) / 1_000_000.0
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(
                "Device sample timestamps must increase; "
                f"samples {start_index}->{end_index} give {duration_s!r} s."
            )
        heading_delta_rad: float | None = None
        if use_gyro:
            heading_delta_rad = 0.0
            for index in range(start_index + 1, end_index + 1):
                previous = samples[index - 1]
                current = samples[index]
                previous_flags = SensorFlags(previous.sensor_flags)
                current_flags = SensorFlags(current.sensor_flags)
                if (
                    not previous_flags & SensorFlags.IMU_VALID
                    or not current_flags & SensorFlags.IMU_VALID
                    or previous_flags & SensorFlags.GYRO_SATURATED
                    or current_flags & SensorFlags.GYRO_SATURATED
                ):
                    continue
                dt_s = (
                    current.sample_timestamp_us - previous.sample_timestamp_us
                ) / 1_000_000.0
                previous_rate = (
                    previous.gyro_z_urad_s
                    / 1_000_000.0
                    * calibration.gyro_z_sign
                )
                current_rate = (
                    current.gyro_z_urad_s
                    / 1_000_000.0
                    * calibration.gyro_z_sign
                )
                heading_delta_rad += 0.5 * (previous_rate + current_rate) * dt_s
        segments.append(
            ReplaySegment(
                duration_s=duration_s / speed_scale,
                left_m_s=(
                    (end.left_encoder_count - start.left_encoder_count)
                    * metres_per_count_left
                    / duration_s
                    * speed_scale
                ),
                right_m_s=(
                    (end.right_encoder_count - start.right_encoder_count)
                    * metres_per_count_right
                    / duration_s
                    * speed_scale
                ),
                heading_delta_rad=heading_delta_rad,
            )
        )
    return segments


def validate_segments(
    controller: MotionController,
    segments: list[ReplaySegment],
) -> None:
    maximum = controller.limits.max_wheel_velocity_m_s
    for index, segment in enumerate(segments):
        observed = max(abs(segment.left_m_s), abs(segment.right_m_s))
        if observed > maximum:
            raise ValueError(
                f"Replay segment {index} needs {observed:.3f} m/s, above the "
                f"configured wheel limit {maximum:.3f} m/s. Record more slowly "
                "or replay with --speed-scale below 1."
            )


def gyro_corrected_wheel_speeds(
    controller: MotionController,
    segment: ReplaySegment,
    *,
    target_heading_rad: float,
    measured_heading_rad: float,
) -> tuple[float, float]:
    """Apply bounded heading feedback without exceeding either wheel limit."""

    error_rad = math.atan2(
        math.sin(target_heading_rad - measured_heading_rad),
        math.cos(target_heading_rad - measured_heading_rad),
    )
    desired_correction = max(
        -min(
            _GYRO_MAX_CORRECTION_RAD_S,
            controller.limits.max_angular_velocity_rad_s,
        ),
        min(
            min(
                _GYRO_MAX_CORRECTION_RAD_S,
                controller.limits.max_angular_velocity_rad_s,
            ),
            error_rad * _GYRO_HEADING_KP_S_INV,
        ),
    )
    half_track = controller.limits.wheel_track_m / 2.0
    maximum = controller.limits.max_wheel_velocity_m_s
    correction_low = max(
        (segment.left_m_s - maximum) / half_track,
        (-maximum - segment.right_m_s) / half_track,
    )
    correction_high = min(
        (segment.left_m_s + maximum) / half_track,
        (maximum - segment.right_m_s) / half_track,
    )
    correction = max(correction_low, min(correction_high, desired_correction))
    return (
        segment.left_m_s - correction * half_track,
        segment.right_m_s + correction * half_track,
    )


def _enter_pressed() -> bool:
    readable, _, _ = select.select((sys.stdin,), (), (), 0.0)
    if not readable:
        return False
    # Consume the complete line so it cannot leak into the later REPLAY
    # confirmation prompt. EOF also means the operator input is unavailable.
    return sys.stdin.readline() is not None


def record_samples(controller: MotionController) -> list[TeachSample]:
    print("正在记录；手动推动车辆，完成后按 Enter……")
    samples: list[TeachSample] = []
    last_print_ns = 0
    try:
        while not _enter_pressed():
            try:
                message = controller.receive_message(timeout=0.05)
            except TimeoutError:
                continue
            if not isinstance(message, OdometryImu):
                continue
            sample = TeachSample.from_message(message)
            samples.append(sample)
            if message.received_timestamp_ns - last_print_ns >= 100_000_000:
                print(
                    "\r"
                    f"样本={len(samples):6d}  "
                    f"编码器=({message.left_encoder_count:+9d},"
                    f"{message.right_encoder_count:+9d})  "
                    f"gyro_z={message.gyro_z_rad_s:+7.3f} rad/s",
                    end="",
                    flush=True,
                )
                last_print_ns = message.received_timestamp_ns
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，结束录制并保存日志。", file=sys.stderr)
    print()
    return samples


def replay_segments(
    controller: MotionController,
    segments: list[ReplaySegment],
    *,
    calibration: OdometryCalibration,
    use_gyro: bool,
) -> bool:
    print("回放中；按 Enter 可随时软刹车停止。")
    gyro_tracker = GyroHeadingTracker(calibration.gyro_z_sign) if use_gyro else None
    if gyro_tracker is not None:
        deadline = time.monotonic() + _GYRO_INITIAL_SAMPLE_TIMEOUT_S
        while not gyro_tracker.ready:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError("Timed out waiting for an initial replay gyro sample.")
            try:
                message = controller.receive_message(timeout=min(0.05, remaining_s))
            except TimeoutError:
                continue
            if isinstance(message, OdometryImu):
                gyro_tracker.observe(message)
    started_ns = time.monotonic_ns()
    segment_started_s = 0.0
    target_heading_start_rad = 0.0
    try:
        for index, segment in enumerate(segments):
            segment_end_s = segment_started_s + segment.duration_s
            while True:
                if _enter_pressed():
                    return False
                now_ns = time.monotonic_ns()
                elapsed_s = (now_ns - started_ns) / 1_000_000_000.0
                if elapsed_s >= segment_end_s:
                    break
                left_m_s = segment.left_m_s
                right_m_s = segment.right_m_s
                if gyro_tracker is not None and gyro_tracker.available:
                    assert segment.heading_delta_rad is not None
                    progress = max(
                        0.0,
                        min(
                            1.0,
                            (elapsed_s - segment_started_s) / segment.duration_s,
                        ),
                    )
                    target_heading_rad = (
                        target_heading_start_rad
                        + segment.heading_delta_rad * progress
                    )
                    left_m_s, right_m_s = gyro_corrected_wheel_speeds(
                        controller,
                        segment,
                        target_heading_rad=target_heading_rad,
                        measured_heading_rad=gyro_tracker.heading_rad,
                    )
                controller.set_wheel_speeds(
                    left_m_s=left_m_s,
                    right_m_s=right_m_s,
                    min_wheel_velocity_m_s=0.0,
                )
                controller.update()
                messages = controller.drain_messages()
                if gyro_tracker is not None:
                    for message in messages:
                        if isinstance(message, OdometryImu):
                            gyro_tracker.observe(message)
                    gyro_tracker.require_recent(time.monotonic_ns())
                if controller.emergency_stop_latched:
                    raise RuntimeError(
                        "STM32 emergency stop became latched during replay."
                    )
                if controller.needs_synchronization or controller.link_degraded:
                    raise RuntimeError(
                        "STM32 link lost synchronization during replay."
                    )
                remaining_s = segment_end_s - elapsed_s
                time.sleep(min(_CONTROL_POLL_S, max(0.0, remaining_s)))
            segment_started_s = segment_end_s
            if segment.heading_delta_rad is not None:
                target_heading_start_rad += segment.heading_delta_rad
            print(
                f"\r回放 {index + 1}/{len(segments)}  "
                f"left={segment.left_m_s:+.3f} right={segment.right_m_s:+.3f} m/s"
                f"  gyro={'on' if gyro_tracker is not None else 'off'}",
                end="",
                flush=True,
            )
        print()
        return True
    finally:
        controller.soft_brake()


def _default_log_path(log_dir: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return (log_dir / f"teach_replay_{stamp}.jsonl").resolve()


def _confirm_replay(path: Path) -> bool:
    try:
        answer = input(
            f"回放日志：{path}\n"
            "把车辆放到回放起点并清空周围，输入 REPLAY 后按 Enter 开始（其它输入退出）： "
        )
    except EOFError:
        return False
    return answer.strip() == "REPLAY"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record hand-pushed encoder/IMU telemetry and replay it later."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--replay",
        type=Path,
        help="Replay an existing teach_replay JSONL instead of recording a new one.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="Replay speed multiplier in (0, 1]; distance is unchanged.",
    )
    parser.add_argument(
        "--use-gyro",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use recorded and live gyro heading feedback during replay "
            "(default: --no-use-gyro)."
        ),
    )
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help="Confirm a physical emergency stop and continuous supervision are ready.",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if not args.supervised_physical_stop_ready:
        parser.error(
            "pass --supervised-physical-stop-ready only when a physical emergency "
            "stop and continuous supervision are ready"
        )
    if not math.isfinite(args.speed_scale) or not 0.0 < args.speed_scale <= 1.0:
        parser.error("--speed-scale must be finite and in (0, 1]")

    config_path = args.config.expanduser().resolve()
    config = load_runtime_config(config_path)
    calibration = config.motion.odometry.build_calibration()
    if not config.uart.enabled or not config.motion.enabled or calibration is None:
        raise RuntimeError(
            "Teach/replay requires uart.enabled=true, motion.enabled=true, and "
            "motion.odometry.enabled=true with complete calibration."
        )
    channel = config.uart.build_channel()
    controller = config.motion.build_controller(channel)
    if channel is None or controller is None:
        raise RuntimeError("Runtime config did not build a UART motion controller.")

    replay_path = args.replay.expanduser().resolve() if args.replay else None
    samples = load_teach_log(replay_path, calibration) if replay_path else None
    output_path: Path | None = None

    if samples is None:
        with channel:
            try:
                print(
                    "正在同步 STM32；录制阶段保持软刹车，请手动缓慢推动车辆。"
                )
                controller.synchronize(
                    timeout_s=config.motion.synchronization_timeout_s
                )
                if controller.emergency_stop_latched:
                    raise RuntimeError(
                        "STM32 emergency stop is latched; reset it first."
                    )
                samples = record_samples(controller)
            finally:
                controller.soft_brake()
        output_path = _default_log_path(args.log_dir.expanduser().resolve())
        write_teach_log(output_path, calibration, samples)
        print(f"日志已保存到 {output_path}")
    final_path = output_path or replay_path
    assert final_path is not None

    if args.use_gyro:
        unusable_gyro_samples = sum(
            1
            for sample in samples
            if (
                not SensorFlags(sample.sensor_flags) & SensorFlags.IMU_VALID
                or SensorFlags(sample.sensor_flags) & SensorFlags.GYRO_SATURATED
            )
        )
        print(
            "陀螺仪航向修正已启用；"
            f"日志中跳过 {unusable_gyro_samples}/{len(samples)} 个不可用 IMU 样本。"
        )
    segments = build_replay_segments(
        samples,
        calibration,
        speed_scale=args.speed_scale,
        use_gyro=args.use_gyro,
    )
    validate_segments(controller, segments)
    # Human confirmation must happen while UART is closed. STM32 telemetry is
    # high-rate, so blocking on input with the reader active fills its bounded
    # receive queue in a few seconds.
    if not _confirm_replay(final_path):
        print(f"未执行回放。日志：{final_path}")
        return

    with channel:
        try:
            print("正在重新打开 UART 并同步 STM32，准备回放。")
            controller.synchronize(
                timeout_s=config.motion.synchronization_timeout_s
            )
            if controller.emergency_stop_latched:
                raise RuntimeError(
                    "STM32 emergency stop is latched; reset it first."
                )
            completed = replay_segments(
                controller,
                segments,
                calibration=calibration,
                use_gyro=args.use_gyro,
            )
            print("回放完成。" if completed else "回放已由操作员停止。")
        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，正在软刹车。", file=sys.stderr)
        finally:
            controller.soft_brake()

    print(f"日志：{final_path}")


if __name__ == "__main__":
    main()
