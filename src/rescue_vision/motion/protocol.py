"""树莓派与 STM32 固定长度二进制命令及遥测协议。"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Protocol

from rescue_vision.communication import ReceivedUartFrame


MAX_DECODED_FRAME_BYTES = 64
STM32_UART_BAUDRATE = 115200
_CRC_STRUCT = struct.Struct("<H")
_WHEEL_COMMAND = struct.Struct("<Hhh")
_SEQUENCE_COMMAND = struct.Struct("<H")
_GRIPPER_COMMAND = struct.Struct("<HHH")
_COMMAND_REPLY = struct.Struct("<HBB")
_ODOMETRY_IMU = struct.Struct("<HQqqiiiiiihH")
_SYSTEM_STATUS = struct.Struct("<HQHIHBHH")
_NO_MOTION_COMMAND_AGE = 0xFFFFFFFF


class ControllerProtocolError(ValueError):
    """一帧数据不符合冻结的 STM32 协议。"""


class MessageType(IntEnum):
    SET_WHEEL_SPEED = 0x10
    SOFT_BRAKE = 0x11
    EMERGENCY_STOP = 0x12
    SET_GRIPPER = 0x13
    QUERY_STATUS = 0x14
    COMMAND_REPLY = 0x80
    ODOMETRY_IMU = 0x81
    SYSTEM_STATUS = 0x82


class CommandResult(IntEnum):
    ACCEPTED = 0
    OUT_OF_RANGE = 1
    EMERGENCY_STOP_LATCHED = 2
    DEVICE_UNAVAILABLE = 3
    SEQUENCE_OLD = 4


class SensorFlags(IntFlag):
    IMU_VALID = 1 << 0
    LEFT_ENCODER_VALID = 1 << 2
    RIGHT_ENCODER_VALID = 1 << 3
    GYRO_SATURATED = 1 << 4
    ACCEL_SATURATED = 1 << 5
    SAMPLE_OVERRUN = 1 << 6


class SystemFlags(IntFlag):
    PROTOCOL_READY = 1 << 0
    EMERGENCY_STOP_LATCHED = 1 << 1
    MOTOR_OUTPUT_ENABLED = 1 << 2
    GRIPPER_OUTPUT_AVAILABLE = 1 << 3
    REPLY_QUEUE_FULL = 1 << 4
    TX_DEGRADED = 1 << 5
    RX_DEGRADED = 1 << 6


class CarStopReason(IntEnum):
    STARTUP = 0
    RUNNING = 1
    SOFT_BRAKE = 2
    WATCHDOG_TIMEOUT = 3
    EMERGENCY_STOP = 4


class CarFrameChannel(Protocol):
    """运动层所需的最小 COBS 帧通道接口。"""

    def send_frame(self, payload: bytes) -> None: ...

    def receive_frame(
        self,
        timeout: float | None = None,
    ) -> ReceivedUartFrame: ...


def crc16_ccitt_false(data: bytes) -> int:
    """计算 CRC-16/CCITT-FALSE。"""

    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, got {type(data).__name__}.")
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def pack_protocol_frame(message_type: MessageType, payload: bytes) -> bytes:
    """构造 COBS 编码前的 ``type | payload | crc16_le``。"""

    if not isinstance(message_type, MessageType):
        raise TypeError("message_type must be a MessageType.")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes.")
    body = bytes((message_type.value,)) + payload
    frame = body + _CRC_STRUCT.pack(crc16_ccitt_false(body))
    if len(frame) > MAX_DECODED_FRAME_BYTES:
        raise ValueError(
            f"Decoded protocol frame has {len(frame)} bytes; "
            f"maximum is {MAX_DECODED_FRAME_BYTES}."
        )
    return frame


def _command_sequence(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 0xFFFF
    ):
        raise ValueError(
            f"command_sequence must be an integer in [0, 65535], got {value!r}."
        )
    return value


def _finite(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return float(value)


def encode_wheel_speed_command(
    command_sequence: int,
    left_m_s: float,
    right_m_s: float,
) -> bytes:
    """编码左右轮目标速度，线路单位为 mm/s。"""

    sequence = _command_sequence(command_sequence)
    speeds: list[int] = []
    for name, value in (("left_m_s", left_m_s), ("right_m_s", right_m_s)):
        mm_s = round(_finite(value, name) * 1000.0)
        if not -32768 <= mm_s <= 32767:
            raise ValueError(f"{name} is outside the protocol int16 range.")
        speeds.append(mm_s)
    return pack_protocol_frame(
        MessageType.SET_WHEEL_SPEED,
        _WHEEL_COMMAND.pack(sequence, speeds[0], speeds[1]),
    )


def encode_soft_brake_command(command_sequence: int) -> bytes:
    return pack_protocol_frame(
        MessageType.SOFT_BRAKE,
        _SEQUENCE_COMMAND.pack(_command_sequence(command_sequence)),
    )


def encode_emergency_stop_command(command_sequence: int) -> bytes:
    return pack_protocol_frame(
        MessageType.EMERGENCY_STOP,
        _SEQUENCE_COMMAND.pack(_command_sequence(command_sequence)),
    )


def encode_gripper_command(
    command_sequence: int,
    left_angle_deg: float,
    right_angle_deg: float,
) -> bytes:
    sequence = _command_sequence(command_sequence)
    angles: list[int] = []
    for name, value in (
        ("left_angle_deg", left_angle_deg),
        ("right_angle_deg", right_angle_deg),
    ):
        angle = _finite(value, name)
        if not 0.0 <= angle <= 180.0:
            raise ValueError(f"{name} must be in [0, 180], got {angle!r}.")
        angles.append(round(angle * 100.0))
    return pack_protocol_frame(
        MessageType.SET_GRIPPER,
        _GRIPPER_COMMAND.pack(sequence, angles[0], angles[1]),
    )


def encode_state_query_command(command_sequence: int) -> bytes:
    return pack_protocol_frame(
        MessageType.QUERY_STATUS,
        _SEQUENCE_COMMAND.pack(_command_sequence(command_sequence)),
    )


def _non_negative_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{location} must be a non-negative integer, got {value!r}."
        )
    return value


def _bounded_int(value: object, minimum: int, maximum: int, location: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{location} must be an integer in [{minimum}, {maximum}], "
            f"got {value!r}."
        )
    return value


@dataclass(frozen=True, slots=True)
class CarCommandReply:
    uart_sequence: int
    received_timestamp_ns: int
    command_sequence: int
    command_type: MessageType
    result: CommandResult

    def __post_init__(self) -> None:
        _non_negative_int(self.uart_sequence, "uart_sequence")
        _non_negative_int(self.received_timestamp_ns, "received_timestamp_ns")
        _command_sequence(self.command_sequence)
        if not isinstance(
            self.command_type,
            MessageType,
        ) or self.command_type not in {
            MessageType.SET_WHEEL_SPEED,
            MessageType.SOFT_BRAKE,
            MessageType.EMERGENCY_STOP,
            MessageType.SET_GRIPPER,
            MessageType.QUERY_STATUS,
        }:
            raise ValueError("command_type must identify a controller command.")
        if not isinstance(self.result, CommandResult):
            raise ValueError("result must be a CommandResult.")


@dataclass(frozen=True, slots=True)
class OdometryImu:
    uart_sequence: int
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
    sensor_flags: SensorFlags

    def __post_init__(self) -> None:
        _non_negative_int(self.uart_sequence, "uart_sequence")
        _non_negative_int(self.received_timestamp_ns, "received_timestamp_ns")
        _bounded_int(self.telemetry_sequence, 0, 0xFFFF, "telemetry_sequence")
        _bounded_int(
            self.sample_timestamp_us,
            0,
            0xFFFFFFFFFFFFFFFF,
            "sample_timestamp_us",
        )
        for name in ("left_encoder_count", "right_encoder_count"):
            _bounded_int(getattr(self, name), -(1 << 63), (1 << 63) - 1, name)
        for name in (
            "gyro_x_urad_s",
            "gyro_y_urad_s",
            "gyro_z_urad_s",
            "accel_x_mm_s2",
            "accel_y_mm_s2",
            "accel_z_mm_s2",
        ):
            _bounded_int(getattr(self, name), -(1 << 31), (1 << 31) - 1, name)
        _bounded_int(
            self.imu_temperature_cdeg,
            -(1 << 15),
            (1 << 15) - 1,
            "imu_temperature_cdeg",
        )
        if not isinstance(self.sensor_flags, SensorFlags):
            raise ValueError("sensor_flags must be SensorFlags.")
        if int(self.sensor_flags) & ~_known_flag_mask(SensorFlags):
            raise ValueError("sensor_flags contains undefined bits.")

    @property
    def gyro_z_rad_s(self) -> float:
        return self.gyro_z_urad_s / 1_000_000.0


@dataclass(frozen=True, slots=True)
class CarSystemStatus:
    uart_sequence: int
    received_timestamp_ns: int
    status_sequence: int
    controller_timestamp_us: int
    watchdog_timeout_ms: int
    last_motion_command_age_ms: int | None
    system_flags: SystemFlags
    stop_reason: CarStopReason
    servo_left_target_cdeg: int
    servo_right_target_cdeg: int

    def __post_init__(self) -> None:
        _non_negative_int(self.uart_sequence, "uart_sequence")
        _non_negative_int(self.received_timestamp_ns, "received_timestamp_ns")
        _bounded_int(self.status_sequence, 0, 0xFFFF, "status_sequence")
        _bounded_int(
            self.controller_timestamp_us,
            0,
            0xFFFFFFFFFFFFFFFF,
            "controller_timestamp_us",
        )
        _bounded_int(self.watchdog_timeout_ms, 1, 0xFFFF, "watchdog_timeout_ms")
        if self.last_motion_command_age_ms is not None:
            _bounded_int(
                self.last_motion_command_age_ms,
                0,
                _NO_MOTION_COMMAND_AGE - 1,
                "last_motion_command_age_ms",
            )
        if not isinstance(self.system_flags, SystemFlags):
            raise ValueError("system_flags must be SystemFlags.")
        if int(self.system_flags) & ~_known_flag_mask(SystemFlags):
            raise ValueError("system_flags contains undefined bits.")
        if not isinstance(self.stop_reason, CarStopReason):
            raise ValueError("stop_reason must be a CarStopReason.")
        for name in ("servo_left_target_cdeg", "servo_right_target_cdeg"):
            value = getattr(self, name)
            _bounded_int(value, 0, 18000, name)

    @property
    def protocol_ready(self) -> bool:
        return bool(self.system_flags & SystemFlags.PROTOCOL_READY)

    @property
    def watchdog_armed(self) -> bool:
        """Whether the controller has accepted a motion heartbeat.

        v2 no longer allocates a ``watchdog_armed`` status bit.  The presence
        of a motion-command age is the only protocol-level indication that
        the communication watchdog has received a motion command.
        """

        return self.last_motion_command_age_ms is not None

    @property
    def emergency_stop_latched(self) -> bool:
        return bool(self.system_flags & SystemFlags.EMERGENCY_STOP_LATCHED)

    @property
    def motor_output_enabled(self) -> bool:
        return bool(self.system_flags & SystemFlags.MOTOR_OUTPUT_ENABLED)

    @property
    def gripper_output_available(self) -> bool:
        return bool(self.system_flags & SystemFlags.GRIPPER_OUTPUT_AVAILABLE)

    @property
    def reply_queue_full(self) -> bool:
        return bool(self.system_flags & SystemFlags.REPLY_QUEUE_FULL)

    @property
    def tx_degraded(self) -> bool:
        return bool(self.system_flags & SystemFlags.TX_DEGRADED)

    @property
    def rx_degraded(self) -> bool:
        return bool(self.system_flags & SystemFlags.RX_DEGRADED)

    @property
    def servo_left_deg(self) -> float:
        return self.servo_left_target_cdeg / 100.0

    @property
    def servo_right_deg(self) -> float:
        return self.servo_right_target_cdeg / 100.0


ParsedCarMessage = CarCommandReply | OdometryImu | CarSystemStatus


def _known_flag_mask(enum_type: type[IntFlag]) -> int:
    mask = 0
    for flag in enum_type:
        mask |= int(flag)
    return mask


def _verify_frame(frame: ReceivedUartFrame) -> tuple[MessageType, bytes]:
    if not isinstance(frame, ReceivedUartFrame):
        raise TypeError(
            f"frame must be ReceivedUartFrame, got {type(frame).__name__}."
        )
    if len(frame.payload) < 3:
        raise ControllerProtocolError("Protocol frame is shorter than type and CRC.")
    if len(frame.payload) > MAX_DECODED_FRAME_BYTES:
        raise ControllerProtocolError(
            f"Protocol frame has {len(frame.payload)} decoded bytes; "
            f"maximum is {MAX_DECODED_FRAME_BYTES}."
        )
    body = frame.payload[:-2]
    expected_crc = _CRC_STRUCT.unpack(frame.payload[-2:])[0]
    actual_crc = crc16_ccitt_false(body)
    if actual_crc != expected_crc:
        raise ControllerProtocolError(
            f"CRC mismatch: received 0x{expected_crc:04X}, "
            f"calculated 0x{actual_crc:04X}."
        )
    try:
        message_type = MessageType(body[0])
    except ValueError as exc:
        raise ControllerProtocolError(
            f"Unknown message type 0x{body[0]:02X}."
        ) from exc
    return message_type, body[1:]


def _unpack_exact(
    layout: struct.Struct,
    payload: bytes,
    message_type: MessageType,
) -> tuple[int, ...]:
    if len(payload) != layout.size:
        raise ControllerProtocolError(
            f"{message_type.name} payload has {len(payload)} bytes; "
            f"expected {layout.size}."
        )
    return layout.unpack(payload)


def parse_controller_frame(frame: ReceivedUartFrame) -> ParsedCarMessage:
    """校验 CRC、类型、固定长度、枚举和值域并解析一帧 STM32 回传。"""

    message_type, payload = _verify_frame(frame)
    if message_type is MessageType.COMMAND_REPLY:
        command_sequence, command_type_raw, result_raw = _unpack_exact(
            _COMMAND_REPLY, payload, message_type
        )
        try:
            command_type = MessageType(command_type_raw)
            result = CommandResult(result_raw)
        except ValueError as exc:
            raise ControllerProtocolError(
                "COMMAND_REPLY contains an unknown enum."
            ) from exc
        try:
            return CarCommandReply(
                frame.sequence,
                frame.received_timestamp_ns,
                command_sequence,
                command_type,
                result,
            )
        except ValueError as exc:
            raise ControllerProtocolError(str(exc)) from exc

    if message_type is MessageType.ODOMETRY_IMU:
        values = _unpack_exact(_ODOMETRY_IMU, payload, message_type)
        try:
            return OdometryImu(
                uart_sequence=frame.sequence,
                received_timestamp_ns=frame.received_timestamp_ns,
                telemetry_sequence=values[0],
                sample_timestamp_us=values[1],
                left_encoder_count=values[2],
                right_encoder_count=values[3],
                gyro_x_urad_s=values[4],
                gyro_y_urad_s=values[5],
                gyro_z_urad_s=values[6],
                accel_x_mm_s2=values[7],
                accel_y_mm_s2=values[8],
                accel_z_mm_s2=values[9],
                imu_temperature_cdeg=values[10],
                sensor_flags=SensorFlags(values[11]),
            )
        except ValueError as exc:
            raise ControllerProtocolError(str(exc)) from exc

    if message_type is MessageType.SYSTEM_STATUS:
        values = _unpack_exact(_SYSTEM_STATUS, payload, message_type)
        try:
            return CarSystemStatus(
                uart_sequence=frame.sequence,
                received_timestamp_ns=frame.received_timestamp_ns,
                status_sequence=values[0],
                controller_timestamp_us=values[1],
                watchdog_timeout_ms=values[2],
                last_motion_command_age_ms=(
                    None if values[3] == _NO_MOTION_COMMAND_AGE else values[3]
                ),
                system_flags=SystemFlags(values[4]),
                stop_reason=CarStopReason(values[5]),
                servo_left_target_cdeg=values[6],
                servo_right_target_cdeg=values[7],
            )
        except ValueError as exc:
            raise ControllerProtocolError(str(exc)) from exc

    raise ControllerProtocolError(
        f"Received command-only message type {message_type.name} from STM32."
    )
