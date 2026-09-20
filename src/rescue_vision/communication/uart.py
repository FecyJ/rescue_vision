"""协议无关的 COBS UART 帧收发与生命周期管理。"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rescue_vision.exception_notes import add_exception_note


class UartError(RuntimeError):
    """UART 通道无法继续可靠工作。"""


class UartReceiveOverflowError(UartError):
    """接收方未及时消费，导致有界队列溢出。"""


class CobsDecodeError(ValueError):
    """一段非零字节不是合法 COBS 编码。"""


@runtime_checkable
class _SerialPort(Protocol):
    def read(self, size: int = 1) -> bytes: ...

    def write(self, data: bytes) -> int: ...

    def close(self) -> None: ...


def cobs_encode(payload: bytes) -> bytes:
    """对任意 bytes 执行标准 COBS 编码，不附加 ``0x00`` 定界符。"""

    if not isinstance(payload, bytes):
        raise TypeError(f"payload must be bytes, got {type(payload).__name__}.")
    encoded = bytearray((0,))
    code_index = 0
    code = 1
    for value in payload:
        if value == 0:
            encoded[code_index] = code
            code_index = len(encoded)
            encoded.append(0)
            code = 1
            continue
        encoded.append(value)
        code += 1
        if code == 0xFF:
            encoded[code_index] = code
            code_index = len(encoded)
            encoded.append(0)
            code = 1
    if payload and payload[-1] != 0 and code == 1:
        del encoded[code_index]
    else:
        encoded[code_index] = code
    return bytes(encoded)


def cobs_decode(encoded: bytes) -> bytes:
    """解码不含 ``0x00`` 定界符的标准 COBS 数据。"""

    if not isinstance(encoded, bytes):
        raise TypeError(f"encoded must be bytes, got {type(encoded).__name__}.")
    if not encoded:
        raise CobsDecodeError("COBS frame must be non-empty.")
    decoded = bytearray()
    index = 0
    while index < len(encoded):
        code = encoded[index]
        if code == 0:
            raise CobsDecodeError("COBS data must not contain zero bytes.")
        index += 1
        block_end = index + code - 1
        if block_end > len(encoded):
            raise CobsDecodeError("COBS code exceeds the encoded frame length.")
        decoded.extend(encoded[index:block_end])
        index = block_end
        if code != 0xFF and index < len(encoded):
            decoded.append(0)
    return bytes(decoded)


@dataclass(frozen=True, slots=True)
class ReceivedUartFrame:
    """一帧已完成 COBS 解码的数据及树莓派接收完成时间。"""

    sequence: int
    received_timestamp_ns: int
    payload: bytes

    def __post_init__(self) -> None:
        for name, value in (
            ("sequence", self.sequence),
            ("received_timestamp_ns", self.received_timestamp_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}."
                )
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("payload must be non-empty bytes.")


class UartFrameFramer:
    """按 ``0x00`` 定界并 COBS 解码，可在损坏或超长帧后恢复。"""

    def __init__(self, *, max_frame_bytes: int) -> None:
        if (
            isinstance(max_frame_bytes, bool)
            or not isinstance(max_frame_bytes, int)
            or max_frame_bytes <= 0
        ):
            raise ValueError(
                "max_frame_bytes must be a positive integer, "
                f"got {max_frame_bytes!r}."
            )
        self.max_frame_bytes = max_frame_bytes
        self.max_encoded_frame_bytes = (
            max_frame_bytes + max_frame_bytes // 254 + 1
        )
        self._buffer = bytearray()
        self._discarding = False
        self.discarded_frames = 0

    @property
    def pending_bytes(self) -> bytes:
        return bytes(self._buffer)

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        """消费任意非空字节块，只返回合法且不超长的已解码帧。"""

        if not isinstance(chunk, bytes) or not chunk:
            raise ValueError("chunk must be non-empty bytes.")
        frames: list[bytes] = []
        for value in chunk:
            if value == 0:
                if self._discarding:
                    self._discarding = False
                    self._buffer.clear()
                    continue
                if not self._buffer:
                    self.discarded_frames += 1
                    continue
                encoded = bytes(self._buffer)
                self._buffer.clear()
                try:
                    decoded = cobs_decode(encoded)
                except CobsDecodeError:
                    self.discarded_frames += 1
                    continue
                if not decoded or len(decoded) > self.max_frame_bytes:
                    self.discarded_frames += 1
                    continue
                frames.append(decoded)
                continue
            if self._discarding:
                continue
            self._buffer.append(value)
            if len(self._buffer) > self.max_encoded_frame_bytes:
                self._buffer.clear()
                self._discarding = True
                self.discarded_frames += 1
        return tuple(frames)


def _open_pyserial(
    *,
    device: str,
    baudrate: int,
    read_timeout_s: float,
    write_timeout_s: float,
) -> _SerialPort:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for real UART communication; "
            "install requirements.txt."
        ) from exc

    return serial.Serial(
        port=device,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=read_timeout_s,
        write_timeout=write_timeout_s,
        exclusive=True,
    )


class UartFrameChannel:
    """单后台读线程、有界接收队列的 8N1 COBS 帧通道。"""

    def __init__(
        self,
        *,
        device: str,
        baudrate: int,
        read_timeout_s: float,
        write_timeout_s: float,
        receive_queue_capacity: int,
        max_frame_bytes: int,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        serial_factory: Callable[..., _SerialPort] = _open_pyserial,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string.")
        if (
            isinstance(baudrate, bool)
            or not isinstance(baudrate, int)
            or baudrate <= 0
        ):
            raise ValueError(
                f"baudrate must be a positive integer, got {baudrate!r}."
            )
        for name, value in (
            ("read_timeout_s", read_timeout_s),
            ("write_timeout_s", write_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < float(value) < float("inf")
            ):
                raise ValueError(f"{name} must be finite and > 0, got {value!r}.")
        if (
            isinstance(receive_queue_capacity, bool)
            or not isinstance(receive_queue_capacity, int)
            or receive_queue_capacity <= 0
        ):
            raise ValueError(
                "receive_queue_capacity must be a positive integer, "
                f"got {receive_queue_capacity!r}."
            )
        UartFrameFramer(max_frame_bytes=max_frame_bytes)

        self.device = device.strip()
        self.baudrate = baudrate
        self.read_timeout_s = float(read_timeout_s)
        self.write_timeout_s = float(write_timeout_s)
        self.receive_queue_capacity = receive_queue_capacity
        self.max_frame_bytes = max_frame_bytes
        self._monotonic_ns = monotonic_ns
        self._serial_factory = serial_factory
        self._serial: _SerialPort | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._received = queue.Queue[ReceivedUartFrame](
            maxsize=receive_queue_capacity
        )
        self._reader_error: BaseException | None = None
        self._started = False
        self.received_frames = 0
        self.discarded_frames = 0
        self.sent_bytes = 0

    @property
    def started(self) -> bool:
        return self._started

    def check_health(self) -> None:
        self._require_healthy()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("UART channel is already started.")
        self._stop_event.clear()
        self._reader_error = None
        self._received = queue.Queue(maxsize=self.receive_queue_capacity)
        self.received_frames = 0
        self.discarded_frames = 0
        self.sent_bytes = 0
        try:
            serial_port = self._serial_factory(
                device=self.device,
                baudrate=self.baudrate,
                read_timeout_s=self.read_timeout_s,
                write_timeout_s=self.write_timeout_s,
            )
            if not isinstance(serial_port, _SerialPort):
                close = getattr(serial_port, "close", None)
                if callable(close):
                    close()
                raise TypeError(
                    "serial_factory must return an object with read, write "
                    "and close methods."
                )
            self._serial = serial_port
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name="rescue-uart-reader",
                daemon=True,
            )
            self._started = True
            self._reader_thread.start()
        except BaseException as error:
            self._started = False
            if self._serial is not None:
                try:
                    self._serial.close()
                except BaseException as cleanup_error:
                    add_exception_note(
                        error,
                        f"UART open cleanup also failed: {cleanup_error!r}",
                    )
            self._serial = None
            self._reader_thread = None
            raise

    def send(self, payload: bytes) -> None:
        """完整写出非空字节；并发调用按整次写操作串行化。"""

        if not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be non-empty bytes.")
        # A receive-thread failure still permits the shutdown path to attempt
        # one final SOFT_BRAKE before the port closes. Read-side callers and
        # check_health() continue to surface the original failure immediately.
        serial_port = self._require_open()
        with self._write_lock:
            offset = 0
            try:
                while offset < len(payload):
                    written = serial_port.write(payload[offset:])
                    if not isinstance(written, int) or written <= 0:
                        raise UartError(
                            f"UART write made no progress at byte {offset}."
                        )
                    offset += written
                    self.sent_bytes += written
            except BaseException as exc:
                if isinstance(exc, UartError):
                    raise
                raise UartError(
                    f"UART write failed for {self.device!r} at byte "
                    f"{offset}/{len(payload)}: {type(exc).__name__}: {exc}"
                ) from exc

    def send_frame(self, payload: bytes) -> None:
        """COBS 编码一个非空解码态帧并附加 ``0x00``。"""

        if not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be non-empty bytes.")
        if len(payload) > self.max_frame_bytes:
            raise ValueError(
                f"UART frame has {len(payload)} bytes; "
                f"maximum is {self.max_frame_bytes}."
            )
        self.send(cobs_encode(payload) + b"\x00")

    def receive_frame(self, timeout: float | None = None) -> ReceivedUartFrame:
        """等待一帧已完成 COBS 解码的数据；超时抛出 ``TimeoutError``。"""

        self._require_healthy()
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 <= float(timeout) < float("inf")
        ):
            raise ValueError(f"timeout must be finite and >= 0, got {timeout!r}.")
        timeout_s = None if timeout is None else float(timeout)
        if timeout_s == 0:
            try:
                frame = self._received.get_nowait()
            except queue.Empty as exc:
                self._raise_reader_error()
                raise TimeoutError(
                    f"Timed out waiting for UART frame from {self.device!r}."
                ) from exc
            self._raise_reader_error()
            return frame

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        last_empty: queue.Empty | None = None
        while True:
            self._raise_reader_error()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait_s = min(remaining, 0.05)
            else:
                wait_s = 0.05
            try:
                frame = self._received.get(timeout=wait_s)
            except queue.Empty as exc:
                last_empty = exc
                continue
            self._raise_reader_error()
            return frame

        self._raise_reader_error()
        error = TimeoutError(
            f"Timed out waiting for UART frame from {self.device!r}."
        )
        if last_empty is None:
            raise error
        raise error from last_empty

    def stop(self) -> None:
        """停止后台读取并关闭串口；可重复调用。"""

        if not self._started:
            return
        self._stop_event.set()
        serial_port = self._serial
        thread = self._reader_thread
        assert serial_port is not None
        assert thread is not None

        cleanup_errors: list[tuple[str, BaseException]] = []
        cancel_read = getattr(serial_port, "cancel_read", None)
        if callable(cancel_read):
            try:
                cancel_read()
            except BaseException:
                pass
        thread.join(timeout=self.read_timeout_s + 1.0)
        try:
            serial_port.close()
        except BaseException as error:
            cleanup_errors.append(("close", error))
        if thread.is_alive():
            thread.join(timeout=1.0)

        self._serial = None
        self._reader_thread = None
        self._started = False
        if thread.is_alive():
            cleanup_errors.append(
                ("reader_thread", RuntimeError("UART reader thread did not stop."))
            )
        if cleanup_errors:
            location, error = cleanup_errors[0]
            for later_location, later_error in cleanup_errors[1:]:
                add_exception_note(
                    error,
                    f"{later_location} also failed: {later_error!r}",
                )
            raise RuntimeError(f"UART cleanup failed in {location}.") from error

    def _reader_loop(self) -> None:
        framer = UartFrameFramer(max_frame_bytes=self.max_frame_bytes)
        sequence = 0
        try:
            assert self._serial is not None
            while not self._stop_event.is_set():
                chunk = self._serial.read(256)
                if not chunk:
                    continue
                if not isinstance(chunk, bytes):
                    raise TypeError(
                        f"UART read must return bytes, got {type(chunk).__name__}."
                    )
                for payload in framer.feed(chunk):
                    frame = ReceivedUartFrame(
                        sequence=sequence,
                        received_timestamp_ns=self._monotonic_ns(),
                        payload=payload,
                    )
                    try:
                        self._received.put_nowait(frame)
                    except queue.Full as exc:
                        raise UartReceiveOverflowError(
                            "UART receive queue is full at capacity "
                            f"{self.receive_queue_capacity}; consumer is too slow."
                        ) from exc
                    sequence += 1
                    self.received_frames += 1
                self.discarded_frames = framer.discarded_frames
        except BaseException as exc:
            if not self._stop_event.is_set():
                self._reader_error = exc
                self._stop_event.set()

    def _raise_reader_error(self) -> None:
        if self._reader_error is None:
            return
        if isinstance(self._reader_error, UartError):
            raise self._reader_error
        raise UartError(
            f"UART reader failed for {self.device!r}: "
            f"{type(self._reader_error).__name__}: {self._reader_error}"
        ) from self._reader_error

    def _require_healthy(self) -> _SerialPort:
        serial_port = self._require_open()
        self._raise_reader_error()
        return serial_port

    def _require_open(self) -> _SerialPort:
        if not self._started or self._serial is None:
            raise RuntimeError("UART channel is not started.")
        return self._serial

    def __enter__(self) -> UartFrameChannel:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.stop()
        except BaseException as cleanup_error:
            if isinstance(exc, BaseException):
                add_exception_note(
                    exc,
                    f"UartFrameChannel cleanup also failed: {cleanup_error!r}",
                )
                return
            raise
