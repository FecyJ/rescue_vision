"""协议无关的 UART 行分帧、收发与生命周期管理。"""

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


class UartLineTooLongError(UartError):
    """收到超过配置上限且尚未完成的 UART 行。"""


@runtime_checkable
class _SerialPort(Protocol):
    def read(self, size: int = 1) -> bytes: ...

    def write(self, data: bytes) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReceivedUartLine:
    """一条完整 UART 行；时间为树莓派接收完成时的单调时钟 ns。"""

    sequence: int
    received_timestamp_ns: int
    payload: bytes

    def __post_init__(self) -> None:
        for name, value in (
            ("sequence", self.sequence),
            ("received_timestamp_ns", self.received_timestamp_ns),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}."
                )
        if not isinstance(self.payload, bytes):
            raise TypeError(
                f"payload must be bytes, got {type(self.payload).__name__}."
            )

    def decode(self, encoding: str = "ascii") -> str:
        """按调用方指定编码严格解码，协议层自行处理失败。"""

        return self.payload.decode(encoding, errors="strict")


class UartLineFramer:
    """把任意分块的字节流拆成 LF 或 CRLF 结尾的原始行。"""

    def __init__(self, *, max_line_bytes: int) -> None:
        if (
            isinstance(max_line_bytes, bool)
            or not isinstance(max_line_bytes, int)
            or max_line_bytes <= 0
        ):
            raise ValueError(
                "max_line_bytes must be a positive integer, "
                f"got {max_line_bytes!r}."
            )
        self.max_line_bytes = max_line_bytes
        self._buffer = bytearray()

    @property
    def pending_bytes(self) -> bytes:
        return bytes(self._buffer)

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        """接收一个非空字节块，返回其中所有新完成的行。"""

        if not isinstance(chunk, bytes) or not chunk:
            raise ValueError("chunk must be non-empty bytes.")
        self._buffer.extend(chunk)
        lines: list[bytes] = []
        while True:
            newline_index = self._buffer.find(b"\n")
            if newline_index < 0:
                break
            raw_line = bytes(self._buffer[:newline_index])
            del self._buffer[: newline_index + 1]
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            if len(raw_line) > self.max_line_bytes:
                raise UartLineTooLongError(
                    f"UART line has {len(raw_line)} bytes; "
                    f"maximum is {self.max_line_bytes}."
                )
            lines.append(raw_line)

        pending_payload_length = len(self._buffer)
        if self._buffer.endswith(b"\r"):
            pending_payload_length -= 1
        if pending_payload_length > self.max_line_bytes:
            raise UartLineTooLongError(
                "Incomplete UART line exceeds maximum "
                f"{self.max_line_bytes} bytes."
            )
        return tuple(lines)


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
    )


class UartLineChannel:
    """单后台读线程、有界接收队列的 8N1 UART 行通道。"""

    def __init__(
        self,
        *,
        device: str,
        baudrate: int,
        read_timeout_s: float,
        write_timeout_s: float,
        receive_queue_capacity: int,
        max_line_bytes: int,
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

        self.device = device.strip()
        self.baudrate = baudrate
        self.read_timeout_s = float(read_timeout_s)
        self.write_timeout_s = float(write_timeout_s)
        self.receive_queue_capacity = receive_queue_capacity
        self.max_line_bytes = max_line_bytes
        self._monotonic_ns = monotonic_ns
        self._serial_factory = serial_factory
        self._serial: _SerialPort | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._received = queue.Queue[ReceivedUartLine](
            maxsize=receive_queue_capacity
        )
        self._reader_error: BaseException | None = None
        self._started = False
        self.received_lines = 0
        self.sent_bytes = 0

        # Validate the framing limit before opening hardware.
        UartLineFramer(max_line_bytes=max_line_bytes)

    @property
    def started(self) -> bool:
        return self._started

    def check_health(self) -> None:
        """确认通道已启动且后台读取没有失败。"""

        self._require_healthy()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("UART channel is already started.")
        self._stop_event.clear()
        self._reader_error = None
        self._received = queue.Queue(maxsize=self.receive_queue_capacity)
        self.received_lines = 0
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
        serial_port = self._require_healthy()
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
                raise UartError(f"UART write failed for {self.device!r}.") from exc

    def send_line(self, payload: bytes) -> None:
        """发送不含 CR/LF 的协议负载，并统一添加 CRLF。"""

        if not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be non-empty bytes.")
        if b"\r" in payload or b"\n" in payload:
            raise ValueError("UART line payload must not contain CR or LF.")
        if len(payload) > self.max_line_bytes:
            raise ValueError(
                f"UART line has {len(payload)} bytes; "
                f"maximum is {self.max_line_bytes}."
            )
        self.send(payload + b"\r\n")

    def receive_line(self, timeout: float | None = None) -> ReceivedUartLine:
        """等待一条完整行；超时抛出 ``TimeoutError``。"""

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
                line = self._received.get_nowait()
            except queue.Empty as exc:
                self._raise_reader_error()
                raise TimeoutError(
                    f"Timed out waiting for UART line from {self.device!r}."
                ) from exc
            self._raise_reader_error()
            return line

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
                line = self._received.get(
                    timeout=wait_s,
                )
            except queue.Empty as exc:
                last_empty = exc
                continue
            self._raise_reader_error()
            return line

        self._raise_reader_error()
        error = TimeoutError(
            f"Timed out waiting for UART line from {self.device!r}."
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
                # Optional wake-up only; the configured read timeout remains
                # the portable shutdown path.
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
        framer = UartLineFramer(max_line_bytes=self.max_line_bytes)
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
                    line = ReceivedUartLine(
                        sequence=sequence,
                        received_timestamp_ns=self._monotonic_ns(),
                        payload=payload,
                    )
                    try:
                        self._received.put_nowait(line)
                    except queue.Full as exc:
                        raise UartReceiveOverflowError(
                            "UART receive queue is full at capacity "
                            f"{self.receive_queue_capacity}; consumer is too slow."
                        ) from exc
                    sequence += 1
                    self.received_lines += 1
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
            f"UART reader failed for {self.device!r}."
        ) from self._reader_error

    def _require_healthy(self) -> _SerialPort:
        if not self._started or self._serial is None:
            raise RuntimeError("UART channel is not started.")
        self._raise_reader_error()
        return self._serial

    def __enter__(self) -> UartLineChannel:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.stop()
        except BaseException as cleanup_error:
            if isinstance(exc, BaseException):
                add_exception_note(
                    exc,
                    f"UartLineChannel cleanup also failed: {cleanup_error!r}",
                )
                return
            raise
