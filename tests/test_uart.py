from __future__ import annotations

import queue
import time
from collections.abc import Callable

import pytest

from rescue_vision.communication.uart import (
    ReceivedUartLine,
    UartError,
    UartLineChannel,
    UartLineFramer,
    UartLineTooLongError,
    UartReceiveOverflowError,
)


class FakeSerial:
    def __init__(
        self,
        *,
        maximum_write_size: int | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.maximum_write_size = maximum_write_size
        self.close_error = close_error
        self.incoming: queue.Queue[bytes | BaseException] = queue.Queue()
        self.written = bytearray()
        self.closed = False

    def read(self, size: int = 1) -> bytes:
        try:
            item = self.incoming.get(timeout=0.05)
        except queue.Empty:
            return b""
        if isinstance(item, BaseException):
            raise item
        return item

    def write(self, data: bytes) -> int:
        if self.closed:
            raise OSError("serial port closed")
        count = (
            len(data)
            if self.maximum_write_size is None
            else min(len(data), self.maximum_write_size)
        )
        self.written.extend(data[:count])
        return count

    def cancel_read(self) -> None:
        self.incoming.put(b"")

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def make_channel(
    fake: FakeSerial,
    *,
    receive_queue_capacity: int = 4,
    max_line_bytes: int = 32,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> UartLineChannel:
    return UartLineChannel(
        device="/dev/test-uart",
        baudrate=115200,
        read_timeout_s=0.05,
        write_timeout_s=0.05,
        receive_queue_capacity=receive_queue_capacity,
        max_line_bytes=max_line_bytes,
        monotonic_ns=monotonic_ns,
        serial_factory=lambda **kwargs: fake,
    )


def wait_for_failure(
    channel: UartLineChannel,
    expected: type[BaseException],
) -> BaseException:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            channel.check_health()
        except expected as exc:
            return exc
        time.sleep(0.001)
    pytest.fail(f"UART channel did not fail with {expected.__name__}")


def test_line_framer_handles_fragmented_and_coalesced_crlf() -> None:
    framer = UartLineFramer(max_line_bytes=16)

    assert framer.feed(b"t12") == ()
    assert framer.feed(b"3,0.1\r\nOK\r\npartial") == (
        b"t123,0.1",
        b"OK",
    )
    assert framer.pending_bytes == b"partial"
    assert framer.feed(b"\n") == (b"partial",)


def test_received_line_validates_identity_and_decodes_strictly() -> None:
    line = ReceivedUartLine(0, 123, b"imu,1,2,3")

    assert line.decode() == "imu,1,2,3"
    with pytest.raises(ValueError, match="sequence"):
        ReceivedUartLine(-1, 123, b"bad")
    with pytest.raises(UnicodeDecodeError):
        ReceivedUartLine(0, 123, b"\xff").decode()


def test_line_framer_rejects_unbounded_input() -> None:
    framer = UartLineFramer(max_line_bytes=4)

    with pytest.raises(UartLineTooLongError, match="maximum"):
        framer.feed(b"12345")


def test_channel_receives_raw_lines_with_monotonic_identity() -> None:
    fake = FakeSerial()
    timestamps = iter([100, 200, 300])
    channel = make_channel(fake, monotonic_ns=lambda: next(timestamps))

    with channel:
        fake.incoming.put(b"OK m=0.2")
        fake.incoming.put(b",0.2\r\nt123,0.1,0.1\r\nimu,1,2,3\r\n")

        first = channel.receive_line(timeout=1.0)
        second = channel.receive_line(timeout=1.0)
        third = channel.receive_line(timeout=1.0)

    assert [first.sequence, second.sequence, third.sequence] == [0, 1, 2]
    assert [first.received_timestamp_ns, second.received_timestamp_ns] == [
        100,
        200,
    ]
    assert first.payload == b"OK m=0.2,0.2"
    assert second.payload.startswith(b"t123")
    assert third.payload == b"imu,1,2,3"
    assert channel.received_lines == 3
    assert fake.closed


def test_channel_serializes_partial_writes_and_adds_crlf() -> None:
    fake = FakeSerial(maximum_write_size=2)
    channel = make_channel(fake)

    with channel:
        channel.send_line(b"m0.2,0.2")

    assert bytes(fake.written) == b"m0.2,0.2\r\n"
    assert channel.sent_bytes == len(fake.written)


@pytest.mark.parametrize("payload", [b"", b"a\nb", b"a\rb"])
def test_send_line_rejects_invalid_payload(payload: bytes) -> None:
    channel = make_channel(FakeSerial())

    with channel, pytest.raises(ValueError):
        channel.send_line(payload)


def test_receive_queue_overflow_is_a_visible_channel_failure() -> None:
    fake = FakeSerial()
    channel = make_channel(fake, receive_queue_capacity=1)

    with channel:
        fake.incoming.put(b"one\ntwo\n")
        error = wait_for_failure(channel, UartReceiveOverflowError)

    assert "consumer is too slow" in str(error)


def test_reader_error_is_reported_and_port_still_closes() -> None:
    fake = FakeSerial()
    channel = make_channel(fake)

    with channel:
        fake.incoming.put(OSError("USB disconnected"))
        error = wait_for_failure(channel, UartError)

    assert isinstance(error.__cause__, OSError)
    assert fake.closed


def test_indefinite_receive_is_woken_by_reader_failure() -> None:
    fake = FakeSerial()
    channel = make_channel(fake)

    with channel:
        fake.incoming.put(OSError("USB disconnected"))
        with pytest.raises(UartError) as raised:
            channel.receive_line(timeout=None)

    assert isinstance(raised.value.__cause__, OSError)


def test_receive_timeout_and_lifecycle_errors_are_explicit() -> None:
    fake = FakeSerial()
    channel = make_channel(fake)

    with pytest.raises(RuntimeError, match="not started"):
        channel.send(b"x")
    with channel:
        with pytest.raises(TimeoutError, match="Timed out"):
            channel.receive_line(timeout=0.01)
        fake.incoming.put(b"ready\n")
        assert channel.receive_line(timeout=1.0).payload == b"ready"
        with pytest.raises(TimeoutError, match="Timed out"):
            channel.receive_line(timeout=0)
    with pytest.raises(RuntimeError, match="not started"):
        channel.receive_line(timeout=0)


def test_stop_clears_lifecycle_state_when_serial_close_fails() -> None:
    fake = FakeSerial(close_error=OSError("close failed"))
    channel = make_channel(fake)
    channel.start()

    with pytest.raises(RuntimeError, match="cleanup failed in close"):
        channel.stop()

    assert fake.closed
    assert not channel.started
