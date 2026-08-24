from __future__ import annotations

import queue
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

import rescue_vision.communication.uart as uart_module

from rescue_vision.communication import (
    CobsDecodeError,
    ReceivedUartFrame,
    UartError,
    UartFrameChannel,
    UartFrameFramer,
    UartReceiveOverflowError,
    cobs_decode,
    cobs_encode,
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
        del size
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
    max_frame_bytes: int = 64,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> UartFrameChannel:
    return UartFrameChannel(
        device="/dev/test-uart",
        baudrate=115200,
        read_timeout_s=0.05,
        write_timeout_s=0.05,
        receive_queue_capacity=receive_queue_capacity,
        max_frame_bytes=max_frame_bytes,
        monotonic_ns=monotonic_ns,
        serial_factory=lambda **kwargs: fake,
    )


def wait_for_failure(
    channel: UartFrameChannel,
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


@pytest.mark.parametrize(
    ("payload", "encoded"),
    [
        (b"", b"\x01"),
        (b"\x00", b"\x01\x01"),
        (b"\x11\x22\x00\x33", b"\x03\x11\x22\x02\x33"),
        (bytes(range(1, 255)), b"\xff" + bytes(range(1, 255))),
    ],
)
def test_cobs_known_vectors_round_trip(payload: bytes, encoded: bytes) -> None:
    assert cobs_encode(payload) == encoded
    assert cobs_decode(encoded) == payload


@pytest.mark.parametrize("encoded", [b"", b"\x00", b"\x03\x11"])
def test_cobs_decode_rejects_malformed_data(encoded: bytes) -> None:
    with pytest.raises(CobsDecodeError):
        cobs_decode(encoded)


def test_frame_framer_handles_fragmentation_and_recovers_after_damage() -> None:
    framer = UartFrameFramer(max_frame_bytes=8)
    first = cobs_encode(b"\x80\x01\x00") + b"\x00"
    second = cobs_encode(b"\x82\x02") + b"\x00"

    assert framer.feed(first[:2]) == ()
    assert framer.feed(first[2:] + b"\x05\x11\x00" + second) == (
        b"\x80\x01\x00",
        b"\x82\x02",
    )
    assert framer.pending_bytes == b""
    assert framer.discarded_frames == 1


def test_frame_framer_discards_overlong_input_until_next_delimiter() -> None:
    framer = UartFrameFramer(max_frame_bytes=4)
    valid = cobs_encode(b"okay") + b"\x00"

    assert framer.feed(b"\x01" * 20 + b"\x00" + valid) == (b"okay",)
    assert framer.discarded_frames == 1


def test_received_frame_validates_identity_and_payload() -> None:
    frame = ReceivedUartFrame(0, 123, b"\x80")
    assert frame.payload == b"\x80"
    with pytest.raises(ValueError, match="sequence"):
        ReceivedUartFrame(-1, 123, b"bad")
    with pytest.raises(ValueError, match="non-empty"):
        ReceivedUartFrame(0, 123, b"")


def test_channel_receives_cobs_frames_with_monotonic_identity() -> None:
    fake = FakeSerial()
    timestamps = iter([100, 200, 300])
    channel = make_channel(fake, monotonic_ns=lambda: next(timestamps))
    payloads = (b"first\x00frame", b"second", b"third")
    wire = b"".join(cobs_encode(payload) + b"\x00" for payload in payloads)

    with channel:
        fake.incoming.put(wire[:7])
        fake.incoming.put(wire[7:])
        frames = [channel.receive_frame(timeout=1.0) for _ in payloads]

    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert [frame.received_timestamp_ns for frame in frames] == [100, 200, 300]
    assert tuple(frame.payload for frame in frames) == payloads
    assert channel.received_frames == 3
    assert fake.closed


def test_channel_serializes_partial_writes_and_adds_cobs_delimiter() -> None:
    fake = FakeSerial(maximum_write_size=2)
    channel = make_channel(fake)
    payload = b"\x10\x00\x01"

    with channel:
        channel.send_frame(payload)

    assert bytes(fake.written) == cobs_encode(payload) + b"\x00"
    assert channel.sent_bytes == len(fake.written)


def test_real_serial_open_requests_exclusive_device(monkeypatch) -> None:
    fake = FakeSerial()
    captured: dict[str, object] = {}

    def serial_constructor(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setitem(
        sys.modules,
        "serial",
        SimpleNamespace(
            EIGHTBITS=8,
            PARITY_NONE="N",
            STOPBITS_ONE=1,
            Serial=serial_constructor,
        ),
    )
    result = uart_module._open_pyserial(
        device="/dev/test-uart",
        baudrate=115200,
        read_timeout_s=0.05,
        write_timeout_s=0.05,
    )

    assert result is fake
    assert captured == {
        "port": "/dev/test-uart",
        "baudrate": 115200,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout": 0.05,
        "write_timeout": 0.05,
        "exclusive": True,
    }


@pytest.mark.parametrize("payload", [b"", b"x" * 65])
def test_send_frame_rejects_invalid_payload(payload: bytes) -> None:
    channel = make_channel(FakeSerial())
    with channel, pytest.raises(ValueError):
        channel.send_frame(payload)


def test_invalid_cobs_frame_is_discarded_without_channel_failure() -> None:
    fake = FakeSerial()
    channel = make_channel(fake)
    valid = cobs_encode(b"ready") + b"\x00"

    with channel:
        fake.incoming.put(b"\x04\x11\x00" + valid)
        assert channel.receive_frame(timeout=1.0).payload == b"ready"
        channel.check_health()

    assert channel.discarded_frames == 1


def test_receive_queue_overflow_is_a_visible_channel_failure() -> None:
    fake = FakeSerial()
    channel = make_channel(fake, receive_queue_capacity=1)
    wire = cobs_encode(b"one") + b"\x00" + cobs_encode(b"two") + b"\x00"

    with channel:
        fake.incoming.put(wire)
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


def test_reader_failure_still_allows_final_safety_write() -> None:
    fake = FakeSerial()
    channel = make_channel(fake)
    safety_frame = b"\x11\x00\x00\xcf\xb8"

    with channel:
        fake.incoming.put(OSError("receive failed"))
        wait_for_failure(channel, UartError)
        channel.send_frame(safety_frame)

    assert bytes(fake.written) == cobs_encode(safety_frame) + b"\x00"


def test_indefinite_receive_is_woken_by_reader_failure() -> None:
    fake = FakeSerial()
    channel = make_channel(fake)

    with channel:
        fake.incoming.put(OSError("USB disconnected"))
        with pytest.raises(UartError) as raised:
            channel.receive_frame(timeout=None)

    assert isinstance(raised.value.__cause__, OSError)


def test_receive_timeout_and_lifecycle_errors_are_explicit() -> None:
    fake = FakeSerial()
    channel = make_channel(fake)

    with pytest.raises(RuntimeError, match="not started"):
        channel.send(b"x")
    with channel:
        with pytest.raises(TimeoutError, match="Timed out"):
            channel.receive_frame(timeout=0.01)
        fake.incoming.put(cobs_encode(b"ready") + b"\x00")
        assert channel.receive_frame(timeout=1.0).payload == b"ready"
        with pytest.raises(TimeoutError, match="Timed out"):
            channel.receive_frame(timeout=0)
    with pytest.raises(RuntimeError, match="not started"):
        channel.receive_frame(timeout=0)


def test_stop_clears_lifecycle_state_when_serial_close_fails() -> None:
    fake = FakeSerial(close_error=OSError("close failed"))
    channel = make_channel(fake)
    channel.start()

    with pytest.raises(RuntimeError, match="cleanup failed in close"):
        channel.stop()

    assert fake.closed
    assert not channel.started
