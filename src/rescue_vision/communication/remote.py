"""带长度分帧、严格序号和有界队列的远程 TCP 消息通道。"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable


PROTOCOL_SCHEMA_VERSION = 2
_FRAME_MAGIC = b"RVM2"
_FRAME_PREFIX = struct.Struct("!4sII")

RemoteAttributeValue: TypeAlias = str | int | float | bool | None


class RemoteError(RuntimeError):
    """远程连接无法继续可靠工作。"""


class RemoteProtocolError(RemoteError):
    """远端发送了损坏、越界或不受支持的帧。"""


class RemotePolicyError(RemoteError):
    """远端违反当前 observe-only/debug-control 权限。"""


class RemoteQueueOverflowError(RemoteError):
    """可靠控制队列已满，不能静默丢弃消息。"""


class RemoteDisconnectedError(RemoteError):
    """TCP 对端已关闭连接。"""


class RemoteStream(str, Enum):
    CONTROL = "control"
    OBSERVATION = "observation"


class RemoteAccessMode(str, Enum):
    """比赛仅允许观察；远程控制只在明确调试配置下启用。"""

    OBSERVE_ONLY = "observe_only"
    DEBUG_CONTROL = "debug_control"


class RemoteRole(str, Enum):
    SERVER = "server"
    CLIENT = "client"


@runtime_checkable
class _ConnectedSocket(Protocol):
    def recv(self, size: int) -> bytes: ...

    def sendall(self, data: bytes) -> None: ...

    def settimeout(self, timeout: float | None) -> None: ...

    def shutdown(self, how: int) -> None: ...

    def close(self) -> None: ...


def _positive_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{location} must be a positive integer, got {value!r}."
        )
    return value


def _non_negative_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{location} must be a non-negative integer, got {value!r}."
        )
    return value


def _finite_positive_float(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < float(value) < float("inf")
    ):
        raise ValueError(f"{location} must be finite and > 0, got {value!r}.")
    return float(value)


def _bounded_text(value: object, location: str, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string.")
    stripped = value.strip()
    if len(stripped.encode("utf-8")) > maximum_bytes:
        raise ValueError(
            f"{location} must be at most {maximum_bytes} UTF-8 bytes."
        )
    return stripped


def _validate_attributes(
    attributes: Mapping[str, RemoteAttributeValue] | None,
) -> dict[str, RemoteAttributeValue]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping):
        raise ValueError("attributes must be a mapping.")
    validated: dict[str, RemoteAttributeValue] = {}
    for key, value in attributes.items():
        normalized_key = _bounded_text(key, "attribute key", 64)
        if isinstance(value, float) and not (
            -float("inf") < value < float("inf")
        ):
            raise ValueError(
                f"attributes[{normalized_key!r}] must be finite."
            )
        if value is not None and not isinstance(
            value,
            (str, int, float, bool),
        ):
            raise ValueError(
                f"attributes[{normalized_key!r}] has unsupported value "
                f"{value!r}."
            )
        validated[normalized_key] = value
    return validated


@dataclass(frozen=True, slots=True)
class ReceivedRemoteMessage:
    """一个通过结构校验和序号校验的远程消息。"""

    stream: RemoteStream
    topic: str
    content_type: str
    sequence: int
    sender_timestamp_ns: int
    received_timestamp_ns: int
    attributes: Mapping[str, RemoteAttributeValue]
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.stream, RemoteStream):
            raise ValueError(f"stream must be RemoteStream, got {self.stream!r}.")
        object.__setattr__(self, "topic", _bounded_text(self.topic, "topic", 128))
        object.__setattr__(
            self,
            "content_type",
            _bounded_text(self.content_type, "content_type", 128),
        )
        _non_negative_int(self.sequence, "sequence")
        _non_negative_int(self.sender_timestamp_ns, "sender_timestamp_ns")
        _non_negative_int(self.received_timestamp_ns, "received_timestamp_ns")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(_validate_attributes(self.attributes)),
        )
        if not isinstance(self.payload, bytes):
            raise TypeError(
                f"payload must be bytes, got {type(self.payload).__name__}."
            )


@dataclass(frozen=True, slots=True)
class _DecodedRemoteMessage:
    stream: RemoteStream
    topic: str
    content_type: str
    sequence: int
    sender_timestamp_ns: int
    attributes: dict[str, RemoteAttributeValue]
    payload: bytes


class RemoteMessageCodec:
    """增量编码/解码带长度前缀的 TCP 二进制消息帧。"""

    def __init__(
        self,
        *,
        max_header_bytes: int,
        max_payload_bytes: int,
    ) -> None:
        self.max_header_bytes = _positive_int(
            max_header_bytes,
            "max_header_bytes",
        )
        self.max_payload_bytes = _positive_int(
            max_payload_bytes,
            "max_payload_bytes",
        )
        self._buffer = bytearray()

    def encode(
        self,
        *,
        stream: RemoteStream,
        topic: str,
        content_type: str,
        sequence: int,
        sender_timestamp_ns: int,
        attributes: Mapping[str, RemoteAttributeValue] | None,
        payload: bytes,
    ) -> bytes:
        if not isinstance(stream, RemoteStream):
            raise ValueError(f"stream must be RemoteStream, got {stream!r}.")
        topic = _bounded_text(topic, "topic", 128)
        content_type = _bounded_text(content_type, "content_type", 128)
        _non_negative_int(sequence, "sequence")
        _non_negative_int(sender_timestamp_ns, "sender_timestamp_ns")
        attributes_document = _validate_attributes(attributes)
        if not isinstance(payload, bytes):
            raise TypeError(
                f"payload must be bytes, got {type(payload).__name__}."
            )
        if len(payload) > self.max_payload_bytes:
            raise ValueError(
                f"payload has {len(payload)} bytes; maximum is "
                f"{self.max_payload_bytes}."
            )

        header_document = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "stream": stream.value,
            "topic": topic,
            "content_type": content_type,
            "sequence": sequence,
            "sender_timestamp_ns": sender_timestamp_ns,
            "attributes": attributes_document,
        }
        header = json.dumps(
            header_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if len(header) > self.max_header_bytes:
            raise ValueError(
                f"header has {len(header)} bytes; maximum is "
                f"{self.max_header_bytes}."
            )
        prefix = _FRAME_PREFIX.pack(_FRAME_MAGIC, len(header), len(payload))
        return prefix + header + payload

    def feed(self, chunk: bytes) -> tuple[_DecodedRemoteMessage, ...]:
        if not isinstance(chunk, bytes) or not chunk:
            raise ValueError("chunk must be non-empty bytes.")
        self._buffer.extend(chunk)
        decoded: list[_DecodedRemoteMessage] = []
        while len(self._buffer) >= _FRAME_PREFIX.size:
            magic, header_length, payload_length = _FRAME_PREFIX.unpack(
                self._buffer[: _FRAME_PREFIX.size]
            )
            if magic != _FRAME_MAGIC:
                raise RemoteProtocolError(
                    f"Invalid remote frame magic {magic!r}."
                )
            if header_length > self.max_header_bytes:
                raise RemoteProtocolError(
                    f"Remote header length {header_length} exceeds "
                    f"{self.max_header_bytes}."
                )
            if payload_length > self.max_payload_bytes:
                raise RemoteProtocolError(
                    f"Remote payload length {payload_length} exceeds "
                    f"{self.max_payload_bytes}."
                )
            total_length = (
                _FRAME_PREFIX.size
                + header_length
                + payload_length
            )
            if len(self._buffer) < total_length:
                break
            frame = bytes(self._buffer[:total_length])
            del self._buffer[:total_length]
            header_start = _FRAME_PREFIX.size
            header_end = header_start + header_length
            header_payload = frame[header_start:header_end]
            payload = frame[header_end : header_end + payload_length]
            decoded.append(self._decode_header(header_payload, payload))
        return tuple(decoded)

    @staticmethod
    def _decode_header(
        header_payload: bytes,
        payload: bytes,
    ) -> _DecodedRemoteMessage:
        try:
            header = json.loads(header_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteProtocolError(
                "Remote frame header must be valid UTF-8 JSON."
            ) from exc
        expected_keys = {
            "schema_version",
            "stream",
            "topic",
            "content_type",
            "sequence",
            "sender_timestamp_ns",
            "attributes",
        }
        if not isinstance(header, dict) or set(header) != expected_keys:
            raise RemoteProtocolError(
                f"Remote frame header keys must be exactly "
                f"{sorted(expected_keys)}."
            )
        if (
            isinstance(header["schema_version"], bool)
            or not isinstance(header["schema_version"], int)
            or header["schema_version"] != PROTOCOL_SCHEMA_VERSION
        ):
            raise RemoteProtocolError(
                "Unsupported remote protocol schema_version "
                f"{header['schema_version']!r}."
            )
        try:
            stream = RemoteStream(header["stream"])
            topic = _bounded_text(header["topic"], "topic", 128)
            content_type = _bounded_text(
                header["content_type"],
                "content_type",
                128,
            )
            sequence = _non_negative_int(header["sequence"], "sequence")
            sender_timestamp_ns = _non_negative_int(
                header["sender_timestamp_ns"],
                "sender_timestamp_ns",
            )
            attributes = _validate_attributes(header["attributes"])
        except (TypeError, ValueError) as exc:
            raise RemoteProtocolError(
                f"Invalid remote frame header: {exc}"
            ) from exc
        return _DecodedRemoteMessage(
            stream=stream,
            topic=topic,
            content_type=content_type,
            sequence=sequence,
            sender_timestamp_ns=sender_timestamp_ns,
            attributes=attributes,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class _OutboundRemoteMessage:
    stream: RemoteStream
    topic: str
    content_type: str
    sender_timestamp_ns: int
    attributes: dict[str, RemoteAttributeValue]
    payload: bytes


class _LatestByTopicQueue:
    """有界按 topic 合并队列；同 topic 只保留最新消息。"""

    def __init__(self, *, capacity: int) -> None:
        self.capacity = _positive_int(capacity, "capacity")
        self._items: OrderedDict[str, object] = OrderedDict()
        self._condition = threading.Condition()

    def put_latest(self, topic: str, item: object) -> int:
        """写入最新值，返回因此丢弃的旧消息数。"""

        dropped = 0
        with self._condition:
            if topic in self._items:
                self._items[topic] = item
                dropped = 1
            else:
                if len(self._items) >= self.capacity:
                    self._items.popitem(last=False)
                    dropped = 1
                self._items[topic] = item
            self._condition.notify()
        return dropped

    def get_nowait(self) -> object:
        with self._condition:
            if not self._items:
                raise queue.Empty
            _topic, item = self._items.popitem(last=False)
            return item

    def get(self, timeout: float | None = None) -> object:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)
            _topic, item = self._items.popitem(last=False)
            return item


class RemoteMessageConnection:
    """一个 TCP 连接上的双向控制/观察消息通道。"""

    def __init__(
        self,
        connection: _ConnectedSocket,
        *,
        io_timeout_s: float,
        control_queue_capacity: int,
        observation_queue_capacity: int,
        max_header_bytes: int,
        max_payload_bytes: int,
        allow_inbound_control: bool,
        allow_outbound_control: bool,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(connection, _ConnectedSocket):
            raise TypeError(
                "connection must provide recv, sendall, settimeout, "
                "shutdown and close."
            )
        self.io_timeout_s = _finite_positive_float(
            io_timeout_s,
            "io_timeout_s",
        )
        self.control_queue_capacity = _positive_int(
            control_queue_capacity,
            "control_queue_capacity",
        )
        self.observation_queue_capacity = _positive_int(
            observation_queue_capacity,
            "observation_queue_capacity",
        )
        if not isinstance(allow_inbound_control, bool):
            raise ValueError("allow_inbound_control must be a boolean.")
        if not isinstance(allow_outbound_control, bool):
            raise ValueError("allow_outbound_control must be a boolean.")
        self.allow_inbound_control = allow_inbound_control
        self.allow_outbound_control = allow_outbound_control
        self._socket = connection
        self._socket.settimeout(self.io_timeout_s)
        self._encoder = RemoteMessageCodec(
            max_header_bytes=max_header_bytes,
            max_payload_bytes=max_payload_bytes,
        )
        self._decoder = RemoteMessageCodec(
            max_header_bytes=max_header_bytes,
            max_payload_bytes=max_payload_bytes,
        )
        self.max_payload_bytes = max_payload_bytes
        self._monotonic_ns = monotonic_ns
        self._outbound_reliable = queue.Queue[_OutboundRemoteMessage](
            maxsize=self.control_queue_capacity
        )
        self._outbound_observation = _LatestByTopicQueue(
            capacity=self.observation_queue_capacity
        )
        self._inbound_control = queue.Queue[ReceivedRemoteMessage](
            maxsize=self.control_queue_capacity
        )
        self._inbound_observation = _LatestByTopicQueue(
            capacity=self.observation_queue_capacity
        )
        self._stop_event = threading.Event()
        self._error_lock = threading.Lock()
        self._worker_error: BaseException | None = None
        self._reader_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self.sent_messages = 0
        self.received_messages = 0
        self.dropped_outbound_observations = 0
        self.dropped_inbound_observations = 0

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Remote connection is already started.")
        if self._closed:
            raise RuntimeError("Remote connection is already closed.")
        self._stop_event.clear()
        self._worker_error = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="rescue-remote-reader",
            daemon=True,
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="rescue-remote-writer",
            daemon=True,
        )
        self._started = True
        try:
            self._reader_thread.start()
            self._writer_thread.start()
        except BaseException:
            self._started = False
            self._stop_event.set()
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            if (
                self._reader_thread is not None
                and self._reader_thread.is_alive()
            ):
                self._reader_thread.join(timeout=self.io_timeout_s + 1.0)
            self._closed = True
            raise

    def check_health(self) -> None:
        self._require_healthy()

    def send_control(
        self,
        topic: str,
        payload: bytes,
        *,
        content_type: str = "application/json",
        attributes: Mapping[str, RemoteAttributeValue] | None = None,
        sender_timestamp_ns: int | None = None,
    ) -> None:
        """非阻塞提交可靠控制；队列满时显式失败。"""

        if not self.allow_outbound_control:
            raise RemotePolicyError(
                "Outbound control is disabled by remote access mode."
            )
        message = self._make_outbound(
            RemoteStream.CONTROL,
            topic,
            payload,
            content_type,
            attributes,
            sender_timestamp_ns,
        )
        try:
            self._outbound_reliable.put_nowait(message)
        except queue.Full as exc:
            raise RemoteQueueOverflowError(
                "Outbound remote control queue is full; command was not sent."
            ) from exc

    def send_observation(
        self,
        topic: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
        attributes: Mapping[str, RemoteAttributeValue] | None = None,
        sender_timestamp_ns: int | None = None,
    ) -> None:
        """非阻塞提交观察数据；队列满时丢旧保新。"""

        message = self._make_outbound(
            RemoteStream.OBSERVATION,
            topic,
            payload,
            content_type,
            attributes,
            sender_timestamp_ns,
        )
        self.dropped_outbound_observations += (
            self._outbound_observation.put_latest(message.topic, message)
        )

    def send_reliable_observation(
        self,
        topic: str,
        payload: bytes,
        *,
        content_type: str = "application/json",
        attributes: Mapping[str, RemoteAttributeValue] | None = None,
        sender_timestamp_ns: int | None = None,
    ) -> None:
        """提交不可被视频等最新值观察覆盖的可靠观察消息。"""

        message = self._make_outbound(
            RemoteStream.OBSERVATION,
            topic,
            payload,
            content_type,
            attributes,
            sender_timestamp_ns,
        )
        try:
            self._outbound_reliable.put_nowait(message)
        except queue.Full as exc:
            raise RemoteQueueOverflowError(
                "Outbound reliable remote queue is full; observation was "
                "not sent."
            ) from exc

    def receive_control(
        self,
        timeout: float | None = None,
    ) -> ReceivedRemoteMessage:
        return self._receive_from(self._inbound_control, "control", timeout)

    def receive_observation(
        self,
        timeout: float | None = None,
    ) -> ReceivedRemoteMessage:
        return self._receive_observation(
            self._inbound_observation,
            timeout,
        )

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        cleanup_errors: list[tuple[str, BaseException]] = []
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        for name, thread in (
            ("reader_thread", self._reader_thread),
            ("writer_thread", self._writer_thread),
        ):
            if thread is None:
                continue
            thread.join(timeout=self.io_timeout_s + 1.0)
            if thread.is_alive():
                cleanup_errors.append(
                    (name, RuntimeError(f"Remote {name} did not stop."))
                )
        try:
            self._socket.close()
        except BaseException as error:
            cleanup_errors.append(("socket.close", error))
        self._reader_thread = None
        self._writer_thread = None
        self._started = False
        self._closed = True
        if cleanup_errors:
            location, error = cleanup_errors[0]
            for later_location, later_error in cleanup_errors[1:]:
                error.add_note(
                    f"{later_location} also failed: {later_error!r}"
                )
            raise RuntimeError(
                f"Remote connection cleanup failed in {location}."
            ) from error

    def _make_outbound(
        self,
        stream: RemoteStream,
        topic: str,
        payload: bytes,
        content_type: str,
        attributes: Mapping[str, RemoteAttributeValue] | None,
        sender_timestamp_ns: int | None,
    ) -> _OutboundRemoteMessage:
        self._require_healthy()
        topic = _bounded_text(topic, "topic", 128)
        content_type = _bounded_text(content_type, "content_type", 128)
        if not isinstance(payload, bytes):
            raise TypeError(
                f"payload must be bytes, got {type(payload).__name__}."
            )
        if len(payload) > self.max_payload_bytes:
            raise ValueError(
                f"payload has {len(payload)} bytes; maximum is "
                f"{self.max_payload_bytes}."
            )
        timestamp_ns = (
            self._monotonic_ns()
            if sender_timestamp_ns is None
            else _non_negative_int(sender_timestamp_ns, "sender_timestamp_ns")
        )
        return _OutboundRemoteMessage(
            stream=stream,
            topic=topic,
            content_type=content_type,
            sender_timestamp_ns=timestamp_ns,
            attributes=_validate_attributes(attributes),
            payload=payload,
        )

    def _reader_loop(self) -> None:
        expected_sequence = 0
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = self._socket.recv(65_536)
                except socket.timeout:
                    continue
                if not chunk:
                    raise RemoteDisconnectedError("Remote peer disconnected.")
                for decoded in self._decoder.feed(chunk):
                    if decoded.sequence != expected_sequence:
                        raise RemoteProtocolError(
                            f"Expected remote sequence {expected_sequence}, "
                            f"got {decoded.sequence}."
                        )
                    expected_sequence += 1
                    received = ReceivedRemoteMessage(
                        stream=decoded.stream,
                        topic=decoded.topic,
                        content_type=decoded.content_type,
                        sequence=decoded.sequence,
                        sender_timestamp_ns=decoded.sender_timestamp_ns,
                        received_timestamp_ns=self._monotonic_ns(),
                        attributes=decoded.attributes,
                        payload=decoded.payload,
                    )
                    self._route_inbound(received)
                    self.received_messages += 1
        except BaseException as error:
            if not self._stop_event.is_set():
                self._set_worker_error(error)

    def _writer_loop(self) -> None:
        sequence = 0
        try:
            while not self._stop_event.is_set():
                try:
                    message = self._outbound_reliable.get_nowait()
                except queue.Empty:
                    try:
                        queued = self._outbound_observation.get(timeout=0.05)
                        assert isinstance(queued, _OutboundRemoteMessage)
                        message = queued
                    except queue.Empty:
                        continue
                frame = self._encoder.encode(
                    stream=message.stream,
                    topic=message.topic,
                    content_type=message.content_type,
                    sequence=sequence,
                    sender_timestamp_ns=message.sender_timestamp_ns,
                    attributes=message.attributes,
                    payload=message.payload,
                )
                self._socket.sendall(frame)
                sequence += 1
                self.sent_messages += 1
        except BaseException as error:
            if not self._stop_event.is_set():
                self._set_worker_error(error)

    def _route_inbound(self, message: ReceivedRemoteMessage) -> None:
        if message.stream is RemoteStream.CONTROL:
            if not self.allow_inbound_control:
                raise RemotePolicyError(
                    "Inbound control is disabled by remote access mode."
                )
            try:
                self._inbound_control.put_nowait(message)
            except queue.Full as exc:
                raise RemoteQueueOverflowError(
                    "Inbound remote control queue is full."
                ) from exc
            return
        self.dropped_inbound_observations += (
            self._inbound_observation.put_latest(message.topic, message)
        )

    def _receive_observation(
        self,
        target_queue: _LatestByTopicQueue,
        timeout: float | None,
    ) -> ReceivedRemoteMessage:
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
                item = target_queue.get_nowait()
            except queue.Empty as exc:
                self._raise_worker_error()
                raise TimeoutError(
                    "No remote observation message is available."
                ) from exc
            self._raise_worker_error()
            assert isinstance(item, ReceivedRemoteMessage)
            return item

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            self._raise_worker_error()
            if deadline is None:
                wait_s = 0.05
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for remote observation message."
                    )
                wait_s = min(remaining, 0.05)
            try:
                item = target_queue.get(timeout=wait_s)
            except queue.Empty:
                continue
            self._raise_worker_error()
            assert isinstance(item, ReceivedRemoteMessage)
            return item

    def _receive_from(
        self,
        target_queue: queue.Queue[ReceivedRemoteMessage],
        stream_name: str,
        timeout: float | None,
    ) -> ReceivedRemoteMessage:
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
                message = target_queue.get_nowait()
            except queue.Empty as exc:
                self._raise_worker_error()
                raise TimeoutError(
                    f"No remote {stream_name} message is available."
                ) from exc
            self._raise_worker_error()
            return message

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            self._raise_worker_error()
            if deadline is None:
                wait_s = 0.05
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for remote {stream_name} message."
                    )
                wait_s = min(remaining, 0.05)
            try:
                message = target_queue.get(timeout=wait_s)
            except queue.Empty:
                continue
            self._raise_worker_error()
            return message

    def _set_worker_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._worker_error is None:
                self._worker_error = error
        self._stop_event.set()

    def _raise_worker_error(self) -> None:
        if self._worker_error is None:
            return
        if isinstance(self._worker_error, RemoteError):
            raise self._worker_error
        raise RemoteError("Remote connection worker failed.") from self._worker_error

    def _require_healthy(self) -> None:
        if not self._started:
            raise RuntimeError("Remote connection is not started.")
        self._raise_worker_error()

    def __enter__(self) -> RemoteMessageConnection:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.stop()
        except BaseException as cleanup_error:
            if isinstance(exc, BaseException):
                exc.add_note(
                    "RemoteMessageConnection cleanup also failed: "
                    f"{cleanup_error!r}"
                )
                return
            raise


@dataclass(frozen=True, slots=True)
class RemoteConnectionOptions:
    io_timeout_s: float
    control_queue_capacity: int
    observation_queue_capacity: int
    max_header_bytes: int
    max_payload_bytes: int

    def __post_init__(self) -> None:
        _finite_positive_float(self.io_timeout_s, "io_timeout_s")
        _positive_int(self.control_queue_capacity, "control_queue_capacity")
        _positive_int(
            self.observation_queue_capacity,
            "observation_queue_capacity",
        )
        _positive_int(self.max_header_bytes, "max_header_bytes")
        _positive_int(self.max_payload_bytes, "max_payload_bytes")


class RemoteTcpServer:
    """树莓派侧单监听端点；每次 accept 返回一个消息连接。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        access_mode: RemoteAccessMode,
        connection_options: RemoteConnectionOptions,
    ) -> None:
        self.host = _bounded_text(host, "host", 255)
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65_535
        ):
            raise ValueError(f"port must be in [0, 65535], got {port!r}.")
        if not isinstance(access_mode, RemoteAccessMode):
            raise ValueError("access_mode must be RemoteAccessMode.")
        self.port = port
        self.access_mode = access_mode
        self.connection_options = connection_options
        self._listener: socket.socket | None = None

    @property
    def started(self) -> bool:
        return self._listener is not None

    @property
    def bound_port(self) -> int:
        if self._listener is None:
            raise RuntimeError("Remote TCP server is not started.")
        return int(self._listener.getsockname()[1])

    def start(self) -> None:
        if self._listener is not None:
            raise RuntimeError("Remote TCP server is already started.")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(1)
            listener.settimeout(0.1)
        except BaseException:
            listener.close()
            raise
        self._listener = listener

    def accept(self, timeout: float | None = None) -> RemoteMessageConnection:
        if self._listener is None:
            raise RuntimeError("Remote TCP server is not started.")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 <= float(timeout) < float("inf")
        ):
            raise ValueError(f"timeout must be finite and >= 0, got {timeout!r}.")
        timeout_s = None if timeout is None else float(timeout)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for remote TCP client.")
            try:
                connection, _ = self._listener.accept()
                break
            except socket.timeout:
                continue
        try:
            return RemoteMessageConnection(
                connection,
                io_timeout_s=self.connection_options.io_timeout_s,
                control_queue_capacity=(
                    self.connection_options.control_queue_capacity
                ),
                observation_queue_capacity=(
                    self.connection_options.observation_queue_capacity
                ),
                max_header_bytes=self.connection_options.max_header_bytes,
                max_payload_bytes=self.connection_options.max_payload_bytes,
                allow_inbound_control=(
                    self.access_mode is RemoteAccessMode.DEBUG_CONTROL
                ),
                allow_outbound_control=False,
            )
        except BaseException:
            connection.close()
            raise

    def stop(self) -> None:
        if self._listener is None:
            return
        listener = self._listener
        self._listener = None
        listener.close()

    def __enter__(self) -> RemoteTcpServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def connect_remote_client(
    *,
    host: str,
    port: int,
    connect_timeout_s: float,
    access_mode: RemoteAccessMode,
    connection_options: RemoteConnectionOptions,
) -> RemoteMessageConnection:
    """电脑侧直接连接树莓派 TCP 服务端。"""

    host = _bounded_text(host, "host", 255)
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 1 <= port <= 65_535
    ):
        raise ValueError(f"port must be in [1, 65535], got {port!r}.")
    connect_timeout_s = _finite_positive_float(
        connect_timeout_s,
        "connect_timeout_s",
    )
    if not isinstance(access_mode, RemoteAccessMode):
        raise ValueError("access_mode must be RemoteAccessMode.")
    connection = socket.create_connection(
        (host, port),
        timeout=connect_timeout_s,
    )
    try:
        return RemoteMessageConnection(
            connection,
            io_timeout_s=connection_options.io_timeout_s,
            control_queue_capacity=connection_options.control_queue_capacity,
            observation_queue_capacity=(
                connection_options.observation_queue_capacity
            ),
            max_header_bytes=connection_options.max_header_bytes,
            max_payload_bytes=connection_options.max_payload_bytes,
            allow_inbound_control=False,
            allow_outbound_control=(
                access_mode is RemoteAccessMode.DEBUG_CONTROL
            ),
        )
    except BaseException:
        connection.close()
        raise
