"""带认证、完整性校验和有界队列的远程 TCP 消息通道。"""

from __future__ import annotations

import hashlib
import hmac
import json
import queue
import secrets
import socket
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable


PROTOCOL_SCHEMA_VERSION = 1
_FRAME_MAGIC = b"RVM1"
_FRAME_PREFIX = struct.Struct("!4sII")
_AUTH_TAG_BYTES = hashlib.sha256().digest_size
_CLIENT_HELLO_MAGIC = b"RVC1"
_SERVER_HELLO_MAGIC = b"RVS1"
_HANDSHAKE_PART_BYTES = 32
_HANDSHAKE_PACKET_BYTES = 4 + _HANDSHAKE_PART_BYTES * 2

RemoteAttributeValue: TypeAlias = str | int | float | bool | None


class RemoteError(RuntimeError):
    """远程连接无法继续可靠工作。"""


class RemoteAuthenticationError(RemoteError):
    """远端未能证明持有相同预共享密钥。"""


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


def load_remote_authentication_key(path: str | Path) -> bytes:
    """读取至少 32 字节的原始密钥或 64 位十六进制密钥文件。"""

    key_path = Path(path).expanduser().resolve()
    payload = key_path.read_bytes().strip()
    if len(payload) == 64:
        try:
            payload = bytes.fromhex(payload.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            pass
    if len(payload) < 32:
        raise ValueError(
            f"Remote authentication key {key_path} must contain at least "
            "32 bytes or 64 hexadecimal characters."
        )
    return payload


@dataclass(frozen=True, slots=True)
class ReceivedRemoteMessage:
    """一个经认证且完整的远程消息。"""

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
    """增量编码/解码带 HMAC-SHA256 完整性标签的二进制消息帧。"""

    def __init__(
        self,
        *,
        session_key: bytes,
        max_header_bytes: int,
        max_payload_bytes: int,
    ) -> None:
        if not isinstance(session_key, bytes) or len(session_key) < 32:
            raise ValueError("session_key must contain at least 32 bytes.")
        self._session_key = session_key
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
        authenticated = prefix + header + payload
        tag = hmac.new(
            self._session_key,
            authenticated,
            hashlib.sha256,
        ).digest()
        return authenticated + tag

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
                + _AUTH_TAG_BYTES
            )
            if len(self._buffer) < total_length:
                break
            frame = bytes(self._buffer[:total_length])
            del self._buffer[:total_length]
            authenticated = frame[:-_AUTH_TAG_BYTES]
            actual_tag = frame[-_AUTH_TAG_BYTES:]
            expected_tag = hmac.new(
                self._session_key,
                authenticated,
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(actual_tag, expected_tag):
                raise RemoteAuthenticationError(
                    "Remote frame integrity check failed."
                )
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
class _SessionKeys:
    client_to_server: bytes
    server_to_client: bytes


def _derive_session_keys(
    authentication_key: bytes,
    client_nonce: bytes,
    server_nonce: bytes,
) -> _SessionKeys:
    session_material = hmac.new(
        authentication_key,
        b"session\0" + client_nonce + server_nonce,
        hashlib.sha256,
    ).digest()
    return _SessionKeys(
        client_to_server=hmac.new(
            session_material,
            b"client-to-server",
            hashlib.sha256,
        ).digest(),
        server_to_client=hmac.new(
            session_material,
            b"server-to-client",
            hashlib.sha256,
        ).digest(),
    )


def _recv_exact(connection: _ConnectedSocket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise RemoteDisconnectedError(
                "Remote peer disconnected during authentication."
            )
        result.extend(chunk)
    return bytes(result)


def _client_handshake(
    connection: _ConnectedSocket,
    authentication_key: bytes,
) -> _SessionKeys:
    client_nonce = secrets.token_bytes(_HANDSHAKE_PART_BYTES)
    client_proof = hmac.new(
        authentication_key,
        b"client\0" + client_nonce,
        hashlib.sha256,
    ).digest()
    connection.sendall(_CLIENT_HELLO_MAGIC + client_nonce + client_proof)
    response = _recv_exact(connection, _HANDSHAKE_PACKET_BYTES)
    if response[:4] != _SERVER_HELLO_MAGIC:
        raise RemoteAuthenticationError("Invalid server authentication reply.")
    server_nonce = response[4 : 4 + _HANDSHAKE_PART_BYTES]
    actual_proof = response[4 + _HANDSHAKE_PART_BYTES :]
    expected_proof = hmac.new(
        authentication_key,
        b"server\0" + client_nonce + server_nonce,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(actual_proof, expected_proof):
        raise RemoteAuthenticationError(
            "Server did not prove possession of the authentication key."
        )
    return _derive_session_keys(authentication_key, client_nonce, server_nonce)


def _server_handshake(
    connection: _ConnectedSocket,
    authentication_key: bytes,
) -> _SessionKeys:
    request = _recv_exact(connection, _HANDSHAKE_PACKET_BYTES)
    if request[:4] != _CLIENT_HELLO_MAGIC:
        raise RemoteAuthenticationError("Invalid client authentication hello.")
    client_nonce = request[4 : 4 + _HANDSHAKE_PART_BYTES]
    actual_proof = request[4 + _HANDSHAKE_PART_BYTES :]
    expected_proof = hmac.new(
        authentication_key,
        b"client\0" + client_nonce,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(actual_proof, expected_proof):
        raise RemoteAuthenticationError(
            "Client did not prove possession of the authentication key."
        )
    server_nonce = secrets.token_bytes(_HANDSHAKE_PART_BYTES)
    server_proof = hmac.new(
        authentication_key,
        b"server\0" + client_nonce + server_nonce,
        hashlib.sha256,
    ).digest()
    connection.sendall(_SERVER_HELLO_MAGIC + server_nonce + server_proof)
    return _derive_session_keys(authentication_key, client_nonce, server_nonce)


@dataclass(frozen=True, slots=True)
class _OutboundRemoteMessage:
    stream: RemoteStream
    topic: str
    content_type: str
    sender_timestamp_ns: int
    attributes: dict[str, RemoteAttributeValue]
    payload: bytes


class RemoteMessageConnection:
    """一个认证 TCP 连接上的双向控制/观察消息通道。"""

    def __init__(
        self,
        connection: _ConnectedSocket,
        *,
        outbound_session_key: bytes,
        inbound_session_key: bytes,
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
            session_key=outbound_session_key,
            max_header_bytes=max_header_bytes,
            max_payload_bytes=max_payload_bytes,
        )
        self._decoder = RemoteMessageCodec(
            session_key=inbound_session_key,
            max_header_bytes=max_header_bytes,
            max_payload_bytes=max_payload_bytes,
        )
        self.max_payload_bytes = max_payload_bytes
        self._monotonic_ns = monotonic_ns
        self._outbound_reliable = queue.Queue[_OutboundRemoteMessage](
            maxsize=self.control_queue_capacity
        )
        self._outbound_observation = queue.Queue[_OutboundRemoteMessage](
            maxsize=self.observation_queue_capacity
        )
        self._inbound_control = queue.Queue[ReceivedRemoteMessage](
            maxsize=self.control_queue_capacity
        )
        self._inbound_observation = queue.Queue[ReceivedRemoteMessage](
            maxsize=self.observation_queue_capacity
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
        while True:
            try:
                self._outbound_observation.put_nowait(message)
                return
            except queue.Full:
                try:
                    self._outbound_observation.get_nowait()
                    self.dropped_outbound_observations += 1
                except queue.Empty:
                    continue

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
        return self._receive_from(
            self._inbound_observation,
            "observation",
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
                        message = self._outbound_observation.get(timeout=0.05)
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
        while True:
            try:
                self._inbound_observation.put_nowait(message)
                return
            except queue.Full:
                try:
                    self._inbound_observation.get_nowait()
                    self.dropped_inbound_observations += 1
                except queue.Empty:
                    continue

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
    """树莓派侧单监听端点；每次 accept 返回一个认证消息连接。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        authentication_key: bytes,
        handshake_timeout_s: float,
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
        if not isinstance(authentication_key, bytes) or len(authentication_key) < 32:
            raise ValueError(
                "authentication_key must contain at least 32 bytes."
            )
        if not isinstance(access_mode, RemoteAccessMode):
            raise ValueError("access_mode must be RemoteAccessMode.")
        self.port = port
        self.authentication_key = authentication_key
        self.handshake_timeout_s = _finite_positive_float(
            handshake_timeout_s,
            "handshake_timeout_s",
        )
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
            connection.settimeout(self.handshake_timeout_s)
            session_keys = _server_handshake(
                connection,
                self.authentication_key,
            )
            return RemoteMessageConnection(
                connection,
                outbound_session_key=session_keys.server_to_client,
                inbound_session_key=session_keys.client_to_server,
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
    authentication_key: bytes,
    handshake_timeout_s: float,
    access_mode: RemoteAccessMode,
    connection_options: RemoteConnectionOptions,
) -> RemoteMessageConnection:
    """电脑侧连接树莓派并完成双向预共享密钥认证。"""

    host = _bounded_text(host, "host", 255)
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 1 <= port <= 65_535
    ):
        raise ValueError(f"port must be in [1, 65535], got {port!r}.")
    if not isinstance(authentication_key, bytes) or len(authentication_key) < 32:
        raise ValueError("authentication_key must contain at least 32 bytes.")
    handshake_timeout_s = _finite_positive_float(
        handshake_timeout_s,
        "handshake_timeout_s",
    )
    if not isinstance(access_mode, RemoteAccessMode):
        raise ValueError("access_mode must be RemoteAccessMode.")
    connection = socket.create_connection(
        (host, port),
        timeout=handshake_timeout_s,
    )
    try:
        connection.settimeout(handshake_timeout_s)
        session_keys = _client_handshake(connection, authentication_key)
        return RemoteMessageConnection(
            connection,
            outbound_session_key=session_keys.client_to_server,
            inbound_session_key=session_keys.server_to_client,
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
