from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

import rescue_vision.communication.remote as remote_transport
from rescue_vision.communication import (
    DebugMotionCommand,
    MotionControlMode,
    RemoteAccessMode,
    RemoteAuthenticationError,
    RemoteConnectionOptions,
    RemoteDisconnectedError,
    RemoteMessageCodec,
    RemoteMessageConnection,
    RemotePolicyError,
    RemoteProtocolError,
    RemoteQueueOverflowError,
    RemoteStream,
    RemoteTopic,
    load_remote_authentication_key,
)


KEY = b"k" * 32


def options(
    *,
    control_queue_capacity: int = 4,
    observation_queue_capacity: int = 2,
) -> RemoteConnectionOptions:
    return RemoteConnectionOptions(
        io_timeout_s=0.05,
        control_queue_capacity=control_queue_capacity,
        observation_queue_capacity=observation_queue_capacity,
        max_header_bytes=4096,
        max_payload_bytes=1_000_000,
    )


class MemorySocket:
    def __init__(self) -> None:
        self.peer: MemorySocket | None = None
        self.timeout: float | None = None
        self.buffer = bytearray()
        self.condition = threading.Condition()
        self.shutdown_requested = False
        self.peer_closed = False

    def settimeout(self, timeout: float | None) -> None:
        self.timeout = timeout

    def sendall(self, data: bytes) -> None:
        if self.shutdown_requested or self.peer is None:
            raise BrokenPipeError("memory socket is closed")
        with self.peer.condition:
            self.peer.buffer.extend(data)
            self.peer.condition.notify_all()

    def recv(self, size: int) -> bytes:
        deadline = (
            None if self.timeout is None else time.monotonic() + self.timeout
        )
        with self.condition:
            while not self.buffer:
                if self.shutdown_requested or self.peer_closed:
                    return b""
                if deadline is None:
                    self.condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise socket.timeout()
                self.condition.wait(remaining)
            result = bytes(self.buffer[:size])
            del self.buffer[:size]
            return result

    def shutdown(self, how: int) -> None:
        self.shutdown_requested = True
        with self.condition:
            self.condition.notify_all()
        if self.peer is not None:
            with self.peer.condition:
                self.peer.peer_closed = True
                self.peer.condition.notify_all()

    def close(self) -> None:
        self.shutdown(socket.SHUT_RDWR)


class BlockingFirstSendSocket(MemorySocket):
    def __init__(self) -> None:
        super().__init__()
        self.send_entered = threading.Event()
        self.release_send = threading.Event()
        self._blocked_once = False

    def sendall(self, data: bytes) -> None:
        if not self._blocked_once:
            self._blocked_once = True
            self.send_entered.set()
            if not self.release_send.wait(timeout=1.0):
                raise TimeoutError("test did not release blocked send")
        super().sendall(data)


def memory_socket_pair() -> tuple[MemorySocket, MemorySocket]:
    left = MemorySocket()
    right = MemorySocket()
    left.peer = right
    right.peer = left
    return left, right


def authenticated_memory_pair() -> tuple[
    MemorySocket,
    MemorySocket,
    remote_transport._SessionKeys,
]:
    server_socket, client_socket = memory_socket_pair()
    server_result: list[remote_transport._SessionKeys | BaseException] = []

    def authenticate_server() -> None:
        try:
            server_result.append(
                remote_transport._server_handshake(server_socket, KEY)
            )
        except BaseException as exc:
            server_result.append(exc)

    thread = threading.Thread(target=authenticate_server)
    thread.start()
    client_keys = remote_transport._client_handshake(client_socket, KEY)
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert len(server_result) == 1
    if isinstance(server_result[0], BaseException):
        raise server_result[0]
    assert server_result[0] == client_keys
    return server_socket, client_socket, client_keys


def connected_pair(
    access_mode: RemoteAccessMode,
    *,
    robot_mode: RemoteAccessMode | None = None,
) -> tuple[RemoteMessageConnection, RemoteMessageConnection]:
    robot_socket, computer_socket, session_keys = authenticated_memory_pair()
    actual_robot_mode = robot_mode or access_mode
    robot = RemoteMessageConnection(
        robot_socket,
        outbound_session_key=session_keys.server_to_client,
        inbound_session_key=session_keys.client_to_server,
        io_timeout_s=0.05,
        control_queue_capacity=4,
        observation_queue_capacity=2,
        max_header_bytes=4096,
        max_payload_bytes=1_000_000,
        allow_inbound_control=(
            actual_robot_mode is RemoteAccessMode.DEBUG_CONTROL
        ),
        allow_outbound_control=False,
    )
    computer = RemoteMessageConnection(
        computer_socket,
        outbound_session_key=session_keys.client_to_server,
        inbound_session_key=session_keys.server_to_client,
        io_timeout_s=0.05,
        control_queue_capacity=4,
        observation_queue_capacity=2,
        max_header_bytes=4096,
        max_payload_bytes=1_000_000,
        allow_inbound_control=False,
        allow_outbound_control=(
            access_mode is RemoteAccessMode.DEBUG_CONTROL
        ),
    )
    return robot, computer


def wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    pytest.fail("condition was not reached before timeout")


def test_codec_handles_fragmentation_binary_payload_and_integrity() -> None:
    encoder = RemoteMessageCodec(
        session_key=KEY,
        max_header_bytes=4096,
        max_payload_bytes=1024,
    )
    decoder = RemoteMessageCodec(
        session_key=KEY,
        max_header_bytes=4096,
        max_payload_bytes=1024,
    )
    frame = encoder.encode(
        stream=RemoteStream.OBSERVATION,
        topic=RemoteTopic.VIDEO_FRAME.value,
        content_type="image/jpeg",
        sequence=0,
        sender_timestamp_ns=123,
        attributes={"frame_sequence": 5, "width": 320, "height": 240},
        payload=b"\xff\xd8image\xff\xd9",
    )

    assert decoder.feed(frame[:7]) == ()
    decoded = decoder.feed(frame[7:])[0]
    assert decoded.topic == RemoteTopic.VIDEO_FRAME.value
    assert decoded.attributes["frame_sequence"] == 5
    assert decoded.payload == b"\xff\xd8image\xff\xd9"

    tampered = bytearray(frame)
    tampered[-33] ^= 1
    with pytest.raises(RemoteAuthenticationError, match="integrity"):
        RemoteMessageCodec(
            session_key=KEY,
            max_header_bytes=4096,
            max_payload_bytes=1024,
        ).feed(bytes(tampered))


def test_handshake_rejects_peer_with_different_key() -> None:
    server_socket, client_socket = memory_socket_pair()
    server_errors: list[BaseException] = []

    def authenticate_server() -> None:
        try:
            remote_transport._server_handshake(server_socket, KEY)
        except BaseException as exc:
            server_errors.append(exc)
            server_socket.close()

    thread = threading.Thread(target=authenticate_server)
    thread.start()
    with pytest.raises(RemoteDisconnectedError):
        remote_transport._client_handshake(client_socket, b"x" * 32)
    thread.join(timeout=1.0)

    assert len(server_errors) == 1
    assert isinstance(server_errors[0], RemoteAuthenticationError)


def test_authenticated_debug_connection_carries_control_and_observation() -> None:
    robot, computer = connected_pair(RemoteAccessMode.DEBUG_CONTROL)
    robot.start()
    computer.start()
    try:
        command = DebugMotionCommand(
            command_id="drive-001",
            issued_timestamp_ns=10,
            valid_for_ms=200,
            deadman_enabled=True,
            control_mode=MotionControlMode.TWIST,
            linear_velocity_m_s=0.2,
            angular_velocity_rad_s=-0.3,
        )
        computer.send_control(
            RemoteTopic.DEBUG_MOTION.value,
            command.to_payload(),
        )
        received_command = robot.receive_control(timeout=1.0)
        assert DebugMotionCommand.from_payload(received_command.payload) == command

        robot.send_observation(
            RemoteTopic.VIDEO_FRAME.value,
            b"jpeg",
            content_type="image/jpeg",
            attributes={
                "frame_sequence": 12,
                "timestamp_ns": 1234,
                "width": 320,
                "height": 240,
                "coordinate_system": "undistorted_pixel",
            },
        )
        frame = computer.receive_observation(timeout=1.0)
        assert frame.payload == b"jpeg"
        assert frame.attributes["frame_sequence"] == 12
    finally:
        computer.stop()
        robot.stop()


def test_observe_only_client_cannot_submit_control() -> None:
    robot, computer = connected_pair(RemoteAccessMode.OBSERVE_ONLY)
    robot.start()
    computer.start()
    try:
        with pytest.raises(RemotePolicyError, match="Outbound control"):
            computer.send_control(RemoteTopic.DEBUG_MOTION.value, b"{}")
        robot.send_observation(RemoteTopic.MAP_SNAPSHOT.value, b"map")
        assert computer.receive_observation(timeout=1.0).payload == b"map"
    finally:
        computer.stop()
        robot.stop()


def test_robot_observe_only_policy_rejects_misconfigured_debug_client() -> None:
    robot, computer = connected_pair(
        RemoteAccessMode.DEBUG_CONTROL,
        robot_mode=RemoteAccessMode.OBSERVE_ONLY,
    )
    robot.start()
    computer.start()
    try:
        computer.send_control(RemoteTopic.DEBUG_MOTION.value, b"{}")
        with pytest.raises(RemotePolicyError, match="Inbound control"):
            robot.receive_control(timeout=1.0)
    finally:
        computer.stop()
        robot.stop()


def test_observation_receive_queue_drops_old_and_keeps_latest() -> None:
    left, right = memory_socket_pair()
    receiver = RemoteMessageConnection(
        left,
        outbound_session_key=KEY,
        inbound_session_key=KEY,
        io_timeout_s=0.05,
        control_queue_capacity=2,
        observation_queue_capacity=1,
        max_header_bytes=4096,
        max_payload_bytes=1024,
        allow_inbound_control=False,
        allow_outbound_control=False,
    )
    encoder = RemoteMessageCodec(
        session_key=KEY,
        max_header_bytes=4096,
        max_payload_bytes=1024,
    )
    receiver.start()
    try:
        frames = b"".join(
            encoder.encode(
                stream=RemoteStream.OBSERVATION,
                topic="observation/test",
                content_type="application/octet-stream",
                sequence=index,
                sender_timestamp_ns=index,
                attributes={},
                payload=str(index).encode(),
            )
            for index in range(3)
        )
        right.sendall(frames)
        wait_until(lambda: receiver.received_messages == 3)

        assert receiver.receive_observation(timeout=0).payload == b"2"
        assert receiver.dropped_inbound_observations == 2
    finally:
        receiver.stop()
        right.close()


def test_codec_rejects_sequence_gap_at_connection_layer() -> None:
    left, right = memory_socket_pair()
    receiver = RemoteMessageConnection(
        left,
        outbound_session_key=KEY,
        inbound_session_key=KEY,
        io_timeout_s=0.05,
        control_queue_capacity=2,
        observation_queue_capacity=2,
        max_header_bytes=4096,
        max_payload_bytes=1024,
        allow_inbound_control=False,
        allow_outbound_control=False,
    )
    encoder = RemoteMessageCodec(
        session_key=KEY,
        max_header_bytes=4096,
        max_payload_bytes=1024,
    )
    receiver.start()
    try:
        right.sendall(
            encoder.encode(
                stream=RemoteStream.OBSERVATION,
                topic="observation/test",
                content_type="application/octet-stream",
                sequence=1,
                sender_timestamp_ns=0,
                attributes={},
                payload=b"x",
            )
        )
        with pytest.raises(RemoteProtocolError, match="Expected remote sequence"):
            receiver.receive_observation(timeout=1.0)
    finally:
        receiver.stop()
        right.close()


def test_outbound_control_queue_never_silently_drops_commands() -> None:
    sender_socket = BlockingFirstSendSocket()
    peer = MemorySocket()
    sender_socket.peer = peer
    peer.peer = sender_socket
    sender = RemoteMessageConnection(
        sender_socket,
        outbound_session_key=KEY,
        inbound_session_key=KEY,
        io_timeout_s=0.05,
        control_queue_capacity=1,
        observation_queue_capacity=1,
        max_header_bytes=4096,
        max_payload_bytes=1024,
        allow_inbound_control=False,
        allow_outbound_control=True,
    )
    sender.start()
    try:
        sender.send_control("control/test", b"first")
        assert sender_socket.send_entered.wait(timeout=1.0)
        sender.send_control("control/test", b"second")
        with pytest.raises(RemoteQueueOverflowError, match="was not sent"):
            sender.send_control("control/test", b"third")
    finally:
        sender_socket.release_send.set()
        wait_until(lambda: sender.sent_messages == 2)
        sender.stop()
        peer.close()


def test_outbound_observation_queue_drops_old_and_keeps_latest() -> None:
    sender_socket = BlockingFirstSendSocket()
    peer = MemorySocket()
    sender_socket.peer = peer
    peer.peer = sender_socket
    sender = RemoteMessageConnection(
        sender_socket,
        outbound_session_key=KEY,
        inbound_session_key=KEY,
        io_timeout_s=0.05,
        control_queue_capacity=1,
        observation_queue_capacity=1,
        max_header_bytes=4096,
        max_payload_bytes=1024,
        allow_inbound_control=False,
        allow_outbound_control=False,
    )
    sender.start()
    try:
        sender.send_observation("observation/test", b"first")
        assert sender_socket.send_entered.wait(timeout=1.0)
        sender.send_observation("observation/test", b"stale")
        sender.send_observation("observation/test", b"latest")

        assert sender.dropped_outbound_observations == 1
    finally:
        sender_socket.release_send.set()
        wait_until(lambda: sender.sent_messages == 2)
        sender.stop()
        peer.close()


def test_authentication_key_file_accepts_hex_and_rejects_short(
    tmp_path: Path,
) -> None:
    hex_key = tmp_path / "remote.key"
    hex_key.write_text("ab" * 32 + "\n", encoding="ascii")
    short_key = tmp_path / "short.key"
    short_key.write_bytes(b"short")

    assert load_remote_authentication_key(hex_key) == bytes.fromhex("ab" * 32)
    with pytest.raises(ValueError, match="at least 32"):
        load_remote_authentication_key(short_key)
