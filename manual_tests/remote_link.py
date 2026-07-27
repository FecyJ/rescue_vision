"""等待电脑端客户端连接，并打印其发来的 control 消息。"""

from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import replace
from pathlib import Path

from rescue_vision.communication import (
    ReceivedRemoteMessage,
    RemoteDisconnectedError,
    RemoteRole,
    RemoteSessionStatus,
    RemoteTopic,
)
from rescue_vision.config import load_runtime_config


def _print_message(message: ReceivedRemoteMessage) -> None:
    try:
        payload = message.payload.decode("utf-8")
    except UnicodeDecodeError:
        preview = message.payload[:64].hex()
        payload = (
            f"<binary {len(message.payload)} bytes; "
            f"first {len(message.payload[:64])} bytes hex={preview}>"
        )
    print(f"stream={message.stream.value}")
    print(f"topic={message.topic}")
    print(f"content_type={message.content_type}")
    print(f"sequence={message.sequence}")
    print(f"sender_timestamp_ns={message.sender_timestamp_ns}")
    print(f"received_timestamp_ns={message.received_timestamp_ns}")
    print(f"attributes={dict(message.attributes)!r}")
    print(f"payload={payload}")
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Accept a desktop client, send the required session status, "
            "and print received control messages."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    config = load_runtime_config(args.config)
    if not config.remote.enabled:
        raise RuntimeError("remote.enabled must be true")
    if config.remote.role is not RemoteRole.SERVER:
        raise RuntimeError(
            "remote_link.py accepts desktop clients; set remote.role: server"
        )

    server = config.remote.build_server()
    assert server is not None
    server_instance_id = f"remote-link-{uuid.uuid4()}"
    with server:
        print(
            f"listening={config.remote.host}:{server.bound_port} "
            f"access_mode={config.remote.access_mode.value}",
            flush=True,
        )
        connection = server.accept(timeout=args.timeout_seconds)
        with connection:
            session_status = RemoteSessionStatus(
                session_id=f"session-{uuid.uuid4()}",
                server_instance_id=server_instance_id,
                timestamp_ns=time.monotonic_ns(),
                access_mode=config.remote.access_mode,
                motion_control_available=False,
                capture_control_available=False,
                video_stream_available=False,
                map_snapshot_available=False,
                vehicle_state_available=False,
                capture_status_available=False,
                target_heading_control_available=False,
                session_status_period_ms=1000,
                vehicle_state_period_ms=None,
                map_snapshot_period_ms=None,
                capture_status_period_ms=None,
                video_nominal_fps=None,
                max_linear_velocity_m_s=None,
                max_angular_velocity_rad_s=None,
                max_motion_command_valid_for_ms=500,
            )
            connection.send_reliable_observation(
                RemoteTopic.SESSION_STATUS.value,
                session_status.to_payload(),
                content_type="application/json",
            )
            print("client_connected=true", flush=True)

            next_status_ns = time.monotonic_ns() + 1_000_000_000
            try:
                while True:
                    now_ns = time.monotonic_ns()
                    if now_ns >= next_status_ns:
                        session_status = replace(
                            session_status,
                            timestamp_ns=now_ns,
                        )
                        connection.send_reliable_observation(
                            RemoteTopic.SESSION_STATUS.value,
                            session_status.to_payload(),
                            content_type="application/json",
                        )
                        next_status_ns = now_ns + 1_000_000_000
                    try:
                        message = connection.receive_control(timeout=0.1)
                    except TimeoutError:
                        continue
                    _print_message(message)
            except RemoteDisconnectedError:
                print("client_connected=false", flush=True)
            except KeyboardInterrupt:
                print("stopped_by_user=true", flush=True)


if __name__ == "__main__":
    main()
