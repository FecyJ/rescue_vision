"""用两台主机验证直接 TCP 远程观察链路，不发送任何运动控制。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rescue_vision.communication import RemoteRole
from rescue_vision.config import load_runtime_config


DIAGNOSTIC_TOPIC = "observation/diagnostic/roundtrip"
ACK_TOPIC = "observation/diagnostic/roundtrip_ack"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify direct TCP remote observation round-trip without "
            "sending debug control."
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

    if config.remote.role is RemoteRole.SERVER:
        server = config.remote.build_server()
        assert server is not None
        with server:
            print(
                f"Listening on {config.remote.host}:{server.bound_port} "
                f"mode={config.remote.access_mode.value}"
            )
            connection = server.accept(timeout=args.timeout_seconds)
            with connection:
                sent_ns = time.monotonic_ns()
                connection.send_observation(
                    DIAGNOSTIC_TOPIC,
                    json.dumps(
                        {"sent_timestamp_ns": sent_ns},
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    content_type="application/json",
                )
                acknowledgement = connection.receive_observation(
                    timeout=args.timeout_seconds
                )
                if acknowledgement.topic != ACK_TOPIC:
                    raise RuntimeError(
                        f"Expected {ACK_TOPIC!r}, got "
                        f"{acknowledgement.topic!r}."
                    )
                print(
                    "Direct TCP observation round-trip OK, "
                    f"age_ms={(time.monotonic_ns() - sent_ns) / 1_000_000:.1f}"
                )
        return

    connection = config.remote.connect_client()
    assert connection is not None
    with connection:
        message = connection.receive_observation(timeout=args.timeout_seconds)
        if message.topic != DIAGNOSTIC_TOPIC:
            raise RuntimeError(
                f"Expected {DIAGNOSTIC_TOPIC!r}, got {message.topic!r}."
            )
        connection.send_observation(
            ACK_TOPIC,
            message.payload,
            content_type="application/json",
            attributes={"received_sequence": message.sequence},
        )
        deadline = time.monotonic() + args.timeout_seconds
        while connection.sent_messages < 1:
            connection.check_health()
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out sending diagnostic acknowledgement.")
            time.sleep(0.01)
        print("Direct TCP observation acknowledgement sent")


if __name__ == "__main__":
    main()
