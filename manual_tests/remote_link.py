"""连接车端，接收并打印一条远程 observation。"""

from __future__ import annotations

import argparse
from pathlib import Path

from rescue_vision.communication import RemoteRole
from rescue_vision.config import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect to the car and print one remote observation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    config = load_runtime_config(args.config)
    if not config.remote.enabled:
        raise RuntimeError("remote.enabled must be true")
    if config.remote.role is not RemoteRole.CLIENT:
        raise RuntimeError(
            "remote_link.py is a receive-only client; set "
            "remote.role: client and remote.host to the car address"
        )

    connection = config.remote.connect_client()
    assert connection is not None
    with connection:
        message = connection.receive_observation(timeout=args.timeout_seconds)
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


if __name__ == "__main__":
    main()
