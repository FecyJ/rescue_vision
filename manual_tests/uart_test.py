from __future__ import annotations

import argparse

import serial


def main() -> None:
    parser = argparse.ArgumentParser(description="Print raw UART bytes for a manual check.")
    parser.add_argument("--device", default="/dev/ttyAMA10")
    parser.add_argument("--baudrate", type=int, default=230400)
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    with serial.Serial(args.device, args.baudrate, timeout=args.timeout) as port:
        print("Listening...")
        try:
            while True:
                data = port.read(port.in_waiting or 1)
                if data:
                    print("HEX :", data.hex(" "))
                    print("RAW :", data)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
