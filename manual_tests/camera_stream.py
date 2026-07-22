# tests/test_camera_stream.py

from time import perf_counter

import cv2

from rescue_vision.camera.rpicam_source import RpicamSource


def main() -> None:
    camera = RpicamSource(
        image_size=(2304, 1296),
        fps=30,
        lens_position=1.0,
    )

    frame_count = 0
    start_time = perf_counter()

    try:
        camera.start()

        while True:
            frame = camera.read()

            # 在这里调用后续视觉模块
            image = frame.image_bgr

            # 仅用于调试显示，缩小可降低桌面显示开销
            preview = cv2.resize(
                image,
                (1152, 648),
            )

            cv2.imshow("camera", preview)

            frame_count += 1
            elapsed = perf_counter() - start_time

            if elapsed >= 1.0:
                print(
                    f"FPS: {frame_count / elapsed:.1f}, "
                    f"sequence: {frame.sequence}, "
                    f"timestamp_ns: {frame.timestamp_ns}"
                )

                frame_count = 0
                start_time = perf_counter()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
