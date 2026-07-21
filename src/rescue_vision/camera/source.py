from time import sleep

from libcamera import controls
from picamera2 import Picamera2

from .frame import CameraFrame


class PiCameraSource:
    """Raspberry Pi Camera Module 3 NoIR Wide 图像源。"""

    def __init__(
        self,
        image_size: tuple[int, int] = (2304, 1296),
        fps: int = 30,
        lens_position: float = 1.0,
    ) -> None:
        self.image_size = image_size
        self.fps = fps
        self.lens_position = lens_position

        self.camera = Picamera2()
        self.sequence = 0

    def start(self) -> None:
        frame_duration_us = round(1_000_000 / self.fps)

        config = self.camera.create_video_configuration(
            main={
                "size": self.image_size,
                "format": "RGB888",
            },
            # 显式选择 2304×1296 的 IMX708 传感器模式，
            # 避免由较小输出流触发其他裁剪模式。
            raw={
                "size": self.image_size,
            },
            controls={
                "FrameDurationLimits": (
                    frame_duration_us,
                    frame_duration_us,
                ),
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": self.lens_position,
            },
            buffer_count=4,
        )

        self.camera.configure(config)
        self.camera.start()

        sleep(1.0)

    def read(self) -> CameraFrame:
        with self.camera.captured_request() as request:
            image = request.make_array("main")
            metadata = request.get_metadata()

        frame = CameraFrame(
            sequence=self.sequence,
            timestamp_ns=int(metadata["SensorTimestamp"]),
            image_bgr=image,
        )

        self.sequence += 1
        return frame

    def stop(self) -> None:
        self.camera.stop()
        self.camera.close()