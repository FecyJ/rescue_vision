from __future__ import annotations

import signal
import subprocess
import threading
import time

import cv2
import numpy as np

from .frame import CameraFrame


class RpicamSource:
    """
    使用官方 rpicam-vid 持续采集 YUV420。

    相机进程在应用运行期间只启动一次。
    后台线程持续排空 stdout，只保留最新帧。
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (2304, 1296),
        fps: int = 20,
        lens_position: float = 1.0,
    ) -> None:
        if len(image_size) != 2 or any(value <= 0 for value in image_size):
            raise ValueError(f"image_size must be positive, got {image_size}.")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        if not np.isfinite(lens_position) or lens_position < 0:
            raise ValueError(
                f"lens_position must be finite and non-negative, got "
                f"{lens_position}."
            )
        self.width, self.height = image_size
        self.fps = fps
        self.lens_position = lens_position

        # YUV420 每帧占 width × height × 1.5 字节。
        self.frame_size = (
            self.width * self.height * 3 // 2
        )

        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None

        self._condition = threading.Condition()
        self._latest_yuv: bytearray | None = None
        self._latest_timestamp_ns: int | None = None
        self._latest_sequence = -1
        self._delivered_sequence = -1
        self._reader_error: BaseException | None = None

        self._running = False

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def start(self) -> None:
        with self._condition:
            if self._running or self._process is not None:
                raise RuntimeError("Camera source is already started.")
            self._latest_yuv = None
            self._latest_timestamp_ns = None
            self._latest_sequence = -1
            self._delivered_sequence = -1
            self._reader_error = None

        command = [
            "rpicam-vid",
            "--nopreview",
            "--timeout", "0",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.fps),
            "--codec", "yuv420",
            "--no-raw",
            "--buffer-count", "6",
            "--lens-position", str(self.lens_position),
            "--flush",
            "--output", "-",
        ]

        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,

            # 让 rpicam 日志直接显示在当前终端。
            # 不使用 PIPE，避免 stderr 未读取导致阻塞。
            stderr=None,

            # 不增加 Python 层输出缓冲。
            bufsize=0,
        )

        self._running = True

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="rpicam-reader",
            daemon=True,
        )
        self._reader_thread.start()

        # 等待第一帧，确保启动确实成功。
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._latest_yuv is not None
                    or self._reader_error is not None
                ),
                timeout=5.0,
            )

        if not ready:
            self.stop()
            raise TimeoutError("等待相机第一帧超时")

        if self._reader_error is not None:
            self.stop()
            raise RuntimeError(
                "相机采集进程启动失败"
            ) from self._reader_error

    def read(self, timeout: float = 1.0) -> CameraFrame:
        """
        等待一张比上次 read() 更新的帧。

        若处理速度落后，只返回当前最新帧。
        """
        if timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}.")

        with self._condition:
            if not self._running:
                raise RuntimeError("Camera source is not started.")
            ready = self._condition.wait_for(
                lambda: (
                    self._latest_sequence
                    > self._delivered_sequence
                    or self._reader_error is not None
                ),
                timeout=timeout,
            )

            if not ready:
                raise TimeoutError("等待相机帧超时")

            if self._reader_error is not None:
                raise RuntimeError(
                    "相机采集进程异常退出"
                ) from self._reader_error

            raw_yuv = self._latest_yuv
            timestamp_ns = self._latest_timestamp_ns
            sequence = self._latest_sequence
            self._delivered_sequence = sequence

        assert raw_yuv is not None
        assert timestamp_ns is not None

        yuv = np.frombuffer(
            raw_yuv,
            dtype=np.uint8,
        ).reshape(
            self.height * 3 // 2,
            self.width,
        )

        image_bgr = cv2.cvtColor(
            yuv,
            cv2.COLOR_YUV2BGR_I420,
        )

        return CameraFrame(
            sequence=sequence,

            # 这是后台线程收到完整帧的单调时钟时间，
            # 不是传感器曝光开始时间。
            timestamp_ns=timestamp_ns,

            image_bgr=image_bgr,
            metadata={
                "source": "rpicam-vid",
                "configured_fps": self.fps,
                "lens_position": self.lens_position,
            },
        )

    def stop(self) -> None:
        process = self._process

        if process is None:
            self._running = False
            return

        # rpicam-vid 显式处理 SIGINT，并执行正常停止流程。
        if process.poll() is None:
            process.send_signal(signal.SIGINT)

            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.terminate()

                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        self._running = False

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)

        if process.stdout is not None:
            process.stdout.close()

        self._reader_thread = None
        self._process = None

    def _reader_loop(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None

        sequence = 0

        try:
            while self._running:
                frame = self._read_exact(
                    self._process.stdout,
                    self.frame_size,
                )

                if frame is None:
                    return_code = self._process.poll()

                    raise RuntimeError(
                        "rpicam-vid 输出流结束，"
                        f"returncode={return_code}"
                    )

                with self._condition:
                    # 直接覆盖旧帧，不让视觉延迟积累。
                    self._latest_yuv = frame
                    self._latest_timestamp_ns = time.monotonic_ns()
                    self._latest_sequence = sequence
                    self._condition.notify_all()

                sequence += 1

        except BaseException as error:
            with self._condition:
                if self._running:
                    self._reader_error = error

                self._condition.notify_all()

    @staticmethod
    def _read_exact(
        stream,
        size: int,
    ) -> bytearray | None:
        """
        从 stdout 恰好读取一帧。

        pipe 的一次 read 不保证返回请求的全部数据，
        因此必须循环读取。
        """
        buffer = bytearray(size)
        view = memoryview(buffer)
        offset = 0

        while offset < size:
            count = stream.readinto(view[offset:])

            if not count:
                return None

            offset += count

        return buffer

    def __enter__(self) -> RpicamSource:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
