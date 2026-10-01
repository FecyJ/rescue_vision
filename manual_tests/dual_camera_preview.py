"""双 CSI 原图预览；也为 perception 手动脚本提供最新帧显示循环。"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from threading import Condition, Thread
from time import monotonic, monotonic_ns, sleep

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.config import load_runtime_config


class PreviewWorker:
    """两路各一张待处理帧，轮流处理，共享一个推理后端。"""

    def __init__(self, process: Callable[[int, CameraFrame], CameraFrame]) -> None:
        self.process = process
        self.condition = Condition()
        self.pending: dict[int, CameraFrame] = {}
        self.results: dict[int, CameraFrame] = {}
        self.error: BaseException | None = None
        self.stopping = False
        self.thread = Thread(target=self._run, name="dual-camera-perception", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, stream: int, frame: CameraFrame) -> None:
        with self.condition:
            self.pending[stream] = frame
            self.condition.notify()

    def latest(self) -> dict[int, CameraFrame]:
        with self.condition:
            if self.error is not None:
                raise RuntimeError("双相机 perception 处理失败") from self.error
            return dict(self.results)

    def stop(self) -> None:
        with self.condition:
            self.stopping = True
            self.pending.clear()
            self.condition.notify()
        self.thread.join()

    def _run(self) -> None:
        previous_stream = 1
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping or self.pending)
                    if self.stopping:
                        return
                    stream = 1 - previous_stream
                    if stream not in self.pending:
                        stream = next(iter(self.pending))
                    frame = self.pending.pop(stream)
                processing_started_ns = monotonic_ns()
                result = self.process(stream, frame)
                completed_ns = monotonic_ns()
                result = replace(result, metadata={
                    **result.metadata,
                    "result_timestamp_ns": completed_ns,
                    "processing_ms": (completed_ns - processing_started_ns) / 1e6,
                    "capture_to_result_ms": (completed_ns - frame.timestamp_ns) / 1e6,
                })
                with self.condition:
                    self.results[stream] = result
                previous_stream = stream
        except BaseException as error:
            with self.condition:
                self.error = error


def preview_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.match.yaml"))
    parser.add_argument("--main-csi-port", type=int, choices=(0, 1), help="覆盖 camera.csi_port")
    parser.add_argument("--near-csi-port", type=int, choices=(0, 1), help="覆盖 near_camera.csi_port")
    parser.add_argument("--near-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--headless", action="store_true", help="只取帧/打印，不打开窗口；须指定时长")
    parser.add_argument("--preview-width", type=int, default=720, help="每路显示宽度，保持纵横比")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.headless and args.duration_seconds is None:
        parser.error("--headless 必须同时指定 --duration-seconds")
    if args.duration_seconds is not None and (
        not np.isfinite(args.duration_seconds) or args.duration_seconds <= 0
    ):
        parser.error(f"时长必须为有限正数：{args.duration_seconds}")
    if args.preview_width <= 0:
        parser.error(f"预览宽度必须为正数：{args.preview_width}")


def display_frame(title: str, frame: CameraFrame, width: int) -> None:
    image = frame.image_bgr
    scale = min(1.0, width / image.shape[1])
    preview = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    age_ms = (monotonic_ns() - frame.timestamp_ns) / 1e6
    cv2.putText(preview, f"seq={frame.sequence} age={age_ms:.0f}ms", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow(title, preview)


def run_preview(args: argparse.Namespace, config, process: Callable[[int, CameraFrame], CameraFrame] | None = None) -> None:
    if config.near_camera is None:
        raise ValueError("双相机预览需要 YAML near_camera 配置。")
    main_port = config.camera.csi_port if args.main_csi_port is None else args.main_csi_port
    near_port = config.near_camera.csi_port if args.near_csi_port is None else args.near_csi_port
    if main_port == near_port:
        raise ValueError(f"两个相机不能使用相同的 csi_port={main_port}。")
    sources = []
    for settings, port, size in (
        (config.camera, main_port, config.camera.image_size),
        (config.near_camera, near_port, tuple(args.near_size) if args.near_size else config.near_camera.image_size),
    ):
        source_class = Picamera2Source if settings.backend == "picamera2" else RpicamSource
        sources.append(source_class(image_size=size, fps=settings.fps,
                                    lens_position=settings.lens_position, csi_port=port))
    labels = [f"main CAM/DISP{main_port}", f"near CAM/DISP{near_port}"]
    counts = [0, 0]
    processed_sequences: dict[int, int] = {}
    try:
        with ExitStack() as stack:
            for source in sources:
                stack.enter_context(source)
            worker = PreviewWorker(process) if process is not None else None
            if worker is not None:
                worker.start()
                stack.callback(worker.stop)
            started = report_time = monotonic()
            last_received = [started, started]
            titles: set[str] = set()
            perception_metrics = {}
            print("两路均已启动；窗口 Q/Esc 或 Ctrl+C 退出。", flush=True)
            while args.duration_seconds is None or monotonic() - started < args.duration_seconds:
                for stream, source in enumerate(sources):
                    try:
                        frame = source.read(timeout=0)
                    except TimeoutError:
                        if monotonic() - last_received[stream] > 3:
                            raise TimeoutError(f"{labels[stream]} 超过 3 s 没有新帧")
                        continue
                    counts[stream] += 1
                    last_received[stream] = monotonic()
                    if worker is not None:
                        worker.submit(stream, frame)
                    if not args.headless:
                        title = f"{labels[stream]} raw_pixel"
                        display_frame(title, frame, args.preview_width)
                        titles.add(title)
                if worker is not None:
                    for stream, result in worker.latest().items():
                        processed_sequences[stream] = result.sequence
                        perception_metrics[labels[stream]] = {
                            "sequence": result.sequence,
                            "capture_age_ms": round((monotonic_ns() - result.timestamp_ns) / 1e6, 1),
                            "result_age_ms": round((monotonic_ns() - result.metadata["result_timestamp_ns"]) / 1e6, 1),
                            "capture_to_result_ms": round(result.metadata["capture_to_result_ms"], 1),
                            "processing_ms": round(result.metadata["processing_ms"], 1),
                            "blue_danger_detections": result.metadata.get("blue_danger_detections"),
                            "detections": {name: result.metadata.get(f"{name}_detections", 0)
                                           for name in ("green_supply", "black_core", "orange_injured", "blue_danger")},
                        }
                        if not args.headless:
                            title = f"{labels[stream]} perception"
                            display_frame(title, result, args.preview_width)
                            titles.add(title)
                now = monotonic()
                if now - report_time >= 1:
                    print(f"elapsed_s={now-started:.1f} main_frames={counts[0]} near_frames={counts[1]} "
                          f"perception={perception_metrics}", flush=True)
                    report_time = now
                if not args.headless:
                    if cv2.waitKey(5) & 0xFF in (27, ord("q"), ord("Q")):
                        break
                    if any(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1 for title in titles):
                        break
                else:
                    sleep(0.005)
            if args.headless and (not all(counts) or (worker is not None and len(processed_sequences) != 2)):
                raise RuntimeError(f"时长内未完成两路验证：frames={counts}, perception={processed_sequences}")
            print(f"完成：main_frames={counts[0]} near_frames={counts[1]} perception_sequences={processed_sequences}", flush=True)
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


def main() -> None:
    parser = preview_parser("双 CSI 相机原图实时预览（不加载标定、不使用 UART）。")
    args = parser.parse_args()
    validate_args(parser, args)
    config = load_runtime_config(args.config)
    try:
        run_preview(args, config)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
