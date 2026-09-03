"""Hailo YOLO26 Pose 的最小独立推理后端。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np

from rescue_vision.geometry.camera_model import IMAGE_BORDER_FILL_VALUE
from rescue_vision.geometry.types import UndistortedPixel
from rescue_vision.perception.types import (
    ModelDetection,
    POSE_MODEL_CLASSES,
    PoseKeypoint,
    PoseModelClass,
    UndistortedBoundingBox,
)


_ONNX_DUPLICATE_SCHEMA_PREFIX = (
    b"Schema error: Trying to register schema with name "
)
_ONNX_DUPLICATE_SCHEMA_MARKER = b" but it is already registered from file "


def _is_duplicate_onnx_schema_line(line: bytes) -> bool:
    """识别系统 ONNX Runtime 重复注册 schema 的已知噪声行。"""

    content = line.rstrip(b"\r\n")
    return content.startswith(_ONNX_DUPLICATE_SCHEMA_PREFIX) and (
        _ONNX_DUPLICATE_SCHEMA_MARKER in content
    )


def _write_all(file_descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:
            raise OSError("stderr forwarding made no progress")
        view = view[written:]


def _forward_onnx_session_stderr(read_fd: int, saved_stderr_fd: int) -> None:
    """转发会话初始化 stderr，仅过滤精确匹配的重复 schema 行。"""

    buffer = bytearray()
    suppress_following_blank = False

    def forward_line(line: bytes) -> None:
        nonlocal suppress_following_blank
        if _is_duplicate_onnx_schema_line(line):
            suppress_following_blank = True
            return
        if suppress_following_blank and not line.strip(b"\r\n"):
            suppress_following_blank = False
            return
        suppress_following_blank = False
        try:
            _write_all(saved_stderr_fd, line)
        except OSError:
            # If stderr has already been closed, continue draining the pipe so
            # the ONNX constructor cannot deadlock on a full pipe.
            return

    try:
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                newline_index = buffer.find(b"\n")
                if newline_index < 0:
                    break
                line_end = newline_index + 1
                forward_line(bytes(buffer[:line_end]))
                del buffer[:line_end]
        if buffer:
            forward_line(bytes(buffer))
    except OSError:
        # The owning context closes the pipe during exceptional teardown.
        return


@contextmanager
def _filter_duplicate_onnx_schema_stderr() -> Iterator[None]:
    """仅在 ONNX 会话构造期间过滤已知的重复 schema 注册噪声。"""

    saved_stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    reader = Thread(
        target=_forward_onnx_session_stderr,
        args=(read_fd, saved_stderr_fd),
        name="rescue-onnx-stderr-filter",
        daemon=True,
    )
    redirected = False
    try:
        reader.start()
        os.dup2(write_fd, 2)
        redirected = True
        os.close(write_fd)
        write_fd = -1
        yield
    finally:
        if redirected:
            os.dup2(saved_stderr_fd, 2)
        if write_fd >= 0:
            os.close(write_fd)
        reader.join(timeout=2.0)
        if reader.is_alive():
            raise RuntimeError("ONNX stderr filter worker did not stop.")
        os.close(read_fd)
        os.close(saved_stderr_fd)


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    original_size: tuple[int, int]
    model_size: tuple[int, int]
    scale: float
    x_offset: int
    y_offset: int

    def point_to_original(self, u: float, v: float) -> UndistortedPixel:
        width, height = self.original_size
        return UndistortedPixel(
            float(np.clip((u - self.x_offset) / self.scale, 0.0, width - 1.0)),
            float(np.clip((v - self.y_offset) / self.scale, 0.0, height - 1.0)),
        )

    def box_to_original(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> UndistortedBoundingBox:
        width, height = self.original_size
        left = float(np.clip((x_min - self.x_offset) / self.scale, 0.0, width))
        top = float(np.clip((y_min - self.y_offset) / self.scale, 0.0, height))
        right = float(np.clip((x_max - self.x_offset) / self.scale, 0.0, width))
        bottom = float(np.clip((y_max - self.y_offset) / self.scale, 0.0, height))
        return UndistortedBoundingBox(left, top, right, bottom)


def letterbox_bgr_to_rgb(
    image_bgr: np.ndarray,
    model_size: tuple[int, int],
) -> tuple[np.ndarray, LetterboxTransform]:
    """将 BGR 原图转换为模型所需的 RGB letterbox uint8 输入。"""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"image_bgr must have shape (height, width, 3), got {image_bgr.shape}."
        )

    model_width, model_height = model_size
    if model_width <= 0 or model_height <= 0:
        raise ValueError(f"model_size must be positive, got {model_size!r}.")

    original_height, original_width = image_bgr.shape[:2]
    scale = min(
        model_width / original_width,
        model_height / original_height,
    )

    # 使用 round，而不是直接 int 截断。
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized_bgr = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    horizontal_padding = model_width - resized_width
    vertical_padding = model_height - resized_height

    # 与 Ultralytics 居中 LetterBox 的取整方式保持一致。
    x_offset = int(round(horizontal_padding / 2 - 0.1))
    y_offset = int(round(vertical_padding / 2 - 0.1))

    output_rgb = np.full(
        (model_height, model_width, 3),
        IMAGE_BORDER_FILL_VALUE,
        dtype=np.uint8,
    )
    output_rgb[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized_rgb

    return output_rgb, LetterboxTransform(
        original_size=(original_width, original_height),
        model_size=model_size,
        scale=scale,
        x_offset=x_offset,
        y_offset=y_offset,
    )


def map_hef_outputs_to_onnx_inputs(
    hailo_outputs: dict[str, np.ndarray],
    tensor_mapping: dict[str, object],
) -> dict[str, np.ndarray]:
    """按部署 JSON 映射输出，并在需要时从 NHWC 转为 NCHW。"""

    onnx_inputs: dict[str, np.ndarray] = {}
    for hef_name, mapping in tensor_mapping.items():
        if (
            not isinstance(mapping, list)
            or len(mapping) != 2
            or not isinstance(mapping[0], str)
            or not isinstance(mapping[1], list)
        ):
            raise ValueError(f"Invalid tensor mapping for {hef_name!r}: {mapping!r}.")
        onnx_name = mapping[0]
        expected_shape = mapping[1]
        if hef_name not in hailo_outputs:
            raise ValueError(
                f"Missing HEF output {hef_name!r}; available "
                f"{sorted(hailo_outputs)}."
            )
        tensor = np.asarray(hailo_outputs[hef_name])
        if tensor.dtype != np.float32:
            raise ValueError(
                f"HEF output {hef_name!r} must be float32, got {tensor.dtype}."
            )
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(
                f"HEF output {hef_name!r} must have batch-1 rank-4 shape, "
                f"got {tensor.shape}."
            )
        if list(tensor.shape[1:]) == expected_shape:
            mapped = tensor
        elif [tensor.shape[3], tensor.shape[1], tensor.shape[2]] == expected_shape:
            mapped = np.transpose(tensor, (0, 3, 1, 2))
        else:
            raise ValueError(
                f"HEF output {hef_name!r} shape {tensor.shape} does not match "
                f"expected {expected_shape}."
            )
        onnx_inputs[onnx_name] = mapped
    return onnx_inputs


def parse_yolo26_pose_output(
    output: np.ndarray,
    *,
    kpt_shape: tuple[int, int],
    transform: LetterboxTransform,
    score_threshold: float,
    max_detections: int,
) -> list[ModelDetection]:
    """解析 YOLO Pose v3 end-to-end 输出的三个固定关键点。"""

    if kpt_shape != (3, 3):
        raise ValueError(
            f"Pose v3 kpt_shape must be [3, 3], got {kpt_shape!r}."
        )
    detections = np.asarray(output)
    if detections.ndim == 3:
        if detections.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {detections.shape}.")
        detections = detections[0]
    expected_width = 6 + kpt_shape[0] * 3
    if detections.ndim != 2 or detections.shape[1] != expected_width:
        raise ValueError(
            f"Pose output must have shape (N, {expected_width}), got "
            f"{detections.shape}."
        )
    scores = detections[:, 4]
    valid = detections[np.isfinite(scores) & (scores >= score_threshold)]
    if valid.size:
        valid = valid[np.argsort(-valid[:, 4], kind="stable")[:max_detections]]

    parsed: list[ModelDetection] = []
    for row in valid:
        class_value = float(row[5])
        class_id = int(class_value)
        if not np.isfinite(class_value) or class_value != class_id or class_id < 0:
            raise ValueError(f"Invalid model class ID {class_value!r}.")
        if class_id >= len(POSE_MODEL_CLASSES):
            raise ValueError(f"Invalid YOLO Pose v3 class ID {class_id}.")
        try:
            box = transform.box_to_original(
                *[float(value) for value in row[:4]]
            )
        except ValueError:
            # 填充区或垃圾输出裁剪后可能退化为零面积框；单条坏输出
            # 不应击穿整帧实时推理。
            continue
        keypoints: list[PoseKeypoint] = []
        for index in range(3):
            offset = 6 + index * 3
            values = row[offset : offset + 3]
            confidence = float(values[2]) if np.isfinite(values[2]) else 0.0
            point = (
                transform.point_to_original(float(values[0]), float(values[1]))
                if confidence > 0.0 and np.isfinite(values).all()
                else None
            )
            keypoints.append(PoseKeypoint(point, confidence if point is not None else 0.0))
        model_class = POSE_MODEL_CLASSES[class_id]
        if (
            model_class is PoseModelClass.SAFE_ZONE
            and keypoints[1].point is not None
            and keypoints[2].point is not None
            and keypoints[1].point.u > keypoints[2].point.u
        ):
            # 模型可能交换两个几何上相同的角点；它们连同置信度作为一组
            # 交换，规范化后的 K1/K2 仍对应当前图像的左右语义。
            keypoints[1], keypoints[2] = keypoints[2], keypoints[1]
        if model_class is not PoseModelClass.SAFE_ZONE:
            # 训练标签中的未使用槽位为 0 0 0，但 Pose head 仍会为每个
            # 类别输出三个回归槽；对非安全区类别只保留 K0，避免网络对
            # 监督为零的槽位产生的任意值污染领域契约。
            keypoints[1] = PoseKeypoint(None, 0.0)
            keypoints[2] = PoseKeypoint(None, 0.0)
        parsed.append(
            ModelDetection(
                model_class_id=class_id,
                confidence=float(row[4]),
                box=box,
                keypoints=(keypoints[0], keypoints[1], keypoints[2]),
            )
        )
    return parsed


class HailoYolo26PoseBackend:
    """HEF 主干加 ONNX 后处理的同步单帧后端。"""

    def __init__(
        self,
        *,
        hef_path: Path,
        postprocess_onnx_path: Path,
        output_mapping_path: Path,
        class_count: int,
        max_detections: int,
        score_threshold: float,
    ) -> None:
        paths = (hef_path, postprocess_onnx_path, output_mapping_path)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Model asset does not exist: {path}.")
        config = json.loads(output_mapping_path.read_text(encoding="utf-8"))
        if config.get("output_format") != "yolo26_pose":
            raise ValueError("output_mapping output_format must be 'yolo26_pose'.")
        params = config.get("postprocess_params")
        if not isinstance(params, dict):
            raise ValueError("output_mapping.postprocess_params must be an object.")
        input_size = params.get("input_size")
        if (
            isinstance(input_size, bool)
            or not isinstance(input_size, int)
            or input_size <= 0
        ):
            raise ValueError("postprocess_params.input_size must be a positive integer.")
        self._configured_input_size = input_size
        if params.get("num_classes") != class_count:
            raise ValueError(
                "postprocess_params.num_classes does not match configured raw classes."
            )
        kpt_value = params.get("kpt_shape")
        if not isinstance(kpt_value, list) or len(kpt_value) != 2:
            raise ValueError("postprocess_params.kpt_shape must be [K, 3].")
        self._kpt_shape = (int(kpt_value[0]), int(kpt_value[1]))
        if self._kpt_shape != (3, 3):
            raise ValueError("YOLO Pose v3 requires kpt_shape [3, 3].")
        if class_count != len(POSE_MODEL_CLASSES):
            raise ValueError(
                f"YOLO Pose v3 requires {len(POSE_MODEL_CLASSES)} classes, "
                f"got {class_count}."
            )
        tensor_mapping = config.get("output_tensor_mapping")
        if not isinstance(tensor_mapping, dict) or not tensor_mapping:
            raise ValueError("output_tensor_mapping must be a non-empty object.")
        self._tensor_mapping = tensor_mapping

        try:
            import onnxruntime as ort
            from hailo_platform import (
                FormatOrder,
                FormatType,
                HailoSchedulingAlgorithm,
                VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hailo backend requires hailo_platform and onnxruntime."
            ) from exc

        self._max_detections = max_detections
        self._score_threshold = score_threshold
        self._closed = False
        self._last_job = None

        try:
            device_params = VDevice.create_params()
            device_params.scheduling_algorithm = (
                HailoSchedulingAlgorithm.ROUND_ROBIN
            )
            device_params.group_id = "SHARED"
            self._device = VDevice(device_params)
            self._infer_model = self._device.create_infer_model(str(hef_path))
            self._infer_model.set_batch_size(1)
            output_shapes = {
                output.name: tuple(output.shape)
                for output in self._infer_model.outputs
            }
            if set(output_shapes) != set(self._tensor_mapping):
                raise ValueError(
                    "HEF output names do not exactly match output_tensor_mapping."
                )
            for hef_name, mapping in self._tensor_mapping.items():
                expected_chw = tuple(mapping[1])
                actual_hwc = output_shapes[hef_name]
                if (
                    len(actual_hwc) != 3
                    or (actual_hwc[2], actual_hwc[0], actual_hwc[1])
                    != expected_chw
                ):
                    raise ValueError(
                        f"HEF output {hef_name!r} shape {actual_hwc} does not "
                        f"match configured CHW {expected_chw}."
                    )

            # 显式要求 HailoRT 输出 NHWC
            input_stream = self._infer_model.input()
            input_stream.set_format_type(FormatType.UINT8)
            input_stream.set_format_order(FormatOrder.NHWC)

            for output in self._infer_model.outputs:
                output_stream = self._infer_model.output(output.name)
                output_stream.set_format_type(FormatType.FLOAT32)
                output_stream.set_format_order(FormatOrder.NHWC)

            self._configure_context = self._infer_model.configure()
            self._configured_model = self._configure_context.__enter__()
            input_shape = tuple(self._infer_model.input().shape)
            if len(input_shape) != 3 or input_shape[2] != 3:
                raise ValueError(
                    f"HEF input must be HWC with 3 channels, got {input_shape}."
                )
            self._model_size = (int(input_shape[1]), int(input_shape[0]))
            expected_model_size = (
                self._configured_input_size,
                self._configured_input_size,
            )
            if self._model_size != expected_model_size:
                raise ValueError(
                    f"HEF model size {self._model_size} does not match "
                    f"postprocess input_size {expected_model_size}."
                )
            with _filter_duplicate_onnx_schema_stderr():
                self._onnx_session = ort.InferenceSession(str(postprocess_onnx_path))
        except BaseException:
            self.close()
            raise

    def infer(self, image_bgr: np.ndarray) -> list[ModelDetection]:
        if self._closed:
            raise RuntimeError("HailoYolo26PoseBackend is closed.")
        preprocessed, transform = letterbox_bgr_to_rgb(image_bgr, self._model_size)
        output_buffers = {
            output.name: np.empty(output.shape, dtype=np.float32)
            for output in self._infer_model.outputs
        }
        bindings = self._configured_model.create_bindings(
            output_buffers=output_buffers
        )
        bindings.input().set_buffer(preprocessed)
        completion_error: list[BaseException] = []

        def completed(completion_info: Any) -> None:
            if completion_info.exception is not None:
                completion_error.append(completion_info.exception)

        self._configured_model.wait_for_async_ready(timeout_ms=10_000)
        self._last_job = self._configured_model.run_async([bindings], completed)
        self._last_job.wait(10_000)
        if completion_error:
            raise RuntimeError("Hailo inference failed.") from completion_error[0]

        raw_outputs = {
            name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
            for name in output_buffers
        }
        onnx_inputs = map_hef_outputs_to_onnx_inputs(
            raw_outputs, self._tensor_mapping
        )
        output_names = [output.name for output in self._onnx_session.get_outputs()]
        results = self._onnx_session.run(output_names, onnx_inputs)
        if len(results) != 1:
            raise ValueError(f"Expected one ONNX output, got {len(results)}.")
        return parse_yolo26_pose_output(
            results[0],
            kpt_shape=self._kpt_shape,
            transform=transform,
            score_threshold=self._score_threshold,
            max_detections=self._max_detections,
        )

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            if self._last_job is not None:
                self._last_job.wait(10_000)
        except BaseException as error:
            first_error = error
        try:
            context = getattr(self, "_configure_context", None)
            if context is not None:
                context.__exit__(None, None, None)
        except BaseException as error:
            if first_error is None:
                first_error = error
        try:
            device = getattr(self, "_device", None)
            if device is not None and hasattr(device, "release"):
                device.release()
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise RuntimeError("Hailo backend shutdown failed.") from first_error

    def __enter__(self) -> HailoYolo26PoseBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
