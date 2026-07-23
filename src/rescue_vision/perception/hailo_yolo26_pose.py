"""Hailo YOLO26 Pose 的最小独立推理后端。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rescue_vision.geometry.types import UndistortedPixel
from rescue_vision.perception.types import ModelDetection, UndistortedBoundingBox


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


def letterbox_bgr(
    image_bgr: np.ndarray,
    model_size: tuple[int, int],
) -> tuple[np.ndarray, LetterboxTransform]:
    """按参考运行时使用 114 灰色填充并保持宽高比。"""

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"image_bgr must have shape (height, width, 3), got {image_bgr.shape}."
        )
    model_width, model_height = model_size
    if model_width <= 0 or model_height <= 0:
        raise ValueError(f"model_size must be positive, got {model_size!r}.")
    original_height, original_width = image_bgr.shape[:2]
    scale = min(model_width / original_width, model_height / original_height)
    resized_width = int(original_width * scale)
    resized_height = int(original_height * scale)
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    x_offset = (model_width - resized_width) // 2
    y_offset = (model_height - resized_height) // 2
    output = np.full((model_height, model_width, 3), 114, dtype=np.uint8)
    output[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return output, LetterboxTransform(
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
    """解析 end-to-end 输出；旧三点模型也只读取 K0。"""

    if kpt_shape not in {(1, 3), (3, 3)}:
        raise ValueError(
            f"Pose kpt_shape must be [1, 3] or legacy [3, 3], got {kpt_shape!r}."
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
    valid = detections[detections[:, 4] >= score_threshold]
    if valid.size:
        valid = valid[np.argsort(-valid[:, 4], kind="stable")[:max_detections]]

    parsed: list[ModelDetection] = []
    for row in valid:
        class_value = float(row[5])
        class_id = int(class_value)
        if not np.isfinite(class_value) or class_value != class_id or class_id < 0:
            raise ValueError(f"Invalid model class ID {class_value!r}.")
        box = transform.box_to_original(*[float(value) for value in row[:4]])
        k0_confidence = float(row[8])
        k0 = (
            transform.point_to_original(float(row[6]), float(row[7]))
            if np.isfinite(row[6:9]).all()
            else None
        )
        parsed.append(
            ModelDetection(
                model_class_id=class_id,
                confidence=float(row[4]),
                box=box,
                k0=k0,
                k0_confidence=k0_confidence,
            )
        )
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HailoYolo26PoseBackend:
    """HEF 主干加 ONNX 后处理的同步单帧后端。"""

    def __init__(
        self,
        *,
        hef_path: Path,
        postprocess_onnx_path: Path,
        output_mapping_path: Path,
        model_version: str,
        model_sha256: str,
        class_count: int,
        max_detections: int,
        score_threshold: float,
    ) -> None:
        paths = (hef_path, postprocess_onnx_path, output_mapping_path)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Model asset does not exist: {path}.")
        actual_checksum = _sha256(hef_path)
        if actual_checksum != model_sha256.lower():
            raise ValueError(
                f"HEF checksum mismatch for {hef_path}: expected "
                f"{model_sha256}, got {actual_checksum}."
            )
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
        if self._kpt_shape not in {(1, 3), (3, 3)}:
            raise ValueError(
                "Only [1, 3] or the legacy [3, 3] pose layout is supported."
            )
        tensor_mapping = config.get("output_tensor_mapping")
        if not isinstance(tensor_mapping, dict) or not tensor_mapping:
            raise ValueError("output_tensor_mapping must be a non-empty object.")
        self._tensor_mapping = tensor_mapping

        try:
            import onnxruntime as ort
            from hailo_platform import (
                FormatType,
                HailoSchedulingAlgorithm,
                VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hailo backend requires hailo_platform and onnxruntime."
            ) from exc

        self._model_version = model_version
        self._model_sha256 = actual_checksum
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
            for output in self._infer_model.outputs:
                self._infer_model.output(output.name).set_format_type(
                    FormatType.FLOAT32
                )
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
            self._onnx_session = ort.InferenceSession(str(postprocess_onnx_path))
        except BaseException:
            self.close()
            raise

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def infer(self, image_bgr: np.ndarray) -> list[ModelDetection]:
        if self._closed:
            raise RuntimeError("HailoYolo26PoseBackend is closed.")
        preprocessed, transform = letterbox_bgr(image_bgr, self._model_size)
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
        if self._last_job is not None:
            self._last_job.wait(10_000)
        context = getattr(self, "_configure_context", None)
        if context is not None:
            context.__exit__(None, None, None)
        device = getattr(self, "_device", None)
        if device is not None and hasattr(device, "release"):
            device.release()

    def __enter__(self) -> HailoYolo26PoseBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
