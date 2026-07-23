"""可替换任务目标推理后端协议和无硬件假实现。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from rescue_vision.perception.types import ModelDetection


@runtime_checkable
class InferenceBackend(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def model_sha256(self) -> str: ...

    def infer(self, image_bgr: np.ndarray) -> Sequence[ModelDetection]: ...

    def close(self) -> None: ...


class FakeInferenceBackend:
    """按调用顺序返回合成检测的测试后端。"""

    def __init__(
        self,
        batches: Sequence[Sequence[ModelDetection]],
        *,
        model_version: str = "fake-v1",
        model_sha256: str = "0" * 64,
    ) -> None:
        if not model_version.strip():
            raise ValueError("model_version must be non-empty.")
        self._batches = [tuple(batch) for batch in batches]
        self._index = 0
        self._model_version = model_version
        self._model_sha256 = model_sha256
        self.closed = False

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def infer(self, image_bgr: np.ndarray) -> Sequence[ModelDetection]:
        if self.closed:
            raise RuntimeError("FakeInferenceBackend is closed.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(
                f"image_bgr must have shape (height, width, 3), got {image_bgr.shape}."
            )
        if self._index >= len(self._batches):
            return ()
        batch = self._batches[self._index]
        self._index += 1
        return batch

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeInferenceBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
