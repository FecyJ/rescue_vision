from __future__ import annotations

import json
import math
from types import SimpleNamespace

from manual_tests.cross_localization import (
    _camera_frames,
    _frames,
    _localization_record,
    _prior_pose,
)
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import (
    CenterCrossPoseCandidate,
    CenterCrossPoseObservation,
    CenterCrossSelectionSource,
    FieldPose2D,
)


def test_manual_cross_localization_serializes_selected_pose() -> None:
    pose = FieldPose2D(FieldPoint(10.0, -20.0), math.pi / 2.0)
    candidate = CenterCrossPoseCandidate(
        pose,
        quarter_turn_index=0,
        position_uncertainty_mm=20.0,
        heading_uncertainty_rad=math.radians(3.0),
    )
    candidates = tuple(
        CenterCrossPoseCandidate(
            FieldPose2D(FieldPoint(float(index), 0.0), index * math.pi / 2.0),
            quarter_turn_index=index,
            position_uncertainty_mm=20.0,
            heading_uncertainty_rad=math.radians(3.0),
        )
        for index in range(4)
    )
    observation = CenterCrossPoseObservation(
        frame_sequence=3,
        capture_timestamp_ns=100,
        result_timestamp_ns=120,
        candidates=(candidate,) + candidates[1:],
        terminals=(),
        selected_pose=pose,
        selection_source=CenterCrossSelectionSource.PRIOR,
        confidence=0.8,
        quality=frozenset(),
    )

    record = _localization_record(observation)

    assert record["selected_pose"] == {
        "position_field_mm": [10.0, -20.0],
        "heading_rad": math.pi / 2.0,
        "heading_deg": 90.0,
    }
    assert record["selection_source"] == "prior"
    json.dumps(record, allow_nan=False)


def test_manual_cross_localization_prior_and_image_input(tmp_path) -> None:
    prior = _prior_pose([100.0, -50.0, 180.0])
    assert prior is not None
    assert prior.position == FieldPoint(100.0, -50.0)
    assert prior.heading_rad == math.pi

    import cv2
    import numpy as np

    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(
        str(image_path),
        np.zeros((8, 12, 3), dtype=np.uint8),
    )
    with _frames(image_path) as frames:
        frame = next(frames)
    assert frame.sequence == 0
    assert frame.image_bgr.shape == (8, 12, 3)


def test_manual_cross_localization_camera_context_closes_source() -> None:
    class FakeSource:
        def __init__(self) -> None:
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.exited = True

        def read(self, timeout: float = 1.0):
            assert timeout == 1.0
            return SimpleNamespace(sequence=7)

    source = FakeSource()
    config = SimpleNamespace(camera=SimpleNamespace(backend="picamera2"))
    with _camera_frames(
        config,
        source_factory=lambda _: source,
    ) as frames:
        assert next(frames).sequence == 7
        assert source.entered is True
        assert source.exited is False
    assert source.exited is True
