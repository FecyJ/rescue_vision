from __future__ import annotations

import numpy as np
import pytest

import rescue_vision.camera.viewer as viewer_module
from rescue_vision.camera.viewer import OpenCvFrameViewer, playback_delay_ms


def test_viewer_resizes_once_and_handles_quit(monkeypatch) -> None:
    calls: list[tuple] = []
    keys = iter((-1, ord("q")))
    monkeypatch.setattr(
        viewer_module.cv2,
        "namedWindow",
        lambda *args: calls.append(("named", *args)),
    )
    monkeypatch.setattr(
        viewer_module.cv2,
        "resizeWindow",
        lambda *args: calls.append(("resize", *args)),
    )
    monkeypatch.setattr(
        viewer_module.cv2,
        "imshow",
        lambda *args: calls.append(("show", *args)),
    )
    monkeypatch.setattr(
        viewer_module.cv2,
        "waitKey",
        lambda delay: calls.append(("wait", delay)) or next(keys),
    )
    monkeypatch.setattr(
        viewer_module.cv2,
        "destroyWindow",
        lambda *args: calls.append(("destroy", *args)),
    )

    viewer = OpenCvFrameViewer("capture", maximum_size=(100, 100))
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert viewer.show(image, delay_ms=5)
    assert not viewer.show(image, delay_ms=10)
    viewer.close()
    viewer.close()

    assert [call[0] for call in calls].count("named") == 1
    assert ("resize", "capture", 100, 50) in calls
    assert ("wait", 5) in calls
    assert ("wait", 10) in calls
    assert [call[0] for call in calls].count("destroy") == 1


def test_viewer_and_playback_timing_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="maximum_size"):
        OpenCvFrameViewer("capture", maximum_size=(0, 720))
    viewer = OpenCvFrameViewer("capture")
    with pytest.raises(ValueError, match="3 BGR channels"):
        viewer.show(np.zeros((10, 10, 4), dtype=np.uint8))
    assert playback_delay_ms(None, 100, speed=1.0) == 1
    assert playback_delay_ms(100, 50_000_100, speed=2.0) == 25
    assert playback_delay_ms(0, 10_000_000_000, speed=1.0) == 1000
    with pytest.raises(ValueError, match="positive and finite"):
        playback_delay_ms(0, 1, speed=0.0)
