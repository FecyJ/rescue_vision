from __future__ import annotations
import sys
import re
from pathlib import Path
import numpy as np
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.app.session_log import _begin_time_named_log, _end_time_named_log

def test_overlay_local_preview_copies_frame_and_adds_state_and_reason() -> None:
    from rescue_vision.app.match_observers import _overlay_local_preview_status

    original = np.zeros((120, 640, 3), dtype=np.uint8)
    frame = CameraFrame(sequence=8, timestamp_ns=11, image_bgr=original)

    result = _overlay_local_preview_status(
        frame,
        state_text="transport_forward",
        reason_text="target_locked",
        localization_lines=(
            "position=(+1350,+1200)mm",
            "heading=-90.0deg",
            "error=not_estimated",
        ),
        process_timestamp_ms=1234.5,
    )

    assert result is not None
    assert result.sequence == frame.sequence
    assert result.timestamp_ns == frame.timestamp_ns
    assert np.any(result.image_bgr != 0)
    assert np.any(result.image_bgr[:, 320:] != 0)
    assert np.array_equal(original, np.zeros((120, 640, 3), dtype=np.uint8))


def test_time_named_log_tees_output_and_restores_streams(tmp_path: Path) -> None:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        stream, stdout_before, stderr_before = _begin_time_named_log(tmp_path)
        assert stream is not None
        print("teed-log-line", flush=True)
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        assert re.fullmatch(r"\d{8}_\d{4}\.log", log_files[0].name)
        content = log_files[0].read_text(encoding="utf-8")
        assert "logging to " in content
        assert "teed-log-line" in content
        _end_time_named_log(stream, stdout_before, stderr_before)
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    # 不传目录时不动标准流、不创建任何文件。
    assert _begin_time_named_log(None) == (None, sys.stdout, sys.stderr)
    assert list(tmp_path.glob("*.log")) == log_files
