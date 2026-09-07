"""按会话时间记录终端日志并恢复标准流。"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import TextIO

class _TeeStream:
    """把标准流同时写到控制台和日志文件；flush 同步刷新两侧。"""

    def __init__(
        self,
        primary: TextIO,
        secondary: TextIO,
        *,
        line_prefix: str = "",
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._line_prefix = line_prefix
        self._at_line_start = True

    def write(self, text: str) -> int:
        original_count = len(text)
        if self._line_prefix:
            prefixed: list[str] = []
            for part in text.splitlines(keepends=True):
                if self._at_line_start and part:
                    prefixed.append(self._line_prefix)
                prefixed.append(part)
                self._at_line_start = part.endswith(("\n", "\r"))
            if text and not prefixed:
                prefixed.extend((self._line_prefix, text))
            if text and not text.endswith(("\n", "\r")):
                self._at_line_start = False
            text = "".join(prefixed)
        primary_count = self._primary.write(text)
        self._secondary.write(text)
        return original_count if self._line_prefix else primary_count

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()


def _begin_time_named_log(
    log_dir: Path | None,
    *,
    file_prefix: str = "",
    line_prefix: str = "",
) -> tuple[TextIO | None, TextIO, TextIO]:
    """可选地把 stdout/stderr tee 到 ``log_dir/<YYYYmmdd_HHMM>.log``。

    返回 ``(日志流, 原 stdout, 原 stderr)``；``log_dir`` 为 ``None`` 时不动
    标准流并返回 ``(None, sys.stdout, sys.stderr)``。日志文件按行缓冲追加，
    同分钟重跑会继续写入同一文件；进程异常退出时已写入内容仍在文件中。
    ``file_prefix`` 只用于文件名；``line_prefix`` 可选地用于每个输出行的前缀。
    """

    if log_dir is None:
        return None, sys.stdout, sys.stderr
    if not isinstance(log_dir, Path):
        raise TypeError("log_dir must be a Path or None.")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{file_prefix}{time.strftime('%Y%m%d_%H%M')}.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stream = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(
        original_stdout,
        stream,
        line_prefix=line_prefix,
    )
    sys.stderr = _TeeStream(
        original_stderr,
        stream,
        line_prefix=line_prefix,
    )
    print(f"logging to {path}", flush=True)
    return stream, original_stdout, original_stderr


def _end_time_named_log(
    log_stream: TextIO | None,
    original_stdout: TextIO,
    original_stderr: TextIO,
) -> None:
    """恢复标准流并关闭按时间命名的日志文件。"""

    if log_stream is None:
        return
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_stream.close()
