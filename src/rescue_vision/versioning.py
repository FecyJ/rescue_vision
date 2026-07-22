"""可追溯产物使用的代码版本标识。"""

from __future__ import annotations

import subprocess


def git_version() -> str:
    """返回提交哈希；工作区有修改时附加 ``-dirty``。"""

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = revision.stdout.strip()
    if revision.returncode != 0 or not value:
        return "unknown"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode == 0 and status.stdout.strip():
        return f"{value}-dirty"
    return value
