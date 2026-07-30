"""跨 Python 3.10+ 保留异常清理上下文。"""

from __future__ import annotations


def add_exception_note(error: BaseException, note: str) -> None:
    """使用 3.11 ``add_note``，并在 3.10 保存同形 ``__notes__``。"""

    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
        return
    notes = getattr(error, "__notes__", None)
    if notes is None:
        error.__notes__ = [note]
    else:
        notes.append(note)
