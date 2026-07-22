"""离线评测命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rescue_vision.evaluation.report import evaluate_records
from rescue_vision.versioning import git_version


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate matched object records and write a JSON report."
    )
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--code-version", default=None)
    args = parser.parse_args()

    report = evaluate_records(
        _read_jsonl(args.records),
        model_version=args.model_version,
        dataset_version=args.dataset_version,
        code_version=args.code_version or git_version(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
