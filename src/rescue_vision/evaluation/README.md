# `evaluation`：任务目标离线评测

本包评测已经完成目标匹配的逐对象记录，输出分类别 precision/recall/F1、混淆矩阵、地面点误差、端到端时延和失败样例。它不读取模型输出张量，也不自行猜测真值与预测的对应关系。

## 常用函数和命令

| 入口 | 用途 |
| --- | --- |
| `TargetAnnotation` | 当前 sample 的人工类别、框和可选地面真值 |
| `observations_to_evaluation_records()` | 按框 IoU 将 `TargetObservation` 与 `TargetAnnotation` 一对一匹配 |
| `evaluate_records()` | 校验 schema v1 逐对象记录并计算完整报告 |
| `rescue-vision-evaluate` | 从 JSONL 读取记录并写 JSON 报告 |
| `git_version()` | CLI 未指定代码版本时记录当前提交和 dirty 状态 |

`observations_to_evaluation_records()` 位于 `perception/evaluation_adapter.py`，因为它理解观测与框；指标计算位于本包。

## 推荐命令

评测流水线应先生成版本化的逐对象 `evaluation.jsonl`，然后执行：

```bash
rescue-vision-evaluate \
  reports/evaluation.jsonl \
  --output reports/evaluation-report.json \
  --model-version target-pose-v2 \
  --dataset-version rescue-targets-2026-07-23-v1
```

未传 `--code-version` 时自动记录当前 Git 提交和 dirty 状态。发布报告前应确保工作区状态、模型 HEF 哈希和数据集版本都能追溯。

## 在评测流水线中调用

```python
import json
from pathlib import Path

from rescue_vision.config import load_runtime_config
from rescue_vision.evaluation import evaluate_records
from rescue_vision.versioning import git_version

config = load_runtime_config("configs/runtime.yaml")
if config.hailo.model_version is None:
    raise RuntimeError("runtime.yaml 未声明待评测模型版本")

records_path = Path("reports/evaluation.jsonl")
records = [
    json.loads(line)
    for line in records_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

report = evaluate_records(
    records,
    model_version=config.hailo.model_version,
    dataset_version="rescue-targets-2026-07-23-v1",
    code_version=git_version(),
)

Path("reports/evaluation-report.json").write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)
```

这里读取的是实际评测 JSONL，而不是为调用函数临时构造几条理想记录。数据版本应与生成这些记录时使用的 manifest 一致。

## 从观测生成逐对象记录

以下片段位于已经加载当前 `frame`、`sample_id/sample_tags`、人工 `annotations` 和模型 `observations` 的评测循环中：

```python
from time import monotonic_ns

from rescue_vision.perception.evaluation_adapter import (
    TargetAnnotation,
    observations_to_evaluation_records,
)

# annotations 来自当前 sample 的正式人工标注；
# observations 来自 TargetPoseDetector 对同一帧的输出。
result_timestamp_ns = (
    observations[0].result_timestamp_ns
    if observations
    else monotonic_ns()
)
evaluation_records = observations_to_evaluation_records(
    sample_id=sample_id,
    annotations=annotations,
    observations=observations,
    capture_timestamp_ns=frame.timestamp_ns,
    result_timestamp_ns=result_timestamp_ns,
    iou_threshold=0.5,
    tags=sample_tags,
)
```

调用方必须保证全部观测属于同一帧。无预测时，适配器为每个真值生成漏检；无真值但有预测时生成误检。若正式评测需要比“类别无关 IoU 贪心匹配”更复杂的关联规则，应修改适配器并版本化，而不是在报告阶段暗中改匹配。

## 输入与输出语义

- `ground_truth_class: null` 表示误检。
- `predicted_class: null` 表示漏检。
- 二者不能同时为 `null`。
- 真值和预测地面点同时存在时才统计毫米误差。
- 捕获时间和结果时间同时存在时按 sample 统计端到端时延。
- 同一 `(sample_id, object_id)` 不能重复。

报告中的 `per_class` 必须逐类审阅，危险类 `blue_danger` 单列。总体平均值不能掩盖危险目标漏检；`failures` 中的样例应回到原图、标注和模型输出联合复盘。

完整 JSONL schema 见 [`docs/数据集与评测.md`](../../../docs/数据集与评测.md)。
