# `evaluation`：离线评测

本包评测已经完成目标匹配的逐对象记录，输出分类别 precision/recall/F1、混淆矩阵、地面点误差、端到端时延和失败样例。

## 最简示例

```python
from rescue_vision.evaluation import evaluate_records

report = evaluate_records(
    [{
        "schema_version": 1,
        "sample_id": "session_001/frame_00000001",
        "object_id": "target_01",
        "ground_truth_class": "hazard",
        "predicted_class": "hazard",
        "confidence": 0.95,
    }],
    model_version="baseline-v1",
    dataset_version="rescue-targets-v1",
    code_version="git-sha",
)

print(report["per_class"]["hazard"]["recall"])
```

## 命令行用法

```bash
rescue-vision-evaluate evaluation.jsonl \
  --output report.json \
  --model-version baseline-v1 \
  --dataset-version rescue-targets-v1
```

未指定 `--code-version` 时自动记录当前 Git 提交和 dirty 状态。

## 输入语义

- `ground_truth_class: null`：误检。
- `predicted_class: null`：漏检。
- 二者不能同时为 `null`。
- 地面真值和预测同时存在时统计毫米误差。
- 捕获与结果时间同时存在时统计每帧端到端时延。

目标匹配不在本包内猜测，应由检测器评测适配器提前完成。危险类必须单独审阅，不能只看总体平均值。完整 JSONL schema 见 `docs/数据集与评测.md`。
