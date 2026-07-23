# `data`：数据清单与划分

本包把 `FrameRecorder` 会话转换为严格 JSONL 数据集清单，再按 `recording_id` 整组划分，避免连续视频帧泄漏到不同集合。

## 最简示例

```python
from pathlib import Path

from rescue_vision.data.build_manifest import build_dataset_records
from rescue_vision.data.split_manifest import split_records

records = build_dataset_records(
    [Path("recordings/session_001")],
    dataset_root=Path("recordings"),
    dataset_version="rescue-targets-v1",
)
split_samples, report = split_records(records, seed="rescue-vision-v1")
```

输入会话必须标记为完成，包含全部分层标签，且图像默认通过 SHA-256 验证。

## 命令行用法

```bash
rescue-vision-check-recording recordings/session_001 \
  --report reports/session_001.json \
  --require-picamera2-metadata

rescue-vision-manifest \
  --dataset-root recordings \
  --dataset-version rescue-targets-v1 \
  --output dataset.jsonl \
  recordings/session_001

rescue-vision-split dataset.jsonl \
  --output dataset.split.jsonl \
  --report split-report.json
```

单会话检查会完整回放并验证图像哈希，报告有效帧率、记录旁路丢帧率、序号缺口、元数据覆盖和亮度诊断。它是采集后的快速门禁；正式数据清单仍由 `rescue-vision-manifest` 严格生成。

发布数据版本时不要使用 `--skip-image-verification`。该选项只用于已知数据完整、需要快速本地迭代的场景。

## 必需分层标签

```text
lighting, distance, occlusion, motion_blur,
background, target_pose, contact_state
```

缺失信息显式写为 `unknown`，不能省略或写空字符串。同一录像、连拍或相同物理布置必须共享 `recording_id`。

完整 schema 见仓库的 `docs/数据集与评测.md`；采集命令、现场覆盖和排障统一见 `docs/数据采集工具使用.md`。
