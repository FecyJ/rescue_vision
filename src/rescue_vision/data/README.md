# `data`：采集检查、数据清单与划分

本包把 `FrameRecorder` 会话验收为可用资产，生成严格 JSONL 清单，再按 `recording_id` 整组划分，避免相邻视频帧泄漏到不同集合。

## 常用函数和命令

| 入口 | 用途 |
| --- | --- |
| `inspect_recording()` | 完整回放、验证哈希和统计采集健康信息 |
| `check_requirements()` | 按最小帧数、FPS、丢帧和元数据门限给出失败原因 |
| `PICAMERA2_METADATA` | 正式 Picamera2 会话要求的逐帧元数据名 |
| `build_dataset_records()` | 把一个或多个合格 session 转为数据清单记录 |
| `split_records()` | 按 `recording_id` 确定性分配 train/validation/test |
| `assign_split()` | 查询单个会话在给定 seed 下的分区 |
| `rescue-vision-check-recording` | 单会话验收及可选回放 |
| `rescue-vision-manifest` | 生成 schema v2 数据集 JSONL |
| `rescue-vision-split` | 写入分区字段并输出分布报告 |

任务数据输入必须满足：

- session 已正常完成且统计与帧清单一致；
- 坐标系为 `undistorted_pixel`；
- 有内参指纹、`valid_pixel_ratio` 和 `undistort_fill_value: 114`；
- 包含全部必需分层标签；
- 默认逐图验证 SHA-256。

原图 session 只用于标定或诊断，不能进入任务目标 manifest。

## 推荐的命令工作流

先使用 `configs/runtime.yaml` 录制，再依次检查、生成清单和划分：

```bash
# 1. 完整回放并验收一次会话；--display 只用于目视检查。
rescue-vision-check-recording \
  recordings/session_001 \
  --report reports/session_001.json \
  --minimum-frames 180 \
  --maximum-drop-ratio 0.05 \
  --minimum-fps-ratio 0.80 \
  --require-picamera2-metadata \
  --display

# 2. 把明确列出的合格会话组成一个版本化数据集。
rescue-vision-manifest \
  --dataset-root /data/rescue-targets \
  --dataset-version rescue-targets-2026-07-23-v1 \
  --output /data/rescue-targets/manifests/rescue-targets-2026-07-23-v1.jsonl \
  /data/rescue-targets/recordings/session_001 \
  /data/rescue-targets/recordings/session_002

# 3. 同一 recording_id 整组进入同一个集合。
rescue-vision-split \
  /data/rescue-targets/manifests/rescue-targets-2026-07-23-v1.jsonl \
  --output /data/rescue-targets/manifests/rescue-targets-2026-07-23-v1.split.jsonl \
  --report /data/rescue-targets/reports/rescue-targets-2026-07-23-v1.split.json \
  --seed rescue-targets-2026-07-23-v1
```

`--display` 按采集时间回放；`--playback-speed 2` 可二倍速。按 `Q/Esc` 只关闭窗口，剩余帧仍在后台完成校验。

发布数据版本时不能使用 `--skip-image-verification`。该选项只允许在已知资产未变化的本地迭代中临时加速。

## 在程序中组合数据流程

```python
from pathlib import Path

from rescue_vision.data.build_manifest import build_dataset_records
from rescue_vision.data.check_recording import (
    PICAMERA2_METADATA,
    check_requirements,
    inspect_recording,
)
from rescue_vision.data.split_manifest import split_records

recording = Path("/data/rescue-targets/recordings/session_001")

# inspect_recording() 会读取全部图片，而不是只相信 session.json。
health = inspect_recording(recording)
failures = check_requirements(
    health,
    minimum_frames=180,
    maximum_drop_ratio=0.05,
    minimum_fps_ratio=0.80,
    required_metadata=PICAMERA2_METADATA,
    require_undistorted=True,
)
if failures:
    raise RuntimeError(f"recording rejected: {failures}")

records = build_dataset_records(
    [recording],
    dataset_root=Path("/data/rescue-targets"),
    dataset_version="rescue-targets-2026-07-23-v1",
)
split_records_with_name, split_report = split_records(
    records,
    seed="rescue-targets-2026-07-23-v1",
)
```

领域函数返回 Python 对象，不自行写文件；CLI 负责 JSON/JSONL 的读写。这使数据发布脚本可以组合规则，同时让核心校验保持可测试。

## 必需分层标签

```text
lighting, distance, occlusion, motion_blur,
background, target_pose, contact_state
```

缺失信息显式写为 `unknown`，不能省略或留空。同一录像、连拍或相同物理布置必须共享 `recording_id`；不要为了让样本“更随机”而按单帧改组。

完整字段定义见 [`docs/数据集与评测.md`](../../../docs/数据集与评测.md)，现场采集、覆盖维度和排障见 [`docs/数据采集工具使用.md`](../../../docs/数据采集工具使用.md)。
