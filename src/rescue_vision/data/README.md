# `data`：采集检查、数据清单与划分

本包把 `FrameRecorder` 会话验收为可用资产，生成严格 JSONL 清单，再按 `recording_id` 整组划分，避免相邻视频帧泄漏到不同集合。

## 常用函数和命令

| 入口 | 用途 |
| --- | --- |
| `inspect_recording()` | 完整回放并统计采集健康信息 |
| `check_requirements()` | 按最小帧数、FPS、丢帧和元数据门限给出失败原因 |
| `PICAMERA2_METADATA` | 正式 Picamera2 会话要求的逐帧元数据名 |
| `build_dataset_records()` | 把一个或多个合格 session 转为数据清单记录 |
| `split_records()` | 按 `recording_id` 确定性分配 train/validation/test |
| `assign_split(group_id, *, seed, train_ratio, validation_ratio)` | 查询单个会话在给定 seed 和比例下的分区 |
| `rescue-vision-check-recording` | 单会话验收及可选回放 |
| `rescue-vision-manifest` | 生成数据集 JSONL |
| `rescue-vision-split` | 写入分区字段并输出分布报告 |

任务数据输入必须满足：

- session 已正常完成且统计与帧清单一致；
- 坐标系为 `undistorted_pixel`；
- 有可读 `calibration_id`、`valid_pixel_ratio` 和 `undistort_fill_value: 114`；
- 包含全部必需分层标签；
- `supervised_manual_motion` 会话包含完整 `manual_motion` 辅助流，且其单调
  时间范围覆盖全部图像；
- 每个图像路径存在且可由后续检查正常解码。

原图 session 只用于标定或诊断，不能进入任务目标 manifest。

## 1. 检查一次正式记录

先使用 `configs/runtime.yaml` 录制，再完整回放并检查一次会话：

```bash
rescue-vision-check-recording \
  recordings/session_001 \
  --report reports/session_001.json \
  --minimum-frames 180 \
  --maximum-drop-ratio 0.05 \
  --minimum-fps-ratio 0.80 \
  --require-picamera2-metadata \
  --display
```

## 2. 从合格记录生成清单

只有上一步通过的会话才能进入 manifest：

```bash
rescue-vision-manifest \
  --dataset-root /data/rescue-targets \
  --output /data/rescue-targets/manifests/rescue-targets-2026-07-23-v1.jsonl \
  /data/rescue-targets/recordings/session_001 \
  /data/rescue-targets/recordings/session_002
```

## 3. 按会话划分数据集

下文输入是前一步生成的 manifest；同一 `recording_id` 整组进入同一集合：

```bash
rescue-vision-split \
  /data/rescue-targets/manifests/rescue-targets-2026-07-23-v1.jsonl \
  --output /data/rescue-targets/manifests/rescue-targets-2026-07-23-v1.split.jsonl \
  --report /data/rescue-targets/reports/rescue-targets-2026-07-23-v1.split.json \
  --seed rescue-targets-2026-07-23-v1
```

`--display` 按采集时间回放；`--playback-speed 2` 可二倍速。按 `Q/Esc` 只关闭窗口，剩余帧仍在后台完成校验。

划分报告始终显式包含 train/validation/test 的样本数、会话数和实际比例。
任一集合为空时 `warnings` 会列出 `empty_split`，CLI 在写完报告后以退出码
1 结束；必须增加独立会话或调整 seed 后重新划分，不能把空测试集当作可用结果。
确定性分配在会话少于约 20 个时方差很大，四类各一个会话不构成可靠划分。

## 4. 在程序中检查记录

```python
from pathlib import Path

from rescue_vision.data.check_recording import (
    PICAMERA2_METADATA,
    check_requirements,
    inspect_recording,
)

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
```

## 5. 在程序中生成和划分记录

以下片段只在前文 `failures` 为空后执行，并继续使用同一个 `recording`：

```python
from rescue_vision.data.build_manifest import build_dataset_records
from rescue_vision.data.split_manifest import split_records

records = build_dataset_records(
    [recording],
    dataset_root=Path("/data/rescue-targets"),
)
split_records_with_name, split_report = split_records(
    records,
    seed="rescue-targets-2026-07-23-v1",
)
```

领域函数返回 Python 对象，不自行写文件；CLI 负责 JSON/JSONL 的读写。这使数据发布脚本可以组合规则，同时让核心校验保持可测试。

## 6. 必需分层标签

```text
lighting, distance, occlusion, motion_blur,
background, target_pose, contact_state
```

缺失信息显式写为 `unknown`，不能省略或留空。同一录像、连拍或相同物理布置必须共享 `recording_id`；不要为了让样本“更随机”而按单帧改组。

完整字段定义见 [`docs/数据集与评测.md`](../../../docs/数据集与评测.md)，现场采集、覆盖维度和排障见 [`docs/数据采集工具使用.md`](../../../docs/数据采集工具使用.md)。
