# `camera`：帧源、回放、显示与记录

本包统一真机和离线输入。所有帧源遵循 `FrameSource` 的 `start/read/stop` 生命周期并输出 `CameraFrame`；算法层只依赖这个协议，不直接创建 Picamera2 或外部进程。

## 常用类、函数和命令

| 入口 | 用途 | 关键行为 |
| --- | --- | --- |
| `CameraFrame` | 一帧 BGR 图、序号、采集时间和元数据 | `image_bgr` 只读；时间单位为 ns |
| `FrameSource` | 相机与离线源共同协议 | `read(timeout)` 交付一帧 |
| `Picamera2Source` | 正式采集首选真机源 | 保存曝光、增益、焦点等逐帧元数据 |
| `RpicamSource` | `rpicam-vid` 低开销真机源 | 不承诺完整传感器元数据 |
| `RecordingSource` | 严格回放记录目录 | 验证图片可解码且尺寸一致，保留原序号和时间 |
| `ImageDirectorySource` | 稳定顺序读取图片目录 | 按文件名排序并按 FPS 生成时间 |
| `VideoFileSource` | 读取普通视频 | 使用视频或覆盖 FPS 生成时间 |
| `FrameRecorder` | 有界异步写盘 | 队列满时返回 `False`，不阻塞主链路 |
| `OpenCvFrameViewer` | 可选缩放预览 | `Q/Esc` 返回 `False` |
| `record_session()` | 组合一个 `FrameSource` 和 `FrameRecorder` | 保证异常路径按相机、记录器顺序收尾 |
| `undistort_camera_frame()` | 保留帧身份并附加去畸变元数据 | 供自定义录制主循环复用 |
| `rescue-vision-record` | 配置驱动的正式录制入口 | 自动去畸变并写完整 session |

## 1. 从运行配置装配真机源

```python
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")

# 具体后端只在装配层选择，下游算法仍只接收 FrameSource。
source_class = (
    Picamera2Source
    if config.camera.backend == "picamera2"
    else RpicamSource
)
source = source_class(
    image_size=config.camera.image_size,
    fps=config.camera.fps,
    lens_position=config.camera.lens_position,
)
```

创建 `source` 时尚未打开相机。后续算法只依赖 `FrameSource`，不需要知道
当前选择的是 Picamera2 还是 `rpicam-vid`。

## 2. 打开相机并读取最新帧

以下片段承接前文的 `source`。上下文管理器负责异常路径关闭相机：

```python
with source:
    frame = source.read(timeout=1.0)
    print(frame.sequence, frame.timestamp_ns, frame.image_bgr.shape)
    print(frame.metadata.get("sensor_timestamp_ns"))
```

两个真机源都会在后台持续排空输入，只向调用方交付最新完整帧。处理速度不足
时可能跳过旧序号，但不会积累越来越陈旧的帧。

## 3. 判断帧是否过期

以下片段使用前文读取的 `frame` 和同一份 `config`：

```python
import time

age_ms = frame.age_ns(time.monotonic_ns()) / 1_000_000
if frame.is_stale(
    time.monotonic_ns(),
    config.processing.max_observation_age_ms,
):
    # 下游应保守丢弃过期观测，而不是继续用于定位或决策。
    print(f"stale frame: {age_ms:.1f} ms")
```

## 4. 回放正式记录

```python
from rescue_vision.camera.replay import RecordingSource

with RecordingSource("recordings/session_001") as source:
    while True:
        try:
            frame = source.read()
        except EOFError:
            break

        # 回放帧已经保持 session 声明的 raw_pixel 或
        # undistorted_pixel 身份，不要再次盲目去畸变。
        print(frame.sequence, frame.metadata["image_coordinate_system"])
```

`RecordingSource` 会拒绝记录结构、图片解码、尺寸或坐标元数据错误。
`ImageDirectorySource`、`VideoFileSource` 更适合外部素材导入，不包含
session 配置和统计信息。

## 5. 正式录制命令

日常采集应使用命令，而不是自行拼装 `FrameRecorder`：

```bash
rescue-vision-record \
  --config configs/runtime.yaml \
  --output recordings/session_001 \
  --duration-seconds 10 \
  --display \
  --tag lighting=indoor_bright \
  --tag distance=near \
  --tag occlusion=none \
  --tag motion_blur=low \
  --tag background=official_mat \
  --tag target_pose=upright \
  --tag contact_state=isolated
```

命令会：

1. 按 `camera.backend` 创建真机源；
2. 在 `intrinsics_enabled: true` 时用 `CameraModel` 去畸变；
3. 写入统一 session 格式、`calibration_id`、有效像素比例、填充值和辅助流声明；
4. 使用 `FrameRecorder` 异步编码，正常关闭相机、线程和窗口。

`--display` 展示实际交给记录器的画面，预览缩放不改变保存分辨率；按 `Q/Esc` 正常结束并收尾 session。显示可能降低吞吐，性能门禁应另做一次不带 `--display` 的短录。

## 6. 在程序中使用旁路记录器

只有应用主循环需要同步保留其他状态时才直接使用 `FrameRecorder`。

### 6.1 加载记录所需配置和内参

```python
from pathlib import Path

import cv2
import yaml

from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.camera_model import IMAGE_BORDER_FILL_VALUE
config_path = Path("configs/runtime.yaml").resolve()
config = load_runtime_config(config_path)
camera_model = config.build_camera_model()
if camera_model is None:
    raise RuntimeError("任务目标记录需要启用内参")
```

### 6.2 创建帧源

以下片段承接前文 `config`，只创建对象，尚未打开相机：

```python
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource

source_class = (
    Picamera2Source
    if config.camera.backend == "picamera2"
    else RpicamSource
)
source = source_class(
    image_size=config.camera.image_size,
    fps=config.camera.fps,
    lens_position=config.camera.lens_position,
)
```

### 6.3 创建记录器

构造参数必须与实际提交图像一致。以下片段继续使用前文的 `config_path`、
`config` 和 `camera_model`：

```python
from rescue_vision.camera.recording import FrameRecorder

recorder = FrameRecorder(
    "recordings/session_001",
    image_size=config.camera.image_size,
    # 保存完整配置快照，后续才能复现这次采集。
    config_snapshot=yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    ),
    session_tags={
        "lighting": "indoor_bright",
        "distance": "near",
        "occlusion": "none",
        "motion_blur": "low",
        "background": "official_mat",
        "target_pose": "upright",
        "contact_state": "isolated",
    },
    queue_capacity=config.recording.queue_capacity,
    image_format=config.recording.image_format,
    image_coordinate_system="undistorted_pixel",
    calibration_id=camera_model.calibration.calibration_id,
    valid_pixel_ratio=(
        cv2.countNonZero(camera_model.valid_mask)
        / camera_model.valid_mask.size
    ),
    undistort_fill_value=IMAGE_BORDER_FILL_VALUE,
)
```

### 6.4 去畸变并提交一帧

上下文管理器按异常安全顺序打开和关闭相机、写盘线程。提交帧继续保留原始
采集序号和时间：

```python
from rescue_vision.camera.record_cli import undistort_camera_frame

with source, recorder:
    raw_frame = source.read(timeout=1.0)
    frame_to_store = undistort_camera_frame(
        raw_frame,
        camera_model=camera_model,
    )
    accepted = recorder.record(frame_to_store)
```

`accepted=False` 表示有界队列已满；应用可记录告警，但不能改为无界堆积或阻塞实时感知。完整现场流程和标签要求见 [`docs/数据采集工具使用.md`](../../../docs/数据采集工具使用.md)。

## 文件定位

| 文件 | 用途 |
| --- | --- |
| `frame.py` | `CameraFrame`、`FrameSource` |
| `picamera2_source.py` | 带逐帧元数据的最新帧源 |
| `rpicam_source.py` | 基于 `rpicam-vid` 的最新帧源 |
| `replay.py` | 图片、视频和记录目录回放 |
| `recording.py` | 有界异步记录器 |
| `record_cli.py` | 配置驱动录制工作流 |
| `viewer.py` | 录制与检查命令共用的 OpenCV 查看器 |
