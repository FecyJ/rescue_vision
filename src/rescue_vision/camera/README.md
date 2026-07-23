# `camera`：帧源、回放与记录

本包统一真机和离线输入。所有帧源实现 `FrameSource` 的 `start/read/stop` 生命周期，输出只读的 BGR `CameraFrame`。

## 最简示例：离线读取一张图

```python
from rescue_vision.camera.replay import ImageDirectorySource

with ImageDirectorySource("images", fps=20.0) as source:
    frame = source.read()

print(frame.sequence, frame.timestamp_ns, frame.image_bgr.shape)
```

图片按文件名稳定排序；读完后 `read()` 抛出 `EOFError`。需要修改图像时先调用 `frame.image_bgr.copy()`。

## 真机最新帧

需要逐帧曝光、增益和传感器时间时使用 Picamera2：

```python
from rescue_vision.camera.picamera2_source import Picamera2Source

with Picamera2Source(image_size=(2304, 1296), fps=20) as source:
    frame = source.read(timeout=1.0)
    print(frame.metadata.get("sensor_timestamp_ns"))
```

只需要低开销 BGR 帧时可以使用 `rpicam-vid` 后端：

```python
from rescue_vision.camera.rpicam_source import RpicamSource

with RpicamSource(image_size=(2304, 1296), fps=20) as source:
    frame = source.read(timeout=1.0)
```

两个真机源都会持续排空输入并只保存最新帧；处理速度低于相机帧率时会跳过旧帧，不会积累延迟。

## 回放已有数据

```python
from rescue_vision.camera.replay import RecordingSource, VideoFileSource

with RecordingSource("recordings/session_001") as source:
    recorded_frame = source.read()

with VideoFileSource("clip.mp4", fps_override=20.0) as source:
    video_frame = source.read()
```

`RecordingSource` 保留原序号、时间戳和元数据，并验证图像 SHA-256；视频和图片目录使用帧序生成确定性时间戳。

## 异步记录

日常采集优先使用命令行入口：

```bash
rescue-vision-record \
  --config configs/runtime.example.yaml \
  --output recordings/session_001 \
  --frames 100
```

`rescue-vision-record` 会读取 schema v3 几何配置：

- `intrinsics_enabled: true` 时，写盘前使用 `CameraModel` 去畸变，记录标记为 `undistorted_pixel`；
- `intrinsics_enabled: false` 时保存相机原图，记录标记为 `raw_pixel`；
- `ground_mapping_enabled` 不参与录制，可在尚无地面映射时保持关闭。

任务目标标注、训练和推理统一使用去畸变图，因此正式数据采集必须启用经过验收且与当前分辨率、焦点匹配的内参。去畸变无效边缘统一填充为与 YOLO Letterbox 相同的 BGR `(114, 114, 114)`，并在 session schema v3 的 `undistort_fill_value` 中记录。原图模式只用于标定或诊断，不能生成任务目标 manifest。

程序内也可以旁路提交帧：

```python
from rescue_vision.camera.recording import FrameRecorder

with FrameRecorder(
    "recordings/session_001",
    image_size=source.image_size,
    config_snapshot={},
    versions={"code": "dev"},
) as recorder:
    recorder.record(frame)  # False 表示有界队列已满
```

输出目录必须为空。编码和写盘在线程中完成，队列满时丢弃记录请求并累计 `dropped_frames`，不会阻塞实时主链路。

采集模板默认使用质量 95 的 JPEG；2304×1296、20 FPS 下应先以短录检查确认存储吞吐。PNG 只有在降低帧率并通过同样检查后再使用。

真机短录后使用 `rescue-vision-check-recording` 验证全部图片哈希、有效帧率、丢帧率和元数据覆盖。完整采集流程见仓库的 [`docs/数据采集工具使用.md`](../../../docs/数据采集工具使用.md)。

## 文件定位

| 文件 | 用途 |
| --- | --- |
| `frame.py` | `CameraFrame`、`FrameSource`、元数据值类型 |
| `picamera2_source.py` | 带逐帧传感器元数据的真机源 |
| `rpicam_source.py` | 基于 `rpicam-vid` 的真机源 |
| `replay.py` | 图片目录、视频和记录目录回放 |
| `recording.py` | 有界异步记录器 |
| `record_cli.py` | `rescue-vision-record` 入口 |
