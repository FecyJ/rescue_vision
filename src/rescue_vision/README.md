# `rescue_vision` Python 包

本目录包含可导入的运行代码。公共对象从所属子包导入，不在顶层 `rescue_vision` 重导出；调用处应能直接看出相机、几何、感知或数据依赖来自哪个领域。

## 期望的运行装配

仓库尚未实现最终应用入口，下面是后续主循环应采用的装配方式。配置只在启动时加载一次，具体相机、内参、地面映射、模型和阈值都来自本机的 `configs/runtime.yaml`：

```python
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.config import load_runtime_config
from rescue_vision.perception import TargetPoseDetector

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
backend = config.hailo.build_backend()

if geometry is None:
    raise RuntimeError("任务感知必须在 runtime.yaml 中启用内参")
if backend is None:
    raise RuntimeError("任务感知必须在 runtime.yaml 中启用 Hailo")

# 领域算法依赖统一帧源；只有这里根据配置选择硬件实现。
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

detector = TargetPoseDetector(
    backend,
    class_mapping=config.hailo.model_class_mapping(),
    detection_threshold=config.hailo.detection_threshold,
    semantic_threshold=config.hailo.semantic_threshold,
    k0_threshold=config.hailo.k0_threshold,
    max_observation_age_ms=config.processing.max_observation_age_ms,
    ground_projector=geometry.ground_projector,
)

with source, detector:
    while True:
        raw_frame = source.read(timeout=1.0)
        undistorted_bgr = geometry.camera_model.undistort_image(
            raw_frame.image_bgr
        )
        observations = detector.detect(raw_frame, undistorted_bgr)

        # 后续跟踪、定位和任务状态机只消费 observations，
        # 不应重新解释模型输出或复制标定矩阵。
        for observation in observations:
            print(
                observation.target_class.value,
                observation.ground_point,
                observation.quality,
            )
```

`ground_mapping_enabled: false` 时仍可运行图像检测，但 `observation.ground_point` 为 `None`。相机、Hailo 和窗口都必须通过上下文管理或 `try/finally` 释放。

## 子包与主要入口

| 子包 | 常用入口 | 对接责任 |
| --- | --- | --- |
| [`config`](config/README.md) | `load_runtime_config()`、`AppConfig.build_geometry()`、`HailoConfig.build_backend()` | 启动时严格加载和装配 |
| [`camera`](camera/README.md) | `FrameSource`、`CameraFrame`、`Picamera2Source`、`RecordingSource` | 产生带时间和序号的最新帧 |
| [`geometry`](geometry/README.md) | `CameraModel`、`GroundProjector`、显式坐标类型 | 去畸变及像素/地面/BEV 转换 |
| [`perception`](perception/README.md) | `InferenceBackend`、`TargetPoseDetector`、`TargetObservation` | 模型结果转任务目标观测 |
| [`calibration`](calibration/README.md) | 三个 `python -m` 标定命令 | 生成内参与地面映射产物 |
| [`data`](data/README.md) | `inspect_recording()`、`build_dataset_records()`、`split_records()` | 采集验收、清单和防泄漏划分 |
| [`evaluation`](evaluation/README.md) | `observations_to_evaluation_records()`、`evaluate_records()` | 目标匹配适配和离线指标 |

`versioning.py` 提供 `git_version()`，用于把当前提交及 dirty 状态写入记录和评测产物，不单独建立子包。

## 对接不变量

- `CameraFrame.image_bgr` 是相机原始帧；任务模型消费 `CameraModel` 产生的全尺寸去畸变图。
- 原始像素使用 `RawPixel`，去畸变像素使用 `UndistortedPixel`，只有后者能交给 `GroundProjector`。
- 检测器只产生 `TargetObservation`，不持有跟踪、定位、世界模型或规则状态。
- 实时循环只处理最新帧；录制、显示、日志和通信使用有界旁路。
- 当前正式任务目标模型、跟踪、定位、世界模型、任务状态机、通信和最终应用入口尚未完成。
