# `perception`：任务目标 Pose 感知

本包把全尺寸去畸变图上的模型结果转换为稳定的 `TargetObservation`。它不创建相机、不复制标定矩阵，也不修改跟踪、定位、世界模型或任务状态。

## 常用类和函数

| 入口 | 对接作用 |
| --- | --- |
| `InferenceBackend` | 推理后端协议：`infer()`、模型身份和 `close()` |
| `HailoYolo26PoseBackend` | HEF 推理与 ONNX 后处理的真实后端 |
| `TargetPoseDetector` | 阈值、类别映射、观测年龄和可选 K0 地面投影 |
| `RealtimeDetectionResult` | 实时检测结果，并明确记录是否丢弃了过期帧 |
| `StaleObservationError` | 严格 `detect()` 在结果超过允许年龄时抛出的异常 |
| `ModelDetection` | 后端输出；框和 K0 已反映射到去畸变原尺寸 |
| `TargetObservation` | 下游跟踪、定位和评测消费的统一观测 |
| `TargetClass` | 四类任务目标及运行时 `unknown` |
| `ObservationQuality` | 保留低分类置信度、K0 不可用等降级原因 |
| `UndistortedBoundingBox` | 去畸变图中的 `(left, top, right, bottom)` |
| `TargetAnnotation` | 评测适配器使用的人工目标真值 |
| `observations_to_evaluation_records()` | 观测与人工真值匹配后生成评测记录 |

`FakeInferenceBackend` 只用于无硬件自动测试和上层算法注入，不是部署示例。

## 按运行配置创建检测器

```python
from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.config import load_runtime_config
from rescue_vision.perception import TargetPoseDetector

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
backend = config.hailo.build_backend()

if geometry is None:
    raise RuntimeError("任务感知需要启用内参")
if backend is None:
    raise RuntimeError("任务感知需要启用 Hailo")

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
    raw_frame = source.read(timeout=1.0)
    undistorted_bgr = geometry.camera_model.undistort_image(
        raw_frame.image_bgr
    )
    result = detector.detect_realtime(raw_frame, undistorted_bgr)
    if result.stale_dropped:
        # 记录性能异常即可；过期结果已经被模块清空，不会传给下游。
        print("dropped stale frame:", result.dropped_stale_age_ms)
    observations = result.observations
```

`detect()` 的两个输入必须属于同一采集帧：第一个参数提供原始 `sequence/timestamp_ns`，第二个参数是该帧经当前 `CameraModel` 产生的去畸变图。不能把缓存旧图、裁剪图或另一相机的图像配给当前帧。

实时调用统一使用 `detect_realtime()`：它只兜底
`StaleObservationError`，返回空观测并通过 `stale_dropped` /
`dropped_stale_age_ms` 记录原因。输入尺寸、类别映射、模型或硬件异常仍会抛出。
离线评测和需要严格失败语义的工具可以直接调用 `detect()`。

若 `ground_mapping_enabled: false`，检测仍正常运行，但所有 `ground_point` 为 `None`。后续定位模块不能把 `None` 当作 `(0, 0)`。

## 读取统一观测

```python
from rescue_vision.perception import ObservationQuality, TargetClass

for observation in observations:
    # 低于 semantic_threshold 的检测不会被强行归入某个规则类别，
    # 而是保守输出 UNKNOWN 并附带质量标记。
    if observation.target_class is TargetClass.UNKNOWN:
        print("uncertain target:", observation.box)
        continue

    if ObservationQuality.K0_UNAVAILABLE in observation.quality:
        # 可以保留图像框用于跟踪，但不能用于地面定位。
        print("image-only observation:", observation.box)
        continue

    if observation.ground_point is not None:
        # 这是后续跟踪/定位接口所需的核心信息。
        local_measurement = {
            "target_class": observation.target_class,
            "point_mm": observation.ground_point,
            "capture_timestamp_ns": observation.capture_timestamp_ns,
        }
```

实际下游应重点使用：

| 字段 | 含义 |
| --- | --- |
| `frame_sequence` | 采集帧序号，用于同帧关联 |
| `capture_timestamp_ns` | 相机帧进入应用的单调时间 |
| `result_timestamp_ns` | 推理结果产生时间 |
| `target_class` / `class_probabilities` | 保守类别及概率 |
| `box` | 全尺寸去畸变图中的目标框 |
| `k0` / `k0_confidence` | 目标与地面的接触锚点及置信度 |
| `ground_point` | 可选机器人地面系毫米坐标 |
| `quality` | 不应被静默丢弃的降级原因 |
| `model_version` / `model_sha256` | 结果对应的模型身份 |

## 类别与 K0

| `TargetClass` | 任务语义 |
| --- | --- |
| `GREEN_SUPPLY` / `green_supply` | 普通物资 |
| `BLACK_CORE` / `black_core` | 核心物资 |
| `ORANGE_INJURED` / `orange_injured` | 伤员 |
| `BLUE_DANGER` / `blue_danger` | 危险目标 |
| `UNKNOWN` / `unknown` | 运行时保守降级，不是第五个训练标签 |

公共契约只包含 `K0 bottom_contact_anchor`。后端可以读取旧三关键点部署包，但只消费 K0；新部署包使用 `kpt_shape: [1, 3]`。完整标注和导出约定见 [`docs/Pose视觉模型约定.md`](../../../docs/Pose视觉模型约定.md)。

## Hailo 部署包和配置

部署包包含：

```text
model_bundle/
├── model.hef
├── postprocess.onnx
└── onnx_split_config.json
```

路径、模型版本、HEF SHA-256、原始类别顺序、类别映射和阈值全部来自 `configs/runtime.yaml`：

```yaml
hailo:
  enabled: true
  hef_path: ../models/target_pose/model.hef
  postprocess_onnx_path: ../models/target_pose/postprocess.onnx
  output_mapping_path: ../models/target_pose/onnx_split_config.json
  model_version: target-pose-v2
  hef_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  raw_classes: [green_supply, black_core, orange_injured, blue_danger]
  class_mapping:
    green_supply: green_supply
    black_core: black_core
    orange_injured: orange_injured
    blue_danger: blue_danger
  detection_threshold: 0.25
  semantic_threshold: 0.50
  k0_threshold: 0.50
  max_detections: 100
```

`raw_classes` 的位置就是模型 class ID。`config.hailo.class_mapping` 是内部的顺序元组；创建检测器时必须调用 `config.hailo.model_class_mapping()` 得到整数 ID 映射。`HailoConfig.build_backend()` 会延迟导入 HailoRT 并校验部署资产；`TargetPoseDetector` 从构造开始接管后端，构造校验失败也会关闭它，因此优先使用检测器上下文管理。

真机单图验收命令：

```bash
python manual_tests/hailo_pose.py \
  --config configs/runtime.yaml \
  --undistorted-image recordings/session_001/frames/00000001_时间戳.jpg
```

输入必须已经是与配置内参一致的去畸变图。该命令用于部署连通性检查，不替代正式数据集指标、持续运行时延和温度验收。

实时相机、去畸变、推理和叠加预览：

```bash
python manual_tests/camera_undistort_perception.py
```

窗口中的绿色框为目标框、红点为 K0，启用地面映射后标签会附加机器人地面 `(x, y) mm`。

## 转换为评测输入

人工标注与观测准备好后，通过适配器完成类别无关 IoU 一对一匹配。以下片段位于已取得当前 `raw_frame`、`annotations` 和 `observations` 的评测循环中：

```python
from time import monotonic_ns

from rescue_vision.perception.evaluation_adapter import (
    TargetAnnotation,
    observations_to_evaluation_records,
)

# 非空时使用检测器记录的统一结果时间；空结果时在 detect() 返回后
# 记录当前时间，使漏检样例仍有端到端时延。
result_timestamp_ns = (
    observations[0].result_timestamp_ns
    if observations
    else monotonic_ns()
)
records = observations_to_evaluation_records(
    sample_id="session_001/frame_00000001",
    annotations=annotations,  # list[TargetAnnotation]
    observations=observations,
    capture_timestamp_ns=raw_frame.timestamp_ns,
    result_timestamp_ns=result_timestamp_ns,
    iou_threshold=0.5,
    tags={"lighting": "indoor_bright"},
)
```

生产评测流水线最好在检测编排层显式保存每帧统一的结果时间，避免不同调用方各自推断。
