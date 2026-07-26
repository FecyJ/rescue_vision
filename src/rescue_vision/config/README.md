# `config`：运行配置与对象装配

本包是运行参数的唯一入口。`load_runtime_config()` 加载 schema v4 YAML，拒绝缺失字段、未知字段、错误类型和不一致资产；相对路径以 YAML 所在目录为基准。

本机运行统一读取不提交的 `configs/runtime.yaml`。`configs/runtime.example.yaml` 只用于创建新配置：

```bash
cp configs/runtime.example.yaml configs/runtime.yaml
```

## 常用类和函数

| 入口 | 用途 | 重要返回语义 |
| --- | --- | --- |
| `load_runtime_config(path)` | 严格读取一次 YAML | 返回不可变 `AppConfig` |
| `AppConfig.build_camera_model()` | 按内参开关创建 `CameraModel` | 内参关闭时返回 `None` |
| `AppConfig.build_geometry()` | 创建相机模型和可选地面映射 | 内参关闭时返回 `None`；只有内参时 projector 为 `None` |
| `HailoConfig.build_backend()` | 校验模型资产并创建 Hailo 后端 | Hailo 关闭时返回 `None` |
| `HailoConfig.model_class_mapping()` | 把模型 class ID 映射为 `TargetClass` | 直接传给 `TargetPoseDetector` |
| `TrackingConfig.build_tracker()` | 创建一轮使用的多目标跟踪器 | 初始无轨迹 |
| `WorldRuntimeConfig.build_model()` | 使用静态区域和危险阈值创建世界模型 | 区域可以暂时为空 |
| `MissionConfig.build_state_machine()` | 创建一轮使用的规则状态机 | 初始为 `WAIT_START` |

主要配置 dataclass：

| 类 | 常用字段 |
| --- | --- |
| `CameraConfig` | `backend`、`image_size`、`fps`、`lens_position` |
| `GeometryConfig` | 内参与地面映射各自的开关和路径 |
| `RecordingConfig` | `queue_capacity`、`image_format` |
| `ProcessingConfig` | `max_observation_age_ms` |
| `TrackingConfig` | 关联、确认、滑行、衰减和删除阈值 |
| `WorldRuntimeConfig` | `WorldModelConfig` 与 `StaticRegion` 集合 |
| `MissionConfig` | 比赛计时、安全超时、避让距离和目标优先级 |
| `HailoConfig` | 模型资产、身份、类别映射和阈值 |
| `RuntimeGeometry` | `camera_model`、可选 `ground_projector` |

## 启动时的典型用法

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")

# build_geometry() 会同时检查运行分辨率、标定可用性、
# 相机模型和地面映射中的内参指纹。
geometry = config.build_geometry()
if geometry is None:
    raise RuntimeError("当前功能需要在 runtime.yaml 中启用内参")

camera_model = geometry.camera_model
ground_projector = geometry.ground_projector  # 尚无地面映射时为 None

# 只有真正需要推理时才创建 Hailo 设备。
backend = config.hailo.build_backend()
if backend is None:
    raise RuntimeError("当前功能需要在 runtime.yaml 中启用 Hailo")

class_mapping = config.hailo.model_class_mapping()

# P1 纯逻辑对象同样只装配一次；每轮结束后显式 reset，
# 或为下一轮创建新对象，不能在逐帧循环中反复构造。
tracker = config.tracking.build_tracker()
world_model = config.world.build_model()
mission = config.mission.build_state_machine()
```

配置对象和已构建对象应在进程生命周期内复用，不能在逐帧循环中重新读取 YAML、重新生成去畸变映射或重复创建 Hailo 设备。

## 几何开关组合

```yaml
geometry:
  intrinsics_enabled: true
  intrinsics_path: ../src/rescue_vision/calibration/output/内参目录/selected_calibration.json
  ground_mapping_enabled: false
  ground_mapping_path: null
```

| 内参 | 地面映射 | 结果 |
| --- | --- | --- |
| 关 | 关 | 原图采集或非几何诊断；`build_geometry()` 返回 `None` |
| 开 | 关 | 可去畸变、录制目标数据和运行图像检测 |
| 开 | 开 | 可进一步把 K0 投影到机器人地面毫米坐标 |
| 关 | 开 | 非法配置，加载阶段直接拒绝 |

正式四类目标采集和感知必须启用内参。地面映射尚未完成时保持关闭，不应伪造路径或绕过指纹校验。

## Hailo 装配

`hailo.enabled: false` 时，导入配置和运行无硬件测试不会导入 HailoRT。启用后，`build_backend()` 才会：

1. 检查 HEF、ONNX 后处理和张量映射文件；
2. 核对 HEF SHA-256；
3. 使用 `raw_classes` 数量和推理阈值创建后端；
4. 打开 Hailo 设备资源。

后端必须由调用方 `close()`，通常交给 `TargetPoseDetector` 的上下文管理统一释放。

## schema 与路径

- 当前运行配置为 schema v4。
- schema v3 升级到 v4 时必须增加完整的 `tracking`、`world` 和
  `mission` 段；不会静默套用比赛安全默认值。
- 更旧 schema 的 `geometry.enabled` 不会被静默兼容，应拆成两个独立开关。
- 位于 `configs/` 的 YAML 指向仓库根目录资产时通常以 `../` 开头。
- 路径、类别、阈值和模型哈希只在配置中维护，不在业务模块再次硬编码。

新增配置字段时必须同步严格校验、`configs/runtime.example.yaml`、无硬件测试和受影响模块 README。
