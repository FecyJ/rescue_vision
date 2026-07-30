# `config`：运行配置与对象装配

本包是运行参数的唯一入口。`load_runtime_config()` 加载 schema v11 YAML，拒绝缺失字段、未知字段、错误类型和不一致资产；相对路径以 YAML 所在目录为基准。

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
| `UartConfig.build_channel()` | 创建协议无关 UART 行通道 | UART 关闭时返回 `None`；创建时尚不打开设备 |
| `RemoteConfig.build_server()` | 创建树莓派直接 TCP 服务端 | 远程关闭时返回 `None`；创建时尚不监听 |
| `RemoteConfig.connect_client()` | 仓库内参考客户端建立直接 TCP 连接 | 只用于互操作/人工检查；独立电脑端不得依赖 |
| `MotionRuntimeConfig.build_controller()` | 用已有 UART 通道创建运动控制器 | motion 关闭时返回 `None`；不打开 UART |
| `MotionRuntimeConfig.build_remote_executor()` | 创建远程调试运动执行器 | 复用同一个运动控制器和限速 |
| `HailoConfig.build_backend()` | 校验模型资产并创建 Hailo 后端 | Hailo 关闭时返回 `None` |
| `HailoConfig.model_class_mapping()` | 把模型 class ID 映射为 `TargetClass` | 直接传给 `TargetPoseDetector` |
| `PerceptionConfig.build_field_feature_detector()` | 创建传统视觉场地特征检测器 | `field_features.enabled=false` 时返回 `None` |
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
| `UartConfig` | 设备名、波特率、读写超时、有界接收容量和最大行长度 |
| `RemoteConfig` | 服务端/客户端、观察/调试权限、连接/IO 超时和有界队列 |
| `MotionRuntimeConfig` | 实测轮距、车体/车轮速度上限、单轮加速度上限和远程命令有效期上限 |
| `TrackingConfig` | 关联、确认、滑行、衰减和删除阈值 |
| `WorldRuntimeConfig` | `WorldModelConfig` 与 `StaticRegion` 集合 |
| `MissionConfig` | 比赛计时、安全超时、避让距离和目标优先级 |
| `PerceptionConfig` | 目标检测/K0、ROI HSV 和静态场地特征参数 |
| `HsvColorClassifierConfig` | 四类 HSV 闭区间、颜色证据门槛和掩码去噪参数 |
| `FieldFeatureConfig` | 场地颜色、区域尺寸、点划线和边界候选阈值 |
| `HailoConfig` | 模型资产、身份、类别映射和后端粗筛阈值 |
| `RuntimeGeometry` | `camera_model`、可选 `ground_projector` |

## 1. 加载运行配置

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
```

`config` 是不可变 `AppConfig`。进程启动时读取一次，后续各装配片段复用它；
不要在逐帧循环中反复解析 YAML。

## 2. 装配通信对象

以下片段承接前文 `config`，用于树莓派服务端进程（`remote.role: server`）。
创建对象不等于打开设备或监听端口：

```python
# 尚未打开 UART。
uart_channel = config.uart.build_channel()

# 尚未绑定 TCP 监听端口。
remote_server = config.remote.build_server()

# motion 复用同一个 UART 通道；禁用时返回 None。
motion_controller = config.motion.build_controller(uart_channel)
motion_executor = config.motion.build_remote_executor(motion_controller)
```

UART/TCP 生命周期和 motion 的停止语义分别见相邻模块 README，不要把
`build_*()` 的成功返回误认为硬件已经连通。

## 3. 装配几何对象

```python
# build_geometry() 会同时检查运行分辨率、标定可用性、
# 相机模型和地面映射中的内参指纹。
geometry = config.build_geometry()
if geometry is None:
    raise RuntimeError("当前功能需要在 runtime.yaml 中启用内参")

camera_model = geometry.camera_model
ground_projector = geometry.ground_projector  # 尚无地面映射时为 None
```

## 4. 装配 Hailo 后端

只有真正需要推理时才创建 Hailo 设备。以下片段仍承接同一个 `config`：

```python
# 只有真正需要推理时才创建 Hailo 设备。
backend = config.hailo.build_backend()
if backend is None:
    raise RuntimeError("当前功能需要在 runtime.yaml 中启用 Hailo")

class_mapping = config.hailo.model_class_mapping()
color_classifier = config.perception.color_classifier
```

`backend` 必须由调用方关闭；正常生产路径通常立即交给
`TargetPoseDetector`，再由检测器上下文统一释放。

## 5. 装配纯逻辑对象

```python
# 纯逻辑对象同样只装配一次；每轮结束后显式 reset，
# 或为下一轮创建新对象，不能在逐帧循环中反复构造。
tracker = config.tracking.build_tracker()
world_model = config.world.build_model()
mission = config.mission.build_state_machine()
```

配置对象和已构建对象应在进程生命周期内复用，不能在逐帧循环中重新读取
YAML、重新生成去畸变映射或重复创建 Hailo 设备。

传统视觉场地检测器同样只装配一次；它不创建 Hailo 或修改世界模型：

```python
field_detector = config.perception.build_field_feature_detector(
    max_observation_age_ms=config.processing.max_observation_age_ms,
    ground_projector=ground_projector,
)
```

配置关闭时返回 `None`。没有地面映射时仍可输出部分图像观测，但不能分配
安全区入口视角左右，也不能把像素伪装成毫米坐标。

## 6. 几何开关组合

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

## 7. Hailo 装配语义

`hailo.enabled: false` 时，导入配置和运行无硬件测试不会导入 HailoRT。启用后，`build_backend()` 才会：

1. 检查 HEF、ONNX 后处理和张量映射文件；
2. 核对 HEF SHA-256；
3. 使用 `raw_classes` 数量和推理阈值创建后端；
4. 打开 Hailo 设备资源。

后端必须由调用方 `close()`，通常交给 `TargetPoseDetector` 的上下文管理统一释放。

`hailo.backend_score_threshold` 是后端粗筛，必须小于等于
`perception.detection_threshold`；后者决定模型框是否进入统一观测。
`perception.k0_threshold` 独立控制接触点是否可用。最终任务类别不再由
模型分类分数决定，而由 `perception.color_classifier` 对检测框 ROI 进行
HSV 分类；颜色覆盖不足或多色歧义时输出 `unknown`。

## 8. ROI HSV 配置

OpenCV HSV 的 H 范围为 `0..179`，S/V 为 `0..255`。每个任务类别可配置一个
或多个闭区间；橘色跨 Hue 边界，因此示例使用两个区间。不同类别的区间必须
互不重叠，避免一个像素被解释为多个任务类别。

```yaml
perception:
  detection_threshold: 0.25
  k0_threshold: 0.50
  color_classifier:
    ranges:
      green_supply:
        - lower: [35, 70, 71]
          upper: [84, 255, 255]
      black_core:
        - lower: [0, 0, 0]
          upper: [179, 255, 70]
      orange_injured:
        - lower: [0, 90, 80]
          upper: [20, 255, 255]
        - lower: [170, 90, 80]
          upper: [179, 255, 255]
      blue_danger:
        - lower: [85, 50, 71]
          upper: [110, 255, 255]
    min_color_fraction: 0.15
    min_color_dominance: 0.70
    min_dominance_margin: 0.20
    morphology_kernel_size: 3
    open_iterations: 1
    close_iterations: 1
    min_component_area_fraction: 0.002
```

这些 HSV 范围由官方命题示意图取样后放宽，只是尚未经过现场验证的启动值。
固定相机、曝光/白平衡、光照和实物后，应通过实拍数据同时校准 HSV 范围、
覆盖率、dominance 和去噪参数；危险类必须单独报告失败样例。

## 9. UART 装配语义

```yaml
uart:
  enabled: true
  device: /dev/serial0
  baudrate: 115200
  read_timeout_ms: 100.0
  write_timeout_ms: 100.0
  receive_queue_capacity: 256
  max_line_bytes: 512
```

`build_channel()` 返回协议无关的 `UartLineChannel`；进入上下文时才导入
PySerial 并打开设备。电机命令、轮速和未来 IMU 报文由后续协议层解释，
不能把类别前缀或字段数塞进运行配置。完整生命周期和故障语义见
[`communication` README](../communication/README.md)。

## 10. 远程通信装配语义

`remote.role: server` 用于树莓派监听，`client` 只用于本仓库参考客户端和
双机人工检查。独立电脑端维护自己的配置和协议实现，不读取本仓库 YAML 或
导入 Python 包。
远程协议有意采用 IP 与端口直接连接，不使用 PSK、认证握手、HMAC 或密钥
文件。`access_mode` 只有：

- `observe_only`：禁止电脑提交控制，适用于正式比赛图传/地图观察；
- `debug_control`：预留运动和录制控制，只用于赛外采集和调试。

完整配置、消息 topic、资源生命周期和双机检查见
[`communication` README](../communication/README.md) 和
[电脑端通信协议](../../../docs/电脑端通信协议.md)。`build_server()` 不打开
网络；`connect_client()` 会立即建立 TCP 连接，因此只能在本仓库参考工具
的装配层调用。

## schema 与路径

- 当前运行配置为 schema v11。
- schema v10 升级到 v11 时，必须在 `perception` 下新增完整的
  `field_features` 段。即使暂不运行也必须保留全部严格字段并设置
  `enabled: false`；不会静默使用场地颜色或尺寸默认值。
- schema v9 升级到 v10 时，必须新增完整的 `perception` 段，并从 `hailo`
  删除 `detection_threshold`、`semantic_threshold` 和 `k0_threshold`。
  `detection_threshold`、`k0_threshold` 移入 `perception`；模型类别只保留为
  诊断证据，最终类别由 ROI HSV 决定。旧字段不会被静默兼容。
- schema v8 升级到 v9 时，`motion` 段必须增加正有限数
  `max_wheel_acceleration_m_s2`。示例值 `0.50` 表示单轮目标速度每秒最多
  变化 0.50 m/s；控制器不会静默使用旧配置或猜测真车安全加速度。
- schema v7 升级到 v8 时必须增加完整的 `motion` 段；默认关闭且轮距为
  `null`，不会猜测机械尺寸、打开 UART 或接受远程运动。
- schema v6 升级到 v7 时，删除 `authentication_key_path` 和
  `handshake_timeout_ms`，增加 `connect_timeout_ms`。这是为了降低赛场
  连接复杂度而有意做出的不兼容简化，不提供旧字段兼容。
- schema v5 升级到 v6 时必须增加完整的 `remote` 段；不会静默开放网络或
  远程控制。未使用时设置 `enabled: false`、`access_mode: observe_only`。
- schema v4 升级时还必须增加完整的 `uart` 段；不会静默猜测设备名
  或打开串口。UART 尚未使用时设置 `enabled: false`、`device: null`。
- schema v3 升级时还必须增加完整的 `tracking`、`world` 和 `mission`
  段；不会静默套用比赛安全默认值。
- 更旧 schema 的 `geometry.enabled` 不会被静默兼容，应拆成两个独立开关。
- 位于 `configs/` 的 YAML 指向仓库根目录资产时通常以 `../` 开头。
- 路径、类别、阈值和模型哈希只在配置中维护，不在业务模块再次硬编码。

新增配置字段时必须同步严格校验、`configs/runtime.example.yaml`、无硬件测试和受影响模块 README。
