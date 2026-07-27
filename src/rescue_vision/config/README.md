# `config`：运行配置与对象装配

本包是运行参数的唯一入口。`load_runtime_config()` 加载 schema v7 YAML，拒绝缺失字段、未知字段、错误类型和不一致资产；相对路径以 YAML 所在目录为基准。

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
| `UartConfig` | 设备名、波特率、读写超时、有界接收容量和最大行长度 |
| `RemoteConfig` | 服务端/客户端、观察/调试权限、连接/IO 超时和有界队列 |
| `TrackingConfig` | 关联、确认、滑行、衰减和删除阈值 |
| `WorldRuntimeConfig` | `WorldModelConfig` 与 `StaticRegion` 集合 |
| `MissionConfig` | 比赛计时、安全超时、避让距离和目标优先级 |
| `HailoConfig` | 模型资产、身份、类别映射和阈值 |
| `RuntimeGeometry` | `camera_model`、可选 `ground_projector` |

## 启动时的典型用法

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")

# 设备名和通信参数只来自运行配置；build_channel() 尚不打开设备。
uart_channel = config.uart.build_channel()

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

# 纯逻辑对象同样只装配一次；每轮结束后显式 reset，
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

阈值分为三层：`backend_score_threshold` 是后端粗筛，必须小于等于
`detection_threshold`；`detection_threshold` 决定是否形成观测；
`semantic_threshold` 决定是否保留模型类别，低于它时保守降级为
`unknown`。`k0_threshold` 独立控制地面接触点是否可用。

## UART 装配

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

## 远程通信装配

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

- 当前运行配置为 schema v7。
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
