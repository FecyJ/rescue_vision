# `config`：运行配置

本包加载 schema v3 YAML，并在启动阶段校验未知字段、类型、尺寸、标定质量、相机模型、内参指纹和 Hailo 模型身份。配置示例位于 `configs/runtime.example.yaml`；本机实际值写入不提交的 `configs/runtime.yaml`。

## 最简示例

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.example.yaml")
print(config.camera.backend, config.camera.image_size)
```

配置对象是不可变 dataclass；运行循环应复用它，不要反复读取 YAML。

## 独立装配内参与地面映射

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("path/to/runtime.yaml")
camera_model = config.build_camera_model()
geometry = config.build_geometry()

if geometry is not None and geometry.ground_projector is not None:
    ground_projector = geometry.ground_projector
```

配置使用两个开关：

```yaml
geometry:
  intrinsics_enabled: true
  intrinsics_path: ../path/to/selected_calibration.json
  ground_mapping_enabled: false
  ground_mapping_path: null
```

- `intrinsics_enabled: true`：加载 `CameraModel`；录制命令保存去畸变图。
- `ground_mapping_enabled: true`：在内参基础上再加载 `GroundProjector`。
- 地面映射依赖内参，不能在内参关闭时单独启用。
- 只有内参、暂时没有地面映射时，按上例配置即可正常采集。

加载过程会拒绝不可用内参、错误分辨率以及地面映射模型或指纹不一致。`build_geometry()` 在内参关闭时返回 `None`；只启用内参时返回 `RuntimeGeometry(camera_model, ground_projector=None)`。

相对路径以配置文件所在目录为基准，而不是当前终端目录。若配置文件位于 `configs/`，指向仓库根目录文件通常需要以 `../` 开头。

## 配置分区

| 分区 | 内容 |
| --- | --- |
| `camera` | 后端、`[width, height]`、FPS、固定焦点 |
| `geometry` | 内参和地面映射各自的开关与路径 |
| `recording` | 有界队列容量、图像格式 |
| `processing` | 最大观测年龄 |
| `hailo` | 模型资产、版本、HEF 哈希、类别映射和推理阈值 |

`hailo.enabled: false` 时不会导入 HailoRT。启用后调用 `config.hailo.build_backend()` 才检查三个部署资产、HEF SHA-256 和模型输出约定并创建设备。

schema v2 不会被静默兼容。迁移到 v3 时：

- 原 `geometry.enabled: false` 改成两个开关均为 `false`；
- 原 `geometry.enabled: true` 改成两个开关均为 `true`；
- 只有内参时设置 `intrinsics_enabled: true`、`ground_mapping_enabled: false`。

没有 Hailo 的开发机配置应显式设置 `hailo.enabled: false`。

新增配置字段时要提升或兼容 schema、补充严格校验和无硬件测试，并同步 `configs/runtime.example.yaml`。
