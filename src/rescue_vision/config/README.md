# `config`：运行配置

本包加载 schema v2 YAML，并在启动阶段校验未知字段、类型、尺寸、标定质量、相机模型、内参指纹和 Hailo 模型身份。配置示例位于 `configs/runtime.example.yaml`；本机实际值写入不提交的 `configs/runtime.yaml`。

## 最简示例

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.example.yaml")
print(config.camera.backend, config.camera.image_size)
```

配置对象是不可变 dataclass；运行循环应复用它，不要反复读取 YAML。

## 装配几何对象

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("path/to/runtime.yaml")
geometry = config.build_geometry()

if geometry is not None:
    camera_model = geometry.camera_model
    ground_projector = geometry.ground_projector
```

当 `geometry.enabled: false` 时返回 `None`。启用后必须同时提供内参和地面映射；加载过程会拒绝分辨率、模型或指纹不一致的组合。

相对路径以配置文件所在目录为基准，而不是当前终端目录。若配置文件位于 `configs/`，指向仓库根目录文件通常需要以 `../` 开头。

## 配置分区

| 分区 | 内容 |
| --- | --- |
| `camera` | 后端、`[width, height]`、FPS、固定焦点 |
| `geometry` | 是否启用、内参和地面映射路径 |
| `recording` | 有界队列容量、图像格式 |
| `processing` | 最大观测年龄 |
| `hailo` | 模型资产、版本、HEF 哈希、类别映射和推理阈值 |

`hailo.enabled: false` 时不会导入 HailoRT。启用后调用 `config.hailo.build_backend()` 才检查三个部署资产、HEF SHA-256 和模型输出约定并创建设备。

schema v1 配置不会被静默补默认值；迁移时将 `schema_version` 改为 2，并按模板补齐完整 `hailo` 分区。没有 Hailo 的开发机配置应显式设置 `enabled: false`。

新增配置字段时要提升或兼容 schema、补充严格校验和无硬件测试，并同步 `configs/runtime.example.yaml`。
