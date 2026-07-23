# `rescue_vision` Python 包

本目录包含可导入的运行代码。公共对象从所属子包导入，不在顶层 `rescue_vision` 重导出，以便一眼看出依赖属于哪个领域。

## 最简示例

下面用记录目录回放一帧，并按运行配置决定是否去畸变：

```python
from rescue_vision.camera.replay import RecordingSource
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.example.yaml")
geometry = config.build_geometry()

with RecordingSource("recordings/session_001") as source:
    frame = source.read()

image = frame.image_bgr
if geometry is not None:
    image = geometry.camera_model.undistort_image(image)
```

示例配置默认关闭几何，因此 `geometry` 可以为 `None`。比赛运行前启用几何时，必须填写匹配当前相机条件的内参和地面映射路径。

## 子包索引

| 子包 | 责任 | 使用说明 |
| --- | --- | --- |
| `calibration` | 棋盘采集、内参选择、地面映射 | [README](calibration/README.md) |
| `camera` | 帧、真机源、回放、异步记录 | [README](camera/README.md) |
| `config` | 严格运行配置和几何装配 | [README](config/README.md) |
| `data` | 记录转清单及防泄漏划分 | [README](data/README.md) |
| `evaluation` | 离线分类、地面误差和时延报告 | [README](evaluation/README.md) |
| `geometry` | 坐标类型、去畸变、地面和 BEV 转换 | [README](geometry/README.md) |
| `perception` | 任务目标观测、K0 投影和可替换推理后端 | [README](perception/README.md) |

`versioning.py` 是跨数据记录和评测共享的小模块，用于取得带 dirty 状态的 Git 版本，不单独建立子包。

## 使用边界

- 算法依赖 `FrameSource`，不要在算法内部创建具体相机后端。
- 原始像素先经 `CameraModel`，只有 `UndistortedPixel` 才能传给 `GroundProjector`。
- 配置在启动时加载一次；实时循环不重复读文件。
- 当前没有跟踪、定位、世界模型、任务状态机、通信或应用入口；感知只完成任务目标 Pose 契约与后端，正式模型指标仍待 P1C。
