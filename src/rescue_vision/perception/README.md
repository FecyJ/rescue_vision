# `perception`：任务目标 Pose 感知

本包只负责把已去畸变图像中的 YOLO Pose 结果转换成统一目标观测，不创建相机、不复制标定矩阵，也不修改跟踪、定位或任务状态。

## 类别与 K0

`TargetClass` 与 [`docs/Pose视觉模型约定.md`](../../../docs/Pose视觉模型约定.md) 的训练标签一致：

| 标签 | 任务语义 |
| --- | --- |
| `green_supply` | 普通物资 |
| `black_core` | 核心物资 |
| `orange_injured` | 伤员 |
| `blue_danger` | 危险目标 |
| `unknown` | 仅用于运行时保守降级，不是第五个训练标签 |

公共契约只包含 `K0 bottom_contact_anchor`。后端仍能读取旧的三关键点部署包，但只消费 K0；新部署包使用 `kpt_shape: [1, 3]`。

## 最简示例

```python
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.types import UndistortedPixel
from rescue_vision.perception import (
    FakeInferenceBackend,
    ModelDetection,
    TargetClass,
    TargetPoseDetector,
    UndistortedBoundingBox,
)

backend = FakeInferenceBackend(
    [[
        ModelDetection(
            model_class_id=0,
            confidence=0.91,
            box=UndistortedBoundingBox(10, 20, 80, 100),
            k0=UndistortedPixel(45, 99),
            k0_confidence=0.88,
        )
    ]]
)
detector = TargetPoseDetector(
    backend,
    class_mapping={0: TargetClass.GREEN_SUPPLY},
    detection_threshold=0.25,
    semantic_threshold=0.50,
    k0_threshold=0.50,
    max_observation_age_ms=150.0,
)
image = np.zeros((120, 160, 3), dtype=np.uint8)
frame = CameraFrame(0, 1_000_000_000, image)
observations = detector.detect(
    frame,
    image,
    result_timestamp_ns=1_020_000_000,
)
detector.close()
```

传入的第二个图像必须是 `CameraModel` 产生的去畸变图像，并保持与标注、训练一致的裁剪、方向和宽高比。若向检测器提供 `GroundProjector`，只有达到 K0 阈值的 `UndistortedPixel` 才会被投影为 `GroundPoint`。

## Hailo 部署包

最小部署包为：

```text
model_bundle/
├── model.hef
├── postprocess.onnx
└── onnx_split_config.json
```

文件名不是代码常量，三条路径分别由 `configs/runtime.yaml` 指定。运行配置还必须给出模型版本、HEF SHA-256、原始类别顺序、到 `TargetClass` 的完整映射及阈值。相对路径以 YAML 所在目录为基准。

最小配置结构：

```yaml
hailo:
  enabled: true
  hef_path: ../models/target_pose/model.hef
  postprocess_onnx_path: ../models/target_pose/postprocess.onnx
  output_mapping_path: ../models/target_pose/onnx_split_config.json
  model_version: target-pose-v2
  hef_sha256: 64位小写十六进制
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

`raw_classes` 的顺序就是模型 class ID；映射键必须完整覆盖该列表。决赛现场若模型标签或顺序改变，只替换部署包和这段配置，不修改任务代码。

Hailo 后端直接使用系统 `hailo_platform` 和 ONNX Runtime，具体版本必须以部署包清单为准。模块导入和假后端不会导入 HailoRT；只有调用 `config.hailo.build_backend()` 才创建 VDevice。使用上下文管理或 `try/finally` 调用 `close()`，确保异步作业和设备上下文在异常路径也释放。

真机单图检查：

```bash
python manual_tests/hailo_pose.py \
  --config configs/runtime.yaml \
  --undistorted-image path/to/undistorted.png
```

该脚本不替代路线图 P2 的目标硬件持续运行、观测年龄和温度验收。
