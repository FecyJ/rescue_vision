# `geometry`：相机与地面几何

本包是坐标类型、镜头去畸变、地面投影和 BEV 的唯一权威实现。调用链固定为 `RawPixel → UndistortedPixel → GroundPoint`。

## 最简示例

```python
from rescue_vision.geometry.camera_model import CameraCalibration, CameraModel
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import RawPixel

calibration = CameraCalibration.from_json("selected_calibration.json")
camera = CameraModel(calibration)
projector = GroundProjector.from_json(
    "ground_mapping.json",
    camera_calibration=calibration,
)

undistorted = camera.undistort_pixel(RawPixel(u=1200.0, v=900.0))
ground = projector.pixel_to_ground(undistorted)
print(ground.x, ground.y)  # mm；x 向前，y 向左
```

`GroundProjector.from_json()` 会校验图像尺寸、相机模型和内参指纹，不能混用不同标定批次。

## 去畸变整帧与生成 BEV

```python
undistorted_image = camera.undistort_image(frame.image_bgr)
bev_image = projector.make_bev_image(undistorted_image)
```

完整 BEV 只在场地结构检测或调试时按需生成。少量目标接触点直接调用 `pixel_to_ground()`，不要先生成 BEV 再查坐标。

## 去畸变黑边

`undistort_image()` 保持标定分辨率不变，无法从原图采样的边缘像素填黑。`camera.valid_mask` 给出有效像素，比例可这样查看：

```python
import cv2

valid_ratio = cv2.countNonZero(camera.valid_mask) / camera.valid_mask.size
```

当前不自动裁黑边，因为裁剪会改变图像尺寸、主点、检测框和 K0 坐标，并使 `new_K` 与地面映射失配。标注、训练和推理都保留同一全尺寸黑边；不得在 `valid_mask == 0` 的区域标注目标。若无效区域过大，应重新选择标定的 `new_K`。未来若引入裁剪，必须记录 ROI、生成裁后 `new_K`，并重做地面映射和数据集版本。

## 坐标类型

| 类型 | 含义 |
| --- | --- |
| `RawPixel` | 原始畸变图像，`u` 右、`v` 下 |
| `UndistortedPixel` | 固定 `new_K` 下的去畸变图像 |
| `GroundPoint` | 机器人地面系，`x` 前、`y` 左，单位 mm |
| `BevPixel` | 上方为前、左侧为左的鸟瞰像素 |
| `FieldPoint` | 场地全局点；全局定义冻结后再作为稳定接口 |

禁止用裸 `(u, v)` 跨模块传递坐标。批量接口包括 `undistort_pixels()`、`pixels_to_ground()`、`ground_to_pixels()`、`ground_to_bev_pixels()` 和 `bev_pixels_to_ground()`；空列表会返回空列表。

## 标定条件

相机、分辨率、裁剪、焦点、安装位姿或 `new_K` 改变后必须重新验证相应标定。标定操作见相邻的 `calibration/README.md`。
