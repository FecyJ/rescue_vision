# `geometry`：相机、地面与 BEV 几何

本包是坐标类型、镜头去畸变、去畸变像素到机器人地面以及 BEV 转换的唯一权威。业务模块不得读取标定 JSON 后自行复制矩阵。

## 常用类和函数

| 入口 | 作用 |
| --- | --- |
| `CameraModel.undistort_image()` | 原始整帧转固定 `new_K` 下的全尺寸去畸变图 |
| `CameraModel.undistort_pixel()` / `undistort_pixels()` | `RawPixel` 转 `UndistortedPixel` |
| `CameraModel.valid_mask` | 标记去畸变图中确实来自原图的像素 |
| `GroundProjector.pixel_to_ground()` / `pixels_to_ground()` | 去畸变像素转机器人地面毫米坐标 |
| `GroundProjector.ground_to_pixel()` / `ground_to_pixels()` | 地面点反投影到去畸变图 |
| `GroundProjector.project_robot_point()` / `project_robot_points()` | 机器人系三维点投影到去畸变图 |
| `GroundProjector.pixel_to_horizontal_plane()` | 像素射线与机器人系指定高度平面求交 |
| `GroundProjector.supports_robot_projection` | 地面标定是否包含完整物理相机外参 |
| `GroundProjector.ground_to_bev_pixel()` / `ground_to_bev_pixels()` | 地面点转鸟瞰图像素 |
| `GroundProjector.bev_pixel_to_ground()` / `bev_pixels_to_ground()` | 鸟瞰像素转地面点 |
| `GroundProjector.make_bev_image()` | 按地面映射生成完整 BEV |
| `CameraCalibration.calibration_id` | 内参参数集的非空可读身份；必须与地面映射一致 |

坐标数据结构：

| 类型 | 坐标约定 |
| --- | --- |
| `RawPixel(u, v)` | 原始畸变图；`u` 向右、`v` 向下 |
| `UndistortedPixel(u, v)` | 固定 `new_K` 的去畸变图 |
| `GroundPoint(x, y)` | 机器人地面系；`x` 向前、`y` 向左，单位 mm |
| `RobotPoint3D(x, y, z)` | 机器人三维系；`x` 向前、`y` 向左、`z` 向上，单位 mm |
| `BevPixel(u, v)` | 图像上方为机器人前方，左侧为机器人左方 |
| `FieldPoint(x, y)` | 场地全局点；仅在全局坐标定义明确的模块中使用 |

## 1. 从运行配置装配

实际运行优先通过 `configs/runtime.yaml` 构建对象，而不是在业务代码中写标定路径：

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
if geometry is None:
    raise RuntimeError("此流程需要启用 geometry.intrinsics_enabled")

camera_model = geometry.camera_model
ground_projector = geometry.ground_projector
```

配置装配会检查运行分辨率、标定可用性、相机模型以及地面映射中的
`calibration_id`。
只有内参时 `ground_projector` 合法地为 `None`。下文继续复用这里创建的
`config`、`geometry`、`camera_model` 和 `ground_projector`。

## 2. 去畸变一帧图像

以下片段承接前文配置，但 `raw_frame` 由相机 `FrameSource.read()` 产生。
示例单独展示相机生命周期，离开 `with source` 后帧数据仍可读取：

```python
from rescue_vision.camera.picamera2_source import Picamera2Source
with Picamera2Source(
    image_size=config.camera.image_size,
    fps=config.camera.fps,
    lens_position=config.camera.lens_position,
) as source:
    raw_frame = source.read(timeout=1.0)

# 图像仍对应同一帧号和 timestamp_ns；只改变像素坐标系。
undistorted_bgr = geometry.camera_model.undistort_image(
    raw_frame.image_bgr
)

# 完整 BEV 成本较高，只在确实需要场地结构图时生成。
if geometry.ground_projector is not None:
    bev_bgr = geometry.ground_projector.make_bev_image(undistorted_bgr)
```

不要把去畸变图重新包装成“新的采集帧”并生成新时间戳。下游观测必须继续携带原 `CameraFrame.sequence` 和 `timestamp_ns`。

## 3. 投影 K0 或其他少量地面点

以下片段使用前文的 `ground_projector`，并承接感知模块产生的
`observation`。它的 `K0` 已位于全尺寸 `UndistortedPixel`：

```python
if ground_projector is None:
    raise RuntimeError("runtime.yaml 尚未启用地面映射")

if observation.k0 is not None:
    ground = ground_projector.pixel_to_ground(observation.k0)
    print(f"前方 {ground.x:.0f} mm，左侧 {ground.y:.0f} mm")
```

一帧有多个点时使用 `pixels_to_ground()` 批量转换；空序列会返回空列表。少量目标接触点不要先生成 BEV 再查坐标。

## 4. 投影离地三维点

以下片段承接第 1 节的 `ground_projector`。完整三维投影只在地面标定产物
包含有效物理外参时可用：

```python
from rescue_vision.geometry.types import RobotPoint3D

if ground_projector is None:
    raise RuntimeError("runtime.yaml 尚未启用地面映射")
if not ground_projector.supports_robot_projection:
    raise RuntimeError("当前地面标定缺少完整相机外参，需要重新标定")

apex_robot = RobotPoint3D(x=500.0, y=0.0, z=32.66)
apex_pixel = ground_projector.project_robot_point(apex_robot)

# 已知目标点位于 z=32.66 mm 平面时，反求机器人系三维坐标。
recovered = ground_projector.pixel_to_horizontal_plane(
    apex_pixel,
    z_mm=32.66,
)
```

`pixel_to_ground()` 只适用于 `z=0` 地面点。目标顶部、围栏顶部或其他离地
像素不得强行走地面单应性；必须使用完整外参和已知高度平面。标定 JSON 中的
外参仍只由 `GroundProjector.from_json()` 加载，业务模块不得自行读取矩阵。

## 5. 检查去畸变边缘填充

`CameraModel.undistort_image()` 保持标定分辨率不变，把无法从原图采样的边缘统一填充为 BGR `(114, 114, 114)`，与 YOLO Letterbox 一致。无效位置仍由 `valid_mask == 0` 表示：

```python
import cv2

valid_ratio = (
    cv2.countNonZero(camera_model.valid_mask)
    / camera_model.valid_mask.size
)
```

不自动裁除填充边缘，因为裁剪会改变图像尺寸、主点、检测框和 K0 坐标，并使 `new_K` 与地面映射失配。标注、训练和推理保留同一全尺寸图，不得在无效区标注目标。

填充值在编码前精确为 `114`。JPEG 解码后边界附近可能因有损压缩略有波动；session 中的 `undistort_fill_value: 114` 描述编码前预处理契约。

## 6. 直接加载产物的适用场景

标定工具、资产检查器等尚未加载运行配置的底层程序可以直接使用：

```python
from rescue_vision.geometry.camera_model import CameraModel
from rescue_vision.geometry.ground_projector import GroundProjector

camera_model = CameraModel.from_json("selected_calibration.json")
projector = GroundProjector.from_json(
    "ground_mapping.json",
    camera_calibration=camera_model.calibration,
)
```

业务运行代码仍应优先使用 `AppConfig.build_geometry()`。相机、分辨率、裁剪、焦点、安装位姿或 `new_K` 改变后必须重新验证相应标定；操作步骤见 [`calibration/README.md`](../calibration/README.md)。
