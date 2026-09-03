# `calibration`：相机与地面标定

本页是相机内参与地面映射的唯一操作说明。标定脚本默认使用本目录下的路径，不受终端当前目录影响；现场生成的 `calibration_captures/` 和 `output/` 已被 Git 忽略。运行时结构和坐标边界见[项目结构](../../../docs/项目结构.md)。

## 使用顺序

1. 固定相机、分辨率、裁剪和焦点后采集棋盘图；
2. 用这批图片比较三种模型并选择内参；
3. 固定机器人在场地全局原点，按定位模板把棋盘放到多个站位并拍照；
4. 在 JSON 中记录每张照片的实体参考点全局坐标，运行脚本自动检测标定板角点并求解地面映射；
5. 验收产物后再写入 `configs/runtime.yaml`。

前两步是内参标定，只依赖相机和棋盘；后三步是外参与地面映射标定，必须承接已经
验收的 `selected_calibration.json`，并等相机和机器人安装姿态最终固定后再执行。
不要把不同相机、焦点、分辨率、裁剪或安装条件的产物混用。

## 第一部分：内参标定

内参描述相机自身的投影和镜头畸变，负责 `RawPixel → UndistortedPixel`。这一部分
不使用机器人地面坐标，也不求相机安装姿态。

### 内参命令与产物

| 命令或入口 | 用途 | 主要产物或结果 |
| --- | --- | --- |
| `python -m rescue_vision.calibration.capture_chessboard_images` | 相机条件固定后采集内参样本 | 棋盘原图、检测图、逐帧元数据 |
| `python -m rescue_vision.calibration.calibrate_intrinsics` | 比较三种模型并选择内参 | `selected_calibration.json`、模型比较和诊断图 |
| `CameraModel.from_json()` | 独立加载并检查内参产物 | 默认拒绝 `quality.usable=false` |

### 内参固定条件

同一套内参只适用于相同的相机与镜头个体、`2304 × 1296` 取流分辨率、裁剪模式和
固定 `LensPosition`。改变其中任一条件，或重新选择 `new_K`，都必须重新求解并验证
内参；不能只凭肉眼观察去畸变图判断质量。

### 1.1 采集内参棋盘图

标定板为 12×9 个实体方格、11×8 个内角点。测量实际方格边长，并让棋盘覆盖画面中央、四角、边缘、近处、远处和多种倾角。避免同一姿态重复采样。

```bash
python -m rescue_vision.calibration.capture_chessboard_images \
  --lens-position 1.0 \
  --target 50
```

如果不提供 `--lens-position`，脚本会先执行一次自动对焦再锁焦。比赛运行必须复用 `session.json` 记录的焦点位置。

输出：

```text
calibration_captures/chessboard_2304x1296_时间戳/
├── images/
├── detected/
├── session.json
└── images.jsonl
```

### 1.2 求解并选择内参模型

脚本比较标准针孔、Rational 针孔和 OpenCV Fisheye 模型。选择依据是 K 折验证集重投影 RMSE，而不是只看全量拟合误差。

```bash
python -m rescue_vision.calibration.calibrate_intrinsics \
  --square-size-mm 15.0 \
  --folds 5
```

指定采集批次：

```bash
python -m rescue_vision.calibration.calibrate_intrinsics \
  --session src/rescue_vision/calibration/calibration_captures/chessboard_2304x1296_YYYYMMDD_HHMMSS \
  --square-size-mm 15.0
```

输出目录带时间戳：

```text
output/intrinsics_YYYYMMDD_HHMMSS/
├── comparison.json
├── selected_calibration.json
├── selected_calibration.npz
├── models/
└── diagnostics/
```

运行时只加载 `selected_calibration.json` 的以下字段：

```json
{
  "calibration_id": "intrinsics_YYYYMMDD_HHMMSS",
  "model_type": "pinhole_rational",
  "image_size": [2304, 1296],
  "camera_model": "imx708_wide",
  "sensor_pixel_array_size": [4608, 2592],
  "scaler_crop": [0, 0, 4608, 2592],
  "camera_matrix": [[...], [...], [...]],
  "distortion": [...],
  "new_camera_matrix": [[...], [...], [...]],
  "lens_position": 1.0,
  "quality": {"usable": true}
}
```

`camera_model`、`sensor_pixel_array_size`、`scaler_crop` 和 `lens_position`
共同绑定采集时的相机条件；外参会逐项核对，旧的缺字段内参必须重新生成。
`calibration_id` 是内参参数集的非空、可读身份。内参标定脚本默认使用输出
目录名生成它，也可通过 `--calibration-id` 指定。地面映射必须原样复制这个
值；它用于阻止不同 `K/D/new_K`、模型、分辨率或焦点条件的标定产物混用。
它不是 schema 版本、Git 提交、文件校验和、相机序列号或场地位置。
JSON 重排不需要改变 ID；只要实际标定条件或参数改变，就必须生成新 ID 并
重新制作地面映射。

`comparison.json`、`models/` 和 `diagnostics/` 保存评测与求解细节，不是运行时
内参 JSON。当前格式不再写入 `schema_version`；旧的缺少 `calibration_id` 或
包含旧顶层字段的 JSON 必须重新生成，不做兼容猜测。

部署前检查：

- `selected_calibration.json` 中 `quality.usable` 必须为 `true`；
- 比较交叉验证 RMSE、最大单图误差和位姿求解成功率；
- 查看各模型诊断图，尤其是图像边缘直线和有效区域；
- 用独立于标定集的图片/测量点做验证。

`CameraModel.from_json(...)` 默认拒绝 `quality.usable=false` 的结果。

### 1.3 控制去畸变黑边

去畸变的视野保留越多，边缘越可能出现无法从原图采样的区域。运行时会把这些像素统一填为与 YOLO Letterbox 一致的灰度 `114`；应在生成 `new_K` 时处理有效视野取舍，不要在录制后直接裁图：

```bash
# 针孔 / Rational 模型：减小 alpha 会减少黑边，但也会缩小有效视野
python -m rescue_vision.calibration.calibrate_intrinsics \
  --square-size-mm 15.0 --alpha 0.0

# Fisheye 模型：减小 balance 会减少黑边，但也会缩小有效视野
python -m rescue_vision.calibration.calibrate_intrinsics \
  --square-size-mm 15.0 --balance 0.0
```

默认值均为 `0.35`，应结合诊断图、`CameraModel.valid_mask` 的有效比例和比赛
所需视野选择。参数变化会生成不同的 `new_K`；确定新参数后必须使用新的
`calibration_id`，并重新制作所有依赖旧标定的地面映射和任务数据。不能只
修改 `selected_calibration.json` 中的数值。

## 第二部分：外参与地面映射标定

外参描述相机相对机器人地面系的安装姿态，并生成
`UndistortedPixel ↔ GroundPoint` 所需的地面映射。开始本部分前，必须先完成并验收
第一部分的 `selected_calibration.json`；外参采集会严格核对相机型号、传感器尺寸、
裁剪、分辨率、焦点和 `calibration_id`。

### 外参命令与产物

| 命令或入口 | 用途 | 主要产物或结果 |
| --- | --- | --- |
| `python -m rescue_vision.calibration.capture_extrinsics_ground` | 用完整普通棋盘采集多站位外参图片 | `board_calibration.json`、原图和检测图 |
| `python -m rescue_vision.calibration.capture_extrinsics_charuco` | 用可局部检测的 ChArUco 板采集多站位图片 | `board_calibration.json`、原图和检测图 |
| `python -m rescue_vision.calibration.calibrate_extrinsics_ground` | 站位均衡筛点并求物理外参与地面映射 | `ground_mapping.json`、BEV 和误差诊断 |
| `load_runtime_config(...).build_geometry()` | 实际运行时联合装配内参与地面映射 | 校验分辨率、标定 ID、质量和物理一致性 |

多图外参流程的可测试入口包括 `load_board_calibration()`、
`validate_capture_session()`、`fit_station_robust_homography()`、
`board_points_field_mm()` 和 `field_points_to_robot_ground()`；它们负责严格加载
JSON 与采集条件、站位均衡筛点和留一验证、展开棋盘坐标，以及转换到机器人地面系。
ChArUco 相关入口为 `create_charuco_board()`、`detect_charuco_board()`、
`charuco_points_field_mm()` 和 `charuco_detection_jitter_px()`。

### 外参固定条件与坐标

外参只适用于相机相对机器人未改变的安装位置和姿态。机器人地面系原点为两驱动轮
接地点连线中点，`x` 向前、`y` 向左、`z` 向上，单位 mm。安装位姿、内参或
`new_K` 变化时必须重新制作地面映射。

### 2.1 固定机器人并采集多位置标定板图

此步骤只能在相机和机器人最终固定后进行。以两驱动轮接地点连线中点作为机器人
地面原点，把该点停在场地全局系 `(0, 0)`；机器人
`x` 正方向与场地全局 `y` 正方向重合；为保持右手系，机器人 `y` 正方向对应
全局 `x` 负方向。棋盘长边平行全局 `x`，短边平行全局 `y`。

推荐在地面铺设定位模板，标出 6 个站位，其中 5 个用于拟合、1 个用于独立验收。
每个站位用两条垂直基准线约束平移和方向，并标出棋盘外框左下角的位置。棋盘
必须贴地、不能翘曲；使用外框角点时必须同时提供对应边距，不能把外框角点直接
当作第一个棋盘内角点。

棋盘为 12×9 个实体方格、11×8 个内角点。棋盘外框通常有边距，必须实测并写入
JSON；当使用外框左下角作为基准时，脚本会按 `left` 和 `bottom` 边距自动换算首个
内角点；`right` 和 `top` 同时保留用于尺寸核对。普通黑白
棋盘有 180° 朝向歧义，必须在参考角做不干扰角点检测的标记，或在定位模板上明确
标记参考角，并在 JSON 正确填写 `detected_corner_order`。

使用交互式外参采集命令：

```bash
python -m rescue_vision.calibration.capture_extrinsics_ground \
  --lens-position 1.0 \
  --square-size-mm 15.0 \
  --max-detection-scale 2.0 \
  --session src/rescue_vision/calibration/calibration_captures/ground_mapping_YYYYMMDD_HHMMSS
```

默认用 OpenCV 预览窗口触发拍摄，窗口需要本地显示器才能看到并接收键盘输入。
通过 SSH 或没有可见显示器的会话，加 `--terminal` 改为终端驱动：每个站位输入坐标后，
终端提示“棋盘放好后按 Enter 拍摄”，直接回车即抓拍并检测，输入 `q` 退出并保留已完成
记录；该模式不会创建任何 GUI 窗口。两种模式保存的原图、检测图和 JSON 完全一致。

脚本启动后依次询问长边方向边距、短边方向边距和采集张数。这里假设左右长边
边距相同、上下短边边距相同；最后一张自动标记为 `holdout`，前面的图片标记为
`fit`。每个站位先输入外框左下角全局坐标，再把棋盘放到定位模板对应位置，按
Enter/Space 拍摄；11×8 内角点检测失败时不会计入张数，可调整棋盘后重拍。
每次成功采集都会立即更新 `board_calibration.json`，按 Q/Esc 退出也会保留已完成记录。
`--max-detection-scale` 默认是 `2.0`，只用于检测时临时放大图像，不改变保存的原始图像
坐标；必要时可提高到 `3.0`，但会增加单次按键后的检测时间和内存占用。
检测回退包括原图高精度 SB、放大后的完整棋盘 SB、`CALIB_CB_LARGER` 和传统自适应阈值；
只有完整 11×8（88 个）角点才会接受。任意局部角点无法确定其在整块棋盘中的绝对行列偏移，
贸然使用会产生看似合理但错误的外参，因此当前仍拒绝不完整角点结果。

如果换用 ChArUco 板，使用独立命令：

```bash
# A4 ChArUco
python -m rescue_vision.calibration.capture_extrinsics_charuco \
  --squares-x 7 --squares-y 5 \
  --square-size-mm 35.0 --marker-size-mm 25.0 \
  --dictionary DICT_5X5_100 \
  --minimum-charuco-corners 8 \
  --board-rotation-degrees 0 \
  --max-detection-scale 2.0 \
  --session src/rescue_vision/calibration/calibration_captures/charuco_YYYYMMDD_HHMMSS
```

ChArUco 板必须打印面朝向相机，并在 OpenCV 板坐标原点侧的外框角做永久物理标记；
OpenCV 当前板坐标的 `y` 朝打印图案下方，因此程序会先按打印面做 `y` 轴翻转，再
应用 `--board-rotation-degrees` 指定的 `0/90/180/270` 度平面旋转。每站输入的是
已标记外框角的场地坐标。一张图必须检测到
至少 8 个角点，覆盖至少 3 行和 3 列 ID，并通过连续帧稳定性和清晰度门槛才会保存。
最后一张仍为 `holdout`。
随后仍使用 `calibrate_extrinsics_ground`，无需换求解命令：
它会读取 `board_type: "charuco"`、字典和角点 ID，并用可见角点拟合。

### 2.2 检查或编写多图标定板 JSON

复制 [`board_calibration.example.json`](board_calibration.example.json)（普通棋盘）或
[`board_calibration_charuco.example.json`](board_calibration_charuco.example.json)（ChArUco），将图片路径和
实体参考角场地坐标替换为实测值。路径相对于 JSON 文件所在目录。ChArUco 使用
`reference: "opencv_board_origin_outer_corner"`、
`board_origin_outer_corner_global_mm`、`printed_face: "camera"` 和显式旋转角；
旧的含糊外框左下角格式会被拒绝。

```text
calibration_captures/ground_mapping_<timestamp>/
├── board_calibration.json
└── images/
    ├── board_001.png
    └── ...
```

`coordinate_frame` 必须描述机器人起始位置为全局 `(0, 0)`，并声明两套坐标轴的方向。
`board.reference` 可取 `lower_left_outer_corner` 或
`lower_left_inner_corner`。使用外框基准时，图片字段名为
`reference_outer_corner_global_mm`；使用内角点基准时字段名为
`reference_inner_corner_global_mm`。`board.edge_margin_mm` 依次记录左、右、下、上边距；
其中左、下边距用于外框角到首个内角点的换算。`reference_corner_marked` 必须为
`true`，表示参考角已在棋盘或定位模板上做物理标记；`detected_corner_order` 取
`reference_first` 或 `reference_last`，用于消除检测角点顺序的 180° 歧义。至少需要
3 张 `fit` 图片和 1 张 `holdout` 图片；推荐 5+1 张，覆盖近、远、左、右区域。
ChArUco 配置另需 `chessboard_size_squares`、`marker_size_mm`、`dictionary` 和
`minimum_charuco_corners`；ChArUco 由 ID 固定角点身份，不需要
`reference_corner_marked` 或 `detected_corner_order`。
ChArUco 采集要求当前 OpenCV 构建提供 `cv2.aruco.CharucoBoard` 和
`cv2.aruco.CharucoDetector`；若构建不含 ArUco 模块，脚本会在打开相机前报错。

### 2.3 求解外参与地面映射

脚本默认读取仓库根目录的 `configs/runtime.yaml`，从
`geometry.intrinsics_path` 获取内参，并通过 `load_runtime_config()` 与
`CameraCalibration` 的同一套校验加载 JSON。支持 pinhole、pinhole_rational
和 fisheye，默认拒绝 `quality.usable=false`、分辨率不匹配或配置未启用的内参。

默认用法：

```bash
python -m rescue_vision.calibration.calibrate_extrinsics_ground \
  --config configs/runtime.yaml \
  --session src/rescue_vision/calibration/calibration_captures/ground_mapping_YYYYMMDD_HHMMSS
```

也可以显式指定另一份运行配置：

```bash
python -m rescue_vision.calibration.calibrate_extrinsics_ground \
  --config configs/runtime.yaml \
  --board-calibration src/rescue_vision/calibration/calibration_captures/ground_mapping_YYYYMMDD_HHMMSS/board_calibration.json
```

`--intrinsics` 仅作为诊断或临时覆盖；未指定时不得再按时间戳自动挑选内参：

```bash
python -m rescue_vision.calibration.calibrate_extrinsics_ground \
  --intrinsics src/rescue_vision/calibration/output/intrinsics_YYYYMMDD_HHMMSS/selected_calibration.json
```

脚本会逐张读取 JSON 中的原始图片并自动寻找角点。检测失败、图片尺寸或焦点与内参
不匹配、站位重复/共线、holdout 位于 fit 覆盖外、站位数量不足或参考角顺序未声明时
直接停止，不生成可部署标定。求解必须读取采集目录相邻的 `session.json`。
求解命令同样支持 `--max-detection-scale`；采集和求解应使用相同或更大的值。

脚本每次运行创建新的时间戳目录，不覆盖已部署结果：

```text
output/ground_mapping_YYYYMMDD_HHMMSS_ffffff/
├── ground_mapping.json
├── ground_mapping.npz
└── diagnostics/
    ├── ground_mapping_diagnostics.json
    ├── *_corners.png
    └── bev_preview.png
```

运行时 `ground_mapping.json` 只保留地面投影所需的标识、矩阵、质量、完整
外参和 BEV 范围；`ground_to_image`、4×4 矩阵、姿态单应矩阵和逐点误差等可
由现有字段计算或仅用于验收的内容放在 `ground_mapping_diagnostics.json`。
其顶层 `calibration_id` 和 `model_type` 必须分别等于内参 JSON 的对应字段，
不再重复嵌套 `intrinsics` 对象。

运行时地面映射 JSON 的结构为：

```json
{
  "calibration_id": "intrinsics_YYYYMMDD_HHMMSS",
  "model_type": "pinhole_rational",
  "image_size": [2304, 1296],
  "coordinate_frame": {
    "origin": "midpoint_between_drive_wheel_contact_points",
    "x": "robot_forward",
    "y": "robot_left",
    "z": "up",
    "unit": "mm"
  },
  "image_to_ground": [[...], [...], [...]],
  "quality": {"usable": true, "physically_valid": true},
  "extrinsics": {
    "rotation_robot_to_camera": [[...], [...], [...]],
    "translation_robot_to_camera_mm": [...]
  },
  "bev": {
    "x_min_mm": -300.0,
    "x_max_mm": 2500.0,
    "y_min_mm": -1200.0,
    "y_max_mm": 1200.0,
    "mm_per_pixel": 5.0
  }
}
```

`coordinate_frame` 是运行时强校验的一部分；缺少它、原点不匹配或轴定义不匹配
的旧标定产物必须重新生成。`image_to_ground` 表示去畸变像素到机器人地面毫米坐标，
由站位均衡筛点后优化得到的物理外参唯一推导。直接单应拟合只用于异常点筛选、初始化
和一致性诊断；PnP 外参同时由 `GroundProjector` 统一用于机器人系
三维点重投影和像素射线与已知高度平面求交。部署前必须使用未参与拟合的
保留点实测地面误差、复核 BEV 有效范围，并检查
`pose_reprojection_rmse_px`；只验证地面单应性不足以启用目标三维模板拟合。

多图 JSON 绑定每张原图文件名、尺寸条件和站位坐标；更换任一图片、分辨率、焦点或
棋盘规格后必须重新生成标定。输出同时保存物理有效性、误差门限、
`quality.usable`、`model_type` 与可读 `calibration_id`。内参命令可用
`--calibration-id` 显式命名，未指定时使用输出目录名。默认门限可通过
`--maximum-mean-inlier-error-mm`、`--maximum-inlier-error-mm`、
`--maximum-pose-rmse-px`、`--maximum-holdout-mean-error-mm` 和
`--maximum-holdout-error-mm`、`--maximum-leave-one-out-mean-error-mm`、
`--maximum-mapping-disagreement-mm` 显式调整并落盘。RANSAC 使用去畸变像素阈值；
每站空间均衡取样并要求至少 60% 内点，同时报告逐站和留一站位误差。运行时
`GroundProjector.from_json(...)` 会拒绝 `quality.usable=false`、非物理
有效位姿、错误图像尺寸、模型或 `calibration_id` 不匹配的产物。
基础矩阵往返和 BEV 四角方向已有自动测试；实际安装仍必须使用独立保留点
测量地面误差。

## 产物管理与运行接入

### 3.1 产物管理

原始采集和临时输出不提交 Git。最终部署应保留一份经过验收的配置，并同时记录：

- 相机序列/硬件标识、分辨率和焦点；
- 标定日期和 OpenCV 版本；
- 内参与地面映射误差摘要；
- 可复现的相机机械定位方式。

如果这些元数据无法对应，宁可重新标定，也不要混用两次标定产物。

### 3.2 接入 `configs/runtime.yaml`

内参验收后先启用去畸变；地面映射完成并通过保留点验证后再单独启用：

```yaml
geometry:
  intrinsics_enabled: true
  intrinsics_path: ../src/rescue_vision/calibration/output/intrinsics_YYYYMMDD_HHMMSS/selected_calibration.json
  ground_mapping_enabled: true
  ground_mapping_path: ../src/rescue_vision/calibration/output/ground_mapping_YYYYMMDD_HHMMSS_ffffff/ground_mapping.json
```

从仓库根目录执行一次真实装配检查：

```bash
python - <<'PY'
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
geometry = config.build_geometry()
if geometry is None:
    raise RuntimeError("内参未启用")

print("intrinsics:", geometry.camera_model.calibration.calibration_id)
print("ground mapping:", geometry.ground_projector is not None)
PY
```

这一步比只解析 YAML 更重要：它会真正读取产物并检查运行分辨率、标定质量、
模型类型和 `calibration_id`。应用代码随后只使用 `geometry.camera_model`
与 `geometry.ground_projector`，不再直接解释 JSON 字段。
