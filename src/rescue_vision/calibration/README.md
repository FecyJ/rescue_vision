# 相机标定脚本

目录约定：

```text
src/rescue_vision/calibration/
├── capture_chessboard_images.py
├── calibrate_intrinsics_fisheye.py
├── calibrate_extrinsics_ground.py
├── ground_points.example.json
├── calibration_captures/
└── output/
```

所有默认路径均相对于本目录，不受当前终端工作目录影响。

## 1. 采集内参标定图

相机分辨率固定为 `(2304, 1296)`，棋盘格为 12×9 格、11×8 内角点。

```bash
python -m rescue_vision.calibration.capture_chessboard_images \
  --lens-position 0.8 \
  --target 50
```

采集结果：

```text
calibration_captures/chessboard_2304x1296_时间戳/
├── images/
├── detected/
├── session.json
└── images.jsonl
```

## 2. 计算 Fisheye 内参

将 `15.0` 替换为标定板实测方格边长。

```bash
python -m rescue_vision.calibration.calibrate_intrinsics_fisheye \
  --square-size-mm 15.0 \
  --balance 0.35
```

默认使用最新采集批次。指定批次：

```bash
python -m rescue_vision.calibration.calibrate_intrinsics_fisheye \
  --session src/rescue_vision/calibration/calibration_captures/chessboard_2304x1296_20260721_120000 \
  --square-size-mm 15.0
```

输出：

```text
output/fisheye_intrinsics_2304x1296.json
output/fisheye_intrinsics_2304x1296.npz
output/intrinsics_diagnostics/
```

## 3. 准备外参和地面映射数据

必须等相机安装位置固定后执行，并保持：

- 同一相机及 2304×1296 取流模式；
- 同一个固定 `LensPosition`；
- 同一个内参文件和 `new_K`；
- 机器人坐标约定：x 向前、y 向左、z 向上，单位 mm。

创建：

```text
calibration_captures/ground_mapping/
├── ground_image.png
└── ground_points.json
```

`ground_image.png` 是固定安装后的原始畸变图像。把 `ground_points.example.json` 复制为 `ground_points.json`，根据实际布置修改坐标。建议使用 12～20 个清晰、分布均匀的地面标记点，不要让点集中在一条线或一小片区域。

## 4. 计算外参和地面映射

```bash
python -m rescue_vision.calibration.calibrate_extrinsics_ground
```

首次运行会显示 `ground_image.png`：

- 按 `ground_points.json` 中的顺序点击对应标记中心；
- `Enter` / `Space`：确认当前点击；
- `Backspace`：撤销；
- `Q` / `Esc`：退出。

点击结果保存为：

```text
calibration_captures/ground_mapping/correspondences.json
```

重新点击：

```bash
python -m rescue_vision.calibration.calibrate_extrinsics_ground --recollect
```

输出：

```text
output/ground_mapping.json
output/ground_mapping.npz
output/ground_diagnostics/
├── undistorted_ground_image.png
├── undistorted_correspondences.png
└── bev_preview.png
```

`ground_mapping.json` 中：

- `image_to_ground`：去畸变像素 → 机器人地面毫米坐标，供 `GroundProjector` 使用；
- `ground_to_image`：机器人地面坐标 → 去畸变像素；
- `extrinsics`：机器人坐标系与 OpenCV 相机坐标系的外参；
- `bev`：上方为机器人前方、左侧为机器人左方的 BEV 变换。

直接拟合的 `image_to_ground` 优先用于地面定位；由 PnP 外参推导的 `pose_image_to_ground` 主要用于检查两种结果是否一致。