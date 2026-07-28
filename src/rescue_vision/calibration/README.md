# `calibration`：相机与地面标定

本页是相机内参与地面映射的唯一操作说明。标定脚本默认使用本目录下的路径，不受终端当前目录影响；现场生成的 `calibration_captures/` 和 `output/` 已被 Git 忽略。运行时结构和坐标边界见[项目结构](../../../docs/项目结构.md)。

## 常用命令与产物

| 命令 | 使用时机 | 主要产物 |
| --- | --- | --- |
| `python -m rescue_vision.calibration.capture_chessboard_images` | 相机条件固定后采集内参样本 | 棋盘原图、检测图、逐帧元数据 |
| `python -m rescue_vision.calibration.calibrate_intrinsics` | 比较三种模型并选择内参 | `selected_calibration.json`、诊断图 |
| `python -m rescue_vision.calibration.calibrate_extrinsics_ground` | 相机安装姿态最终固定后 | `ground_mapping.json`、BEV 与误差诊断 |
| `CameraModel.from_json()` | 独立检查标定产物 | 默认拒绝 `quality.usable=false` |
| `load_runtime_config(...).build_geometry()` | 实际运行接入 | 联合验证内参、分辨率和地面映射指纹 |

## 完整工作流速览

1. 固定相机、分辨率、裁剪和焦点后采集棋盘图；
2. 用这批图片比较三种模型并选择内参；
3. 固定最终安装位姿并准备地面对应点；
4. 使用前一步内参求解地面映射；
5. 验收产物后再写入 `configs/runtime.yaml`。

每一步的实际命令在下文独立代码段给出，并承接上一步产物。前两步只依赖相机
和棋盘；地面映射必须等相机安装姿态固定，并准备 `ground_image.png` 与
`ground_points.json`。不要把不同焦点、分辨率或安装条件的产物混用。

## 固定条件

一套标定只对以下条件组合有效：

- 相机与镜头个体；
- `2304 × 1296` 取流分辨率和裁剪模式；
- 固定 `LensPosition`；
- 相机安装位置和姿态；
- 相机模型及 `new_camera_matrix`；
- 地面点坐标定义。

分辨率、焦点、镜头或 `new_K` 改变时至少重新做内参验证；安装位姿改变时必须重新做地面映射。不要只凭肉眼观察去畸变图判断标定质量。

## 1. 采集棋盘图

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

## 2. 求解并选择内参模型

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

部署前检查：

- `selected_calibration.json` 中 `quality.usable` 必须为 `true`；
- 比较交叉验证 RMSE、最大单图误差和位姿求解成功率；
- 查看各模型诊断图，尤其是图像边缘直线和有效区域；
- 用独立于标定集的图片/测量点做验证。

`CameraModel.from_json(...)` 默认拒绝 `quality.usable=false` 的结果。

### 控制去畸变黑边

去畸变的视野保留越多，边缘越可能出现无法从原图采样的区域。运行时会把这些像素统一填为与 YOLO Letterbox 一致的灰度 `114`；应在生成 `new_K` 时处理有效视野取舍，不要在录制后直接裁图：

```bash
# 针孔 / Rational 模型：减小 alpha 会减少黑边，但也会缩小有效视野
python -m rescue_vision.calibration.calibrate_intrinsics \
  --square-size-mm 15.0 --alpha 0.0

# Fisheye 模型：减小 balance 会减少黑边，但也会缩小有效视野
python -m rescue_vision.calibration.calibrate_intrinsics \
  --square-size-mm 15.0 --balance 0.0
```

默认值均为 `0.35`，应结合诊断图、`CameraModel.valid_mask` 的有效比例和比赛所需视野选择。参数变化会生成不同的 `new_K` 和内参指纹；确定新参数后必须重新验证内参，并重新制作所有依赖旧指纹的地面映射和任务数据。不能只修改 `selected_calibration.json` 中的数值。

## 3. 准备地面映射数据

此步骤只能在相机最终固定后进行。创建：

```text
calibration_captures/ground_mapping/
├── ground_image.png
└── ground_points.json
```

`ground_image.png` 是固定安装条件下的原始畸变图。参考 `ground_points.example.json` 建立 12～20 个分布均匀、易准确点击的地面点，覆盖实际工作区域和远近范围，避免共线或集中于局部。

坐标约定为机器人地面系 `x` 向前、`y` 向左、`z` 向上，单位 mm。

## 4. 求解地面映射

脚本复用 `CameraCalibration` / `CameraModel`，支持 pinhole、pinhole_rational 和 fisheye，默认拒绝 `quality.usable=false` 的内参。默认选择最新时间戳目录，也可显式指定：

```bash
python -m rescue_vision.calibration.calibrate_extrinsics_ground \
  --intrinsics src/rescue_vision/calibration/output/intrinsics_YYYYMMDD_HHMMSS/selected_calibration.json
```

首次运行按 `ground_points.json` 顺序点击原图中的点：

- 鼠标左键选择；
- `Enter` / `Space` 确认；
- `Backspace` 撤销；
- `Q` / `Esc` 退出；
- `--recollect` 强制重新选点。

脚本每次运行创建新的时间戳目录，不覆盖已部署结果：

```text
output/ground_mapping_YYYYMMDD_HHMMSS_ffffff/
├── ground_mapping.json
├── ground_mapping.npz
└── diagnostics/
    ├── undistorted_ground_image.png
    ├── undistorted_correspondences.png
    └── bev_preview.png
```

`image_to_ground` 表示去畸变像素到机器人地面毫米坐标。直接单应拟合用于地面点定位；PnP 推导矩阵用于交叉诊断。部署前必须使用未参与拟合的保留点实测地面误差，并复核 BEV 有效范围。

对应点文件绑定原图文件名与 SHA-256；更换或覆盖原图后必须
`--recollect`。输出 schema v3 同时保存物理有效性、误差门限、
`quality.usable`、`model_type` 与内参 SHA-256 指纹。默认门限可通过
`--maximum-mean-inlier-error-mm`、`--maximum-inlier-error-mm` 和
`--maximum-pose-rmse-px` 显式调整并落盘。运行时
`GroundProjector.from_json(...)` 会拒绝 `quality.usable=false`、非物理
有效位姿、错误 schema、图像尺寸、模型或指纹不匹配的产物。
基础矩阵往返和 BEV 四角方向已有自动测试；实际安装仍必须使用独立保留点
测量地面误差。

## 5. 产物管理

原始采集和临时输出不提交 Git。最终部署应保留一份经过验收的配置，并同时记录：

- 相机序列/硬件标识、分辨率和焦点；
- 标定日期、代码提交和 OpenCV 版本；
- 原始结果文件的校验和；
- 内参与地面映射误差摘要；
- 相机安装版本或可复现的机械定位方式。

如果这些元数据无法对应，宁可重新标定，也不要混用两次标定产物。

## 6. 接入 `configs/runtime.yaml`

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

print("intrinsics:", geometry.camera_model.calibration.fingerprint())
print("ground mapping:", geometry.ground_projector is not None)
PY
```

这一步比只解析 YAML 更重要：它会真正读取产物并检查运行分辨率、标定质量、模型类型和指纹。应用代码随后只使用 `geometry.camera_model` 与 `geometry.ground_projector`，不再直接解释 JSON 字段。
