# 工创赛智能救援-上位机视觉
工创赛智能救援赛道视觉工程，运行于 Raspberry Pi 5 + Hailo AI HAT+（Hailo-8L），相机为 Raspberry Pi Camera Module 3 NoIR Wide。

## 环境

- Raspberry Pi OS Trixie
- Python 3.13.5
- 相机分辨率：2304 × 1296
- OpenCV、Picamera2、NumPy、HailoRT 使用系统包
- 其余 Python 依赖见 `requirements.txt`

## 安装

```bash
sudo apt update
sudo apt install -y \
    python3-venv \
    python3-numpy \
    python3-opencv \
    python3-picamera2 \
    dkms \
    hailo-all
```

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pip install -e .
```

## 工程结构

```text
rescue_vision/
├── configs/                    # 相机、地面映射、场地与运行配置
├── src/rescue_vision/
│   ├── camera/                 # Picamera2 图像采集
│   ├── geometry/               # 去畸变、地面投影、坐标类型
│   ├── perception/             # YOLO Pose 与 OpenCV 感知
│   ├── localization/           # 视觉、编码器和 IMU 定位融合
│   ├── tracking/               # 目标跟踪
│   ├── world/                  # 静态地图与动态目标地图
│   └── interfaces/             # STM32 等外部接口
├── scripts/                    # 标定与测试脚本
├── tests/
└── requirements.txt
```

## 几何约定

- 原始像素：`RawPixel`
- 去畸变像素：`UndistortedPixel`
- 机器人地面坐标：`GroundPoint`
- 地面坐标中 `x` 向前，`y` 向左，单位为毫米
- BEV 图像上方对应机器人前方，左侧对应机器人左方

处理链路：

```text
相机原图
→ CameraModel 去畸变
→ GroundProjector 地面映射 / BEV
→ 目标与场地特征检测
→ 定位和世界模型
```

## 运行前注意

相机内参、畸变参数、`new_camera_matrix` 和地面单应矩阵必须在最终的以下条件下重新标定：

- 2304 × 1296 分辨率
- 固定相机安装位置
- 固定 `LensPosition`
- 固定去畸变模型与参数

测试前可先验证：

```bash
python -c "import cv2, numpy, yaml, serial; from picamera2 import Picamera2; print('OK')"
hailortcli fw-control identify
```