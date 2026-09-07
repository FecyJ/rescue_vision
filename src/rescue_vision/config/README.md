# `config`：运行配置与对象装配

`load_runtime_config()` 是当前 YAML 格式的唯一入口。它拒绝未知键、错误类型、越界
数值、不可用标定和不匹配的相机元数据；相对路径以 YAML 文件所在目录为基准。

```bash
cp configs/runtime.example.yaml configs/runtime.yaml
```

正式流程使用 `configs/runtime.match.yaml` 的 `match` 配置节：

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.match.yaml")
match = config.build_match_sequence()
grab_transport = config.build_grab_transport_sequence()
```

两个装配方法只创建纯逻辑状态机，不打开相机、Hailo、串口或网络。车端入口在
`app/match_runtime.py` 创建并关闭这些资源。

## 配置分区

| 分区 | 责任 |
| --- | --- |
| `camera` / `geometry` | 图像尺寸、去畸变内参和地面外参；标定必须匹配当前相机和分辨率 |
| `processing` | 观测年龄和实时处理限制 |
| `uart` / `remote` | 通信设备、队列、观察权限和超时 |
| `motion` | 轮距、速度/加速度限制、里程计机械量、夹爪标定和解团试验参数 |
| `match` | 正式流程启动、目标团、机会抓取、绿色接近、安全区 d1/d2、刹车补偿和退出参数 |
| `tracking` / `world` / `mission` | 轨迹生命周期、静态地图、颜色派生和规则状态机 |
| `perception` / `hailo` | Pose、HSV、中心十字和安全区观测 |
| `localization` | 安全区/中心十字视觉锚点和编码器/IMU 融合 |
| `green_grab` | 赛外像素居中抓取试验 |

## `match` 的关键语义

正式流程使用相对视觉坐标、编码器定距和陀螺仪航向。`robot_footprint_radius_mm` 与
`safety_margin_mm` 是解团/绿色接近路径的联合安全膨胀量；它们属于正式流程，不再从
已删除的模拟流程读取。

目标团选择由 `cluster_*`、机会抓取由 `opportunistic_single_green_*` 控制。绿色目标
使用 `green_*` 和 `green_preclose_*` 门禁：到达 `green_grab_offset_mm` 后，以车体
原点为中心，在 `green_preclose_recheck_range_mm` 内停车复核，不要求目标在车前；
`green_preclose_recheck_hold_ms` 给新露出的目标完成 tracker 确认的时间，达到
`green_preclose_max_carried_blocks` 才停止继续纳入。`action_settle_time_s` 在转向/直线
切换前保持零速。

安全区运输参数分为夹取→d1、d1→d2、d2→末段三段。d1 停稳后，流程使用安全区
K0/K1/K2 和静态地标拟合 `FieldPose2D` 并覆盖当前航位；`safe_zone_d2_braking_overrun_*`
和 `safe_zone_d2_to_final_braking_overrun_mm` 只补偿预计刹车过冲。视觉纠偏当前只消费
d1 停稳阶段的当前观测，不接入采集时刻位姿对齐或跨帧目标记忆。

## 其他可复用模块

`app/scan_target_memory.py` 和 `app/field_target_cluster.py` 的 schema、值域和老化规则
仍由自身代码与测试维护，但尚未接入正式流程。未来接入前必须先确定位姿时间对齐、
重投影失效和记忆目标与实时轨迹的交接策略，不能在入口复制第二套定位逻辑。

## 生命周期与验证

资源由上下文或 `try/finally` 释放，旁路使用有界队列。配置装配不访问硬件，开发机可运行：

```bash
python -m compileall -q src tests manual_tests
python -m pytest tests/test_config.py
```

正式流程的整车停车、看门狗、标定、性能和现场规则验收必须在 `manual_tests/` 与目标
设备完成。
