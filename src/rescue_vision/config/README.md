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
| `match_cc` | CC 严格成团、单块通道净空、单块对准、20 ms 指令间隔和动作速度参数 |
| `tracking` / `world` / `mission` | 轨迹生命周期、静态地图、颜色派生和规则状态机 |
| `perception` / `hailo` | Pose、HSV、中心十字和安全区观测；`unknown_override_confidence_threshold` 控制高置信度模型类别对未知颜色结果的覆盖 |
| `localization` | 安全区/中心十字视觉锚点和编码器/IMU 融合 |
| `near_field_grasp` | 正式流程近场交接及独立绿黑多目标收拢的范围、几何余量、确认窗口和评分权重 |
| `green_grab` | 赛外像素居中抓取试验 |

## `match` 的关键语义

正式流程使用相对视觉坐标、编码器定距和陀螺仪航向。`robot_footprint_radius_mm` 与
`safety_margin_mm` 是解团/绿色接近路径的联合安全膨胀量；它们属于正式流程，不再从
已删除的模拟流程读取。

目标团选择由 `cluster_*`、机会抓取由 `opportunistic_single_green_*` 控制。机会抓取先按
`motion.gripper.transport_*` 固定姿态检查前向走廊；首轮只接受单个绿色，一般阶段允许
绿/黑物资组或单个橙色伤员；没有安全方案才继续目标团解团。固定走廊外的合法目标
先进入停车路由窗口。进入近场后由唯一确认窗口按同一目标 ID 收集不同有效帧，
在安全近场组、远距重接近和解团之间只做一次路由；确认中的几何随该目标的最新有效帧更新，
开爪提交时冻结执行计划，再按正式流程接近，进入 `near_field_grasp.max_range_mm` 后由近场状态机接管。首轮策略（单个
绿色）与后续策略（绿/黑 1～3 个或单橙色）由 `MatchSequence` 按运输次数选择；近场配置同时
用于独立收拢入口。`supply_recovery_frames` 控制单帧未知/质量异常后的稳定物资恢复帧数；
明确危险证据不通过该参数解除。`action_settle_time_s` 在转向/直线切换前保持零速。
普通运动的非零单轮最低速度由 `motion.min_wheel_velocity_m_s` 控制；绿色目标和近场组
精对准使用 `match.green_alignment_min_wheel_velocity_m_s` 的单次覆盖值，当前 match
模板为 0.01 m/s，避免改变启动和普通运输动作的最低速度。
近场组的 `center_tolerance_mm` 是进入允许范围，当前 match 模板为 20 mm；进入滞回范围由
`alignment_hysteresis_mm` 保持，锁定后固定旋转方向；
已锁定目标使用 `max_range_mm + range_hysteresis_mm` 的退出半径，避免边界观测抖动换组；
唯一确认窗口由 `confirmation_frames` 和 `alignment_timeout_ms` 共同限制，计时从 settle 完成后的 observation window 打开开始；
单橙色计划还受配置项 `orange_isolation_radius_mm` 的硬门禁：只有当前帧有地面位置且落入半径的其它目标才拒绝；缺少 K0 的目标不作无条件推定。

安全区运输参数分为夹取→d1、d1→d2、d2→末段三段。d1 停稳后，流程使用安全区
K0/K1/K2 和静态地标拟合 `FieldPose2D` 并覆盖当前航位；`safe_zone_d2_braking_overrun_*`
和 `safe_zone_d2_to_final_braking_overrun_mm` 只补偿预计刹车过冲。绿/黑物资末段使用
`safe_zone_d2_to_final_speed_m_s` 与 `safe_zone_d2_to_final_braking_overrun_mm`；单个橙色
伤员使用独立的 `safe_zone_orange_d2_to_final_speed_m_s` 与
`safe_zone_orange_d2_to_final_braking_overrun_mm`。解团阶段可用
`breakup_max_wheel_acceleration_m_s2` 覆盖 `motion.max_wheel_acceleration_m_s2`，设为
`null` 时继承全局值。视觉纠偏当前只消费 d1 停稳阶段的当前观测，不接入采集时刻位姿对齐
或跨帧目标记忆。
绿/黑物资路线终点由 `safe_zone_fallback_target_field_mm` 配置，单个橙色
伤员由 `safe_zone_injured_target_field_mm` 配置；两者都是 `FieldPoint`，单位 mm。

## `match_cc` 的关键语义

`configs/runtime.cc.yaml` 只在 `match` 节保留 CC 确实复用的启动、近场入口和安全区运输
参数；CC 特有阈值全部位于 `match_cc`。`cluster_neighbor_distance_mm` 定义团内每块
至少拥有两个邻居的距离上限；原解团的5帧均值、单次角度对准、固定接近和前推/后退
参数继续来自 `match.cluster_*`/`breakup_*`，其中 `cluster_align_hold_ms=1000` 是缺团后
重新搜索的最长等待。`isolated_line_clearance_mm` 定义绿/黑物块的原点—K0 线段净空，
橙色净空则逐帧使用颜色掩码的地面投影宽度。`block_alignment_tolerance_mm` 控制单块
y 对准误差。未知键、非有限数值、
超过底盘速度上限的动作参数都会在打开硬件前被拒绝。

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

## 近场收拢试验配置

`AppConfig.near_field_grasp` 为 `NearFieldGraspConfig`，配置缺省值与两个运行模板一致。
`max_targets` 在 1～3 内；`max_candidates` 在 `max_targets`～20 内；
`confirmation_frames` 必须为正。`grasp_commit_max_observation_age_ms` 限制准备结果发布年龄、
静止遥测的间断/年龄，以及没有静止证据时的短龄几何；实际静止期间采集的当前计划使用
`processing.max_observation_age_ms` 作为观测失联上限。`stationary_max_gyro_rad_s` 为有限正数，
默认0.03 rad/s；完整证据条件见 [app README](../app/README.md)。准备器不改写 `capture_timestamp_ns`。`fine_alignment_zone_rad`
与 `fine_alignment_min_wheel_velocity_m_s` 只影响近零角度的单次对准指令，后者允许为零。
规则分值、权重与余量有限且非负，权重和必须有限且大于零；
`orange_priority_weight` 必须大于数量、净空、距离和对准权重之和，以保证单橙严格优先于同分的绿黑组合；
尺寸、范围和确认数量严格校验，未知字段拒绝加载。`max_range_mm` 是 K0 的机器人相对
径向距离，`max_forward_distance_mm` 是底盘行程，不能互换。`target_final_x_mm` 是最远
目标 K0 的期望结束位置，`corridor_start_x_mm` 和 `corridor_lateral_margin_mm` 定义危险
目标 K0 的前进扫掠走廊；`side_neighbor_longitudinal_margin_mm` 与
`side_neighbor_lateral_margin_mm` 定义预测抓取方向下非橙计划的蓝色侧邻、以及单橙计划的
蓝/橙侧邻 K0 中心差门限，明显前后错开的目标不触发该门禁。角度安全端点继续使用
`motion.gripper`；不再从 `match` 读取
车体包络或夹臂扫掠余量。

配置迁移：已删除 `near_field_grasp.sample_frames`、`near_field_grasp.min_valid_frames`、
`near_field_grasp.alignment_stable_frames`、
`near_field_grasp.decision_timeout_ms`、`near_field_grasp.final_verify_settle_time_ms` 和
`near_field_grasp.final_verify_frames`；使用
`confirmation_frames`，独立 `gripper-width` 入口也不再接受 `--sample-frames`。

配置同时服务 `rescue-vision-match`、`rescue-vision-grab-transport` 和独立近场入口。完整选组语义、
归一化公式、CLI 迁移和生产装配见 [app README](../app/README.md#选组与试验几何)，
不在本页重复维护。

正式一般阶段接近种子采用“近场优先、类别配置分值优先、距离优先”，最终组合仍按
近场选择器评分。`orange_isolation_radius_mm` 保留原值，只有完整包络证明扫掠外的
侧后方普通物资可通过精细复核；条件见 [app README](../app/README.md)。无需迁移配置。
