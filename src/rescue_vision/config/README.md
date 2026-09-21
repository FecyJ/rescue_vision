# `config`：运行配置与对象装配

`load_runtime_config()` 是当前 YAML 格式的唯一入口。它拒绝未知键、错误类型、越界
数值、不可用标定和不匹配的相机元数据；相对路径以 YAML 文件所在目录为基准。

```bash
cp configs/runtime.example.yaml configs/runtime.yaml
```

正式流程使用 `configs/runtime.match.yaml` 的 `match` 配置节：

该配置及 `configs/runtime.match_nb.yaml` 的树莓派端车体线速度上限
`motion.max_linear_velocity_m_s` 与单轮速度上限
`motion.max_wheel_velocity_m_s` 均为 `1.5 m/s`；各阶段实际目标速度仍由对应动作参数决定。

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
| `perception` / `hailo` | Pose、HSV、中心十字和安全区观测；模型类别为权威，HSV 只提取几何掩码 |
| `localization` | 安全区/中心十字视觉锚点和编码器/IMU 融合 |
| `near_field_grasp` | 正式流程近场交接及独立绿黑多目标收拢的范围、几何余量、确认窗口和评分权重 |
| `green_grab` | 赛外像素居中抓取试验 |

## `match` 的关键语义

正式流程使用相对视觉坐标、编码器定距和陀螺仪航向。`robot_footprint_radius_mm` 与
`safety_margin_mm` 是解团/绿色接近路径的联合安全膨胀量；它们属于正式流程，不再从
已删除的模拟流程读取。

`match.grasp_task_timeout_ms`（默认20000 ms，有限正数）是选择物理目标后跨接近、
停稳、规划、恢复与再抓取的总预算；状态和 tracker ID 变化不重置。
`near_field_grasp.no_plan_wait_ms` 是无计划的短观察，`alignment_timeout_ms` 是稳定场景
与异步准备的提交窗口，均受任务剩余预算约束。`confirmation_frames=1` 在第一张可靠
静止帧中完成确认，无消费者锁组握手；`breakup_confirmation_frames` 单独约束恢复核心。
正式恢复由同一准备器在抓取确实受阻时规划；带载补夹、单绿联调和CC保持各自限制。
完整规则、机械可达性及时间语义见[正式流程设计](../../../docs/正式流程设计.md)。

`configs/runtime.match_nb.yaml` 的 `match.nb_opening_actions` 还支持 `wheel_turn`：
它直接表达右轮恒速、左轮固定加速度切换、IMU 定角度和里程定距刹停，适用于需要连续
轮速交接的开头动作。该参数只定义主机控制目标，最终制动距离和精度必须用真车标定。
其余已有配置字段和CLI保持不变；模板可省略新字段并使用默认值。

`cluster_search_empty_angular_velocity_rad_s` 是没有可搜索的绿/黑/橙目标时的同向快速扫描
速度；如果当前帧所有有效非蓝目标的 bbox 中心都落在任一安全区 bbox 内，也使用该速度。它必须
不慢于 `cluster_search_angular_velocity_rad_s`。同一速度也用于合爪后的补夹扫描
（`TRANSPORT_GREEDY_SCAN`）：补夹候选为全场绿/黑，不受 `near_field_grasp.max_range_mm` 截断，
按机器人地面距离优先。当前帧没有可补夹的绿/黑信息时快速转动，出现可补夹信息后回到
`close_gripper_spin_angular_velocity_rad_s`，两条速度共用 `spin_angle_rad` 作为本次扫描的
旋转预算；目标一旦确认不因新目标出现而换组，目标失观时才接管下一个候选。目标方向走廊的横向阻挡门限按目标 K0 加
`near_field_grasp.clearance_mm / 2 + corridor_lateral_margin_mm` 逐目标计算，
不再用目标实体半径扩大近场扫掠范围。安全区运输参数分为夹取→d1、
d1→d2、d2→末段三段；d1→d2 严格使用 `safe_zone_d1_to_d2_speed_m_s`，不叠加
`pickup_cruise_speed_scale`。D1 视觉校准距安全区终点的可用区间由
`safe_zone_calibration_min_offset_mm` 和 `safe_zone_calibration_start_offset_mm`
分别定义最小、最大偏移；夹取位置在区间内时原地校准，距安全区过近时先回到最小偏移线。
正式配置 `configs/runtime.match.yaml` 的抓取对准增益 `green_alignment_kp_rad_s=2.0`
（远场和近场共用）、解团对准增益 `cluster_align_kp_rad_s=1.0` 均为原值两倍；
角速度继续随角度误差递减，并分别受原有 `0.6`、`0.4 rad/s` 上限约束。
近场前进的航向纠偏也复用抓取对准增益；此次调参的真机响应与过冲未验证。
抓取接近的最后 50 mm 使用 `pickup_terminal_speed_gain_s_inv`（单位 `s^-1`）按
`目标速度 = 增益 × 剩余距离` 收速；默认 `1.0` 保持原曲线，最低目标速度仍为
`0.005 m/s`。该增益同时用于远场接近终点和近场编码器定距收拢，不影响安全区运输。
安全区两点策略和时间语义以[正式流程设计](../../../docs/正式流程设计.md#安全区视觉纠偏)为准。
`safe_zone_bbox_turn_max_angular_velocity_rad_s` 是单侧裁剪时持续扫描到完整 bbox 入镜的固定角速度；`safe_zone_keypoint_reobserve_timeout_s`
是静止观察窗口。删除旧追框比例增益和中心滞回三字段，外部配置必须同步移除。
`safe_zone_bbox_edge_margin_px` 同时约束完整 bbox 四边和按画面位置选出的对侧两点；
第三点允许缺失，完整框不可缺失。
正式入口的 `localization.safe_zone_corners.max_observation_age_ms` 与 processing 对齐为 800 ms。
两点和静态
地标拟合 `FieldPose2D` 后覆盖当前航位。`safe_zone_d2_braking_overrun_*`
和 `safe_zone_d2_to_final_braking_overrun_mm` 只补偿预计刹车过冲。绿/黑物资末段使用
`safe_zone_d2_to_final_speed_m_s` 与 `safe_zone_d2_to_final_braking_overrun_mm`；单个橙色
伤员使用独立的 `safe_zone_orange_d2_to_final_speed_m_s` 与
`safe_zone_orange_d2_to_final_braking_overrun_mm`。解团阶段可用
`motion.max_linear_acceleration_m_s2`、`max_linear_deceleration_m_s2`、
`max_angular_acceleration_rad_s2`、`max_angular_deceleration_rad_s2` 分别限制车体
直线加速、直线减速、角加速和角减速。解团和 D2→末段可用同名
`breakup_max_*`、`safe_zone_d2_to_final_max_*` 字段逐项覆盖；某项为 `null` 时仅该项
继承全局值。释放并直线倒车退出后直接转向搜索，不再执行退区安全区纠偏或停车等图。视觉纠偏只消费停稳阶段的当前观测；正式 match 的目标对准会使用
有限采集时刻编码器/IMU 位姿历史，不接入跨帧目标记忆。
绿/黑物资路线终点由 `safe_zone_fallback_target_field_mm` 配置，单个橙色
伤员由 `safe_zone_injured_target_field_mm` 配置；两者都是 `FieldPoint`，单位 mm。

## `match_cc` 的关键语义

`configs/runtime.cc.yaml` 只在 `match` 节保留 CC 确实复用的启动、近场入口和安全区运输
参数；CC 特有阈值全部位于 `match_cc`。`cluster_neighbor_distance_mm` 定义团内每块
至少拥有两个邻居的距离上限；CC 的5帧均值、单次角度对准、固定接近和前推/后退
参数继续来自 `match.cluster_*`/`breakup_*`，其中 `cluster_align_hold_ms=1000` 是缺团后
重新搜索的最长等待。正式 match 的 `breakup_confirmation_frames` 控制停稳后接触核心
确认所需的连续有效帧数；停稳后仍选不出任何合法接触计划时，`breakup_no_plan_reobserve_ms`
只给一个有界的重观测窗口就退出本次区域，不占满按确认帧数计的确认预算。
`breakup_forward_distance_m` 和 `breakup_backward_distance_m` 是完整动作行程，
正式配置及默认值分别为 0.5 m、0.3 m，全程闭爪；整段路径不安全时拒绝该方案，不缩短动作。
旧 `breakup_penetration_mm` / `breakup_retry_penetration_mm` 已删除，外部 YAML 必须同步删除旧键。
`breakup_max_attempts` 是同一物理接触团允许的推进尝试次数，取值范围 `[1, 4]`；
同一接触射线重试需要几何改变并能产生更深接触，不能只靠重编号或重启会话。
CC 入口仍把同名前进/后退值作为固定动作距离。
`isolated_line_clearance_mm` 定义绿/黑物块的原点—K0 线段净空，
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
`stopped_scene_new_target_max_bbox_iou` 在 0～1 内，控制停车视野中新目标首帧冻结的
bbox IoU 上限；
`confirmation_frames` 必须为正。`grasp_commit_max_observation_age_ms` 限制准备结果发布年龄、
静止遥测的间断/年龄，以及没有静止证据时的短龄几何；实际静止期间采集的当前计划使用
`processing.max_observation_age_ms` 作为观测失联上限。`stationary_max_gyro_rad_s` 为有限正数，
默认0.03 rad/s；完整证据条件见 [app README](../app/README.md)。准备器不改写 `capture_timestamp_ns`。
规则分值、权重与余量有限且非负，权重和必须有限且大于零；
`orange_priority_weight` 必须大于数量、净空、距离和对准权重之和，以保证单橙严格优先于同分的绿黑组合；
尺寸、范围和确认数量严格校验，未知字段拒绝加载。`max_range_mm` 是 K0 的机器人相对
径向距离，`max_forward_distance_mm` 是底盘行程，不能互换。`target_final_x_mm` 是最远
目标 K0 的期望结束位置；贪心补夹绿/黑组使用独立的 `greedy_target_final_x_mm`，只在贪心
近场策略生效，降低它会增加补夹前进行程。`corridor_start_x_mm` 和 `corridor_lateral_margin_mm` 定义危险
目标 K0 的前进扫掠走廊；`side_neighbor_longitudinal_margin_mm` 与
`side_neighbor_lateral_margin_mm` 定义预测抓取方向下非橙计划的蓝色侧邻、以及单橙计划的
蓝/橙侧邻 K0 中心差门限，明显前后错开的目标不触发该门禁。
角度安全端点继续使用
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
近场选择器评分。`orange_isolation_radius_mm` 当前为 `10 mm`；橙色邻近例外只依据邻近
目标 K0 是否位于实际扫掠走廊内，目标完整外轮廓不扩大扫掠阻挡范围；条件见
[app README](../app/README.md)。已有配置需要同步为 `10 mm`。

`near_field_grasp.orange_target_final_x_mm` 是橙色 K0 的合爪终点；减小该值会增加前进行程。
本次按现场“约三分之一留在爪外”的反馈增加 30 mm 行程：正式 `runtime.match.yaml` 从
138 mm 调为 108 mm，其余运行模板及缺省值从 142 mm 调为 112 mm。该值仍需真车校准。
`black_closed_servo_offset_deg=5` 只在计划包含黑色物块时，把合爪及随后带载保持的左右绝对
舵机命令各增加 5°；张爪几何仍使用原机械标定。

`match.gate_clearance` 是独立的实验模块配置。`front_depth_mm` 定义两个己方安全区门前触发深度；
`side_x_mm` / `sweep_y_mm` 定义 S1/S2，`release_reverse_m` 同时定义暂存点向外偏移和放置后
退出距离，`center_release_y_mm` 定义中场释放线。`sweep_speed_m_s` 仅用于横扫，普通航点用
`transit_speed_m_s`；`observation_timeout_ms` 是暂存物回取观察窗口，`attempt_timeout_s` 是从
首次触发起连续计算的整次清障截止时间。数值必须有限且为正，未知键拒绝加载。

## 夹爪内颜色门禁配置

`configs/runtime.match.yaml` 的 `perception.gripper_color`（其余入口模板同步）默认启用。
`polygon_normalized` 为全尺寸去畸变图的归一化 `(u,v)` 凸多边形，默认只包住夹臂内侧，
当前实拍收紧为 `[(0.47,0.85),(0.53,0.85),(0.57,0.97),(0.43,0.97)]`，排除夹爪尖端前方
及两侧夹臂外的色块；相机/安装改变后须在预览中重新核对。`min_component_fraction=0.03`
为最大连通色块占ROI比例；
`black_min_thickness_fraction=0.12` 为黑色去细线核直径与ROI包围框短边的比例；ROI收紧后
同步增大该比例，以保持去细线核在实拍图上的像素尺度，仍随分辨率缩放。
`shadow_min_value=30` 是夹爪内彩色暗面的OpenCV HSV亮度下限（整数1～255）；H/S继续来自
`perception.color_classifier`。此放宽仅属于误夹色块检查，不改变模型类别或目标几何分割。
`orange_bbox_min_color_fraction=0.15` 要求每个双橙误夹候选的模型框内，橙色 HSV 掩码至少占
整个 bbox 的 15%；未达到时只保留模型观测，不作为误夹数量证据。通过该门限后，
`orange_distinct_max_bbox_iou=0.20` 与 `orange_distinct_min_k0_distance_px=20.0` 共同定义两个明显
不同目标：两者必须同时满足，缺 K0 或只有一项分离不触发张爪，避免将大框误检或同一橙块的
重复检测当作两块。
未知键、无效多边形和非有限/越界阈值拒绝加载；缺省使用上述值，无旧配置必删字段。

调节时以 `gripper_color` 日志的 `black_raw_fraction`、`black_chromatic_fraction` 和
最终 `components` 区分暗面混色、细线和实体黑块；不可只提高面积阈值掩盖误报。
合成测试不代表现场黑块/危险类指标，参数与ROI须用固定相机的实物验证。
