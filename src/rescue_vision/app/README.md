# `app`：可运行应用装配

正式比赛流程由 `MatchSequence` 提供纯逻辑状态机，由 `match_runtime.py` 负责相机、
Hailo、UART、夹爪、观察发布和资源生命周期。纯逻辑对象不创建硬件资源；硬件旁路只
提交最新观测，不阻塞运动控制循环。

## 常用入口

| 入口 | 配置和用途 |
| --- | --- |
| `rescue-vision-match` | 使用 `configs/runtime.match.yaml` 运行正式流程；支持 `--start-area 2/3`、本地预览、observe-only perception 图传和流程日志 |
| `rescue-vision-match-cc` | 使用 `configs/runtime.cc.yaml` 运行 CC 流程；启动、CLI、日志和安全区运输与 match 相同，其余搜索/解团/夹取策略独立 |
| `rescue-vision-grab-transport` | 使用同一配置，从 `(0,0,+90°)` 搜索绿色单物块，并在安全区末端保持张爪推送，不执行解团 |
| `rescue-vision-cluster-breakup` | 赛外固定解团试验，不是正式比赛入口 |
| `rescue-vision-manual-capture` | 受监督手动驾驶、图传和数据采集 |
| `rescue-vision-green-grab` | 无地面标定时的绿色像素居中抓取试验 |
| `rescue-vision-gripper-width` | 受监督的绿黑多目标或单橙色近场收拢；合爪后保持，不掉头 |
| `rescue-vision-motion-sequence` | 受监督 TUI 定距动作：输入 `a1/a2`，自动规划并前进 1.5 m |
| `rescue-vision-teach-replay` | 手推示教时记录编码器/IMU JSONL，并按左右轮轨迹受监督回放 |

## `rescue-vision-match-cc` 流程

CC 流程要求团内每个物块在 K0 地面坐标中都至少有两个不超过 100 mm 的团内邻居。
首次发现团后停车，并按原 match 解团方式逐帧重新成团：收集5帧团中心和最近正向
K0 的平均值，不绑定首次 tracker ID；找不到符合条件的团最多等待 1000 ms，超时重新
旋转搜索。随后按锁定平均中心只对准一次，接近到固定前距，再执行固定距离前推、张爪、
固定距离退出和合爪；舵机与底盘运动指令之间仍至少间隔 20 ms。之后只选择原点至目标
K0 线段净空的绿色单块，并在
`y=0±5 mm` 内锁定；450 mm 外持续更新坐标接近，进入近场后按 `x-150 mm`
定距、使用 `TRANSPORT` 夹爪姿态并合爪。

首次运输退出后每轮依次整圈搜索单橙、单黑/绿和含橙/黑/绿的团。橙色通道宽度来自
与 `gripper_width` 相同的 ROI 颜色掩码地面投影，近场抓取复用同一非对称夹爪反解；
单黑/绿阶段优先黑色、再按距离排序，团阶段按橙、黑、绿优先级选择。橙色安全区
运输终点的 FieldPoint 绝对 `x` 始终为正，其余运输和退出动作沿用 match。

```bash
rescue-vision-match-cc \
  --config configs/runtime.cc.yaml \
  --start-area 2 \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

## `rescue-vision-motion-sequence` 使用方法

该入口默认只打开配置中的 UART 和底盘控制器；增加 `--local-preview` 后，会启动相机和
Hailo Pose，并在 OpenCV 窗口显示最新 perception 识别叠加帧，不启动远程服务。启动前必须
确认车辆处于架空/安全测试环境，物理急停可用，并由操作员全程监督：

```bash
rescue-vision-motion-sequence \
  --config configs/runtime.match.yaml \
  --distance-m 1.5 \
  --supervised-physical-stop-ready \
  --local-preview
```

TUI 输入 `a1`、`a2`：单位均为 m/s²，`a2` 输入正的减速度幅值。`--distance-m` 配置
目标路程，默认 1.5 m。程序按目标路程自动计算三角速度曲线的峰值速度、加速时间和减速
时间，并逐周期下发目标速度。
该入口不采用 `motion` 配置中的最大速度、最大加速度和最小非零速度限制，但仍拒绝超过
STM32 线路 int16 可编码范围的轮速；运行中按 `q` 或 `Esc` 会软刹车退出。该入口是赛外受监督
动作试验，不是正式比赛流程；真车动作前仍需验证轮距、轮向、加速度、制动距离和急停。

## `rescue-vision-teach-replay` 使用方法

该入口打开 UART 后保持软刹车，只读取 100 Hz 编码器/IMU。手动缓慢推动车辆，按 Enter
或 `Ctrl+C` 结束录制；程序把每个原始样本、设备/本机时间戳和质量位写入独立 JSONL，
先打印日志路径，再等待输入 `REPLAY`。确认后按约 50 ms 窗口计算左右轮平均速度并回放；
回放中按 Enter 可随时软刹车：

```bash
rescue-vision-teach-replay \
  --config configs/runtime.match.yaml \
  --log-dir logs \
  --supervised-physical-stop-ready
```

以后可直接复用日志；`--speed-scale 0.5` 会把速度减半、时间加倍，轮行程保持不变：

```bash
rescue-vision-teach-replay \
  --config configs/runtime.match.yaml \
  --replay logs/teach_replay_YYYYmmdd_HHMMSS_ffffff.jsonl \
  --speed-scale 0.5 \
  --use-gyro \
  --supervised-physical-stop-ready
```

日志会绑定编码器每转计数、左右轮半径和陀螺仪方向；当前配置不匹配、编码器质量无效、
采样溢出、设备时间倒退或派生轮速超过配置上限时，程序在电机动作前拒绝回放。
`--no-use-gyro` 是默认行为，只按编码器差分开环回放；`--use-gyro` 会积分日志与回放现场的
Z 轴角速度，用相对航向误差对左右轮作有界差速修正，修正仍受运行配置的角速度和单轮速度
上限约束。日志中的零散无效 IMU 样本会被明确计数并跳过；回放现场短暂无效时暂停
航向修正并重新建立积分基线，连续不可用超过 200 ms 或设备时间不递增才软刹车。地面打滑、
零偏变化、人工推动速度尖峰、电机死区和控制器加速度限制仍会
造成路径误差，该入口不能代替完整定位或无人监督运行。

## `rescue-vision-gripper-width` 使用方法

该入口在近场自动选择 1～3 个绿色/黑色物资，或单个橙色目标，执行「选择当前可达目标 → 静止复核
→ 张爪 → 编码器定距前进 → 合爪 → 完成」。正式 `match` 在 450 mm 近场交接后复用
同一状态机，并按运输趟次把首轮策略限制为单个绿色物资；最近绿色的前进走廊被
蓝色危险/橙色/黑色目标阻挡时先尝试其它安全绿色候选，没有合法抓取方案才进入局部解团。正式流程远场对准和接近保持闭爪，
进入近场后才按计划的实际舵机映射张开；本独立入口不处理运输或交付。
完成仅表示动作指令与计时完成，`capture_confirmed=False`。

```bash
rescue-vision-gripper-width \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs/near_field
```

合爪后底盘保持静止，夹爪保持最后角度，不自动掉头、不搜索下一组。`--once` 在完成后
退出；普通模式等待 `q`、`Esc` 或 `Ctrl+C`。停稳后复核帧数只从 YAML 的
`near_field_grasp.confirmation_frames` 读取，正式配置为 1。旧 CLI 的 `--clearance-mm`、
`--min-mask-pixels` 已迁入 `near_field_grasp.clearance_mm`、
`min_mask_pixels`；不保留第二套参数来源。`--log-dir` 指定按分钟命名的
`gripper_width_YYYYmmdd_HHMM.log` 目录，默认是当前目录下的 `logs`；日志同时保留在终端。

准备线程在候选几何状态变化时立即记录 `grasp_candidate`，稳定状态最多每秒汇总一次，包含 `track_id`、类别、帧号、
采集时间、是否可选、`x0_mm`、`x1_mm`、`depth_mm`、`k0_x_mm`、`opening_mm`、
`target_final_x_mm`、`corridor_start_x_mm`、`corridor_end_x_mm`、
`corridor_half_width_mm`、`forward_distance_mm`、对准角和拒绝原因。`x0_mm`、`x1_mm`、
`depth_mm` 仅用于诊断颜色掩码的纵向投影，不再表示目标自身的硬容纳门禁；不可用地面
包络的目标也会记录，但几何字段为 `none`。

### 选组与试验几何

配置权威为 `near_field_grasp`，近场半径是单次实际收拢的接管条件，不是补夹的全场发现半径；
正式 match 可按现场标定调整，最多 3 个物资。远场补夹目标先沿正常入口接近，进入该半径后
再由近场收拢接管。
为限制后台枚举量，最多从最近 12 个可选物资建立组合（可配置为不超过 20 个）；
其余所有检测仍参加障碍检查或顺带纳入，不会因候选截断而被忽略。

单目标测量通过唯一 `GroundProjector` 投影 ROI 颜色掩码，并包含 K0。整组宽度取
候选包络最左至最右边界的跨度，包含空隙。近场先在当前朝向计算左右边界、独立开度、真实
前进行程和扫掠走廊；若当前朝向已物理可达，包络中心不论是否在 ±5 mm 内都直接进入停稳
提交。仅当前朝向不可达时才选择最小必要转角，也不要求成员
纵向包络整体落入夹爪深度。默认总开口余量 4 mm；开口宽度为
`y1 - y0 + clearance_mm`，其中左夹爪目标位置为 `y1 + clearance_mm/2`，右夹爪
目标位置为 `y0 - clearance_mm/2`，两侧分别反解舵机角度，不再用对称开角。任一侧边界
超过安全舵机行程时直接淘汰，不截断，也不通过超出真实边界的开度伪造可抓候选。必要转角
由 IMU 闭环执行一次有界粗转，进入可夹范围即制动，最多再做一次有依据的修正；没有新计划
时不会持续按上一帧视觉角旋转，`alignment_continue_max_age_ms` 只作为无运动历史时的短
动作上限。
候选通过后立即制动；
真实停稳后的一个当前新帧重新执行原始包络、危险、走廊和静态路径门禁，通过后直接
开爪。`alignment_timeout_ms` 只限制停稳和当前异步结果的总提交时间，选不出方案的空等由
`no_plan_wait_ms` 单独限制；正式配置
`confirmation_frames=1`，不再执行额外中心确认。锁定目标的身份已由 tracker 确认，某一帧
颜色证据不足（`candidate_class=None`）只保留身份和确认预算并暂停清洁几何提交，不升级为
`class_changed`；缺颜色证据不是新的任务类别，也不能作为安全物资或安全路径证据，明确
蓝色危险和颜色冲突仍然立即失效。

### 正式流程局部解团

正式 `match` 使用 `app/breakup_planner.py` 依据实际 K0 成员生成局部解团计划。算术
团中心只用于诊断，不作为撞击点；计划选择能够被夹爪前向扫掠实际接触的成员，并把
蓝色危险成员纳入连带推移和场界/安全区检查。全蓝团跳过，含蓝混合团在存在合法
抓取阻挡时可以处理。计划包含尝试编号、碰撞成员、前进/后退距离、穿入量和采集时间。
`plan_breakup(field_bounds=...)` 必须传物理场界；`_physical_field_bounds()` 从当前静态地图
读取该边界，车体/夹爪取覆盖两者的最大半径再加一次安全余量，物块只扣自身实体半径
与推移余量。正式接近、近场路径、动态解团规划及执行使用相同的物理场界，不能把旧的
`_breakup_allowed_field_bounds()`（已经扣夹爪偏移的点检查边界）再送入实体扫掠检查。
拒绝日志记录 `blocked_path`、实体类别、起终点和实际余量；已有入口计划但复核失败时
同样输出 `rejected_candidates`。
已经放入安全区的物资由 `_ground_in_safe_zone()` 按当前位姿排除在成团和瞄准点候选
之外（`plan_breakup(..., non_contact_ids=...)`），但它们仍留在目标集合里参加推移净空
检查，不能被碰撞也不能被忽略。同一判定也用在首轮优先绿色候选和 CC/联调入口的团选择上。
交付线（`safe_zone_fallback_target_field_mm` 的 y=1115）、静态地图多边形（y≥1200）和实测
地标（y=1137）并不一致，单靠场地位姿反推会把刚推进安全区的物块重新当成候选；
`_target_overlaps_safe_zone_bbox()` 直接按图像上目标框与安全区框是否重叠排除，同时用在
`_graspable_target_is_usable()`、成团 `non_contact_ids` 和补夹候选过滤上，不引入第二套
场地阈值。

正式动态解团只有一个确认闭环：先通过采集位姿历史把 K0 补偿到当前机器人系，
按物理接触射线完成 IMU 对准，再等待真实编码器/IMU停稳；
停稳后的当前帧选择一个可靠接触核心，并用当前目标一次完成危险、场界、安全区和机器人
包络检查。随后用 `match.breakup_confirmation_frames` 个不同有效帧确认同一物理核心，
同团排名改变优先保持原物理瞄准点，停稳帧更新方向时先修正再确认；角容差随接触盘
半径/距离收紧。确认成功立即冻结当前计划，连续推进到穿入终点（1 mm 行程容差），
刹车余量保留在路径外扩中，不再从物料穿入量扣除。不在团前重新停车观察，也不等待近场
抓取准备器。最小推进门槛取 `breakup_min_penetration_mm + breakup_braking_margin_mm`
与局部接触集自身跨度的较小值，浅团只要能整段推穿就成立，行程不足以推穿时仍按当时
门槛淘汰。入口时冻结的待确认计划不算数：执行计划必须由停稳后的当前帧重新生成。
停稳后仍选不出任何合法接触计划时只等
`match.breakup_no_plan_reobserve_ms` 就退出本次区域：停稳后重复观测只有在新一帧检出
新的可接触成员时才可能改变结果，占满按确认帧数计的预算纯属空等。失败记忆按物理
瞄准点记：`_resume_dynamic_search()` 只记录 `plan.aim_field`，`_aim_in_failed_aims()`
和 `_breakup_attempts` 历史都按同一瞄准点比对（容差 `cluster_group_ground_mm`），
不再用 460 mm 邻域连坐整片区域。搜索、对准、近场和重选共用同一份旋转预算：
`_advance_cluster_search_sweep()` 按绝对累计转角计时，只有真正执行完一次抓取或一次
解团推挤（见 `breakup_gripper_complete_resume_search`、`safe_zone_exit_complete_start_search`）
或确认不再有可收集信息时才由 `_reset_rotation_budget()` 重新计时，状态切换不清零。
一圈用完仍未执行任何动作且场上仍有蓝块以外的目标时，`_commit_after_rotation_budget()`
先在失败记忆有效的前提下选择其它合法接触计划并进入正常确认（`rotation_budget_commit`），
不会带上失败瞄准点直接重试。若没有其它合法计划，则启动 `RELOCATE_FORWARD`，仅在完整
换位段经过场界/安全区路径检查后执行有界平移；失败记忆仍保留，只有编码器确认真实平移后
才按改变后的几何重新评估。同一原地转向或新会话不会解除失败记录；只有确实只剩蓝块时才继续
扫描。近场路由宣告几何阻挡但当前帧已经选不出合法接触计划时不停车：
`_enter_breakup_only_search()` 直接按
`near_field_route:reselect:breakup_no_contact_plan` 回到搜索，把该物理瞄准点记入失败
记录，由搜索换目标或换区域；同一输入帧最多求一次接触计划，避免 5～10 ms 控制周期
重复跑边界搜索。逐候选淘汰原因写入
`breakup_failure ... rejected_candidates=[...]` 和 `cluster_target=...,rejection=...`
诊断。冻结后按闭爪前推、停稳张爪、后退、停稳闭爪执行。重复帧不计数，真实运动、
遥测失效、核心变化或当前几何不安全都会阻止冻结；确认超时或区域失败后恢复扫描。前推
距离由首次接触、局部穿入量和配置上限共同限制，后退保留退出净空并在刹车余量前减速。
`breakup_forward_distance_m` 和 `breakup_backward_distance_m` 在正式 match 中是行程上限，
CC 入口仍按固定距离语义。

选择器依次把不同物资作为最远 X 锚点，分别生成不同长度的扫掠走廊；橙色目标不参与
多成员组合，只生成单目标计划。每个走廊会把实际进入其中的合法物资闭包纳入。通过
危险、容量、开口和边界硬门禁后，先最大化成员数量，再按配置权重比较净空、行程和对准角。
前进行程一律使用对准后最前成员 K0 的 `x`：
`forward_distance = max(0, max(member.K0.x) - endpoint)`，其中绿/黑组的 `endpoint`
是 `target_final_x_mm`，单个橙色目标是 `orange_target_final_x_mm`。橙色不再使用
上表面颜色投影的最近端加跨度（`x0 + d`）：颜色掩码的纵向拉长不是真实前进深度，
可靠底面中心 K0 才是权威。这允许目标在揽入过程中相互滑动/转动；
`max_forward_distance_mm` 仍限制底盘动作时长。危险检查使用单个矩形走廊，纵向范围为
`[corridor_start_x_mm, max(left_tip.x, right_tip.x, corridor_start_x_mm) + forward_distance]`，
即前端包含按实际开口反解出的夹爪末端前伸距离；横向范围为
`[right_tip_y - corridor_lateral_margin_mm, left_tip_y + corridor_lateral_margin_mm]`。
`grasp_candidate` 诊断的 `corridor_end_x_mm` 与执行走廊共用同一公式。走廊只覆盖夹爪
张开宽度将要扫过的区域，不再覆盖车体或夹臂的历史全扫掠包络。

绿/黑候选不再因蓝色侧邻被直接淘汰：蓝色只按实际扫掠走廊判定——K0 加
**内切半径**是否进入夹爪开口将要扫过的矩形；没有其它安全方案时正式路由进入解团。
`_side_neighbor_metrics()` 的纵向/横向中心差门禁只用于单橙计划的侧邻检查，明显前后
错开的目标不触发。橙色侧邻不影响绿/黑计划；橙色独立性门禁只约束单橙计划。纯绿黑目标
即使分布密集，也不会因为彼此接近被拆散，仍允许最多三个成员的多目标计划。

蓝色危险目标、缺几何目标及类别冲突目标进入走廊时直接拒绝，不能用评分抵消；蓝块按
内切半径收缩净空，即保证被实体占据的圆盘进入扫掠矩形才判为阻挡，不再用外接半径把只是
近旁的蓝块当成阻挡；当前
扫掠内缺少 K0 的蓝色危险目标也不能证明走廊安全，直接拒绝当前提交；橙色只能
作为单目标计划，不能作为额外成员加入绿黑组合。橙色目标地面中心周围
`orange_isolation_radius_mm` 内有当前可定位的其它目标时拒绝；其它
目标未观测或没有当前地面点时不无条件推定其位于禁区。额外绿黑的 K0 落入走廊时加入组合并
重新检查数量和横向开口；蓝色危险、缺几何及类别
冲突目标只在静止规划阶段影响选组。最终确认通过并提交计划后不再用运动中的检测重检走廊。规划
停车后的近场走廊门禁只使用当前帧仍有可靠 K0 地面点的目标；当前帧缺少地面点或已经
失观的历史轨迹不参与该走廊阻挡，避免侧方静态目标的历史膨胀包络制造幽灵障碍。全局
tracker 仍保留短时历史，明确危险证据也继续在轨迹存续期内保留，但二者不替代近场停车
窗口所需的当前空间证据。

橙色从候选生成阶段起只形成单目标方案；绿/黑只彼此组合。橙色落入绿黑走廊、或绿黑落入
橙色走廊时都只作为阻挡物，不能顺带加入另一类方案。远场 handoff prior 只唯一匹配关联门限内
K0 最近的局部目标；首轮单绿优先合法交接目标，也可选择其它合法单绿。一般阶段在通过
容量、机械开度和完整扫掠检查的组合中，依次优先成员数量、黑色数量、规则总分、单橙优先级、
交接匹配及净空/行程/对准代价。绿黑并列可共同收拢时优先多抓，橙色仍严格单独转运。
补夹阶段的 handoff prior 则是锁定入口目标：全场绿/黑候选按机器人地面距离优先，目标确认
后不因新出现的更近目标改选；锁定目标失观或本次动作明确失败后才接管下一个候选，并重新走同一对准、接近
和近场确认链。

| 次级特征 | 默认权重 | 特征值 |
| --- | ---: | --- |
| 单橙优先 | 0.55 | 单橙为 1，其余为 0 |
| 数量 | 0.10 | 数量 / 3 |
| 净空 | 0.15 | 扣除余量后的最小净空 / 100 mm，上限 1；无邻近障碍取 1 |
| 距离 | 0.12 | `1 − 行程 / 最大行程`，截断至 `[0,1]` |
| 对准 | 0.08 | `1 − |转角| / (π/2)`，截断至 `[0,1]` |

### 可复用接口、时间与生命周期

`measure_target_envelope()` 输出单目标 `TargetGroundEnvelope`，不做居中或规则选择。
`NearFieldGraspSelector.select()` 输入带 tracker ID 的目标和可选的
`NearFieldGraspPolicy`，输出 `GraspSelection`，包含
可选 `NearFieldGraspPlan` 和拒绝原因。计划包含成员、帧号、采集时间、对准点、组合边界、
开口/最大开口、舵机角度、行程、评分分项和动作区域。所有地面点为机器人系 `GroundPoint`，
单位 mm；时间为相机与应用共用的单调时钟 ns。`bounds` 是计划对准后的机器人系，
`regions` 是该计划采集帧的机器人系，不能当作场地坐标。

`GraspPreparationSession` 串行消费新帧，复用 `GraspTargetTracker` 和现有 tracker。
该准备器只服务近场抓取，不属于正式动态解团的确认前置条件。
正式 match 在刹车 settle 完成后打开唯一确认窗口：同一目标/组合 ID 的不同有效帧按
`confirmation_frames` 累计，同一帧不会重复计数；控制循环更快时保留已有进度。首轮
的 handoff prior 把远场已选定的单绿作为近场排序偏好，该目标当前帧无法生成合法计划
（失去颜色包络、越过近场半径或被走廊硬门禁阻挡）时由下一个合法绿色接管，避免为一个
不可交付目标空转到超时；一般阶段继续由
选择器按规则分值选择绿/黑组合或单橙。确认期间同一 ID 的最新有效几何更新开口和行程，
出现明确危险或不合法组合才使确认失效，短暂漏检只等待当前 ID 的新证据。
正式补夹要求近场计划保留 handoff prior 对应的入口目标；目标仍被观测但几何不合法时不换组，
锁定前目标确实失观，或本次动作明确失败后才选择下一个全场绿/黑目标；失败保留物理记录与原扫描预算。
一般阶段交接目标旁有当前可选但尚未完成局部轨迹确认的绿/黑物资时，先在原有有界
窗口内等待相邻物资确认，再选组锁定；首轮单绿不使用此等待。无关候选的拒绝诊断
不清空合法计划的确认。静态场界/安全区路径失败返回搜索重选，不触发解团；入口
路径检查复用近场静态边界门禁，失败日志附估计位置、航向、行程、边界和余量。
开爪提交时冻结执行计划；提交门禁分别记录计划采集年龄和准备结果年龄，过期计划不能
冒充新结果。
生产入口在每个真实 UART `OdometryImu` 到达时调用
`MatchSequence.observe_grasp_motion(message)`；独立入口调用
`GripperWidthPickupSequence.observe_motion(message)`，随后再调用 `step()`。
静止证据要求两轮原始计数不变、IMU有效且 `abs(gyro_z)` 不超过
`near_field_grasp.stationary_max_gyro_rad_s`（默认 0.03 rad/s）；无效、饱和、重复/倒退
样本及超过 `grasp_commit_max_observation_age_ms` 的遥测间断均打断静止区间。
当前计划必须在已证实静止之后采集，且直到当前遥测一直静止；其真实采集年龄使用
`processing.max_observation_age_ms` 作为失联上限（match 配置为800 ms），不再强制低于150 ms。
计划必须来自当前准备快照且成员实际被观测，不能给缓存旧计划换发布时间续期。
准备完成后的排队年龄和最近遥测年龄仍受150 ms配置限制；确认总预算不因新帧重置。
未注入遥测的纯逻辑调用只能使用原有短龄计划路径，不能使用静止有效期。
日志增加 `stationary_since_ms`、`motion_age_ms`、`gyro_z_rad_s`，用于区分正常处理延迟、
底盘仍在移动和遥测断档。零速命令不是静止证据。

NearFieldGraspConfig 提供通用默认值，正式 match 的 YAML 是权威档位。确认只使用同一物理成员的当前有效几何，
不跨目标或跨身份混合；tracker ID 短暂变化时由近场会话按 K0 空间连续性保留锁定身份。
近场锁定后最多执行一次粗转和一次必要修正；提交前使用锁定物理成员的最新有效几何更新
开口和行程。偏置目标只按组几何生成对准预览，不使用中心 ±5 mm 作为提交条件；转角完成后
重新计算横向组宽、夹爪开口和前进扫掠，再做完整障碍检查。
开爪前失效会解锁重选，限定时间仍无有效组则回到带原因的正式搜索，
不会持续停在 `ABORTED`。单帧颜色证据不足或质量异常不再累计疑似/恢复计数：模型类别是
权威，颜色只提供几何；明确危险证据保持到轨迹消失。
确认成功后短暂漏检或旁路异常不单独取消动作。
正式流程为每趟准备结果附加会话 ID；首趟通过 `NearFieldGraspPolicy` 限制为单个绿色，
后续趟次允许绿色/黑色 1～3 个或单独 1 个橙色伤员。`MatchDecision` 可携带精确双舵机角度、
单次轮速下限和 `soft_brake` 意图，运行时据此保持动态开口并避免重复刷写命令。

生产装配见 `gripper_width._run()` 和 `match_runtime._run_hardware()`：从
`configs/runtime.match.yaml` 构造唯一几何对象、
`GraspTargetTracker(config.tracking.build_tracker(), projector, config.near_field_grasp)`、
`NearFieldGraspSelector` 和 `GraspPreparationSession`。后两者不打开硬件。准备器放在
单个后台线程串行调用 `update(snapshot, locked_ids=..., handoff_prior=...,
require_handoff=..., excluded_observation_indices=...)`；索引只改变候选资格，不从 `targets` 删除观测，
因此已携带/已交付物资和蓝色危险仍参加扫掠障碍检查。索引在同帧去重和锁定 ID
空间映射后仍绑定对应物理观测。
队列只保留最新一帧。全候选诊断独立线程使用一个最新结果槽，最多每秒计算一次；
退出上下文时唤醒并回收准备/诊断线程。正式流程还会校验准备结果的近场会话 ID，迟到结果不会跨运输轮次
驱动动作。运动层在自己的循环调用
`GripperWidthPickupSequence.step(now_ns, preparation, cumulative_distance_m=..., heading_rad=...)`，将返回的
速度/角度意图交给已装配的控制器。`GripperWidthPickupResult` 返回成员 ID/类别、完成时间、
最终舵机命令及未确认收拢标志；旧单目标 `GripperWidthPickupPlan` 已替换为多目标计划。

开爪提交时冻结唯一确认窗口形成的执行计划，不再把运动中的目标检测送入走廊或成员包络复核，
因此运动模糊、颜色证据不足或目标遮挡不会单独打断已提交动作。编码器仍按动作计划定距，
并保留关键里程计不可用、编码器方向错误和持续无前进进度时的退出保护；明确急停、硬件
健康故障和进程异常仍走软刹车路径。开爪阶段底盘保持静止，张爪计时结束后直接按冻结
计划进入 `forward_open_loop`。动作退出后必须用退出之后采集的新证据重新选组，不能直接
重放旧计划。
合爪指令提交后只等待计时并保持底盘静止，遮挡不会自行重新抓取；不把视觉消失当成成功
证据。`rx_degraded` 只保留在日志诊断中。

`GraspPreparation` 的 `confirmation_progress`、计划的 `capture_timestamp_ns` 和
`prepared_timestamp_ns` 分别用于确认进度、计划年龄和准备结果年龄，
`result_timestamp_ns` 保留推理结果的真实完成时刻；
两种年龄均来自单调时钟，不能由控制循环改写。

相机/推理、规划、预览和有界终端日志均在旁路。GUI 或日志失败被报告并隔离；规划或感知
不可用时由动作层根据剩余有效证据决定退出。退出先软刹车，再结束准备线程、相机、
UART 并恢复信号处理器，夹爪保持最后一次显式命令。预览标记成员 ID、禁止卷入目标、
动作包络、开口、最大开口和评分；不代表已经验证无碰撞。

运行前必须检查相机标定、Hailo 模型、UART、双舵机安全端点、里程计和轮距。
4 mm 总开口余量、`target_final_x_mm`、`corridor_start_x_mm` 和 10 mm 横向走廊余量都是
试验初值。掩码含离地表面、真实夹臂厚度、车头形状、目标滑动和有效前进行程尚未实测
验证，不能用这些近似几何证明真实抓取安全。实物收拢效果、真机控制响应时间、夹持成功率、
解团效果、危险类表现和目标设备延迟均为**未验证**；验收步骤见 `manual_tests/README.md`
的近场收拢章节。

两个正式流程入口都要求显式的物理急停和全程监督确认，直到固件看门狗和急停闭环完成
真车验收。正常退出、信号、相机、UART、网络或线程异常都进入软刹车、停止旁路和资源
释放路径。

## `rescue-vision-match` 使用方法

先确认 `configs/runtime.match.yaml` 已按当前车辆填写，并满足以下装配条件：

- `match.enabled: true`；
- `motion`、`motion.odometry` 和 `motion.gripper` 已启用并完成机械标定；
- `geometry.ground_mapping_enabled: true` 且地面标定与当前相机分辨率匹配；
- `hailo.enabled: true`，模型路径和六类 v3 输出顺序有效；
- `remote.enabled: true` 时使用 `role: server` 与 `access_mode: observe_only`。

在仓库根目录安装后运行：

```bash
python -m pip install -e '.[dev]'

rescue-vision-match \
  --config configs/runtime.match.yaml \
  --start-area 2 \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

参数含义如下：

| 参数 | 作用 |
| --- | --- |
| `--config PATH` | 必填；正式流程 YAML，通常为 `configs/runtime.match.yaml` |
| `--start-area {2,3}` | 启动区域；2 为地图右上角红方，3 为地图左下角蓝方，默认 2；3 会把场地 `FieldPoint` 和初始位姿绕 `(0,0)` 中心对称 |
| `--supervised-physical-stop-ready` | 确认物理急停已就绪且操作者全程监督；固件看门狗未完成验收前必须提供 |
| `--local-preview` | 打开本地 OpenCV 预览，显示最新感知帧、状态和动作原因；按 `Q`/`Esc` 退出 |
| `--jpeg-quality N` | 观察图传 JPEG 质量，范围 `1..100`，默认 `80` |
| `--observer-image-interval-seconds S` | observe-only 图传最小发布间隔，必须为正数，默认 `1.0` |
| `--log-dir PATH` | 按时间写入流程日志和 D2 遥测，默认 `logs/`；传入 `--log-dir /dev/null` 不适用，需使用有效目录 |

流程启动后会先执行预检、启动转向/直行和目标搜索。搜索态的机会检查使用固定
`TRANSPORT` 夹爪包络的前向走廊：首轮按距离尝试单个绿色 K0；若候选走廊被蓝/橙/黑目标阻挡，先尝试其它安全绿色，只有没有合法抓取方案才进入局部解团；一般阶段支持绿/黑物资组或单个橙色伤员快速入口。固定走廊外的合法目标先生成
近场直接可达性预览；当前方向没有合法方案时才继续目标团解团，不再增加一次近场旋转。远场和近场等待阶段保持 `CLOSED`，不会先发固定开口；近场
计划 ready 后才发送动态开度。规划期间的同帧画面叠加半透明走廊：黄色为对准预览、
青色为复核、绿色为 ready、红色为阻挡；match 中任何已从多个候选中选出的远场目标、解团目标团或近场成员框，
均以橙色粗线和 `SEL#track_id` 标记，
组中心显示十字并沿对准方向画箭头。本地预览和观察图传使用同一叠加帧。终端状态行包含当前状态、动作原因、
编码器距离、航向、目标轮速、绿色走廊诊断、安全区阶段和 D2 遥测丢弃计数。观察图传、
地图状态和本地预览只读取最新数据，不参与运动决策。

安全区 bbox 追踪使用 `safe_zone_bbox_turn_kp_rad_s` 的归一化比例控制和
`safe_zone_bbox_turn_max_angular_velocity_rad_s` 限速；
`safe_zone_bbox_turn_deadband_ratio` 与 `safe_zone_bbox_turn_resume_ratio` 提供中心滞回。
追踪换向先等待轮速停稳，再接受停车后的新帧。关键点缺失时使用
`safe_zone_keypoint_reobserve_timeout_s` 静止重观测；中心 bbox 仍缺点时最多进行一次同向低速扫视，扫描后重新停车；
关键点出现后停车取样，
重复帧不能延长窗口。

安全区末段的绿/黑物资使用 `match.safe_zone_d2_to_final_speed_m_s` 和
`match.safe_zone_d2_to_final_braking_overrun_mm`；单个橙色伤员使用独立的
`match.safe_zone_orange_d2_to_final_speed_m_s` 和
`match.safe_zone_orange_d2_to_final_braking_overrun_mm`。解团阶段的临时最大轮加速度由
`match.breakup_max_wheel_acceleration_m_s2` 配置，`null` 表示继承 `motion` 全局值。

正常结束、`Ctrl+C`、`SIGTERM`、相机/Hailo/UART/网络异常或旁路线程失败都会进入统一
软刹车清理路径。重新运行前应确认车辆已停稳、急停状态已复位，并重新提供显式监督确认。

如果只需联调绿色物块夹取、末端张爪推送和安全区运输，使用 `rescue-vision-grab-transport`；该入口复用
同一配置，从 `(0,0,+90°)` 直接搜索固定 `TRANSPORT` 走廊内的单个绿色 K0，不执行目标团解团。

## `rescue-vision-grab-transport` 使用方法

该入口用于在目标已经摆散时单独联调绿色物资夹取、d1 视觉纠偏、安全区末端张爪推送和退出。
车辆按正式流程先夹取并运输；到达安全区末端后不再重复闭合夹爪，而是保持张开把物块推入安全区，
再保持张开退出。它仍要求与正式流程相同的
Hailo、地面映射、UART、里程计和夹爪标定；配置检查仍使用
`match.enabled` 以及同一组运动安全限制。
末端推进不再重复闭合夹爪；前段夹取、d2 张爪、车辆停稳、视觉校准和路径安全门禁仍按原流程保留。

```bash
rescue-vision-grab-transport \
  --config configs/runtime.match.yaml \
  --supervised-physical-stop-ready \
  --local-preview \
  --log-dir logs
```

入口会覆盖本次联调的受限初始位姿：场地 `FieldPoint(0, 0)`、航向 `+90°`，跳过正式
流程的固定启动转向/直行，预检通过后立即进入绿色目标搜索。只接受固定 `TRANSPORT`
开口前向走廊内的单个绿色 K0，并继续检查安全区和场地边界；搜索不到合格目标时保持原地搜索，
不会转入目标团解团。

命令行参数与 `rescue-vision-match` 的通用参数相同；`--start-area` 仅用于正式比赛入口。
`--config` 指定正式流程配置，
`--supervised-physical-stop-ready` 确认物理急停和全程监督，`--local-preview` 打开本地
预览，`--jpeg-quality` 和 `--observer-image-interval-seconds` 控制观察图传，
`--log-dir` 指定流程日志和 D2 遥测目录。正常退出、`Ctrl+C`、信号或硬件/旁路异常均
使用同一软刹车和资源清理路径。

## 正式流程行为

`MatchSequence` 以 `GroundPoint`、陀螺仪航向和编码器累计距离完成：

- 启动转向/直行，随后搜索目标团；首轮按距离优先尝试可靠单绿，走廊不安全时先换安全候选，一般阶段优先尝试可抓取物资组，没有合法方案才局部解团。
- 搜索不为绿色额外转满一圈；无绿/黑/橙信息，或当前帧所有有效非蓝目标的 bbox 中心都在安全区 bbox 内时快速扫描，
  只有出现安全区 bbox 外的可转运类别信息后才降速并进入统一评估。
- 首轮解团候选优先包含规则要求的绿色，一般阶段只按成员数选择完整目标团；候选的场界、双方安全区和机器人包络门禁失败时
  继续尝试下一完整团；全蓝色危险目标团不进入解团，继续搜索其它候选。
- 解团进入 `BREAKUP_SETTLE` 后保持零速，等待真实停稳；用停稳后的当前帧选择可靠接触核心并完成一次完整危险/边界检查，
  再按 `match.breakup_confirmation_frames` 个不同有效帧确认，冻结后直接进入前推、张爪、后退。它不再执行单独对准、接近、
  接触后第二轮复核，也不依赖近场抓取准备器。近场路由判定阻挡但当前帧选不出接触计划时不停车，直接带原因重选并把
  区域记入失败记录；同一输入帧最多求一次接触计划。远场目标的常规近场抓取仍在 `near_field_grasp.max_range_mm` 后进入近场状态机。
- 搜索旋转预算在一圈内没有有效动作时不绕过失败记忆：先选择其它合法解团计划；没有合法计划时进入经过场界/安全区路径检查的有界
  `RELOCATE_FORWARD` 换位，真实平移后才重新评估同一物理目标。原地转向和新会话都不会解除失败记录。
- 首次运输限制为单个绿色普通物资；首次交付后允许绿色/黑色物资 1～3 个，或单独转运 1 个橙色伤员。蓝色、
  蓝色危险和缺几何目标在当前帧有可靠地面点时按实际扫掠判定，缺少 K0 的当前危险目标不能被当作安全而会阻止提交；橙色隔离半径内的邻居需要按实际包络复核；危险或扫掠侵入仍拒绝橙色计划，缺少 K0 的无关合法目标不参与距离推定。
  候选方向的走廊横向门限按两块实体内切半径之和加实际夹爪余量逐目标计算；只是近旁、不在夹取路径上的异类目标不否决候选，蓝块实体仍参与扫掠阻挡。
- 正式 match 一般阶段绿/黑收拢后进入 `TRANSPORT_GREEDY_SCAN`，闭爪按 `match.spin_angle_rad`（正式配置 270°）扫描：当前帧没有可补夹的绿/黑信息时用 `match.cluster_search_empty_angular_velocity_rad_s` 快速转动，出现可补夹信息后回到 `match.close_gripper_spin_angular_velocity_rad_s`。扫描命中即进入与搜索阶段相同的全局夹取前置链（对准→接近→近场唯一确认窗口）并成为近场交接先验；每命中一次就累计容量并直接携已有物资返程，不再重复整圈扫描（最多 3 个）。失败候选留下物理失败记录，在原扫描预算内继续寻找其它近场或远场目标；预算耗尽后返程。首趟、单橙、CC 和单绿联调保持各自规则。近处已收拢区观测继续参与障碍检查，详见[正式流程设计](../../../docs/正式流程设计.md)。
- 补夹候选范围为全场绿/黑，不受 `near_field_grasp.max_range_mm` 截断，按机器人地面距离优先；
  目标确认后保持交接目标不变，目标失观或动作明确失败后才接管下一个补夹目标，并重新走正常对准、接近和近场确认链。
- 抓取结果为橙色时路线使用 `match.safe_zone_injured_target_field_mm`；绿/黑使用 `match.safe_zone_fallback_target_field_mm`。
  选择 `--start-area 3` 时这些终点和 d1/d2 直线会相对中心十字对称到负 y 侧，动作切换由零速保持和陀螺仪航向保持保护。
- 夹取后若 y 已沿己方方向越过 d1，则原地开始纠偏，否则沿 d1 直线前进。先按误差比例限速旋转到安全区
  bbox 的画面水平中心；中心处若三个关键点仍未出现，则静止重观测并最多同向低速扫视一次，关键点出现后
  停稳，再以 K0/K1/K2 和静态红蓝安全区角点拟合 `FieldPose2D` 并覆盖当前航位；d2 与末段继续沿
  校正位姿积分。释放并直线退出后再次完成同样纠偏，直接进入搜索。

视觉纠偏只使用 d1 停稳期间的当前安全区观测。正式流程当前不接入跨帧视野外目标记忆、
记忆目标接管或旧的 20 分编排；正式 match 的远场/对准链路会使用已记录的采集时刻
编码器/IMU 位姿把目标几何更新到当前机器人坐标。

## 纯逻辑装配

```python
from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.match.yaml")
flow = config.build_match_sequence()
```

联调入口的类型和装配方法为 `GrabTransportSequence` 与
`config.build_grab_transport_sequence()`。它覆盖初始场地位置和航向，仍复用正式流程
配置、跟踪器、夹爪标定和安全区视觉纠偏。

`MatchNBSequence` 与 `config.build_match_nb_sequence()` 对应
`rescue-vision-match-nb`。它直接继承当前 `MatchSequence`，只替换区域 2 开场：原地转向并
前进到 `(100,800)`，张爪，原地转向并前进到 `(100,-900)`，保持当前朝向倒车到
`(100,0)`，随后进入正式目标团搜索。该入口不提供区域 3 中心对称。

两段前进各只在起步前转向一次，倒车前不再转向。每段航向在起步时按当前场地估计固定，
行驶中仅做普通 IMU 航向保持；不做横向前视、制动提前量或途中重新停车对准。每个航点
到位后只保留 `nb_opening_settle_time_s` 的短暂停稳，用于切换动作。

配置字段（`match` 节，全部可独立配置）：

| 字段 | 单位 | 说明 |
| --- | --- | --- |
| `nb_opening_first_target_field_mm` / `nb_opening_first_speed_m_s` | mm / m/s | 第一航点绝对场地坐标与推进速度 |
| `nb_opening_gripper_left_deg` / `nb_opening_gripper_right_deg` | deg | 第一航点到位后的张爪左右角度 |
| `nb_opening_second_target_field_mm` / `nb_opening_second_speed_m_s` | mm / m/s | 第二航点坐标与推进速度 |
| `nb_opening_reverse_target_field_mm` / `nb_opening_reverse_speed_m_s` | mm / m/s | 倒车回退航点坐标与倒车速度 |
| `nb_opening_align_tolerance_mm` | mm | 航点到达容差 |
| `nb_opening_heading_tolerance_rad` | rad | 起步转向完成门限 |
| `nb_opening_heading_kp_rad_s` | rad/s per rad | 转向与行驶航向保持的比例增益 |
| `nb_opening_heading_max_angular_velocity_rad_s` | rad/s | 行驶航向保持的角速度上限 |
| `nb_opening_align_angular_velocity_rad_s` | rad/s | 起步原地转向的角速度上限 |
| `nb_opening_settle_time_s` | s | 每个航点到位后的零速停稳时间 |
| `nb_opening_align_timeout_s` | s | 起步转向超时后停车 |

车端诊断：`MatchNBSequence.nb_opening_route_phase` 返回当前开场阶段，`nb_opening_diagnostic`
返回一行「阶段 / 目标航点 / 估计位置 / 剩余误差 / 航向」。`match_runtime` 在阶段变化时
打印一次，例如：

```
nb_opening=phase=turn_and_move_to_first target=(+100,+800)mm \
  position=(+1350,+1350)mm error=(-1250,-550)mm heading=-90.0deg
```

`error` 是现场判断“车辆实际落点是否偏离航点”的唯一读数：只看 state/reason 横幅无法
区分策略没走到位，还是航位推算本身已经漂移。开场结束后两个属性都返回 `None`，不会在
解团/运输阶段产生误导读数。

## 蓝色优先策略变体

`MatchStrategySequence`（`app/match_strategy.py`）是 `MatchSequence` 的**子类**，
入口为 `python -m rescue_vision.app.match_strategy --config configs/runtime.strategy.yaml`
（当前没有 console script）。它分两个阶段：

- **策略阶段**（`_strategy_formal_phase is False`）：开场两段反向冲刺（右转短冲 →
  左转长冲，转角与速度来自 `startup_turn_*` / `cluster_relocate_*`），随后**只搜蓝色
  危险物块**；能直接夹取则单块转运，否则对全蓝目标团执行动态解团。两趟分别放到对面
  安全区左右 D2 点（`safe_zone_fallback_target_field` / `safe_zone_injured_target_field`），
  在 D2 释放而不推进到安全区深处。
- **正式阶段**：第二趟返回结束（共享实现给出 `FINISH_STOP`）时由 `_step_return_backup`
  拦截并调用 `_begin_formal_match_after_strategy`，恢复己方 y 方向的 D2 端点、重新打开
  首轮单绿机会、复位旋转预算与解团失败记忆，然后进入正式绿块搜索。

继承与委托的边界：正式阶段通过 `_SharedMatchSequence.<method>(self, ...)` 显式委托回基类，
策略阶段使用本地实现；`__init__` / `start` 只 `super()` 委托后追加 `_strategy_*` 字段，
不再复制基类字段清单。策略阶段 `near_field_enabled` 返回 `False` 并临时把
`_near_field_pickup` 置空（真身存在 `_strategy_saved_near_field_pickup`），蓝色近场计划
由 `_strategy_blue_grasp_preparation` 在控制线程内直接生成；翻转时恢复。

`_step_align_green` 是唯一按相位显式分派的对准入口：共享实现在 `_near_field_pickup`
不为 `None` 时会改走 `_step_formal_green_align`，而蓝色运输恰好会重新暴露该序列，
因此变体在 `_strategy_blue_transport` 期间固定走 `_align_green_legacy`（真机验证过的
历史对准路径），其余阶段才走共享实现。

## 可复用但未接入的模块

`scan_target_memory.py` 保存带场地点和老化时间的可靠目标，可在未来正式流程明确采集
时刻位姿策略后接入；当前不参与任何运动决策。`field_target_cluster.py` 提供带身份的
`FieldPoint` 连通聚类，也只保留为可复用逻辑。两者均有独立 pytest，不能被解释为已接入
正式比赛能力。

## 观察与日志

`match_observers.py` 提供不修改原帧的本地叠加和 observe-only 远程发布；
`session_log.py` 把标准流 tee 到按时间命名的日志文件。D2 到安全区末段的
`D2TelemetryLogger` 逐帧记录编码器/IMU 原始数据、时间差、阶段和丢弃计数，旁路队列
拥塞时不阻塞运动循环。

正式流程设计、状态顺序和视觉纠偏验收见 [`docs/正式流程设计.md`](../../../docs/正式流程设计.md)。

### 一般阶段选目标与重复等待修复

正式 `match` 首次有效交付后统一比较固定走廊内外的接近种子：近场范围内优先，
同层按配置的类别分值降序、距离升序。进入近场后仍由同一选择器按组合总分确定
绿黑1～3个或单橙；首轮单绿策略不变。种子、重接近和对准复核复用目标方向走廊，
仅物资组可忽略可共同收拢的绿/黑成员，伤员路径不能忽略其它目标。
正式入口的已确认跟踪点直接用于当前方向可达性检查；停稳后只用一个当前帧复核，不重复等待十帧。

`orange_isolation_radius_mm` 数值未降低。入口只把可信侧后方绿/黑邻居交给精细复核，
不据此提交开爪。`NearFieldGraspSelector` 要求邻居完整当前包络全部位于目标方向的
橙色中心后方，横向完全位于夹爪扫掠宽度及配置余量之外，且两物体包络距离大于
`clearance_mm`。蓝色危险、缺几何、质量异常、缺包络或侵入者不使用例外。
这修正的是圆形邻近区把侧后方物资误判成混运的工程策略，不改变伤员单独运输规则。

### 现场 20260912 停等、空转与组合抓取回归

近场交接清空主选中 ID 后，受阻重选仍从 handoff prior 保存原目标身份和物理失败核心，
不会重新选择同一未变化目标。远场接近目标丢失或重新编号时优先接管当前合法候选；
旧参考点不再反复重置编码器起点。接近复核复用选组入口的可共同收拢绿黑成员规则，
最终开爪仍由近场选择器按实际包络、剩余容量和完整危险扫掠决定。首次仍为单绿，
一般阶段可抓取绿绿、绿黑、黑黑及容量内更大组合；侧邻蓝块不单独否决单绿。

一圈无有效动作后，即使当前视野为空或只剩蓝块也尝试有界换位。当前方向不通时保留
耗尽预算继续找方向；同帧不重复规划。换位前和运动中检查场界、安全区、车体与夹爪
实体前伸、制动余量和目标实体。详细行为与时间语义见[正式流程设计](../../../docs/正式流程设计.md)。
这些回归为无硬件逻辑验证，实际抓取成功率及目标设备端到端时延仍未验证。

搜索候选的场界与路径检查使用采集位姿补偿后的当前 K0；换位途中可交接合法目标。近场锁组前对尚未确认的绿黑同伴使用实际开度和行程检查共同收拢可能性，仍保留原确认截止时间与完整扫掠门禁。
