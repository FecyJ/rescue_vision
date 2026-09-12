# match 1317 修复进度（2026-09-12）

用户因额度不足要求暂停，本文用于续接。**纯逻辑修改已完成自动验证；带载机械效果、真车时延和现场规则闭环仍未验收。所有修改尚未提交 Git。**

## 当前用户目标

依据 `logs/match_20260912_1317.log` 和录像：
`/home/rescue/.codex/attachments/23d4c7d3-5a10-4047-9e27-ea9cc04e09e1/屏幕录制 2026-09-12 132601.mp4`。

1. 录像 1:53 后看得到远场绿黑却反复旋转，有时靠近安全区。用户明确要求：**与安全区 bbox 过滤匹配的物体，从任何搜索和规划完全排除，不再保留为规划障碍。**这条新要求覆盖仓库旧文档中“已交付物资继续参与障碍检查”的对应 bbox 情况。安全区本身的静态路径约束仍保留。
2. 录像 0:14、1:26 面前可一次揽入的双绿只抓一块；0:43 左侧双绿也输给单块。用户要求在机械最大开度、合法路径和最多三个的约束内，**优先一次抓取数量最多的组合**。首次单绿、伤员不混运及场外蓝块危险检查仍有效。
3. 保留此前 1140、1236、1258 三份日志对应的修复，不回退已有工作。
4. 补夹与正常夹取使用同一对准/接近/近场确认链；补夹候选允许全场并按距离优先，目标确认后不因新目标出现而改选，只有目标失观才接管下一个。

## 已确认的证据

- 录像 0:14 对应日志约 252 秒：会话 33 已锁定单块 `(3,)`；主 tracker 的相邻绿块 K0 约 `(459.7,57.5)` mm，另一块约 `(416.7,12.4)` mm。前者距离约 463 mm，超过 450 mm 近场入口半径，但组合所需前进行程可仍在机械限制内。旧选择器逐成员执行半径门禁，导致组合提前淘汰。
- 录像 1:26 截帧约 323955 ms，画面曾显示双成员计划；随后日志会话 45 在约 324924 ms 锁定 `(2,)`，325929 ms 按单块开爪。需继续核对从双成员预览退为单块的具体路径，不能仅凭一张截图断定最终计划一直是双块。
- 录像 1:53 后对应约 353 秒。已查看 114 秒截帧，安全区 bbox 内有已交付绿块。日志后段仍多次搜索/换位，可见远场绿块及解团 `local_push_not_clear_or_no_penetration`。
- `_step_dynamic_cluster_search` 原先在 `_breakup_only=True` 时跳过 `_try_dynamic_grasp`；`_try_dynamic_grasp` 原先遇到有效近场准备结果但无可执行计划时直接返回，可能挡住远场候选。
- 本次录像截帧在 `/tmp/match14.jpg`、`/tmp/match86.jpg`、`/tmp/match114.jpg`，仅辅助检查，不提交。

## 本次已写入、尚未完成验收的修改

### `src/rescue_vision/app/match.py`

- 新增 `planning_perception(perception)`：删除与任一安全区 bbox 重叠的任务观测，保留原图、时间、场地特征；不修改原始 snapshot。
- `step()` 开始先调用上述过滤，主 tracker、搜索、解团、换位等使用过滤后的输入。
- `_target_is_fresh()` 同时排除与当前安全区 bbox 重叠的轨迹，防止刚过滤的目标以 coasting 旧轨迹继续参与判断。
- 动态搜索即便 `_breakup_only=True` 也先尝试合法抓取，物理失败记忆仍有效。
- `_try_dynamic_grasp` 无有效近场计划时继续检查远场/接近种子。

### `src/rescue_vision/app/match_runtime.py`

- 近场 worker 提交前使用同一个 `planning_perception()`，并对过滤后的 snapshot 重新计算 excluded indices，避免索引错位。
- 原始观察显示仍可保留安全区物块。确认这里及所有其它 worker 提交入口没有漏掉过滤。

### `src/rescue_vision/app/near_field_grasp.py`

- `_group_geometry`：绿黑多成员组只要求至少一个成员在近场入口半径内，其余成员由实际最大开口、正向位置和最大前进行程限制；单块仍用原半径规则。
- `_plan_rank` 第一排序项改为 `-len(plan.members)`，其次规则分值、单橙同层优先、handoff 和原有几何评分。
- 注意：当前实现也会让合法多物资组合优先于单橙；这是按用户“尽可能多”实现的选择。文档仍有旧“同分单橙优先于三绿”的文字，必须同步明确最终策略。
- 不允许用扩大机械行程/安全阈值替代组合规划。
- 新增 `require_handoff`：正式补夹在近场计划锁定前必须保留已选交接目标；交接目标失观才返回 `handoff_target_missing` 供上层接管下一个目标。

### 本轮补夹修改

- `TRANSPORT_GREEDY_SCAN` 移除 `max_range_mm` 的发现半径限制，复用正常入口候选门禁，按全场机器人地面距离选择最近绿/黑目标。
- 目标确认后，近场选择器要求计划包含该 handoff；新出现的更近目标不会替换它。目标失观时按同一正常链路重新选择下一个全场目标。
- 目标仍存在但 K0/颜色包络暂时缺失，不视为消失，不切换目标。

### 测试

`tests/test_near_field_grasp.py`：
- 旧“单个 handoff 黑块优先于双绿”测试已改为双绿优先。
- 旧“同 15 分单橙优先”两个参数用例已改为多成员优先。
- 新增跨 450 mm 入口的绿绿/绿黑/黑黑组合、过长行程仍拒绝、三成员优先、整组都在入口范围外仍拒绝。

`tests/test_match_breakup_latency.py`：
- 新增四类安全区 bbox 物体从规划快照完全删除的测试。
- 新增 300/600 ms 延迟、250/400 ms 帧间隔、5 ms 控制轮询下，解团搜索模式仍接近远场绿块的测试。

## 最新测试结果

使用 `.venv/bin/python`；系统 `python` 没有 pytest。

已执行：

```bash
.venv/bin/python -m pytest -q tests/test_near_field_grasp.py tests/test_match_breakup_latency.py
```

结果 **1239 passed**；`compileall` 和 `git diff --check` 均通过。

## 后续必要工作

1. 真车复核补夹全场候选、距离优先和目标锁定/失观接管时序。
2. 继续检查单块锁定是否仍在组合成员短暂未确认、半径抖动或准备结果返回时过早发生。已有邻居确认等待逻辑在 `NearFieldGraspSelector.select()`，不要加无限等待或每帧重置会话。
3. 复核所有搜索入口都可接近合法远场目标、不会因 `_breakup_only` 或近场空计划永久跳过；覆盖同一失败目标未改变仍不重试。
4. 检查 bbox 完全排除贯通主 tracker、近场 worker、解团、换位、补夹和历史轨迹。近场 tracker 会保留 `observed=False` 的历史目标，但 `_eligible` 拒绝它们，`_obstacle_distance` 返回无穷；确认它们不会通过其它途径影响当前规划。
5. 补必要回归：安全区内目标不能阻挡场外组合，场外蓝块侵入仍阻挡；入口半径外的伴随成员可组合但超行程/超开口/超容量仍拒绝；邻近双绿优先于单块 handoff；保持首次单绿规则。
6. 已同步正式流程、应用/配置 README、根 README、人工验收和开放优先级文档：
   - `docs/正式流程设计.md` 约 26 行仍为规则总分/单橙优先；约 86、123 行仍有旧安全区物资处理语义。
   - `src/rescue_vision/app/README.md` 说明 `planning_perception` 入口、原始显示与规划输入、worker 输入索引关系。
   - `README.md`、`docs/后续优先级.md`、`manual_tests/README.md` 更新状态/现场回归项目。
   - 相邻近场模块/配置文档关于范围和排序的受影响内容也检查并同步。
   - 不改写比赛正式规则；这是用户指定的工程选组策略和 bbox 输入过滤。
7. 已完成运行装配、match 各变体、准备线程和近场几何验证：

```bash
.venv/bin/python -m pytest -q \
  tests/test_match_liveness.py tests/test_match_breakup_latency.py \
  tests/test_match_near_field.py tests/test_match.py tests/test_match_actions.py \
  tests/test_match_breakup.py tests/test_match_strategy.py tests/test_match_cc.py \
  tests/test_match_nb.py tests/test_gripper_width_sequence.py tests/test_near_field_grasp.py
```

全量 pytest、compileall 和 `git diff --check` 已通过；pytest 不替代硬件性能证据。
8. 实车录像复测与目标设备端到端时延 **未验证**。交付明确剩余风险，无配置迁移时说明无需调整配置。

## 前序工作（必须保留）

工作树已有上一阶段修改，开始本次前就存在：
- 1140 日志：近场交接清空主 ID 后未记录失败目标，导致静止重开会话。已从 handoff prior 保存 ID 和物理失败核心。
- 1236 日志：空视野/只蓝块转满一圈后无限反向扫描。已改为预算耗尽后选择合法动作或经车体/夹爪/障碍/安全区检查的换位；受阻方向继续扫描，不重开整圈。
- 1258 日志：主目标 track=10 消失后沿旧 reference 前进，每周期重置接近起点。已改为接管当前合法候选，并修复纯参考路径的固定编码器终点。
- 接近复核复用可共同收拢绿黑成员规则，避免把组内成员当作障碍。
- 前序阶段相关 **440 个测试通过**，最后局部调整又复跑 24 和 39 个受影响测试通过。这个数字不代表本次 1317 修改已通过。
- 前序文档修改也都未提交。不要 reset、清理或覆盖；提交必须检查整个累积 diff，用户未授权 push。
