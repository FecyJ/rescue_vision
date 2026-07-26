# `mission`：规则状态机与抽象动作

本包集中实现智能救援初赛规则、安全优先级和任务阶段。详细状态、规则矩阵和回放要求见[规则状态机设计](../../../docs/规则状态机设计.md)；本页说明调用接口。

## 公共入口

| 入口 | 作用 |
| --- | --- |
| `MissionConfig.build_state_machine()` | 从 `runtime.yaml` 创建一轮使用的状态机 |
| `MissionStateMachine.start()` | 记录唯一启动时刻，进入首个普通物资阶段 |
| `MissionStateMachine.step()` | 消费同一时刻的世界、转运、交付和安全证据 |
| `MissionStateMachine.reset()` | 仅在未开始或已经终止时清空本轮状态 |
| `replay_mission()` / `MissionReplayStep` | 从空状态确定性重放完整合成或记录事件 |
| `TransportStatus` | 当前持续接触/推动的稳定 track ID 集合 |
| `DeliveryEvidence` | 带唯一 ID、目的区域和完全进入标记的一次性交付证据 |
| `SafetySignals` | 运动、对手接触、失控、事故、非法载运等非视觉事实 |
| `MissionDecision` | 复合状态、抽象动作、目标 ID 和终止原因 |
| `MissionProgress` | 已验证交付和错区/对方区/出场计数 |

## 生产对接形态

状态机不读取相机或串口。应用层应把接触监测、电控健康和区域验证结果转换为显式输入：

```python
from rescue_vision.config import load_runtime_config
from rescue_vision.mission import (
    DeliveryEvidence,
    SafetySignals,
    TransportStatus,
)

config = load_runtime_config("configs/runtime.yaml")
mission = config.mission.build_state_machine()
mission.start(start_timestamp_ns)

# 以下值由后续接触/机构、电控健康和区域验证适配器提供；
# 所有时间必须与 WorldSnapshot 使用同一 monotonic 时钟。
transport = TransportStatus(
    engaged_track_ids=engaged_track_ids,
    contact_started_ns=contact_started_ns,
)
safety = SafetySignals(
    last_motion_timestamp_ns=last_motion_timestamp_ns,
    opponent_contact_since_ns=opponent_contact_since_ns,
    active_attack=active_attack,
    lost_control=lost_control,
    safety_accident=safety_accident,
    human_touched_after_start=human_touched_after_start,
    target_carried_on_robot=target_carried_on_robot,
    external_stop_requested=external_stop_requested,
)

# 没有新交付边界事件时传 None；不要每帧伪造 delivery_id。
decision = mission.step(
    snapshot,
    transport=transport,
    safety=safety,
    delivery=delivery_evidence_or_none,
)

# 规划/电控只能细化该抽象动作，不能覆盖 STOP。
publish_abstract_action(
    decision.action,
    target_track_id=decision.target_track_id,
)
```

机械和电控尚未完成，因此上述适配器变量目前只能由合成回放或未来真实提供者给出；状态机本身及其契约已经可独立运行。

## 交付证据

```python
from rescue_vision.mission import DeliveryDestination, DeliveryEvidence

evidence = DeliveryEvidence(
    delivery_id="round-1-boundary-0007",
    track_ids=(12,),
    destination=DeliveryDestination.OWN_MATERIAL,
    fully_entered=True,
)
```

`delivery_id` 在一轮内不可复用。完全相同的事件重放是幂等的；同一 ID 携带不同目标、目的地或进入状态会报错。己方得分和首次阶段只接受
`fully_entered=True`，部分进入保持 `VERIFYING_DELIVERY / DELIVER`。

`track_ids` 必须来自 `WorldSnapshot`。当前仍在接触时，交付集合必须与
`TransportStatus.engaged_track_ids` 一致；不一致会进入安全保持。

## 离线事件回放

```python
from rescue_vision.mission import MissionReplayStep, replay_mission

decisions = replay_mission(
    config.mission,
    start_timestamp_ns=start_timestamp_ns,
    steps=tuple(
        MissionReplayStep(
            snapshot=event.snapshot,
            transport=event.transport,
            safety=event.safety,
            delivery=event.delivery,
        )
        for event in recorded_events
    ),
)
```

回放不访问相机、模型、底盘或系统时间；同一配置和事件序列必须产生相同决策序列。事件持久化 schema 尚未冻结，当前调用方保存构成
`MissionReplayStep` 的领域字段即可，不应使用 `pickle` 作为长期格式。

## 状态、动作和恢复

任务阶段 `MissionPhase` 保存首个普通物资约束；`ActivityState` 保存搜索、接近、推动、验证、避让或停止。动作只有：

```text
SEARCH  APPROACH  PUSH  AVOID  DELIVER  STOP
```

- `SAFETY_HOLD / STOP` 可由新鲜视觉、同一 track ID 恢复、接触解除，
  或重新提供机器人场地位置证据恢复；
- `ENDED / STOPPED` 是吸收态，后续高收益事件也只能得到 `STOP`；
- 只有显式 `reset()` 才能开始新一轮，活动中的状态机拒绝复位；
- 时间倒退、未来运动/接触时间和非法输入立即报错。

## 配置和规则来源

`runtime.yaml` 配置比赛时长、无运动超时、对手接触超时、危险避让距离和一般阶段目标优先级。前三项默认分别为官方初赛的 180 秒、15 秒和 10 秒；现场正式文件变化时必须同步核对[赛题约束与视觉需求](../../../docs/赛题约束与视觉需求.md)，不能只调配置而不记录规则来源。

危险目标接触即终止是项目在官方“进入安全区或离场才终止”之前增加的预防策略，`TerminationReason.DANGER_ENGAGED` 明确区分该软件安全停止。

## 当前限制

- `TransportStatus` 和 `DeliveryEvidence` 的真实提供者尚未实现，不能凭状态机测试宣称整车规则验收完成；
- 状态机输出抽象动作，不保证路径可达、推动稳定或串口执行；
- 决赛规则现场公布后可能需要新增事件或替换规则配置，不能静默沿用初赛语义。
