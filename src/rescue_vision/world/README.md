# `world`：最小世界模型

本包维护固定物理场地地图，并把 `TrackedTarget`、抽签后的任务区域、可选场地
坐标和对手占据整理成不可变 `WorldSnapshot`。它是静态地图与规则状态机之间的
唯一世界语义层，不运行检测、不维护比赛得分。

## 常用类和函数

| 入口 | 作用 |
| --- | --- |
| `WorldRuntimeConfig.build_model()` | 从物理静态地图和 `team_color` 派生任务区域并创建 `WorldModel` |
| `StaticFieldMap` | 固定物理区域和中心十字终端语义的唯一权威 |
| `PhysicalStaticRegion` / `PhysicalRegionKind` | 场界、红蓝安全分区和出发区的 `FieldPoint` 多边形 |
| `StaticCenterCross` / `CenterCrossTerminal` | 中心交点及 `±x/±y` 四条射线的终端语义 |
| `TeamColor` | `red`、`blue` 或保守的 `unknown` |
| `WorldModel.update()` | 在单调时间轴上产生一个完整快照 |
| `WorldSnapshot` | 状态机消费的动态目标、区域、对手和不确定性 |
| `WorldSnapshot.target_region_kinds()` | 查询目标所在静态区域；缺少场地坐标时返回 `None` |
| `WorldSnapshot.target_in_opponent_occupancy()` | 查询目标是否落入对手占据；缺少场地坐标时返回 `None` |
| `StaticRegion` / `RegionKind` | `FieldPoint` 多边形及场地语义 |
| `OpponentOccupancy` | 带置信度和时间戳的对手场地占据多边形 |
| `WorldTarget` | 跟踪目标的类别概率、局部/全局位置和危险状态 |
| `HazardState` | `CLEAR`、`SUSPECTED`、`CONFIRMED` |
| `WorldUncertainty` | 视觉过期、缺坐标、未确认或对手信息过期等原因 |

## 1. 从运行配置装配

```python
from time import monotonic_ns

from rescue_vision.config import load_runtime_config

config = load_runtime_config("configs/runtime.yaml")
world_model = config.world.build_model()
```

`world_model` 在进程或一轮生命周期内复用。后续片段继续使用它。

## 2. 用当前证据更新世界快照

`tracks` 来自同帧 tracker 更新。当前没有对手感知提供者，因此显式传空序列：

```python
# tracks 来自同一帧 tracker.update()。
# 当前没有定位时，robot_field_point 和 target_field_points 保持缺省；
# 世界模型会明确标记缺失，而不是伪造 FieldPoint。
snapshot = world_model.update(
    timestamp_ns=monotonic_ns(),
    visual_timestamp_ns=raw_frame.timestamp_ns,
    tracks=tracks,
    opponent_occupancies=(),
)
```

未来对手感知完成后，只把 `opponent_occupancies=()` 替换为同一时刻的
`tuple[OpponentOccupancy, ...]`，不改变世界模型接口。

## 3. 消费世界目标

以下片段承接前文 `snapshot`：

```python
for target in snapshot.targets:
    print(
        target.track_id,
        target.hazard_state,
        target.ground_point,  # 当前机器人地面系，mm
        target.field_point,   # 定位接入前通常为 None
    )
```

`timestamp_ns` 是本次世界更新时刻，`visual_timestamp_ns` 是最近有效视觉帧时刻。两者必须来自同一单调时钟。没有检测目标的有效新帧仍应更新 `visual_timestamp_ns`，否则会被误判为视觉中断。

## 4. 固定物理地图与任务区域

所有不会随机器人运动改变的场地事实统一配置在 `world.static_map`。顶点使用
`FieldPoint` 的 `[x_mm, y_mm]`：原点是中心十字交点，`+x` 沿水平基准线向右，
`+y` 沿竖直基准线指向当前静态地图定义的红色安全区。不能写入机器人局部
`GroundPoint`。

```yaml
world:
  team_color: unknown  # 抽签后改为 red 或 blue
  static_map:
    center_cross:
      intersection_field_mm: [0.0, 0.0]
      terminals:
        positive_x: plain_boundary
        negative_x: plain_boundary
        positive_y: red_safe_zone
        negative_y: blue_safe_zone
    regions:
      - region_id: field
        kind: field
        polygon_field_mm:
          - [-1500.0, -1500.0]
          - [1500.0, -1500.0]
          - [1500.0, 1500.0]
          - [-1500.0, 1500.0]
```

完整的红/蓝物资区、伤员区和 1–4 号出发区启动地图及逐字段注释只在
[`configs/runtime.example.yaml`](../../../configs/runtime.example.yaml) 维护，
不在此复制第二份数值。

物理 `kind` 允许 `field`、`red_material`、`red_injured`、`blue_material`、
`blue_injured` 和 `start_zone`。多边形至少三个点、面积非零，`region_id`
全局唯一；四个出发区使用相同 kind，以 `start-1` 至 `start-4` 区分。

`team_color` 不改变物理地图。`red` 会把红色两个分区派生为
`OWN_MATERIAL/OWN_INJURED`，把蓝色两个分区派生为 `OPPONENT_SAFE`；`blue`
反向映射。`unknown` 只派生 `FIELD`，不会猜测己方/对方。状态机仍只消费派生后的
`StaticRegion/RegionKind`，不解释物理颜色。

中心十字终端是定位消歧的地图真值：同一种终端若配置在多条射线上，本身就不能
唯一确定航向。例如默认 `plain_boundary` 同时位于 `±x`，所以普通边界仍保留
180° 歧义。交点按场地坐标定义必须是 `[0, 0]`，不能用非零值补偿标定误差。

## 5. 危险与不确定性

世界模型集中融合跟踪状态和完整类别概率：

- 曾完成跨帧确认且危险概率达到 `danger_confirm_threshold` 时是
  `CONFIRMED`；短时 `COASTING` 不会抹掉危险历史；
- 危险概率达到较低疑似阈值、未知概率过高、轨迹未确认或处于
  `COASTING`、类别为 `UNKNOWN` 时是 `SUSPECTED`；
- 其余才是 `CLEAR`。

状态机只允许选择 `CLEAR` 目标。`SUSPECTED` 不会被强行改成普通物资。

`WorldSnapshot.uncertainties` 是快照级诊断集合。缺少机器人
`FieldPoint` 不妨碍局部目标跟踪，但不能据此断言机器人“不在对方安全区”；需要区域判断的调用方必须等待定位或其它明确证据。

目标区域查询同样保留三态语义：`True`/非空集合表示已有明确场地证据，
`False`/空集合表示明确不在其中，`None` 表示目标没有 `FieldPoint`、无法判断。
未知 `track_id` 会抛出 `KeyError`。状态机只排除已经明确位于安全区或对手占据
内的候选，不把缺失定位伪装成区域结论。

## 6. 对手占据生命周期

同一 `opponent_id` 的新占据会替换旧值；没有新观测时保留到
`opponent_max_age_ms`，随后删除并标记 `STALE_OPPONENT`。占据多边形属于场地坐标，不能把检测框或 `GroundPoint` 直接传入。

## 错误与降级

- 世界时间倒退、未来视觉/对手时间戳、重复 track ID、未知场地映射 ID 会报错；
- 视觉年龄超过 `max_visual_age_ms` 时保留快照但增加 `STALE_VISION`，由状态机输出保守停止；
- 缺 K0 的目标仍保留图像与类别历史，同时标记
  `TARGET_WITHOUT_GROUND_POINT`，不能交给地面规划；
- 固定物理地图和中心十字视觉位姿观测已实现；现场区域测量、连续定位融合和
  对手检测仍未完成，合成事件不能作为现场能力证据。
