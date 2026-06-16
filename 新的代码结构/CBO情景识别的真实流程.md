# CBO 情景识别的真实流程

## 关键发现：有两个完全不同的"情景识别"概念

### 1️⃣ **BO 在线运行时的情景识别** (Online Context)

**代码位置**：`scenario_experiments.py` 的 `run_scenario_group()` 函数

```python
# 每轮 BO 迭代开始时
for i in range(CFG.BO_ITERATIONS):
    # 第1步：从 ScenarioMonitor 获取当前情景
    state, _, _ = fac.scenario_monitor.get_state(fac.current_time)
    ctx = fac.scenario_monitor.get_context_vector(fac.current_time)
    
    # 第2步：决定是否传给 BO
    ask_state = state if getattr(fac.agent, "use_state_partition", False) else None
    ask_ctx = ctx if getattr(fac.agent, "use_context", False) else None
    
    # 第3步：BO ask（根据情景选择下一个 theta）
    theta_control = fac.agent.ask(state=ask_state, context=ask_ctx)
    
    # 第4步：执行并获取结果
    _, _, _, _, metrics, _ = fac.run_continuous(theta_full, ...)
    
    # 第5步：BO tell（带上情景反馈）
    fac.agent.tell(theta_control, metrics["cost"], 
                   state=state_arg, context=context_arg)
```

### **state 是什么？** (离散状态)

来自 `ScenarioMonitor.get_state()` 的 `StateSignature`：

```python
def get_state(self, now):
    metrics = self.compute_metrics(now)  # 计算最近窗口的指标
    
    # 把连续指标离散化成 3×3×3 = 27 种状态之一
    load_level = self._level(metrics["avg_util"], CFG.UTIL_THRESHOLDS)      # LOW/MID/HIGH
    delay_level = self._level(metrics["avg_delay"], CFG.DELAY_THRESHOLDS)  # LOW/MID/HIGH
    arrival_level = self._level(metrics["arrival_rate"], CFG.ARRIVAL_THRESHOLDS)  # LOW/MID/HIGH
    
    state = StateSignature(load_level, delay_level, arrival_level)
    return state, stable, metrics
```

**关键点**：state 完全基于**当前系统内部运行状态**（队列利用率、延迟、到达率），**不是外部场景**

---

### **context 是什么？** (连续向量)

来自 `ScenarioMonitor.get_context_vector()` 的 7 维向量：

```python
def get_context_vector(self, now, metrics=None):
    metrics = self.compute_metrics(now)
    return self.build_context_vector(metrics)

def build_context_vector(self, metrics=None):
    names = CFG.CONTEXT_FEATURE_NAMES  # = [
                                        #     "arrival_rate",
                                        #     "avg_util", 
                                        #     "backlog",
                                        #     "vio_rate",
                                        #     "rt_arrival_ratio",
                                        #     "batch_arrival_ratio",
                                        #     "ai_arrival_ratio",
                                        # ]
    vec = []
    for name in names:
        vec.append(float(metrics.get(name, 0.0)))
    return vec
```

**关键点**：context 混合了：
- **内部状态**：avg_util, backlog, vio_rate
- **任务结构**：rt/batch/ai_arrival_ratio（但这些来自 window_feedback，是运行过程中的实际任务）
- **压力指标**：arrival_rate（当前到达强度）

---

## ❌ 我之前说的"外部情景"是什么？

那个东西（phase_signature, resource_perturbation_id 等）**不是 CBO 在线用的**！

```python
# 这个是离线实验分析用的，用于给窗口打标签
外部情景描述 = {
    "lambda": float,                    # 实验配置的到达强度
    "task_mix": {...},                  # 实验配置的任务比例
    "deadline_pressure": float,         # 根据配置计算的压力
    "resource_perturbation_id": str,    # 实验参数：资源扰动 ID
    "link_profile_id": str,             # 实验参数：链路配置 ID
    "task_adaptation_enabled": bool,    # 实验参数：是否启用亲和力
}
```

这个**仅用于**：
1. 生成 phase_signature（给窗口分类）
2. 识别"两个窗口是否在同一个外部场景中"
3. 决定是否复用 normalization reference

**与 BO 的 ask/tell 流程无关**！

---

## ✅ CBO 情景识别的实际流程

### 流程图：

```
┌─ 每轮 BO 迭代 ─────────────────────────────┐
│                                            │
├─ step 1: ScenarioMonitor.get_state()      │
│   输出: StateSignature (27 种离散状态)     │
│   基于: avg_util, avg_delay, arrival_rate │
│        （当前系统运行状态）                │
│                                            │
├─ step 2: ScenarioMonitor.get_context()    │
│   输出: 7D 连续向量                        │
│   基于: arrival_rate, avg_util, backlog,  │
│         vio_rate, rt/batch/ai_ratio      │
│        （当前窗口内部指标）                │
│                                            │
├─ step 3: agent.ask(state, context)        │
│   BO 根据当前状态/context 选择 theta      │
│   使用策略:                                │
│   - use_state_partition=True:             │
│     优先查历史中"同状态"的样本             │
│   - use_context=True:                     │
│     通过 k-NN 找相似 context 的历史样本   │
│                                            │
├─ step 4: 执行 theta，得到 metrics         │
│                                            │
├─ step 5: agent.tell(theta, cost,          │
│                     state, context)        │
│   BO 记录这次经验                         │
│   连同 state/context 一起存档             │
│                                            │
└────────────────────────────────────────────┘
```

---

## 🔑 三个关键的情景概念

| 概念 | 维度 | 来源 | 作用 | 何时使用 |
|------|------|------|------|---------|
| **StateSignature** | 27 种离散 | ScenarioMonitor | BO 的样本检索分桶 | BO ask/tell |
| **Context Vector** | 7D 连续 | ScenarioMonitor | BO 的 k-NN 相似查询 | BO ask/tell |
| **Phase Signature** | 字符串 | _phase_external_descriptor | 窗口标记，reference 匹配 | 离线分析 |

---

## 💡 理解 CBO 在线识别的关键

### 问题 1：CBO 如何区分"高压力"与"低压力"？

**答**：通过 context 中的：
- `backlog` - 当前积压任务数
- `avg_util` - 当前平均利用率
- 以及离散 state 中的 load_level

### 问题 2：CBO 如何区分"RT 多"与"AI 多"？

**答**：通过 context 中的：
- `rt_arrival_ratio` - 最近到达的任务中 RT 占比
- `batch_arrival_ratio`
- `ai_arrival_ratio`

**重要**：这些是**当前窗口内实际观测到的任务比例**，不是实验配置。所以 CBO 能看到"原本配置是 30% RT，但最近这个窗口随机到了 50% RT"

### 问题 3：为什么有"lite context"那么多模式？

**答**：实验消融。不同模式对应不同的上下文信息：

```python
"pressure_only": [1, 2, 3, 4]  # 只看压力，不看任务比例
"taskmix": [6, 7, 8]            # 只看任务比例，不看压力
"pressure_taskmix": [1,2,3,4,6,7,8]  # 既看压力又看任务比例
"pressure_taskmix_counts": [1,2,3,4,6,7,8,9,10,11]  # + 上一窗口的任务数
```

通过对比这些模式，可以验证"是否需要看任务比例来改进 BO"

---

## 总结：三层情景概念的清晰分离

### 关键发现：我混淆了 **外部情景划分** 和 **BO 在线识别**

| 层级 | 来源 | 目的 | 是否参与 BO ask/tell |
|------|------|------|-----|
| **外部情景划分** | 实验配置 (lambda_schedule, TASK_TYPE_PROBS) | 给窗口打 phase_signature 标签，聚类相似的外部条件 | ❌ **不直接参与** |
| **离散状态分区** | 运行时观测 (avg_util, delay, arrival_rate) | BO 的 state_partition 分桶 | ✅ **直接参与** |
| **连续 context** | 运行时观测 (arrival_rate, backlog, util, task_ratio) | BO 的 k-NN 相似查询 | ✅ **直接参与** |

---

## 你的三个问题逐个回答

### Q1："现在是进行27种离散状态打标？不是进行7D情景信息标记吗？"

**答**：都做，但用途不同！

- **27种离散状态**：用于 BO 的 `state_partition`
  - 代码：`_collect_samples(state=state)` 按 state 分桶历史
  - 效果：同一离散状态的样本优先被选中训练 GP

- **7D 连续context**：用于 BO 的 context-aware 学习
  - 代码：`_select_pivot_theta(context, records)` 进行 k-NN 查询
  - 效果：根据 context 相似性加权历史样本

**两个不是替代关系，而是组合关系**：
```python
# BO 的 ask() 可能：
ask(state=state_signature,        # 27种之一
    context=7d_context_vector)    # 连续向量
```

### Q2："然后任务比例、任务强度进行不同情景的外部划分么？"

**答**：是的，这是**外部情景划分**的核心：

```python
def _phase_external_descriptor(lam, task_probs):
    """生成外部情景描述，用于给窗口分类"""
    return {
        "lambda": lam,              # 外部配置：到达强度
        "task_mix": {               # 外部配置：任务比例
            "RT": task_probs["RT"],
            "Batch": task_probs["Batch"],
            "AI": task_probs["AI"],
        },
        "deadline_pressure": ...,    # 根据task_probs计算的压力
        "resource_perturbation_id": ...,  # 外部配置：资源扰动
        "link_profile_id": ...,           # 外部配置：链路配置
        "task_adaptation_enabled": ...,   # 外部配置：亲和力
    }
```

然后根据这个 descriptor，计算 `phase_signature`：
```
lam0.1_mix0.40-0.30-0.30_dl0.25_resnormal_linknormal_adapt0
```

**用途**：离线实验分析
- 将多个 LAMBDA_SCHEDULE 中的窗口**聚类到相似的外部情景**
- 给每个聚类打 signature 标签
- 用于选择 normalization reference

### Q3："这个会跟离散状态有关么？"

**答**：**逻辑上无关，但运行中可能间接相关**

#### 无关的原因：
```
外部情景划分 (phase_signature)
    ↓
    基于：lambda, task_mix（实验配置）
    生成：phase_signature（窗口标签）
    用途：离线分析、reference 选择
    ❌ 不影响 BO 的 ask() 调用

离散状态 (StateSignature)
    ↓
    基于：avg_util, avg_delay, arrival_rate（运行时观测）
    生成：27 种状态之一
    用途：BO state_partition
    ✅ 影响 BO 的 ask() 调用
```

#### 间接相关的原因：
```
如果 lambda 从 0.1 变成 0.5（外部配置改变）
  → 系统到达压力增加
  → 平均利用率、延迟可能上升
  → 可能从 LOW_util → HIGH_util 离散状态
  → BO 看到新的离散状态
  → 使用不同的历史样本
```

---

## 现在的情景识别架构

```
┌─ 外部情景配置层 ────────────────┐
│  Lambda Schedule                │
│  TASK_TYPE_PROBS               │  →  _phase_external_descriptor()
│  RESOURCE_PERTURBATION_ID       │  →  assign_phase_reference_signatures()
│  LINK_PROFILE_ID                │
│  USE_TASK_TYPE_ADAPTATION       │
└─────────────────────────────────┘
         ↓
    phase_signature (离线标签)
    [用于离线分析、reference聚类]
         ↓
┌─ BO 在线运行层 ──────────────────┐
│  get_state()                    │
│    → avg_util, avg_delay,       │
│      arrival_rate               │  →  StateSignature (27种)
│    → 用于 state_partition       │
│                                 │
│  get_context_vector()           │
│    → arrival_rate, avg_util,    │
│      backlog, vio_rate,         │
│      rt_ratio, batch_ratio,     │
│      ai_ratio                   │  →  7D context vector
│    → 用于 context-aware 学习     │
│                                 │
│  ask(state=..., context=...)    │
│    → BO 选择下一个 theta        │
└─────────────────────────────────┘
```

---

## 关键区别：配置层 vs 运行层

| 特征 | 外部情景配置 (phase_signature) | 内部离散状态 (StateSignature) | 内部连续context |
|------|---------|---------|---------|
| **来源** | Lambda schedule、TASK_TYPE_PROBS 等实验配置 | 当前窗口观测的 util、delay、arrival_rate | 当前窗口观测的各种指标 |
| **何时确定** | 实验设计时固定 | 每个窗口动态计算 | 每个窗口动态计算 |
| **任务比例** | 来自 CFG.TASK_TYPE_PROBS（配置值） | context 中的 rt_arrival_ratio 等（实时观测值） |
| **用途** | 给窗口贴标签，聚类外部条件 | BO 的 state_partition 和 context-aware 学习 |
| **影响 BO ask** | ❌ 否 | ✅ 是 |

---

## 后续问题

CBO 的在线情景识别是否足够？

- ✅ 能分出"高利用率 vs 低利用率"（离散状态）
- ✅ 能分出"高积压 vs 低积压"（离散状态）  
- ✅ 能分出"RT多 vs AI多"（context 中的 ratio）
- ❌ **不能分出"动态压力变化趋势"**（没有历史差分）
- ❓ **是否需要加入"前几个窗口的状态变化"作为 context？**
- ❓ **是否需要让 phase_signature 影响 BO 的分桶策略？**（目前只用于离线分析）

这可能是后续改进方向。
