# CBO 内外部情景结构修改说明 V1.1

## 1. 修改目标

本次修改不是简单替换情景特征名，而是把 CBO agent 内部的样本结构和训练样本选择流程拆成两层：

- 外部场景：只描述任务强度和任务比例，用于决定哪些历史样本可以参与当前建模。
- 内部情景：只使用当前轮次或上一窗口已经可观测到的系统压力信息，作为 GP 代理模型输入。

这样 CBO 的语义变成：

> 先按外部负载阶段筛历史样本，再在相近外部场景内，用内部压力状态 + 参数 theta 建代理模型并选点。

## 2. 当前外部场景

外部场景向量为 4 维：

```text
arrival_rate_recent
cfg_rt_prob
cfg_batch_prob
cfg_ai_prob
```

含义：

- `arrival_rate_recent`：当前或最近窗口任务到达强度，优先取运行指标里的 `arrival_rate`，否则回退到 workload 当前 lambda。
- `cfg_rt_prob`、`cfg_batch_prob`、`cfg_ai_prob`：当前场景配置中的任务比例。

外部场景不再拼进 GP 输入，只用于历史样本门控。

## 3. 当前内部情景

主 CBO 方法现在使用 `internal_pressure6`，内部情景向量为 6 维：

```text
start_backlog_norm
start_queue_total_norm
start_avg_util
start_max_util
prev_unfinished_rate
unfinished_rate_trend
```

含义：

- `start_backlog_norm`：窗口开始时积压压力，按当前窗口预期到达任务数归一化。
- `start_queue_total_norm`：窗口开始时队列总压力，按当前窗口预期到达任务数归一化。
- `start_avg_util`：窗口开始时平均资源利用率。
- `start_max_util`：窗口开始时最大资源利用率，用于感知节点瓶颈。
- `prev_unfinished_rate`：上一窗口未完成任务比例。
- `unfinished_rate_trend`：未完成率相对上一统计状态的变化趋势。

这些特征会进入 CBO 的上下文 GP，即训练输入为：

```text
[theta, internal_pressure6]
```

## 4. 样本结构变化

CBO 的每条历史样本现在额外保存：

```text
external_context
external_context_feature_names
```

同时仍保留原有：

```text
context
context_feature_names
state
metrics
confidence
```

warm history 导出和读取也同步支持 `external_context`，后续跨实验复用历史时可以继续按外部场景筛选。

## 5. 训练样本选择流程

当前流程为：

1. 每轮先构建外部场景向量 `external_context`。
2. 再构建内部情景向量 `internal_pressure6`。
3. CBO 收集历史样本时，先对样本做外部场景相似度计算。
4. 外部门控默认保留相似度不低于阈值的样本。
5. 如果通过阈值的样本不足，则回退到外部相似度最高的一批样本，避免冷启动阶段无样本可用。
6. 门控后的样本会恢复成时间顺序，再交给原有 recent/window/state/context 训练逻辑。
7. GP 只使用内部情景，不直接使用任务比例和 lambda。

主方法默认参数：

```text
cbo_external_gate_mode = taskmix_intensity
cbo_external_gate_threshold = 0.35
cbo_external_gate_topk = 240
cbo_external_gate_min_samples = 12
context_mode = internal_pressure6
```

## 6. 已修改位置

- `new_tr_split/runtime_patches.py`
  - 增加外部场景构建函数。
  - 增加 `internal_pressure4`、`internal_pressure6` 内部情景模式。
  - CBO 样本记录新增 `external_context`。
  - `_refactor_collect_samples` 增加外部场景门控。
  - warm history CSV 导出/读取支持外部场景。
  - 运行日志新增外部场景和门控诊断字段。

- `new_tr_split/scenario_experiments.py`
  - 主 CBO key `reduced7_cbo_lite_pressure_taskmix_counts` 切换为 `internal_pressure6`。
  - 开启外部任务强度 + 任务比例门控。

- `new_tr_split/diagnostics.py`
  - round summary 输出外部门控计数、fallback、相似度和外部场景向量。

## 7. 后续检查重点

后续重跑实验后，重点看这些字段：

```text
external_gate_raw_count
external_gate_passed_count
external_gate_selected_count
external_gate_fallback_used
external_similarity_mean
selected_external_similarity_mean
Context_Feature_Names_情景特征名
External_Context_Vector_外部情景向量
```

如果 `external_gate_passed_count` 长期很低，说明外部阈值过严或历史覆盖不足；如果 240 窗口仍明显更好，则需要继续判断是内部压力特征不足，还是外部门控后可用样本仍偏少。

