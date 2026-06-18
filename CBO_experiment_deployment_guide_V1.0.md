# CBO 实验部署指南 V1.0

## 1. 目标

本文档用于统一后续 CBO 实验部署口径，避免不同实验之间因为历史窗口、归一化基准、能耗搜索范围、初始探索点等设置不同而导致结论不可比。

后续实验主要分为四类：

1. 静态 108 场景扫描实验
2. BO/CBO 选点机制对比实验
3. 动态阶段切换实验
4. 消融实验与未来联邦扩展实验

当前推荐的主线判断逻辑是：

- 若目标是证明 CBO 相比 BO 的后期优势，优先使用 BO+CBO 双方法对照。
- 若目标是做完整 baseline 总览，再加入 direct/fixed 方法。
- 若目标是研究历史数据选择机制，需要明确区分 recent80、all-history、state-kernel。
- 若目标是动态场景，需要单独使用 dynamic_scenario，不要和静态扫描结论混写。

## 2. 当前主方法

推荐主方法：

```text
BO  = reduced7_bo_greedy
CBO = reduced7_cbo_lite_pressure_taskmix_counts
```

常用别名：

```text
reduced7-bo-greedy -> reduced7_bo_greedy
cbo7               -> reduced7_cbo_lite_pressure_taskmix_counts
```

当前 CBO 方法含义：

- 控制向量：reduced7
- 情景向量：pressure + task mix + counts
- 代理模型：GP
- 选点：greedy posterior mean / greedy mean
- 默认历史：recent_confidence window=80
- 默认置信筛选：confidence_min=0.35，confidence_min_samples=12

## 3. 推荐统一配置

后续除非明确做消融，否则建议保持以下配置：

```text
--bo-iterations 500
--bo-interval 240
--session-duration 120000
--fixed-rng
--fixed-seed <seed>
--reduced7-energy-scale-bounds 0.5,3.0
--feedback-score task_effective_backlog_violation
--cbo-objective-mode normalized_tradeoff
--cbo-reference-mode calibrate
--cbo-shared-reference-policy cbo_first
--cbo-shared-reference-warmup-rounds 5
--cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts
--cbo-backlog-growth-penalty-weight 0
--scheduler-score-norm-mode candidate_minmax_deadline
--task-adaptation
```

说明：

- `cbo_first` 表示先由 CBO 建立归一化基准，其他方法复用该基准。
- warmup 5 轮包含在 500 轮预算内，不额外增加训练轮数。
- 相同场景应复用已建立的 reference。
- backlog growth penalty 当前关闭，避免和“去掉动态堆积”版本冲突。
- reduced7 energy scale 建议统一为 `0.5,3.0`，不要混用旧实验的 `0.5,2.0`。

## 4. 方法组设计

### 4.1 主对照组：BO + CBO

用于回答：

```text
CBO 是否优于 BO？
CBO 后期是否稳定超过 BO？
历史选择机制是否影响 CBO？
```

推荐：

```text
--selected-keys reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts
```

### 4.2 完整 baseline 组

用于回答：

```text
CBO 相比 direct/fixed/BO 的整体位置如何？
```

推荐：

```text
--selected-keys direct_greedy_cost,direct_queue_aware_greedy,reduced7_fixed_mid,reduced7_fixed_tuned,reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts
```

注意：

- baseline 组适合总览，不适合直接分析 BO/CBO 选点机制。
- 如果加入 direct/fixed，结果图更全，但实验时间更长。

## 5. 历史选择口径

这是后续实验最容易混淆的地方。

### 5.1 V8 recent80 口径

用于检验：

```text
在只保留近窗口信息时，CBO 是否仍能后期优于 BO？
```

配置：

```text
--bo-history-mode recent
--bo-recent-window 80
```

此时：

- BO：recent80
- CBO：如果被 CLI 覆盖，通常也会变成 recent80；若只使用方法默认，则是 recent_confidence window=80。

### 5.2 CBO recent window sweep 口径

用于检验：

```text
CBO 困难场景不如 BO，是否因为 CBO 可用历史太短？
```

推荐只改 CBO：

```text
--cbo-recent-window 80
--cbo-recent-window 120
--cbo-recent-window 160
--cbo-recent-window 240
```

注意：

- BO 保持默认 recent80。
- 不要同时传 `--bo-history-mode recent --bo-recent-window 80`，否则可能影响 CBO 方法默认历史模式。

### 5.3 V9 all-history + state-kernel 口径

用于检验：

```text
允许历史复用/情景相似样本后，CBO 是否更稳？
BO 是否会被旧历史拖累？
```

配置：

```text
--bo-history-mode all
--bo-recent-window 80
--cbo-history-select-mode state_gated_kernel
--cbo-state-kernel-topk 100
--cbo-state-kernel-min-rows 20
--cbo-state-kernel-recent-keep 20
--cbo-state-kernel-threshold 0.05
--cbo-state-kernel-fallback recent_context
--cbo-state-kernel-rate-gain 1.0
--cbo-state-kernel-rate-power 1.0
--cbo-state-kernel-max-rate-dist 3.0
```

此时：

- BO：不只是最近 80，会混入 archive/all-history。
- CBO：不是最近 80，而是最近样本 + state/kernel 相似历史样本。

## 6. 静态 108 场景扫描实验

### 6.1 实验目的

108 场景扫描用于回答：

```text
在更广泛的静态场景中，CBO 相比 BO/direct/fixed 的优势在哪些场景成立？
CBO 输在哪些任务比例/到达率组合？
是否存在高 AI、高 RT、高 Batch 或高负载场景的结构性差异？
```

### 6.2 推荐先跑 BO+CBO 主对照

Linux 模板：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python -m new_tr_split \
  --mode pressure_scan \
  --lambda-values 1.8,2.2,2.6,3.0 \
  --selected-keys reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-iterations 500 \
  --bo-interval 240 \
  --session-duration 120000 \
  --fixed-rng \
  --fixed-seed 43 \
  --reduced7-energy-scale-bounds 0.5,3.0 \
  --feedback-score task_effective_backlog_violation \
  --bo-history-mode recent \
  --bo-recent-window 80 \
  --cbo-objective-mode normalized_tradeoff \
  --cbo-reference-mode calibrate \
  --cbo-shared-reference-policy cbo_first \
  --cbo-shared-reference-warmup-rounds 5 \
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
  --cbo-backlog-growth-penalty-weight 0 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --output-root /home/ecs-user/CBO/result/r7_108_bo_cbo_v8_recent80_s43
```

说明：

- 这里是 V8 recent80 口径。
- `lambda-values` 和 task mix 网格由 `pressure_scan` 内部逻辑决定。
- 如果实际 108 场景需要固定 lambda/task mix 网格，应在运行前确认 `pressure_scan` 的扫描网格定义。

### 6.3 完整 baseline 版本

Linux 模板：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python -m new_tr_split \
  --mode pressure_scan \
  --lambda-values 1.8,2.2,2.6,3.0 \
  --selected-keys direct_greedy_cost,direct_queue_aware_greedy,reduced7_fixed_mid,reduced7_fixed_tuned,reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-iterations 500 \
  --bo-interval 240 \
  --session-duration 120000 \
  --fixed-rng \
  --fixed-seed 43 \
  --reduced7-energy-scale-bounds 0.5,3.0 \
  --feedback-score task_effective_backlog_violation \
  --bo-history-mode recent \
  --bo-recent-window 80 \
  --cbo-objective-mode normalized_tradeoff \
  --cbo-reference-mode calibrate \
  --cbo-shared-reference-policy cbo_first \
  --cbo-shared-reference-warmup-rounds 5 \
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
  --cbo-backlog-growth-penalty-weight 0 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --output-root /home/ecs-user/CBO/result/r7_108_full_baselines_v8_recent80_s43
```

### 6.4 V9 历史复用版本

Linux 模板：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python -m new_tr_split \
  --mode pressure_scan \
  --lambda-values 1.8,2.2,2.6,3.0 \
  --selected-keys reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-iterations 500 \
  --bo-interval 240 \
  --session-duration 120000 \
  --fixed-rng \
  --fixed-seed 43 \
  --reduced7-energy-scale-bounds 0.5,3.0 \
  --feedback-score task_effective_backlog_violation \
  --bo-history-mode all \
  --bo-recent-window 80 \
  --cbo-history-select-mode state_gated_kernel \
  --cbo-state-kernel-topk 100 \
  --cbo-state-kernel-min-rows 20 \
  --cbo-state-kernel-recent-keep 20 \
  --cbo-state-kernel-threshold 0.05 \
  --cbo-state-kernel-fallback recent_context \
  --cbo-state-kernel-rate-gain 1.0 \
  --cbo-state-kernel-rate-power 1.0 \
  --cbo-state-kernel-max-rate-dist 3.0 \
  --cbo-objective-mode normalized_tradeoff \
  --cbo-reference-mode calibrate \
  --cbo-shared-reference-policy cbo_first \
  --cbo-shared-reference-warmup-rounds 5 \
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
  --cbo-backlog-growth-penalty-weight 0 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --output-root /home/ecs-user/CBO/result/r7_108_bo_cbo_v9_statekernel_s43
```

## 7. 多 seed 稳健性实验

推荐 seed：

```text
40, 41, 42, 43, 44
```

实验顺序：

1. 先用 seed 43 跑完整流程，确认无报错。
2. 再跑 5 seed。
3. 最后对比 last50、last100、all500。

主要判断：

- CBO last50 是否赢过 BO。
- CBO last100 是否稳定。
- all500 是否被前期学习成本拖累。
- 哪些场景 CBO 输得最多。
- 结果是否集中在某些任务比例上。

## 8. 动态实验

### 8.1 实验目的

动态实验用于回答：

```text
场景变化后，CBO 是否比 BO 更快适应？
历史复用是否会帮助重复场景？
state-kernel 是否会错误召回旧场景样本？
```

### 8.2 推荐动态命令模板

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python -m new_tr_split \
  --mode dynamic_scenario \
  --dynamic-schedule "1.8:30,40,30:100;2.6:60,30,10:100;2.6:10,20,70:100;1.8:30,40,30:100;2.6:50,30,20:100;3.0:60,30,10:100;2.6:10,20,70:100;1.8:40,30,30:100" \
  --dynamic-history-mode state_gated_kernel \
  --dynamic-history-window 80 \
  --dynamic-context-topk 50 \
  --selected-keys reduced7_fixed_tuned,reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-interval 240 \
  --fixed-rng \
  --fixed-seed 43 \
  --reduced7-energy-scale-bounds 0.5,3.0 \
  --feedback-score task_effective_backlog_violation \
  --cbo-objective-mode normalized_tradeoff \
  --cbo-reference-mode calibrate \
  --cbo-shared-reference-policy cbo_first \
  --cbo-shared-reference-warmup-rounds 5 \
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
  --cbo-backlog-growth-penalty-weight 0 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --output-root /home/ecs-user/CBO/result/r7_dynamic_statekernel_s43
```

### 8.3 动态实验必须检查的字段

输出文件：

```text
dynamic_run_config.json
dynamic_round_summary.csv
dynamic_phase_summary.csv
dynamic_transition_summary.csv
dynamic_repeated_phase_summary.csv
```

重点字段：

```text
dynamic_history_mode
effective_history_mode
history_select_mode
selected_recent_count
selected_context_count
selected_elite_count
selected_total_count
state_kernel_selected_count
state_kernel_fallback_used
state_kernel_selected_phase_counts
phase_reference_cache_status
active_reference_id
```

解释规则：

- 如果 `history_select_mode = state_gated_kernel`，CBO 不是 recent80。
- 如果 `selected_total_count > selected_recent_count`，说明混入了历史样本。
- 如果 `state_kernel_selected_phase_counts` 出现旧 phase，说明当前阶段使用了旧阶段经验。
- 如果重复场景表现更快恢复，说明历史复用有效。
- 如果阶段切换初期明显变差，需要检查是否错误召回旧 phase。

## 9. 消融实验接口

建议按单变量原则做消融。

### 9.1 历史窗口消融

```text
CBO recent window = 80 / 120 / 160 / 240
BO recent window 固定 80
```

问题：

```text
CBO 是否因为记忆太短而输给 BO？
```

### 9.2 state-kernel 消融

变量：

```text
--cbo-state-kernel-topk 50 / 100 / 160
--cbo-state-kernel-recent-keep 10 / 20 / 40
--cbo-state-kernel-threshold 0.03 / 0.05 / 0.10
--cbo-state-kernel-fallback recent / recent_context / all
```

问题：

```text
历史相似样本多一点是否更好？
最近样本保护是否足够？
fallback 是否引入错误旧场景？
```

### 9.3 reference 消融

变量：

```text
cbo_first warmup=5
cbo_first warmup=10
旧 calibrate rounds=30
固定 reference
```

问题：

```text
归一化基准是否影响 BO/CBO 公平比较？
```

### 9.4 TR 消融

变量：

```text
TR off
TR adaptive
TR radius init/min/max
TR anchor posterior_mean / recent_best / context_best
```

当前建议：

- TR 不作为主结果。
- 只作为后续稳定性增强模块测试。

## 10. 联邦扩展预留接口

后续如果扩展到联邦 CBO，建议保留以下接口概念：

### 10.1 多客户端/多工厂

新增维度：

```text
client_id
site_id
factory_id
resource_profile_id
link_profile_id
```

每个客户端本地维护：

```text
local_recent
local_archive
local_reference_bank
local_context_statistics
```

全局服务端维护：

```text
global_reference_bank
global_context_cluster
global_theta_candidate_pool
federated_history_index
```

### 10.2 联邦共享内容

建议优先共享低风险摘要，而不是直接共享全部原始任务数据：

```text
theta
normalized score
context vector
phase signature
confidence
sample count
summary metrics
```

可选共享：

```text
GP posterior summary
top-k elite theta
state-kernel selected prototypes
reference baseline statistics
```

### 10.3 联邦实验问题

后续可检验：

```text
跨场景经验共享是否提升冷启动？
相似客户端历史是否能减少动态切换损失？
错误共享是否会造成负迁移？
state-kernel 是否能作为联邦样本选择器？
```

### 10.4 联邦结果字段预留

建议未来输出中加入：

```text
client_id
source_client_id
federated_sample_count
local_sample_count
shared_sample_count
federated_selection_mode
federated_context_similarity
federated_transfer_gain
negative_transfer_flag
```

## 11. 推荐输出分析口径

静态实验：

```text
last50
last100
all500
first50
201-300
301-400
```

动态实验：

```text
per phase mean
phase first50
phase last50
transition first20/first50
repeated phase recovery
rolling50
```

主指标：

```text
normalized_tradeoff_score
Eval_Cost
Avg_Delay
Avg_Delay_RT
Avg_Delay_Batch
Avg_Delay_AI
energy_per_arrival
unfinished_rate
Backlog
Violation_Rate
SLA_Success_Rate
```

历史选择诊断：

```text
effective_history_mode
effective_recent_window
history_select_mode
selected_recent_count
selected_total_count
state_kernel_selected_phase_counts
Feedback_Confidence
```

## 12. 结论写法建议

可以写：

```text
在 fixed5 公平初始探索、CBO-first 共享归一化基准和多 seed 设置下，CBO 在后期窗口中相比 BO 表现出更稳定的综合目标改善。
```

不要写：

```text
CBO 全程优于 BO。
CBO 在所有场景显著大幅优于 BO。
```

更准确的写法：

```text
CBO 前期存在学习成本，但在样本积累后能够利用情景信息获得后期优势。该优势是否来自 recent window、state-kernel 历史复用或 reference 机制，需要通过消融实验进一步区分。
```

## 13. 后续推荐实验顺序

1. V9 seed43：验证 all-history + state-kernel 是否优于 V8 recent80。
2. V9 多 seed：如果 seed43 有改善，再跑 5 seed。
3. 108 场景 BO+CBO recent80：得到主扫描基线。
4. 108 场景 BO+CBO state-kernel：对比历史复用收益。
5. 108 场景 full baseline：补 direct/fixed 总览。
6. 动态 state-kernel：检验阶段切换和重复场景复用。
7. 消融实验：history window、topk、recent_keep、reference、TR。
8. 联邦扩展：先做离线共享历史，再做多客户端在线训练。

