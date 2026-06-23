# CBO 内部情景结构消融与实验指南 V1.4

## 1. 本版要解决的问题

静态 108 场景的 A/B/C/D 对比显示：

- BO + adaptive exploration 本身已经能明显改善固定 BO。
- CBO 单独结构在静态场景里不稳定，尤其前期容易弱于 BO。
- external gate 在静态实验中基本没有筛掉样本，说明静态场景下问题不主要来自外部门控。
- internal6 情景特征不是全 0，也有波动，但当前主 CBO 只是把 internal6 拼进 GP 输入，并没有用内部情景相似度去筛选或补充训练样本。

因此 V1.4 重点验证两个问题：

1. 内部情景维度是否过多，导致 80 个 recent 样本下 GP 更难学。
2. CBO 是否应该用内部情景相似度补充相近历史样本，而不是只用最近窗口。

本版不改主 CBO 方法，新增可选方法键做消融。

## 2. 新增方法

### 2.1 原主 CBO

```text
reduced7_cbo_lite_pressure_taskmix_counts
```

配置：

```text
external gate = taskmix_intensity
internal context = internal_pressure6
history = recent_confidence
recent_window = 80
confidence_min = 0.35
```

含义：外部场景门控 + internal6 输入 GP，但内部情景不参与历史样本筛选。

### 2.2 internal4 低维 CBO

```text
reduced7_cbo_lite_internal4
```

内部情景由 6 维降到 4 维：

```text
start_backlog_norm
start_max_util
prev_unfinished_rate
unfinished_rate_trend
```

其余保持主 CBO 一致：

```text
external gate = taskmix_intensity
history = recent_confidence
recent_window = 80
confidence_min = 0.35
```

验证目的：看 CBO 静态弱势是否来自 internal6 维度偏多、噪声偏多。

### 2.3 internal6 + context history

```text
reduced7_cbo_lite_internal6_context
```

配置：

```text
external gate = taskmix_intensity
internal context = internal_pressure6
history_select_mode = recent_context
recent_window = 80
context_k = 50
context_similarity_threshold = 0.0
```

训练样本来源变为：

```text
最近 80 条样本 + 内部情景相似 top50 历史样本
```

验证目的：看 CBO 是否因为只用 recent window，导致相似内部状态经验没有被复用。

### 2.4 internal4 + context history

```text
reduced7_cbo_lite_internal4_context
```

配置：

```text
external gate = taskmix_intensity
internal context = internal_pressure4
history_select_mode = recent_context
recent_window = 80
context_k = 50
context_similarity_threshold = 0.0
```

验证目的：同时降低内部维度，并补充相似内部状态历史样本。这个是本轮最重要的组合变体。

## 3. 重要说明

本版 `recent_context` 是硬选择，不是软加权：

```text
选择最近 80 + 相似 top50
```

目前还没有对低相似样本设置更低信任度，也没有给 GP 增加 sample weight 或 heteroscedastic noise。原因是这类修改会改变模型训练机制，影响更大。V1.4 先验证“相似历史补充”这个方向是否有效。

如果 `internal4_context` 明显好于主 CBO，再考虑下一版做软权重：

```text
similarity 高 -> 低噪声 / 高信任
similarity 低 -> 高噪声 / 低信任
```

## 4. 修改文件

需要上传/同步：

```text
D:\CBOv2\新的代码结构\去掉动态堆积\new_tr_split\scenario_experiments.py
D:\CBOv2\新的代码结构\去掉动态堆积\new_tr_split\runtime_patches.py
D:\CBOv2\run_static108_v16_bc_ablation.sh
D:\CBOv2\run_static108_v17_internal_context_adaptive.sh
D:\CBOv2\run_static_v17_internal_context_selected.sh
D:\CBOv2\analyze_static108_abcd_matched.py
D:\CBOv2\CBO内部情景结构消融与实验指南_V1.4.md
```

服务器推荐覆盖：

```text
D:\CBOv2\新的代码结构\去掉动态堆积\new_tr_split
-> /home/ecs-user/CBO/new_tr_split
```

以及上传根目录脚本：

```text
run_static108_v16_bc_ablation.sh
run_static108_v17_internal_context_adaptive.sh
run_static_v17_internal_context_selected.sh
analyze_static108_abcd_matched.py
```

## 5. 服务器检查

```bash
cd /home/ecs-user/CBO
source env.sh

python -m py_compile new_tr_split/*.py

grep -n 'reduced7_cbo_lite_internal4' new_tr_split/scenario_experiments.py
grep -n 'cbo7_i4ctx' new_tr_split/runtime_patches.py
```

也可以检查命令是否能识别新 key：

```bash
python -m new_tr_split \
  --mode pressure_scan \
  --selected-keys reduced7_cbo_lite_internal4_context \
  --lambda-values 1.8 \
  --task-probs 10,10,80 \
  --bo-iterations 1 \
  --output-root /tmp/cbo_key_check
```

## 6. 推荐先跑：selected12 小规模验证

不要一上来直接跑 108。V1.4 先用 12 个代表场景验证方向：

```bash
cd /home/ecs-user/CBO
source env.sh

MAX_JOBS=6 bash run_static_v17_internal_context_selected.sh 43
```

默认包含 12 个场景：

```text
lambda=3.0, rt=10, batch=40, ai=50
lambda=3.0, rt=30, batch=60, ai=10
lambda=3.0, rt=10, batch=30, ai=60
lambda=2.6, rt=20, batch=10, ai=70
lambda=2.6, rt=70, batch=20, ai=10
lambda=2.6, rt=40, batch=40, ai=20
lambda=1.8, rt=10, batch=10, ai=80
lambda=1.8, rt=80, batch=10, ai=10
lambda=3.0, rt=40, batch=40, ai=20
lambda=2.6, rt=10, batch=80, ai=10
lambda=1.8, rt=30, batch=30, ai=40
lambda=3.0, rt=70, batch=10, ai=20
```

包含方法：

```text
reduced7_bo_adaptive
reduced7_cbo_lite_pressure_taskmix_counts
reduced7_cbo_lite_internal4
reduced7_cbo_lite_internal6_context
reduced7_cbo_lite_internal4_context
```

结果目录：

```text
result/static_v17_internal_context_selected12_s43
```

完成标准：

```text
pressure summaries = 12 / 12
round summaries    = 60 / 60
```

selected12 分析：

```bash
python analyze_static108_abcd_matched.py \
  --variant B=result/static_v17_internal_context_selected12_s43=BO_ADAPTIVE \
  --variant D0=result/static_v17_internal_context_selected12_s43=CBO \
  --variant D4=result/static_v17_internal_context_selected12_s43=CBO_I4 \
  --variant D6C=result/static_v17_internal_context_selected12_s43=CBO_I6CTX \
  --variant D4C=result/static_v17_internal_context_selected12_s43=CBO_I4CTX \
  --comparison B:D0 \
  --comparison B:D4 \
  --comparison B:D6C \
  --comparison B:D4C \
  --comparison D0:D4 \
  --comparison D0:D6C \
  --comparison D0:D4C \
  --output result/static_v17_internal_context_selected12_analysis_s43
```

如果 selected12 中 `D4C` 相比 `D0` 和 `B` 没有改善，就先不要跑 108，应该回头检查 internal similarity 的样本选择是否真的生效。

## 7. 实验一：无自适应探索的 CBO 结构消融

目的：回答“CBO 结构本身为什么不如 BO/为什么前期弱”。

运行：

```bash
cd /home/ecs-user/CBO
source env.sh

MAX_JOBS=10 VARIANT=cmods bash run_static108_v16_bc_ablation.sh 43
```

包含方法：

```text
reduced7_cbo_lite_pressure_taskmix_counts
reduced7_cbo_lite_internal4
reduced7_cbo_lite_internal6_context
reduced7_cbo_lite_internal4_context
```

结果目录：

```text
result/static108_v16_cmods_noadaptive_s43
```

完成标准：

```text
pressure summaries = 108 / 108
round summaries    = 432 / 432
```

## 8. 实验二：带 adaptive exploration 的结构消融

目的：回答“完整 CBO 中，internal4/context history 是否进一步增强 D 组”。

运行：

```bash
cd /home/ecs-user/CBO
source env.sh

MAX_JOBS=10 bash run_static108_v17_internal_context_adaptive.sh 43
```

包含方法：

```text
reduced7_bo_adaptive
reduced7_cbo_lite_pressure_taskmix_counts
reduced7_cbo_lite_internal4
reduced7_cbo_lite_internal6_context
reduced7_cbo_lite_internal4_context
```

所有 CBO 变体都使用：

```text
--cbo-sigma-calibration-use-in-acq adaptive
```

结果目录：

```text
result/static108_v17_internal_context_adaptive_s43
```

完成标准：

```text
pressure summaries = 108 / 108
round summaries    = 540 / 540
```

## 9. 运行进度查询

V16：

```bash
cd /home/ecs-user/CBO

find result/static108_v16_cmods_noadaptive_s43 \
  -name pressure_scan_summary_all.csv -type f | wc -l

find result/static108_v16_cmods_noadaptive_s43 \
  -name '*round_summary*.csv' -type f | wc -l

pgrep -fc 'python.*new_tr_split'
```

V17：

```bash
cd /home/ecs-user/CBO

find result/static108_v17_internal_context_adaptive_s43 \
  -name pressure_scan_summary_all.csv -type f | wc -l

find result/static108_v17_internal_context_adaptive_s43 \
  -name '*round_summary*.csv' -type f | wc -l

pgrep -fc 'python.*new_tr_split'
```

内存和负载：

```bash
free -h
uptime
pgrep -fc 'python.*new_tr_split'
```

10 并行在 64GB 内存服务器上通常可以承受；如果 load 长期明显高于 vCPU 数或内存 available 低于 10GB，再降到 7-8 并行。

## 10. V17 分析命令

```bash
cd /home/ecs-user/CBO
source env.sh

python analyze_static108_abcd_matched.py \
  --variant B=result/static108_v17_internal_context_adaptive_s43=BO_ADAPTIVE \
  --variant D0=result/static108_v17_internal_context_adaptive_s43=CBO \
  --variant D4=result/static108_v17_internal_context_adaptive_s43=CBO_I4 \
  --variant D6C=result/static108_v17_internal_context_adaptive_s43=CBO_I6CTX \
  --variant D4C=result/static108_v17_internal_context_adaptive_s43=CBO_I4CTX \
  --comparison B:D0 \
  --comparison B:D4 \
  --comparison B:D6C \
  --comparison B:D4C \
  --comparison D0:D4 \
  --comparison D0:D6C \
  --comparison D0:D4C \
  --output result/static108_v17_internal_context_analysis_s43
```

重点看：

```text
001-050
051-100
101-200
451-500
all500
```

核心判断：

- `D4 - D0 < 0`：说明降维 internal4 有帮助。
- `D6C - D0 < 0`：说明内部相似历史补充有帮助。
- `D4C - D0 < 0`：说明两者组合有效。
- `D4C - B < 0`：说明完整 CBO 变体优于 adaptive BO。

指标优先级：

```text
Eval_Cost
normalized_tradeoff_score
Backlog / unfinished_rate / avg_delay
energy_norm
```

如果 normalized 差距很小，但 backlog、unfinished、delay 明显改善，仍然说明调度安全性和服务质量有实际收益。

## 11. V16 分析命令

如果要专门看 CBO 结构本身：

```bash
cd /home/ecs-user/CBO
source env.sh

python analyze_static108_abcd_matched.py \
  --variant C0=result/static108_v16_cmods_noadaptive_s43=CBO \
  --variant C4=result/static108_v16_cmods_noadaptive_s43=CBO_I4 \
  --variant C6C=result/static108_v16_cmods_noadaptive_s43=CBO_I6CTX \
  --variant C4C=result/static108_v16_cmods_noadaptive_s43=CBO_I4CTX \
  --comparison C0:C4 \
  --comparison C0:C6C \
  --comparison C0:C4C \
  --output result/static108_v16_cmods_analysis_s43
```

判断方式：

- 如果 C4C 明显好于 C0，说明原主 CBO 的内部结构确实有改进空间。
- 如果 C4C 仍然弱，说明静态场景下 CBO 的主要收益不是结构划分，而是 adaptive exploration；CBO 的优势应更多放在动态切换和跨相似场景复用上证明。

## 12. 预期结论口径

如果实验结果支持，论文/报告里可以这样表述：

```text
在静态场景中，CBO 的外部门控不会产生明显筛选作用，因为外部负载状态保持稳定。
此时 CBO 的收益主要来自内部运行状态建模和自适应探索。
原 internal6 直接拼接到 GP 输入会增加样本稀疏性，因此在 80-window 小样本下可能造成前期弱势。
V1.4 通过 internal4 降维和 internal-context history 补充相似内部状态样本，验证 CBO 是否能够更有效复用相近运行状态经验。
```

动态实验中再强调：

```text
CBO 的核心价值不只是在单一静态场景中超过 BO，而是在外部场景切换时减少跨场景负迁移，并在相似内部状态之间复用经验。
```
selected12：

```bash
cd /home/ecs-user/CBO

find result/static_v17_internal_context_selected12_s43 \
  -name pressure_scan_summary_all.csv -type f | wc -l

find result/static_v17_internal_context_selected12_s43 \
  -name '*round_summary*.csv' -type f | wc -l

pgrep -fc 'python.*new_tr_split'
```

