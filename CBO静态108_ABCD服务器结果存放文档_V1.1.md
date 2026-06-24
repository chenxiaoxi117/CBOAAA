# CBO 静态 108 服务器结果与补充实验记录 V1.1

更新时间：2026-06-23

本文档记录服务器 `/home/ecs-user/CBO` 下已经确认存在的原始实验数据目录，以及后续建议补充的实验输出位置和运行命令。

## 1. 已确认的原始数据目录

原始数据指每个场景目录下的：

```text
*_round_summary_轮次汇总.csv
pressure_scan_summary_all.csv
refactor_run_config.json
```

不是 `inventory` 汇总表。

## 2. 静态 108 全基线目录

位置：

```text
/home/ecs-user/CBO/result/static108_v11_sigma_calibrated_s43
```

该目录包含完整 108 场景：

```text
direct_greedy_cost                              108
direct_queue_aware_greedy                       108
reduced7_fixed_mid                              108
reduced7_fixed_tuned                            108
reduced7_bo_greedy                              108
reduced7_cbo_lite_pressure_taskmix_counts       108
```

用途：

```text
直接调度方法
固定权重方法
原 BO / A0
原 CBO 的早期结果
```

后续论文中如果需要“直接调度 / 固定权重 / 原 BO”的原始数据，优先从这个目录取。

## 3. 多 seed 原 BO/CBO 目录

位置：

```text
/home/ecs-user/CBO/result/static108_v15_adaptive_core_s43
/home/ecs-user/CBO/result/static108_v15_adaptive_core_s44
/home/ecs-user/CBO/result/static108_v15_adaptive_core_s45
```

每个目录包含：

```text
reduced7_bo_greedy                              108
reduced7_cbo_lite_pressure_taskmix_counts       108
```

用途：

```text
BO vs CBO 的多 seed 稳健性分析
完整 CBO 主方法 D 的 108 场景结果
```

注意：这里的 `reduced7_cbo_lite_pressure_taskmix_counts` 与其他目录中文件名前缀相同，但运行配置可能不同。分析时必须同时记录：

```text
variant label + result root + method name
```

不能只看 method name。

## 4. BO 自适应探索目录

位置：

```text
/home/ecs-user/CBO/result/static108_v16_bc_s43
```

包含完整 108 场景：

```text
reduced7_bo_adaptive                            108
reduced7_cbo_lite_pressure_taskmix_counts       108
```

用途：

```text
B = BO + adaptive exploration
C = 原 CBO 结构，不含 adaptive acquisition
```

该目录用于回答：

```text
adaptive exploration 单独加到 BO 上是否有效
原 CBO 结构相对 BO adaptive 是否有效
```

## 5. V17 selected12 新内部情景结构目录

位置：

```text
/home/ecs-user/CBO/result/static_v17_internal_context_selected12_newmods_s43
```

包含 12 个代表场景：

```text
reduced7_cbo_lite_internal4                     12
reduced7_cbo_lite_internal6_context             12
reduced7_cbo_lite_internal4_context             12
```

方法含义：

```text
D4   = internal4，仅改变内部情景维度
D6C  = internal6 + recent_context，相似内部状态样本补充
D4C  = internal4 + recent_context，相似内部状态样本补充
```

注意：这三组是已经带 adaptive exploration 的结果，不是纯结构消融。

## 6. 当前建议补充的实验

不建议马上重跑所有旧实验。当前最缺的是两类补充：

```text
补充实验 1：新内部情景结构 + adaptive exploration 的完整 108 场景
补充实验 2：新内部情景结构但不加 adaptive exploration 的完整 108 场景
```

这样可以拆开两个问题：

```text
内部情景结构本身是否有效
adaptive exploration 是否进一步加快前期收敛并保持后期稳定
```

## 7. 补充实验 1：V17 full108 adaptive newmods

推荐输出目录：

```text
/home/ecs-user/CBO/result/static108_v17_internal_context_adaptive_s43
```

推荐跑完整 5 组，虽然 B/D 已经有历史结果，但同一脚本同一批次更容易做严格对齐：

```text
B    = reduced7_bo_adaptive
D0   = reduced7_cbo_lite_pressure_taskmix_counts
D4   = reduced7_cbo_lite_internal4
D6C  = reduced7_cbo_lite_internal6_context
D4C  = reduced7_cbo_lite_internal4_context
```

运行命令：

```bash
cd /home/ecs-user/CBO
source env.sh

tmux new -s static108_v17_adaptive
```

进入 tmux 后运行：

```bash
cd /home/ecs-user/CBO
source env.sh

MAX_JOBS=10 \
OUT=/home/ecs-user/CBO/result/static108_v17_internal_context_adaptive_s43 \
bash run_static108_v17_internal_context_adaptive.sh 43
```

预期完成数量：

```text
pressure summaries = 108 / 108
refactor configs   = 108 / 108
round summaries    = 540 / 540
failed records     = 0
```

如果为了节省时间，只补 D4/D6C/D4C 三个新方法，可以运行：

```bash
cd /home/ecs-user/CBO
source env.sh

MAX_JOBS=10 \
METHODS=reduced7_cbo_lite_internal4,reduced7_cbo_lite_internal6_context,reduced7_cbo_lite_internal4_context \
OUT=/home/ecs-user/CBO/result/static108_v17_internal_context_newmods_s43 \
bash run_static108_v17_internal_context_adaptive.sh 43
```

预期完成数量：

```text
pressure summaries = 108 / 108
refactor configs   = 108 / 108
round summaries    = 324 / 324
failed records     = 0
```

论文主结果更推荐完整 5 组；快速补数据可以用只跑新方法版本。

## 8. 补充实验 2：V17 full108 no-adaptive 结构消融

推荐输出目录：

```text
/home/ecs-user/CBO/result/static108_v17_cmods_noadaptive_s43
```

包含方法：

```text
C    = reduced7_cbo_lite_pressure_taskmix_counts
D4   = reduced7_cbo_lite_internal4
D6C  = reduced7_cbo_lite_internal6_context
D4C  = reduced7_cbo_lite_internal4_context
```

运行命令：

```bash
cd /home/ecs-user/CBO
source env.sh

tmux new -s static108_v17_noadaptive
```

进入 tmux 后运行：

```bash
cd /home/ecs-user/CBO
source env.sh

MAX_JOBS=10 \
VARIANT=cmods \
OUT=/home/ecs-user/CBO/result/static108_v17_cmods_noadaptive_s43 \
bash run_static108_v16_bc_ablation.sh 43
```

预期完成数量：

```text
pressure summaries = 108 / 108
refactor configs   = 108 / 108
round summaries    = 432 / 432
failed records     = 0
```

该实验用于回答：

```text
不依赖 adaptive exploration 时，internal4 / recent_context 是否仍然改善 CBO
D4C 的优势来自内部情景结构，还是主要来自 adaptive exploration
```

## 9. 运行进度检查命令

### 9.1 检查 V17 adaptive full108

```bash
cd /home/ecs-user/CBO

OUT=result/static108_v17_internal_context_adaptive_s43

echo running=$(pgrep -fc 'python.*new_tr_split')
echo configs=$(find "$OUT" -name refactor_run_config.json -type f 2>/dev/null | wc -l)/108
echo pressure=$(find "$OUT" -name pressure_scan_summary_all.csv -type f 2>/dev/null | wc -l)/108
echo rounds=$(find "$OUT" -name '*round_summary*.csv' -type f 2>/dev/null | wc -l)/540
echo failed=$(awk 'NF{c++} END{print c+0}' "$OUT/failed_jobs.txt" 2>/dev/null || echo 0)
```

如果只跑新方法，把目录和 round 数改成：

```bash
OUT=result/static108_v17_internal_context_newmods_s43
echo rounds=$(find "$OUT" -name '*round_summary*.csv' -type f 2>/dev/null | wc -l)/324
```

### 9.2 检查 V17 no-adaptive full108

```bash
cd /home/ecs-user/CBO

OUT=result/static108_v17_cmods_noadaptive_s43

echo running=$(pgrep -fc 'python.*new_tr_split')
echo configs=$(find "$OUT" -name refactor_run_config.json -type f 2>/dev/null | wc -l)/108
echo pressure=$(find "$OUT" -name pressure_scan_summary_all.csv -type f 2>/dev/null | wc -l)/108
echo rounds=$(find "$OUT" -name '*round_summary*.csv' -type f 2>/dev/null | wc -l)/432
echo failed=$(awk 'NF{c++} END{print c+0}' "$OUT/failed_jobs.txt" 2>/dev/null || echo 0)
```

### 9.3 查看最近日志是否还在更新

```bash
find "$OUT/logs" -type f -mmin -10 -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -n 20
```

### 9.4 查看当前运行的输出目录

```bash
pgrep -af 'python.*new_tr_split' \
  | sed -n 's/.*--selected-keys \([^ ]*\).*--output-root \([^ ]*\).*/methods=\1\nout=\2\n/p'
```

## 10. 原始文件清单生成命令

生成主要原始轮次文件清单：

```bash
cd /home/ecs-user/CBO

find result/static108_v11_sigma_calibrated_s43 \
     result/static108_v15_adaptive_core_s43 \
     result/static108_v15_adaptive_core_s44 \
     result/static108_v15_adaptive_core_s45 \
     result/static108_v16_bc_s43 \
     result/static_v17_internal_context_selected12_newmods_s43 \
     result/static108_v17_internal_context_adaptive_s43 \
     result/static108_v17_cmods_noadaptive_s43 \
  -name '*round_summary*.csv' -type f 2>/dev/null | sort \
  > result/raw_round_summary_file_list_main_v1_1.txt
```

如果只跑了新方法 adaptive 目录，用：

```bash
result/static108_v17_internal_context_newmods_s43
```

替换上面的：

```bash
result/static108_v17_internal_context_adaptive_s43
```

## 11. Inventory 命令

```bash
cd /home/ecs-user/CBO
source env.sh

python inventory_cbo_results.py result \
  --output result/cbo_result_inventory_v4.csv
```

当前 v4 已确认：

```text
static108_v11_sigma_calibrated_s43：直接调度 + fixed + A0 BO + CBO，完整 108
static108_v15_adaptive_core_s43/s44/s45：A0 BO + CBO，多 seed 完整 108
static108_v16_bc_s43：BO adaptive + CBO，完整 108
static_v17_internal_context_selected12_newmods_s43：D4/D6C/D4C，selected12
```

补充实验跑完后，需要重新运行 inventory，并输出：

```bash
python inventory_cbo_results.py result \
  --output result/cbo_result_inventory_after_v17_full108.csv
```

## 12. 建议分析顺序

先分析 full108：

```text
A0 vs 直接调度/fixed：证明 BO 自动调参相对固定规则的意义
A0 vs B：证明 adaptive exploration 单独加到 BO 上的效果
A0/B vs C：证明原 CBO 结构是否优于 BO
C vs D4/D6C/D4C no-adaptive：证明内部情景结构本身是否有效
D4/D6C/D4C no-adaptive vs D4/D6C/D4C adaptive：证明 adaptive exploration 是否进一步加速收敛
B/D0 vs D4C adaptive：证明最终推荐 CBO 是否优于 BO adaptive 和旧 CBO
```

再分析 selected12 图像趋势：

```text
rolling50 normalized_tradeoff_score
rolling50 Eval_Cost
rolling50 unfinished_rate
rolling50 avg_delay
rolling50 energy_norm
```

不要只看每段均值，要重点看：

```text
什么时候低于 BO
什么时候进入稳定区间
最终是否收敛到更低或相近位置
是否出现后期 rebound
```

