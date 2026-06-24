# CBO 静态 108 ABCD 服务器结果存放文档 V1.0

## 1. 固定标签口径

后续所有静态 108 和 selected12 分析，建议固定使用下面标签：

```text
A0 = 原 BO / reduced7_bo_greedy
B  = BO + adaptive exploration / reduced7_bo_adaptive
C  = 原 CBO 结构，无 adaptive exploration / reduced7_cbo_lite_pressure_taskmix_counts
D  = 完整 CBO，含 adaptive exploration / reduced7_cbo_lite_pressure_taskmix_counts
```

注意：`C` 和 `D` 的方法名前缀相同，都是：

```text
reduced7_cbo_lite_pressure_taskmix_counts
```

它们的区别来自结果目录和运行配置，不是文件名前缀：

```text
C = static108_v16_bc_s43 中的 CBO
D = static108_v15_adaptive_core_s43 中的 CBO
```

因此以后做分析时必须同时标记：

```text
variant label + result root + method label
```

不能只看 method name。

## 2. Seed 43 四组主结果位置

服务器根目录：

```text
/home/ecs-user/CBO
```

### A0：原 BO

推荐使用：

```text
result/static108_v11_sigma_calibrated_s43
```

方法：

```text
reduced7_bo_greedy
```

文件模式：

```text
reduced7_bo_greedy*round_summary*.csv
```

说明：这是之前作为原 BO 基准使用的位置。

可复核位置：

```text
result/static108_v15_adaptive_core_s43
```

该目录也有 108 个 `reduced7_bo_greedy`，但为了保持既有分析口径，默认仍把 `static108_v11_sigma_calibrated_s43` 作为 A0。

### B：BO + adaptive exploration

位置：

```text
result/static108_v16_bc_s43
```

方法：

```text
reduced7_bo_adaptive
```

文件模式：

```text
reduced7_bo_adaptive*round_summary*.csv
```

说明：这是在 BO 基础上加入 adaptive exploration 的消融组。

### C：原 CBO 结构，无 adaptive exploration

位置：

```text
result/static108_v16_bc_s43
```

方法：

```text
reduced7_cbo_lite_pressure_taskmix_counts
```

文件模式：

```text
reduced7_cbo_lite_pressure_taskmix_counts*round_summary*.csv
```

说明：这是 external gate + internal6 的 CBO 结构，但未开启 adaptive acquisition。它用于回答“CBO 结构本身是否优于 BO”。

### D：完整 CBO，含 adaptive exploration

位置：

```text
result/static108_v15_adaptive_core_s43
```

方法：

```text
reduced7_cbo_lite_pressure_taskmix_counts
```

文件模式：

```text
reduced7_cbo_lite_pressure_taskmix_counts*round_summary*.csv
```

说明：这是当前更接近主方法的完整 CBO。它用于回答“CBO 结构 + adaptive exploration 是否优于 BO”。

## 3. 多 seed 可用结果

当前服务器上已经发现这些完整 108 目录：

### 原 BO / reduced7_bo_greedy

```text
108 result/static108_v15_adaptive_core_s45
108 result/static108_v15_adaptive_core_s44
108 result/static108_v15_adaptive_core_s43
108 result/static108_v11_sigma_calibrated_s43
```

### BO adaptive / reduced7_bo_adaptive

```text
108 result/static108_v16_bc_s43
```

当前只确认 seed 43 有完整 B 组。

### 原 CBO / 完整 CBO / reduced7_cbo_lite_pressure_taskmix_counts

```text
108 result/static108_v16_bc_s43
108 result/static108_v15_adaptive_core_s45
108 result/static108_v15_adaptive_core_s44
108 result/static108_v15_adaptive_core_s43
108 result/static108_v11_sigma_calibrated_s43
```

解释：

- `static108_v16_bc_s43` 中的 CBO 是 C 组，即无 adaptive exploration。
- `static108_v15_adaptive_core_s43/s44/s45` 中的 CBO 是 D 组，即完整 CBO。
- `static108_v11_sigma_calibrated_s43` 是较早 sigma 校准实验目录，默认不作为 D 组主口径。

## 4. 新 V17 selected12 变体位置

新内部情景结构消融 selected12 结果：

```text
result/static_v17_internal_context_selected12_newmods_s43
```

包含 12 个代表场景，3 个新方法：

```text
D4  = reduced7_cbo_lite_internal4
D6C = reduced7_cbo_lite_internal6_context
D4C = reduced7_cbo_lite_internal4_context
```

完成标准：

```text
pressure summaries = 12 / 12
round summaries    = 36 / 36
configs            = 12 / 12
failed             = 0
```

## 5. 服务器计数检查命令

### A0 原 BO

```bash
cd /home/ecs-user/CBO

find result/static108_v11_sigma_calibrated_s43 \
  -name 'reduced7_bo_greedy*round_summary*.csv' | wc -l
```

期望：

```text
108
```

### B BO adaptive

```bash
find result/static108_v16_bc_s43 \
  -name 'reduced7_bo_adaptive*round_summary*.csv' | wc -l
```

期望：

```text
108
```

### C 原 CBO 无 adaptive

```bash
find result/static108_v16_bc_s43 \
  -name 'reduced7_cbo_lite_pressure_taskmix_counts*round_summary*.csv' | wc -l
```

期望：

```text
108
```

### D 完整 CBO

```bash
find result/static108_v15_adaptive_core_s43 \
  -name 'reduced7_cbo_lite_pressure_taskmix_counts*round_summary*.csv' | wc -l
```

期望：

```text
108
```

### V17 selected12 newmods

```bash
OUT=result/static_v17_internal_context_selected12_newmods_s43

echo configs=$(find "$OUT" -name refactor_run_config.json -type f 2>/dev/null | wc -l)/12
echo pressure=$(find "$OUT" -name pressure_scan_summary_all.csv -type f 2>/dev/null | wc -l)/12
echo rounds=$(find "$OUT" -name '*round_summary*.csv' -type f 2>/dev/null | wc -l)/36
echo failed=$(awk 'NF{c++} END{print c+0}' "$OUT/failed_jobs.txt" 2>/dev/null || echo 0)
```

## 6. 查找目录命令

如果忘记结果在哪，可以用：

```bash
cd /home/ecs-user/CBO

find result -name 'reduced7_bo_greedy*round_summary*.csv' \
  | awk -F'/lambda_' '{print $1}' \
  | sort | uniq -c | sort -nr | head -n 20
```

```bash
find result -name 'reduced7_bo_adaptive*round_summary*.csv' \
  | awk -F'/lambda_' '{print $1}' \
  | sort | uniq -c | sort -nr | head -n 20
```

```bash
find result -name 'reduced7_cbo_lite_pressure_taskmix_counts*round_summary*.csv' \
  | awk -F'/lambda_' '{print $1}' \
  | sort | uniq -c | sort -nr | head -n 30
```

## 7. full108 ABCD 分析命令

```bash
cd /home/ecs-user/CBO
source env.sh

python analyze_static108_abcd_matched.py \
  --variant A0=result/static108_v11_sigma_calibrated_s43=BO \
  --variant B=result/static108_v16_bc_s43=BO_ADAPTIVE \
  --variant C=result/static108_v16_bc_s43=CBO \
  --variant D=result/static108_v15_adaptive_core_s43=CBO \
  --comparison A0:B \
  --comparison A0:C \
  --comparison A0:D \
  --comparison B:C \
  --comparison B:D \
  --comparison C:D \
  --output result/static108_abcd_storage_check_s43
```

解释：

- `A0:B`：adaptive exploration 对 BO 的贡献。
- `A0:C`：CBO 结构本身相对原 BO 的表现。
- `A0:D`：完整 CBO 相对原 BO 的表现。
- `B:C`：原 CBO 结构是否优于 BO adaptive。
- `B:D`：完整 CBO 是否优于 BO adaptive。
- `C:D`：adaptive exploration 对 CBO 的贡献。

## 8. selected12 与 full108 ABCD 对齐分析

selected12 只跑了 12 个场景，不能直接和 full108 的 108 行混在一个汇总里。必须先生成 selected12 场景列表，再用 `--scene-list` 过滤 full108。

### 8.1 生成 selected12 场景列表

```bash
cd /home/ecs-user/CBO

find result/static_v17_internal_context_selected12_newmods_s43 \
  -name pressure_scan_summary_all.csv \
  | sed 's#result/static_v17_internal_context_selected12_newmods_s43/##' \
  | sed 's#/pressure_scan_summary_all.csv##' \
  | sort > result/static_v17_selected12_scenes.txt
```

检查：

```bash
wc -l result/static_v17_selected12_scenes.txt
cat result/static_v17_selected12_scenes.txt
```

期望：

```text
12
```

### 8.2 selected12 对齐分析

```bash
python analyze_static108_abcd_matched.py \
  --scene-list result/static_v17_selected12_scenes.txt \
  --variant A0=result/static108_v11_sigma_calibrated_s43=BO \
  --variant B=result/static108_v16_bc_s43=BO_ADAPTIVE \
  --variant C=result/static108_v16_bc_s43=CBO \
  --variant D=result/static108_v15_adaptive_core_s43=CBO \
  --variant D4=result/static_v17_internal_context_selected12_newmods_s43=CBO_I4 \
  --variant D6C=result/static_v17_internal_context_selected12_newmods_s43=CBO_I6CTX \
  --variant D4C=result/static_v17_internal_context_selected12_newmods_s43=CBO_I4CTX \
  --comparison A0:B \
  --comparison A0:C \
  --comparison A0:D \
  --comparison A0:D4 \
  --comparison A0:D6C \
  --comparison A0:D4C \
  --comparison B:C \
  --comparison B:D \
  --comparison B:D4 \
  --comparison B:D6C \
  --comparison B:D4C \
  --comparison C:D \
  --comparison C:D4 \
  --comparison C:D6C \
  --comparison C:D4C \
  --comparison D:D4 \
  --comparison D:D6C \
  --comparison D:D4C \
  --output result/static_v17_selected12_internal_context_analysis_with_abcd_s43
```

## 9. 结论读取顺序

建议按下面顺序读结果：

1. `A0:B`：确认 adaptive exploration 单独是否有效。
2. `A0:C`：确认 CBO 结构单独是否有效。
3. `C:D`：确认 adaptive exploration 对 CBO 的贡献。
4. `B:D`：确认完整 CBO 是否优于 BO adaptive。
5. `C:D4/D6C/D4C`：确认 V17 内部情景结构是否优于原 CBO 结构。
6. `D:D4/D6C/D4C`：确认 V17 新结构是否优于完整 CBO 旧结构。
7. `B:D4C`：确认最终推荐的新 CBO 是否优于 BO adaptive。

重点指标：

```text
Eval_Cost
normalized_tradeoff_score
Backlog
unfinished_rate
avg_delay
energy_norm
```

如果 normalized 分数差距很小，但 backlog、unfinished_rate、avg_delay 明显下降，说明调度稳定性和服务质量仍然有实际收益。

