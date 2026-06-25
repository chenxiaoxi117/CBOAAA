# CBO 动态外部-内部情景筛选修改说明 V1.5

## 1. 修改背景

前一版动态实验中，`global_context_threshold` 主要是在全局历史样本中按内部情景相似度筛选。它能扩大历史数据来源，但存在一个核心问题：它没有严格执行“先外部场景、再内部情景”的两级筛选。

这会导致动态场景中，CBO 可能从全局历史中直接寻找内部状态相似样本，而这些样本可能来自不同任务强度或任务比例阶段。这样虽然样本数量变多，但可能引入跨阶段样本污染，使代理模型学习到不对应当前生产阶段的反馈关系。

因此 V1.5 的目标是把 CBO 历史样本选择流程改成：

```text
全局历史记录
  -> 外部场景相似筛选
  -> 内部情景相似筛选
  -> 样本不足时才补弱相似/最近样本
  -> 构建 GP 代理模型
```

## 2. 新增模式

新增 CBO 历史筛选模式：

```bash
--cbo-history-select-mode external_internal_threshold
```

该模式不依赖旧的 `cbo_external_gate_mode` 是否开启，而是在 CBO 训练样本选择阶段内部直接执行两级筛选。

## 3. 当前逻辑

V1.5 下，CBO 每轮仍然保留全局历史样本，但构建 GP 训练集时按以下顺序处理：

1. 外部场景筛选：
   使用外部情景向量判断阶段相似性，当前包含 `lambda + RT/Batch/AI 任务比例`。

2. 内部情景筛选：
   在外部相似样本池中，再根据 internal4/internal6 内部状态相似度筛选。

3. 强相似样本优先：
   内部相似度达到阈值的样本优先进入训练集。

4. 样本不足时补充：
   先补外部相似但内部相似度较弱的样本，再补最近样本，最后用最近尾部样本做冷启动兜底。

这对应的含义是：

```text
外部场景用于识别生产阶段；
内部情景用于区分同一或相近阶段中的窗口波动状态；
历史经验可以跨相似阶段复用，但不再无条件混用全局历史。
```

## 4. 新增参数

新增 CLI 参数：

```bash
--cbo-external-sim-threshold
--cbo-external-topk
--cbo-external-min-rows
--cbo-external-recent-keep
```

推荐初始配置：

```bash
--cbo-history-select-mode external_internal_threshold \
--cbo-external-sim-threshold 0.70 \
--cbo-external-topk 400 \
--cbo-external-min-rows 40 \
--cbo-external-recent-keep 20 \
--cbo-context-sim-threshold 0.70 \
--cbo-context-k 200 \
--cbo-context-min-rows 40 \
--cbo-context-weak-fallback-k 40
```

## 5. 代码修改位置

本次修改涉及：

```text
新的代码结构/去掉动态堆积/new_tr_split/core_config.py
新的代码结构/去掉动态堆积/new_tr_split/cli.py
新的代码结构/去掉动态堆积/new_tr_split/runtime_patches.py
新的代码结构/去掉动态堆积/new_tr_split/scenario_experiments.py
```

主要修改点：

```text
core_config.py
  增加 external_internal_threshold 默认参数。

cli.py
  增加 external_internal_threshold 模式和外部筛选参数。

runtime_patches.py
  增加 CBO 两级历史样本筛选逻辑。
  增加外部/内部筛选过程诊断字段。

scenario_experiments.py
  在 refactor_run_config.json 和 dynamic_run_config.json 中记录新增参数。
```

语法检查已通过：

```bash
python -m py_compile core_config.py cli.py runtime_patches.py scenario_experiments.py
```

## 6. 新增诊断字段

运行后可在 `dynamic_round_summary.csv` 中检查：

```text
external_internal_threshold_enabled
external_internal_external_threshold
external_internal_internal_threshold
external_internal_raw_records
external_internal_external_passed_count
external_internal_external_pool_count
external_internal_internal_scored_count
external_internal_internal_strong_count
external_internal_selected_strong_count
external_internal_selected_weak_count
external_internal_selected_recent_count
external_internal_final_fallback_count
external_internal_selected_external_similarity_mean
external_internal_selected_external_similarity_min
external_internal_selected_internal_similarity_mean
external_internal_selected_internal_similarity_min
external_internal_selected_phase_counts
```

判断是否符合预期：

```text
external_pool_count > 0：
  说明外部阶段筛选生效。

internal_strong_count 随阶段变化：
  说明内部情景相似度确实参与了筛选。

selected_recent_count / final_fallback_count 不长期过高：
  说明不是一直靠最近样本兜底。

selected_phase_counts 更集中：
  说明跨阶段污染减少。
```

## 7. 新增实验脚本

新增服务器运行脚本：

```text
run_dynamic_v22_external_internal_multiseed.sh
```

上传到服务器 `/home/ecs-user/CBO/` 后执行：

```bash
cd /home/ecs-user/CBO
chmod +x run_dynamic_v22_external_internal_multiseed.sh

tmux new -d -s dynamic_v22_extint \
'cd /home/ecs-user/CBO && MAX_JOBS=6 SEEDS="43 44 45" bash run_dynamic_v22_external_internal_multiseed.sh'
```

默认实验组合：

```text
A_ext070_ctx070_k200
B_ext075_ctx070_k200
C_ext065_ctx070_k200
D_ext070_ctx070_k300
```

每组包含：

```text
BO adaptive
CBO old
CBO internal6 context
CBO internal4 context
```

## 8. 进度检查

```bash
cd /home/ecs-user/CBO

echo "running:"
pgrep -af 'python.*new_tr_split' | wc -l

for d in result/dynamic_v22_*_s4[3-5]; do
  [ -d "$d" ] || continue
  if [ -f "$d/dynamic_round_summary.csv" ]; then
    rows=$(python - <<PY
import pandas as pd
print(len(pd.read_csv("$d/dynamic_round_summary.csv", low_memory=False)))
PY
)
    echo "OK rows=$rows $d"
  else
    echo "RUN/MISS $d"
  fi
done
```

每个 V22 动态实验如果为 12 个阶段、每阶段 150 轮、4 个方法，则 `dynamic_round_summary.csv` 应为：

```text
12 × 150 × 4 = 7200 行
```

## 9. 后续分析重点

V22 不只看最后 100/200 轮，还要看阶段切换恢复能力：

```text
all1800
first50 global
last100 global
每阶段 first20
每阶段 first50
每阶段 last50
阶段切换后前 20/50 轮恢复速度
```

重点比较：

```text
BO adaptive
旧 CBO
CBO internal6 context
CBO internal4 context
```

如果 V22 生效，期望观察到：

```text
1. A4C/internal4 context 在相似阶段复现时恢复更快；
2. 与旧 CBO 相比，跨阶段负迁移减少；
3. selected_phase_counts 更集中于相似外部阶段；
4. selected_internal_similarity_mean 高于旧全局筛选；
5. 最近样本兜底比例下降，不再主要依赖 recent fallback。
```

## 10. 当前结论定位

V1.5 的核心不是简单增加历史样本数量，而是修正历史样本选择结构：

```text
从“全局内部相似筛选”
改为“外部阶段优先门控 + 内部窗口状态细分 + 不足时弱相似补充”。
```

这更符合 CBO 的论文表述：

```text
外部情景识别不同生产阶段；
内部情景识别相似阶段内的窗口状态差异；
通过相似历史经验复用提高动态场景下的恢复速度，并减少跨阶段负迁移。
```
