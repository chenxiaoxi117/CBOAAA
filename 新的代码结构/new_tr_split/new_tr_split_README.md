# new_tr_split

`new_tr_split` 是 `new_TR.py` 的拆分版实现，用于云-边-端制造系统中的上下文贝叶斯优化调度实验。

本目录只维护拆分后的实验代码。项目根目录中的 `new_TR.py` 保留为原始 monolithic 版本，不建议在后续实验中继续修改。后续新增 CBO 方法、调度器、diagnostics、跨场景分析和联邦迁移机制，优先在 `new_tr_split` 目录下完成。

## 1. 项目目标

本项目研究面向多工厂云-边-端制造场景的上下文贝叶斯优化调度方法。核心问题是：

- 在不同任务结构下，例如 RT-heavy、Batch-heavy、AI-heavy、balanced 场景中，自适应学习调度控制参数；
- 在不同负载强度下，例如 `lambda=2.6` 和 `lambda=3.0`，保持较低的综合调度代价；
- 在 CBO 后期预测不准、探索漂移或跨场景迁移不匹配时，通过 deployment guard 降低负迁移和策略退化；
- 为后续 cross-factory warm-start、federated-inspired experience sharing 和更严格的 federated CBO 奠定实验框架。

当前实验主线不是简单证明所有 CBO 版本都无条件超过 fixed baseline，而是重点比较：

1. fixed tuned baseline；
2. 原始 pressure-only CBO；
3. 加入 prev-unfinished context 和 prediction-aware deployment guard 后的新 CBO；
4. 不同任务比例、不同到达率和不同随机种子下的全场景稳定性。

## 2. 代码结构

`new_tr_split` 通过 `runtime.py` 按原始 `new_TR.py` 的执行顺序加载各功能模块。这样可以保持旧脚本中共享全局变量和 late runtime patch 的兼容性。

主要文件如下：

```text
new_tr_split/
├── __init__.py
├── __main__.py
├── runtime.py
├── cli.py
├── core_config.py
├── simulation_entities.py
├── schedulers.py
├── agents.py
├── factory.py
├── basic_outputs.py
├── diagnostics.py
├── scenario_experiments.py
├── runtime_patches.py
├── offline_export.py
└── sensitivity.py
```

各文件职责：

### `core_config.py`

全局配置文件。包括：

- `ExperimentConfig`
- 节点拓扑配置
- 任务类型配置
- BO 控制向量定义
- 调度器权重边界
- CBO / TR / prediction guard / warm-start / denoise 默认参数
- `CFG` 全局配置对象

常见需要修改的内容：

- 任务到达率；
- RT / Batch / AI 任务比例；
- BO 迭代轮数；
- scheduler tradeoff mode；
- CBO guard 默认参数；
- 控制变量搜索边界。

### `simulation_entities.py`

仿真基础实体，包括：

- `Event`
- `Task`
- `Node`
- 工作负载生成逻辑

如果要改任务属性、任务生成机制、节点状态更新逻辑，可以优先看这个文件。

### `schedulers.py`

调度器实现文件。包括：

- Boltzmann scheduler；
- constrained Boltzmann scheduler；
- direct heuristic scheduler；
- RoundRobin scheduler；
- alpha-direct latency-energy tradeoff scoring；
- queue / risk / cloud gate / opportunity window 等调度评分项。

如果要改底层节点选择公式，例如：

```text
score = alpha * latency + (1 - alpha) * energy + risk + queue
```

优先修改这里。

### `agents.py`

BO / CBO agent 文件。包括：

- `FederatedBOAgent`
- GP surrogate 训练；
- contextual BO 输入拼接；
- candidate selection；
- 本地历史样本；
- warm-start history 接入；
- denoise / outlier filter hook；
- state signature、cohort、scenario monitor 等辅助结构。

如果要改 BO surrogate、GP 训练数据、history selection、federated aggregation 或 warm-start 数据注入，优先修改这里。

### `factory.py`

事件驱动仿真主流程。包括：

- `ConnectedFactory`
- 每个 window 内的任务生成、调度、执行、统计；
- 调度结果反馈；
- 节点、队列、任务完成情况和代价计算。

如果要改仿真行为、任务流转、窗口统计方式，优先看这里。

### `scenario_experiments.py`

实验方法组和场景实验入口。包括：

- fixed baseline；
- vanilla BO；
- CBO pressure-only；
- CBO pressure + prev-unfinished context；
- alpha-direct CBO；
- no-risk ablation；
- scenario experiment；
- pressure scan；
- ratio grid experiment；
- selected key / alias 管理。

常用方法 key 包括：

```text
reduced6_fixed_tuned
reduced6_cbo_lite_pressure_only
reduced6_cbo_lite_pressure_prev_unfinished
reduced6_cbo_alpha_direct
reduced6_cbo_alpha_direct_no_risk
reduced6_cbo_alpha_direct_unfinished_context
reduced6_cbo_alpha_direct_prev_unfinished_context
```

当前跨场景实验主线建议重点比较：

```text
reduced6_fixed_tuned
reduced6_cbo_lite_pressure_only
reduced6_cbo_lite_pressure_prev_unfinished
```

其中：

- `reduced6_fixed_tuned` 是固定强基线；
- `reduced6_cbo_lite_pressure_only` 是原始 CBO baseline；
- `reduced6_cbo_lite_pressure_prev_unfinished` 是加入 prev-unfinished context 的新 CBO 方法，适合作为当前主方法分析。

### `runtime_patches.py`

CBO 运行时增强和 patch 文件。包括：

- deploy / history policy override；
- dual feedback；
- trust region 逻辑；
- residual / condition adaptive TR；
- good-region guard；
- prediction-error-aware deployment guard；
- candidate 与 deployed theta 分离；
- active guard 诊断字段；
- BO prediction error / surprise 统计。

如果要改 CBO 后期稳定性、部署回退、prediction guard、good-region guard、candidate deployment 逻辑，优先修改这里。

### `diagnostics.py`

诊断输出文件。包括：

- round summary；
- allocation diagnostics；
- key metric summary；
- diagnostic plots；
- guard / TR / alpha-direct / denoise 等字段写出。

如果新增了运行时变量，但 CSV 中没有导出，应优先检查这里。

### `basic_outputs.py`

基础输出和绘图辅助文件。包括：

- 旧版 plotting；
- log aggregation；
- metric summary；
- parameter scan helper。

### `offline_export.py`

离线分析与短文件名导出辅助文件。包括：

- offline window noise diagnostic；
- short-name export helper。

其中 short-name export 不是主实验必须功能。只有显式使用 `--export-short-names` 时，才会把结果复制到 `_short_export` 并额外打包。若不需要该压缩包，不要传入 `--export-short-names`。

### `cli.py`

命令行入口。所有命令行参数在这里注册，并最终调用不同运行模式：

```text
scenario
pressure_scan
ratio_grid
offline_noise
sensitivity
param
scan
extreme
```

### `runtime.py`

兼容加载器。它会按固定顺序执行拆分后的各个文件，保持原始 `new_TR.py` 的运行行为。

不要随意改变 `runtime.py` 中的加载顺序，除非同步处理相关 shared globals 和 runtime patches。

### `__main__.py`

支持以下方式运行：

```bash
python -m new_tr_split
```

它会加载 `runtime.py`，再执行 `cli.py`。

## 3. 环境准备

建议使用 Python 3.10 或更新版本。

主要依赖包括：

```text
numpy
pandas
matplotlib
scikit-learn
torch
gpytorch
botorch
```

建议在项目根目录运行：

```bash
python -m new_tr_split --help
```

如果 help 能正常显示，说明拆分版入口可用。

## 4. 快速运行

在 Windows / PowerShell 中，推荐从项目根目录运行：

```powershell
cd D:\CBOv2
python -m new_tr_split --help
```

运行默认 scenario 实验：

```powershell
python -m new_tr_split --mode scenario
```

运行指定方法：

```powershell
python -m new_tr_split `
  --mode scenario `
  --selected-keys reduced6_fixed_tuned,reduced6_cbo_lite_pressure_prev_unfinished `
  --bo-iterations 500 `
  --bo-interval 20 `
  --session-duration 10000 `
  --fixed-rng `
  --fixed-seed 43 `
  --output-root results\scene_sweep_main
```

运行原始 CBO baseline 与新 CBO 对照：

```powershell
python -m new_tr_split `
  --mode scenario `
  --selected-keys reduced6_fixed_tuned,reduced6_cbo_lite_pressure_only,reduced6_cbo_lite_pressure_prev_unfinished `
  --bo-iterations 500 `
  --bo-interval 20 `
  --session-duration 10000 `
  --fixed-rng `
  --fixed-seed 43 `
  --output-root results\compare_original_cbo_vs_guard
```

## 5. 当前推荐的全场景分析设计

当前建议做全场景分析，而不是只看单个 P1 或 λ=3.0 场景。

推荐场景矩阵：

```text
S1: RT60 / Batch30 / AI10, lambda=2.6
S2: RT10 / Batch20 / AI70, lambda=3.0
S3: RT20 / Batch70 / AI10, lambda=2.6
S4: RT33 / Batch33 / AI34, lambda=2.6
S5: RT60 / Batch30 / AI10, lambda=3.0
```

推荐方法矩阵：

```text
fixed_tuned
original pressure-only CBO
prev-unfinished + prediction guard CBO
```

推荐 seeds：

```text
43
44
45
```

如果时间有限，可以先跑 seed 43 做主趋势，再补 seed 44 / 45 做稳定性验证。

## 6. 推荐统计指标

每个场景、每个方法都建议统计：

```text
mean
first100
tail100
last50
rolling50_min
rolling50_min_iter
rebound_pct
```

其中：

- `mean` 表示全程平均；
- `first100` 表示 cold-start 前期表现；
- `tail100` 表示后期稳定性；
- `last50` 表示最终收敛阶段；
- `rolling50_min` 表示最好窗口潜力；
- `rebound_pct` 表示从最好窗口到最终窗口的反弹程度。

对于 CBO / guard 方法，还建议统计：

```text
deploy_source
prediction_guard_active_triggered
prediction_guard_should_trigger
prediction_guard_reason
prediction_guard_active_reason
prediction_error_valid
prediction_guard_recent_bias
prediction_guard_recent_mae
prediction_guard_underestimate_rate
```

这些字段用于判断 guard 是真正因为 prediction error 触发，还是因为预测无效、history 不足或保守 fallback 触发。

## 7. 推荐论文分析逻辑

当前结果适合写成三层结论。

### 第一层：fixed_tuned 是强基线

`fixed_tuned` 是人工调好的固定策略，在部分场景中仍然非常强。因此论文中不要简单声称 CBO 在所有场景都超过 fixed_tuned。

更稳妥的说法是：

```text
The fixed tuned policy is a strong manually calibrated baseline. The proposed contextual BO method aims to improve adaptability and robustness across heterogeneous scenarios rather than only overfitting to a single fixed workload.
```

### 第二层：原始 pressure-only CBO 存在退化

原始 `reduced6_cbo_lite_pressure_only` 在高负载或任务结构变化场景中可能出现明显退化，尤其是后期漂移、服务质量下降、backlog 增长或 delay 增大。

这部分可以作为新方法的 motivation。

### 第三层：prev-unfinished context + prediction guard 显著修复退化

新方法通过引入上一窗口 unfinished / backlog 状态，以及 prediction-aware deployment guard，在多个场景下缓解了原始 CBO 的退化。

尤其在 λ=3.0 高负载场景中，如果新方法相对 fixed_tuned 仍略弱，也可以强调：

```text
Although the guarded CBO has not fully surpassed the fixed tuned baseline under the heaviest RT-dominant workload, it substantially reduces the degradation observed in the original pressure-only CBO.
```

推荐中文表述：

```text
在高负载 RT-heavy 场景中，原始 pressure-only CBO 相对 fixed_tuned 存在明显退化。引入 prev-unfinished context 与 prediction-aware deployment guard 后，该退化显著缓解。虽然新方法在 λ=3.0 场景下仍略弱于 fixed_tuned，但相对原始 CBO 已经大幅缩小差距，说明该机制有效提升了 CBO 在高压力场景下的部署稳定性。
```

## 8. 关于 short export / shortexport 压缩包

`offline_export.py` 中包含 short-name export helper，用于把结果文件复制到 `_short_export` 目录，并生成英文短文件名映射，避免 Windows 长路径、中文文件名或网盘下载时出错。

这不是主实验必需功能。

如果不想生成 `_short_export` 或压缩包，不要在命令行中使用：

```bash
--export-short-names
```

也请检查你的 PowerShell runner 脚本中是否包含该参数。

如果想从代码层面彻底禁用，可以在 `cli.py` 最后找到类似逻辑：

```python
if getattr(args, "export_short_names", False):
    export_root = args.output_root or SCENARIO_SAVE_DIR
    export_short_named_results(export_root)
```

然后删除或注释掉这几行。

更推荐的做法是：保留代码功能，但默认不使用该参数。这样以后遇到中文路径或长文件名导出问题时，还可以临时开启。

## 9. 输出文件说明

常见输出包括：

```text
*_round_summary_轮次汇总.csv
key_metric_summary_核心指标统计.csv
scenario_experiment_summary_实验汇总.csv
refactor_run_config.json
*_context_debug_情景调试.csv
*_alloc_debug_节点分配调试.csv
*_alloc_by_type_summary_任务类型节点分配汇总.csv
```

最重要的是：

```text
*_round_summary_轮次汇总.csv
```

里面包含每轮 Eval_Cost、Delay、Energy、Backlog、control vector、scheduler diagnostics、CBO diagnostics、guard diagnostics 等字段。

全场景分析时，优先读取所有 `round_summary` 文件，并排除 `_short_export` 目录。

推荐过滤逻辑：

```python
files = [
    p for p in root.rglob("*round_summary*csv")
    if "_short_export" not in str(p)
]
```

## 10. 开发建议

### 修改方法组

改 `scenario_experiments.py`。

适合新增：

- 新方法 key；
- 新 context mode；
- 新 baseline；
- 新 ablation；
- 场景 sweep；
- ratio grid；
- pressure scan。

### 修改 CBO 运行逻辑

改 `runtime_patches.py`。

适合新增：

- prediction guard；
- good-region guard；
- candidate/deployed theta 分离；
- trust-region 更新；
- source credibility；
- warm-start guard；
- federated CBO 聚合逻辑。

### 修改底层调度公式

改 `schedulers.py`。

适合新增：

- 新 score；
- latency-energy tradeoff；
- queue penalty；
- risk penalty；
- cloud gate；
- Boltzmann 随机性消融。

### 修改 BO surrogate / history

改 `agents.py`。

适合新增：

- GP / sparse GP / neural surrogate；
- history selection；
- warm-start；
- denoise；
- outlier filter；
- source weighting；
- federated aggregation。

### 修改输出字段

改 `diagnostics.py`。

如果实验已经运行，但 CSV 中看不到想要的字段，优先检查这里。

### 修改默认参数

改 `core_config.py`。

包括：

- BO 迭代数；
- CBO guard 默认值；
- scheduler 默认模式；
- 搜索边界；
- 任务比例；
- 负载参数；
- 节点配置。

## 11. 后续联邦学习升级方向

当前 warm-start 如果共享 `bo_warm_history.csv`，本质是 cross-factory experience sharing，不是严格 privacy-preserving federated learning。

更严格的方向是：

1. 工厂本地保留原始任务日志、节点状态、窗口级轨迹和本地 BO history；
2. 每个工厂只在本地训练 surrogate；
3. 服务端不接收 `theta-context-cost` 原始样本；
4. 服务端只接收模型更新、安全聚合统计量、差分隐私扰动后的摘要，或 surrogate 在公共 support set 上的预测均值和方差；
5. 新工厂使用 global prior / source prediction 辅助 cold-start；
6. target 根据早期 prediction error 动态校准 source credibility；
7. 部署层继续保留 prediction-error-aware guard，避免 source 经验造成负迁移。

对当前代码来说，最现实的升级路线是：

```text
cross-factory warm-start baseline
→ surrogate prediction summary sharing
→ source credibility calibration
→ prediction-error-aware deployment guard
→ federated CBO
```

## 12. 推荐提交前检查

提交代码前建议运行：

```powershell
python -m py_compile new_tr_split\*.py
python -m new_tr_split --help
```

快速实验建议：

```powershell
python -m new_tr_split `
  --mode scenario `
  --selected-keys reduced6_fixed_tuned,reduced6_cbo_lite_pressure_prev_unfinished `
  --bo-iterations 10 `
  --bo-interval 20 `
  --session-duration 200 `
  --fixed-rng `
  --fixed-seed 43 `
  --output-root results\quick_test
```

检查重点：

- 是否生成 round_summary；
- `Eval_Cost_最终评估Cost` 是否非空；
- control vector 是否正常；
- prediction guard 字段是否存在；
- `BO_Training_Cost` 是否与当前设定一致；
- 没有 NaN / inf；
- 没有误生成 `_short_export` 或压缩包。

## 13. 当前推荐主线

当前建议优先完成：

1. 全场景分析；
2. fixed_tuned vs 原始 pressure-only CBO vs 新 guarded CBO；
3. seed 43/44/45 多随机种子验证；
4. λ=3.0 高负载场景单独分析；
5. guard 触发原因与服务指标分解；
6. 论文中将当前方法定位为 robust cross-scenario CBO，而不是严格 federated learning；
7. 后续再扩展为 surrogate-output-based federated CBO。

当前最稳妥的论文结论是：

```text
The proposed guarded contextual BO does not merely optimize a single fixed workload. Instead, it improves the robustness of CBO under heterogeneous workload structures and mitigates the severe degradation observed in the original pressure-only CBO, especially under high-load scenarios. The fixed tuned policy remains a strong baseline, but the guarded CBO substantially narrows the performance gap while preserving adaptability across scenarios.
```
