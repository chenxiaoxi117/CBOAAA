# CBO 自适应探索修正与 V13 实验指南 V1.4

## 1. V12 结论

V12 自适应模式成功消除了固定 soft 的后期恶化，但没有保留前期加速。诊断显示前 50 轮平均：

```text
adaptive_beta = 0.349
adaptive_eta = 0.029
adaptive_safety_factor = 0.185
```

固定 soft 为 `beta=3.0, eta=0.25`。原因是旧安全门把稳定的 unfinished rate 和正常高利用率直接当成危险，导致探索几乎关闭。

## 2. V1.4 修正

### 2.1 动态压力风险

不再使用四项风险最大值。只有 backlog 或 unfinished trend 上升时，unfinished 和 max utilization 才参与抑制：

```text
dynamic_pressure = max(backlog_risk, unfinished_trend_risk)

risk = max(
    backlog_risk,
    min(unfinished_risk, dynamic_pressure),
    min(max_util_risk, dynamic_pressure)
)
```

因此，高利用率但 backlog 稳定时仍允许探索；真正出现积压或未完成趋势增长时会自动收紧。

### 2.2 数据不足探索

```text
data_need_linear = clip(1 - effective_samples / 80, 0, 1)
data_need = sqrt(data_need_linear)
```

这里依据的是当前外部场景筛选后真正进入 GP 的有效样本数，不是固定迭代轮数。动态场景、过滤后样本不足或返回旧场景时，行为可能不同。

### 2.3 后期再探索

```text
reexplore_need = 0.25 * stagnation * uncertainty_need
demand_target = max(data_need, reexplore_need) * safety_factor
```

样本成熟后，停滞触发的探索只有原强度的四分之一，避免重新变成固定 soft。

### 2.4 候选合理性门槛

```text
plausible_margin = 2.0 * residual_RMSE
```

相比 V12 的 `1×RMSE`，允许更多预测上仍可能改善的候选获得不确定性奖励，同时继续排除明显较差的纯高 sigma 候选。

## 3. 默认 V1.4 参数

```bash
--cbo-sigma-calibration on
--cbo-sigma-calibration-use-in-acq adaptive
--cbo-adaptive-exploration-beta-max 3.0
--cbo-adaptive-exploration-eta-max 0.25
--cbo-adaptive-exploration-window 30
--cbo-adaptive-exploration-sample-target 80
--cbo-adaptive-exploration-smoothing 0.20
--cbo-adaptive-exploration-progress-pct 0.01
--cbo-adaptive-exploration-reexplore-gain 0.25
--cbo-adaptive-exploration-plausible-margin-mult 2.0
```

默认主实验模式仍是 `false`；只有显式传入 `adaptive` 才使用 V1.4。

## 4. 上传

上传并覆盖：

```text
D:\CBOv2\新的代码结构\去掉动态堆积\new_tr_split
    -> /home/ecs-user/CBO/new_tr_split
```

另外上传：

```text
D:\CBOv2\run_top5_v12_adaptive_exploration_s43.sh
D:\CBOv2\run_top5_v13_adaptive_exploration_s43.sh
D:\CBOv2\analyze_top5_sigma_abc.py
```

检查：

```bash
cd /home/ecs-user/CBO
source env.sh
python -m py_compile new_tr_split/*.py
python -m new_tr_split --help | grep -E 'reexplore-gain|plausible-margin-mult'
```

## 5. V13 运行

建议使用 tmux：

```bash
tmux new -s cbo_v13
cd /home/ecs-user/CBO
source env.sh
MAX_JOBS=3 bash run_top5_v13_adaptive_exploration_s43.sh 43
```

脱离：`Ctrl+B`，再按 `D`。

结果目录：

```text
/home/ecs-user/CBO/result/top5_v13_adaptive_exploration_s43
```

完成标准：`15` 个 pressure summary、`30` 个 round summary。

## 6. 分析

```bash
python analyze_top5_sigma_abc.py \
  result/top5_v13_adaptive_exploration_s43 \
  --variants V13-A_diag V13-B_soft V13-C_adaptive
```

目标不是要求 adaptive 全阶段等同 soft，而是同时满足：

1. `001-050` 和 `051-100` 明显优于 A，缩小与 fixed soft 的差距。
2. `401-500` 和 `451-500` 接近 A，不能重现 fixed soft 的约 4%-5% 恶化。
3. 前期 `adaptive_beta` 应接近 2-3；样本成熟后应明显下降。
4. backlog 或 unfinished trend 增长时，`adaptive_safety_factor` 应及时下降。

本地 15 轮烟雾实验中，V1.4 平均值为：

```text
adaptive_safety_factor = 0.980
adaptive_beta = 2.795
adaptive_eta = 0.233
adaptive_dynamic_risk = 0.020
```

这证明安全门不再错误关闭前期探索，但最终收益仍必须由 V13 完整实验验证。
