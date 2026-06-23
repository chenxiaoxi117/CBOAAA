# CBO 自适应探索机制与实验指南 V1.3

## 1. 修改目标

V11 证明固定 soft 校准能让 CBO 前 100 轮更快，但会让 401-500 轮继续探索并恶化约 4%-5%。V1.3 不再规定固定探索轮数，而是根据数据量、优化进展、模型不确定性和运行压力闭环调整探索强度。

本次只修改 ask 阶段的候选评分。GP 输入仍为 `theta + internal_pressure6`，外部门控、训练样本筛选、reference、tell 和 calibration buffer 均保持不变。默认模式仍是 `false`。

## 2. 自适应计算

每个外部场景独立保存控制器状态。场景键使用配置中的到达强度和任务比例，不使用带泊松波动的单窗口实测到达率，因此同一静态场景不会因随机到达波动被拆开：

```text
data_need = max(0, 1 - effective_GP_samples / sample_target)
stagnation = 1 - positive_recent_improvement / progress_threshold
uncertainty_need = (sigma_scale - 1) / (sigma_scale_default - 1)
safety_factor = 1 - max(backlog_risk, unfinished_risk, trend_risk, utilization_risk)

demand_target = max(data_need, stagnation * uncertainty_need) * safety_factor
demand_t = (1-smoothing) * demand_(t-1) + smoothing * demand_target
beta_t = beta_max * demand_t
eta_t = eta_max * demand_t
```

采集评分为：

```text
sigma_acq = raw_sigma + eta_t * (sigma_calibrated - raw_sigma)
score = predicted_reward + beta_t * sigma_acq
```

当 calibration buffer 至少有 10 条记录后，只给预测收益距离贪心最优点不超过 residual RMSE 的候选增加不确定性奖励，避免选择“预测已经很差、只是 sigma 很大”的点。

## 3. 默认参数

```bash
--cbo-sigma-calibration on
--cbo-sigma-calibration-use-in-acq adaptive
--cbo-adaptive-exploration-beta-max 3.0
--cbo-adaptive-exploration-eta-max 0.25
--cbo-adaptive-exploration-window 30
--cbo-adaptive-exploration-sample-target 40
--cbo-adaptive-exploration-smoothing 0.20
--cbo-adaptive-exploration-progress-pct 0.01
--cbo-adaptive-exploration-backlog-ref 1.0
--cbo-adaptive-exploration-unfinished-ref 0.10
--cbo-adaptive-exploration-trend-ref 0.05
--cbo-adaptive-exploration-max-util-start 0.80
```

这些是 V12 待验证参数，不应在验证前替换 `false` 主配置。

## 4. 上传文件

上传并覆盖整个本地目录：

```text
D:\CBOv2\新的代码结构\去掉动态堆积\new_tr_split
```

到服务器：

```text
/home/ecs-user/CBO/new_tr_split
```

另外上传：

```text
D:\CBOv2\run_top5_v12_adaptive_exploration_s43.sh
D:\CBOv2\analyze_top5_sigma_abc.py
```

## 5. 服务器检查

```bash
cd /home/ecs-user/CBO
source .venv/bin/activate
python -m py_compile new_tr_split/*.py
python -m new_tr_split --help | grep -A1 cbo-adaptive-exploration-beta-max
```

## 6. V12 Top5 三组实验

三组保持 BO、CBO、场景、seed、范围和 500 轮配置一致：

```text
V12-A_diag: calibration=on, acquisition=false
V12-B_soft: calibration=on, acquisition=soft, eta=0.25
V12-C_adaptive: calibration=on, acquisition=adaptive
```

运行：

```bash
cd /home/ecs-user/CBO
source .venv/bin/activate
MAX_JOBS=3 bash run_top5_v12_adaptive_exploration_s43.sh 43
```

完成标准：

```text
pressure summaries = 15 / 15
refactor configs   = 15 / 15
round summaries    = 30 / 30
```

分析：

```bash
python analyze_top5_sigma_abc.py \
  result/top5_v12_adaptive_exploration_s43 \
  --variants V12-A_diag V12-B_soft V12-C_adaptive
```

## 7. 重点判断标准

比较 `001-050`、`051-100`、`101-200`、`201-300`、`301-400`、`401-500` 和 `451-500`：

1. C 组前 100 轮是否接近或优于固定 soft。
2. C 组 401-500 是否回到 A 组附近，不再出现固定 soft 的后期恶化。
3. `adaptive_beta`、`adaptive_eta` 是否随有效样本增加而下降，并能在停滞且低压时重新升高。
4. backlog、unfinished 或 max utilization 较高时，`adaptive_safety_factor` 是否下降。
5. `adaptive_plausible_fraction` 是否在校准成熟后低于 1，阻止纯高 sigma 远点。

如果 C 组只改善前期且后期仍差，先降低 `beta_max` 到 2.0；不要先修改窗口或写死探索轮数。如果 C 组几乎等同 A 组，再检查安全门是否长期接近 0，或 residual RMSE 门槛是否过窄。

## 8. 后续实验顺序

Top5 单 seed 通过后，再按以下顺序扩大：

1. Top5 多 seed：43、44、45。
2. 静态 108 场景：A 与 C 主对比，B 作为固定探索对照。
3. 动态阶段切换：重点统计切换后 1-50 轮恢复速度和旧场景复用效果。
4. 消融：依次关闭 safety factor、stagnation 和 RMSE candidate gate，确认收益来自哪一部分。
