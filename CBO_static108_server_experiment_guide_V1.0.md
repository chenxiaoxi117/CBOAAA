# CBO 静态 108 场景服务器实验指南 V1.0

## 1. 实验目标

本轮不再先跑 top5，直接运行完整静态 108 场景，对比传统调度、固定参数、BO 和新版 CBO。

场景网格：

```text
lambda = 1.8, 2.6, 3.0
RT/Batch/AI 比例以 10% 为步长
三类比例均至少为 10%，总和为 100%
每个 lambda 有 36 个任务比例
总场景数 = 3 * 36 = 108
```

旧结果 `r7_107_main_s43` 实际只有 107 个场景，因为 `lambda=3.0, RT=10, Batch=10, AI=80` 缺失。本脚本会完整生成 108 个场景。

## 2. 对比方法

沿用旧静态扫描中的 6 个方法：

| 方法 key | 作用 |
|---|---|
| `direct_greedy_cost` | 直接贪心代价基线 |
| `direct_queue_aware_greedy` | 队列感知直接贪心基线 |
| `reduced7_fixed_mid` | Reduced7 固定中间参数 |
| `reduced7_fixed_tuned` | Reduced7 旧实验调优固定参数 |
| `reduced7_bo_greedy` | 新范围、fixed5 冷启动、recent=80 的 BO |
| `reduced7_cbo_lite_pressure_taskmix_counts` | 新范围、external gate、internal_pressure6、sigma 诊断校准的 CBO |

前 4 个 baseline 保持现有代码定义不变。只有 BO/CBO 使用新的搜索范围。CBO 保留后验校准 buffer，但默认 `use_in_acq=false`，采集使用 raw sigma。

## 3. 公平性设置

每个场景统一：

```text
BO iterations = 500
BO interval = 240
session duration = 120000
seed = 43
fixed RNG = on
BO recent window = 80
backlog growth penalty = 0
TR = off
```

BO/CBO 共用 Reduced7 新范围：

```text
Latency = [0.1, 7.0]
Queue = [0.0, 3.0]
Risk = [0.0, 8.0]
Cloud Gate = [0.01, 0.95]
Energy Scale = [0.25, 2.0]
```

CBO 使用前 5 轮建立该场景共享归一化基准。代码会把 CBO 自动调整到方法执行顺序第一位；这 5 轮包含在 500 轮预算中，不额外增加实验轮数。后续 BO 和其他需要归一化的方法复用同一基准。

## 4. 上传文件

上传：

```text
D:\CBOv2\新的代码结构\去掉动态堆积\new_tr_split
D:\CBOv2\run_static108_v11_sigma_calibrated_s43.sh
D:\CBOv2\CBO_static108_server_experiment_guide_V1.0.md
D:\CBOv2\CBO_environment_deployment_guide_V1.2.md
```

服务器放置为：

```text
/home/ecs-user/CBO/new_tr_split/
/home/ecs-user/CBO/run_static108_v11_sigma_calibrated_s43.sh
/home/ecs-user/CBO/CBO_static108_server_experiment_guide_V1.0.md
```

不要上传本地 `results`、图片和 `__pycache__`。

## 5. 服务器预检查

推荐 Python 3.11，不建议用 Python 3.14 跑正式实验。

```bash
cd /home/ecs-user/CBO
source .venv/bin/activate

python --version
python -m py_compile new_tr_split/*.py
python -m new_tr_split --help | grep -E 'reduced7-latency-weight|cbo-sigma-calibration'

grep -n "y_val = -float(cost)" new_tr_split/agents.py
grep -n "predicted_cost = -float(mu_np\[best_idx\])" new_tr_split/runtime_patches.py
grep -n "prediction_error = float(actual_cost) - predicted_cost" new_tr_split/runtime_patches.py
```

## 6. 启动完整 108 场景

先赋予执行权限：

```bash
cd /home/ecs-user/CBO
chmod +x run_static108_v11_sigma_calibrated_s43.sh
bash -n run_static108_v11_sigma_calibrated_s43.sh
```

建议首次并发 4 个场景：

```bash
tmux new -s cbo_static108
MAX_JOBS=4 bash run_static108_v11_sigma_calibrated_s43.sh 43
```

确认 CPU、内存稳定后，`ecs.g9i.4xlarge` 可改为：

```bash
MAX_JOBS=6 bash run_static108_v11_sigma_calibrated_s43.sh 43
```

脚本支持断点续跑。已存在且非空的 `pressure_scan_summary_all.csv` 会被跳过。不要同时启动两个脚本写入同一个输出目录。

默认主实验结果目录：

```text
/home/ecs-user/CBO/result/static108_v12_sigma_acq_false_eta0p25_s43
```

主实验明确使用 raw sigma 采集，同时保留 calibration buffer：

```bash
SIGMA_ACQ_MODE=false SIGMA_ETA=0.25 MAX_JOBS=4 \
  bash run_static108_v11_sigma_calibrated_s43.sh 43
```

soft 消融会自动写入另一个目录：

```bash
SIGMA_ACQ_MODE=soft SIGMA_ETA=0.25 MAX_JOBS=4 \
  bash run_static108_v11_sigma_calibrated_s43.sh 43
```

`SIGMA_ACQ_MODE=true` 仅用于复现完全校准的负向消融，不作为主实验。

## 7. 运行监控

```bash
htop
free -h
df -h
tail -f result/static108_v11_sigma_calibrated_s43/logs/*.stdout.log
```

退出 tmux：`Ctrl-b d`

恢复：

```bash
tmux attach -t cbo_static108
```

## 8. 完整性标准

实验完成后应有：

```text
pressure_scan_summary_all.csv = 108
refactor_run_config.json = 108
round_summary CSV = 108 * 6 = 648
failed_jobs.txt = 空文件
```

检查：

```bash
OUT=/home/ecs-user/CBO/result/static108_v12_sigma_acq_false_eta0p25_s43
find "$OUT" -name pressure_scan_summary_all.csv -type f | wc -l
find "$OUT" -name refactor_run_config.json -type f | wc -l
find "$OUT" -name '*round_summary*.csv' -type f | wc -l
cat "$OUT/failed_jobs.txt"
```

## 9. 配置抽查

任选一个场景检查 `refactor_run_config.json`：

```text
selected_keys 包含全部 6 个方法
bo_iterations = 500
bo_recent_window = 80
cbo_shared_reference_policy = cbo_first
cbo_shared_reference_warmup_rounds = 5
cbo_reference_source_method_key = reduced7_cbo_lite_pressure_taskmix_counts
cbo_backlog_growth_penalty_weight = 0
sigma_calibration = on
sigma_calibration_buffer_size = 50
sigma_calibration_use_in_acq = false
sigma_calibration_eta = 0.25
sigma_floor = 0.03
reduced7_effective_bounds = 新范围
```

CBO round summary 还应满足：

```text
context_mode = internal_pressure6
model_input_dim = 13（运行日志中检查）
calibration_buffer_size 最终达到 50
sigma_acq = raw_sigma
predicted_cost = -selected_candidate_mu
prediction_error = actual_cost - predicted_cost
```

## 10. 后续分析口径

优先比较实际指标，而不是只看 normalized score：

```text
Eval_Cost
Avg_Delay
Violation_Rate
unfinished_rate
Backlog
energy_per_arrival
rolling50 后期均值
收敛速度与后期稳定性
```

108 场景先按 lambda、任务主导类型和压力等级分组，再统计 CBO 相对 BO 的胜率、均值差、中位数差和最差场景。单个 seed 用于主扫描，稳健性结论仍需后续增加种子。
