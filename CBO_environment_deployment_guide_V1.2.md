# CBO 服务器环境部署指南 V1.2

> 当前正式实验计划已调整为直接运行完整静态 108 场景，不再先跑 top5。实验方法、脚本和完整性检查以 `CBO_static108_server_experiment_guide_V1.0.md` 为准；本文中的 smoke test 仅用于部署排错，不是本轮正式实验前置要求。

## 1. 本版本对应代码

当前主方法：

```text
BO  = reduced7_bo_greedy
CBO = reduced7_cbo_lite_pressure_taskmix_counts
```

CBO 结构：

```text
外部门控：任务到达强度 + RT/Batch/AI 任务比例
GP context：internal_pressure6
GP 输入维度：theta(7) + context(6) = 13
后验校准：最近 50 条 (prediction_error, raw_sigma)
```

本版本不会修改 GP 输入维度，也没有新增风险模型。

## 2. 残差符号口径

代码内部 BO/CBO 以 reward 视角训练：

```text
y = -cost
mu_reward = GP posterior mean
predicted_cost = -mu_reward
prediction_error = actual_cost - predicted_cost
```

`sigma` 是标准差，不变号。

服务器上传后必须确认以下代码仍存在：

```bash
grep -n "y_val = -float(cost)" new_tr_split/agents.py
grep -n "predicted_cost = -float(mu_np\[best_idx\])" new_tr_split/runtime_patches.py
grep -n "prediction_error = float(actual_cost) - predicted_cost" new_tr_split/runtime_patches.py
```

三条均有输出才说明服务器代码与本地校准版一致。

## 3. 推荐服务器

正式实验推荐：

```text
实例：ecs.g9i.4xlarge
CPU：16 vCPU
内存：64 GB
系统：Ubuntu 22.04 LTS
Python：3.10 或 3.11
磁盘：100 GB ESSD/SSD
GPU：不需要
Swap：8-16 GB
```

单个 GP 实验进程主要使用一个 CPU 核。首次并行建议 4 个进程，稳定后增加到 6-8 个；不要直接启动 16 个进程。

## 4. 上传内容

本地代码目录：

```text
D:\CBOv2\新的代码结构\去掉动态堆积\new_tr_split
```

必须上传：

```text
new_tr_split/
CBO_V10_参数范围与后验不确定性校准修改说明_V1.1.md
CBO_environment_deployment_guide_V1.2.md
```

推荐服务器结构：

```text
/home/ecs-user/CBO/
  new_tr_split/
  CBO_V10_参数范围与后验不确定性校准修改说明_V1.1.md
  CBO_environment_deployment_guide_V1.2.md
  result/
  logs/
```

不要上传：

```text
results/
图片/
pic_core_v2/
pic_scenario_trust_region/
__pycache__/
*.pyc
*.bak_*
旧 tar.gz 备份
```

Windows 的 `.ps1` 脚本不能直接在 Linux 执行，只用于核对参数。

## 5. Python 环境

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip tmux htop

cd /home/ecs-user/CBO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn torch gpytorch botorch openpyxl seaborn tqdm
```

运行前设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
```

基础检查：

```bash
cd /home/ecs-user/CBO
source .venv/bin/activate
python -m py_compile new_tr_split/*.py
python -m new_tr_split --help | grep -E "reduced7-latency-weight|cbo-sigma-calibration"
```

## 6. 当前推荐参数范围

```text
Latency weights : 0.1,7.0
W_Queue         : 0.0,3.0
W_Risk_Scale    : 0.0,8.0
Cloud_Gate      : 0.01,0.95
W_Energy_Scale  : 0.25,2.0
```

后验校准参数：

```text
buffer_size = 50
min_samples = 10
default_scale = 4.0
scale_range = [1.0, 6.0]
sigma_floor = 0.03
```

## 7. 服务器冒烟测试

冒烟测试使用 18 轮。前 5 轮是固定初始点，之后需要至少 10 条有效 GP 预测才能观察历史校准开始生效；只跑 10 轮无法完整验证该机制。

```bash
cd /home/ecs-user/CBO
source .venv/bin/activate

python -m new_tr_split \
  --mode pressure_scan \
  --lambda-values 2.6 \
  --task-probs 70,20,10 \
  --selected-keys reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-iterations 18 \
  --bo-interval 240 \
  --session-duration 4320 \
  --fixed-rng \
  --fixed-seed 43 \
  --reduced7-latency-weight-bounds 0.1,7.0 \
  --reduced7-queue-weight-bounds 0.0,3.0 \
  --reduced7-risk-scale-bounds 0.0,8.0 \
  --reduced7-cloud-gate-bounds 0.01,0.95 \
  --reduced7-energy-scale-bounds 0.25,2.0 \
  --feedback-score task_effective_backlog_violation \
  --bo-history-mode recent \
  --bo-recent-window 80 \
  --cbo-objective-mode normalized_tradeoff \
  --cbo-reference-mode calibrate \
  --cbo-shared-reference-policy cbo_first \
  --cbo-shared-reference-warmup-rounds 5 \
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
  --cbo-backlog-growth-penalty-weight 0 \
  --cbo-sigma-calibration on \
  --cbo-sigma-calibration-buffer-size 50 \
  --cbo-sigma-calibration-min-samples 10 \
  --cbo-sigma-calibration-use-in-acq false \
  --cbo-sigma-calibration-eta 0.25 \
  --cbo-sigma-scale-default 4.0 \
  --cbo-sigma-scale-min 1.0 \
  --cbo-sigma-scale-max 6.0 \
  --cbo-sigma-floor 0.03 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --output-root /home/ecs-user/CBO/result/smoke_v10_sigma_calibrated_s43
```

日志中必须出现：

```text
control_dim=7 context_dim=6 model_input_dim=13
```

## 8. 冒烟结果检查

检查配置：

```bash
find result/smoke_v10_sigma_calibrated_s43 -name refactor_run_config.json -print
grep -R '"sigma_calibration": "on"' result/smoke_v10_sigma_calibrated_s43
grep -R '"sigma_floor": 0.03' result/smoke_v10_sigma_calibrated_s43
```

round summary 应包含：

```text
predicted_cost
actual_cost
prediction_error
raw_sigma
sigma_scale
sigma_calibrated
sigma_acq
sigma_calibration_use_in_acq
sigma_calibration_eta
raw_surprise
calibrated_surprise
calibration_buffer_size
```

预期行为：

1. 固定初始点没有 GP 预测时，这些预测字段可以为空。
2. 前 10 条有效校准样本使用 `sigma_scale=4.0`。
3. buffer 达到 10 条后，下一轮开始出现历史估计和平滑后的 scale。
4. `prediction_error` 必须等于 `actual_cost-predicted_cost`。
5. 通常 `abs(calibrated_surprise)` 应小于 `abs(raw_surprise)`；若 scale 裁剪到 1，则两者可能接近。

## 9. 正式实验顺序

1. 先运行校准版 top5，seed 43，BO recent=80。
2. 检查实际指标、校准字段和参数边界命中率。
3. 使用至少 3 个种子复验 top5。
4. 再运行完整 108 静态场景。
5. 静态结论稳定后再进行动态实验和联邦扩展。

本地对应脚本：

```text
run_window240_nogrowth_top5_bo_cbo_v10_internal6_externalgate.ps1
run_window240_nogrowth_top5_bo_cbo_v10_bo_window_sweep.ps1
```

这两个脚本已经使用独立的 `sigma_calibrated` 输出目录，不会覆盖旧 V10 结果。

## 10. 后台运行与监控

```bash
tmux new -s cbo_sigma_cal
# 在 tmux 中运行实验命令
```

断开：`Ctrl-b d`

恢复：

```bash
tmux attach -t cbo_sigma_cal
```

监控：

```bash
htop
free -h
df -h
```

每次正式实验必须保存运行命令、代码版本、seed、开始/结束时间、stdout、stderr、`refactor_run_config.json` 和结果摘要。
