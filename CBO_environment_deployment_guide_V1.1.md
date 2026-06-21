# CBO 服务器环境部署指南 V1.1

## 1. 当前适用版本

本指南对应当前 CBO 主线：

```text
外部情景：任务到达强度 + RT/Batch/AI 任务比例，用于历史样本门控
内部情景：internal_pressure6，作为 CBO GP 的上下文输入
BO：recent history
CBO：external gate + recent_confidence + internal_pressure6
```

主方法 key：

```text
reduced7_bo_greedy
reduced7_cbo_lite_pressure_taskmix_counts
```

V8/V9 命令保留为历史复现实验。当前静态验证统一命名为 V10。

## 2. 推荐服务器

正式实验推荐：

```text
实例：ecs.g9i.4xlarge 或同等级通用/计算型实例
CPU：16 vCPU，高主频 x86_64
内存：64 GB
磁盘：200 GB ESSD/SSD
系统：Ubuntu 22.04 LTS
Python：3.10 或 3.11
GPU：不需要
Swap：8～16 GB
```

选择建议：

- `ecs.g9i.4xlarge`（16C/64G）：推荐，用于 top5 多 seed、窗口消融和 108 场景并行扫描。
- `ecs.c9i.4xlarge`（16C/32G）：仅在明显更便宜且主要顺序运行时选择。
- 不推荐突发性能实例；不需要 GPU 实例。

当前 BoTorch/GPyTorch 使用 CPU 精确 GP。单个实验进程主要使用一个 CPU 核，因此核心数只有在启动多个独立场景/seed 进程时才会发挥作用。

推荐并发：

```text
首次运行：4 个并发进程
稳定后：6～8 个并发进程
确认内存和负载稳定后：最多尝试 10～12 个
```

不要直接按 16 vCPU 启动 16 个进程。运行时使用 `htop`、`free -h` 和 `df -h` 观察 CPU、内存和磁盘。

## 3. 上传内容

Windows 本地核心目录：

```text
D:\CBOv2\新的代码结构\去掉动态堆积\new_tr_split
```

Linux 推荐放置为：

```text
/home/ecs-user/CBO/new_tr_split
```

同时上传：

```text
CBO_internal_external_context_modification_V1.1.md
CBO_environment_deployment_guide_V1.1.md
```

不需要上传：

```text
results/
图片/
__pycache__/
*.bak_*
旧 tar.gz 备份
```

运行时必须位于包的上一级目录：

```bash
cd /home/ecs-user/CBO
python -m new_tr_split --help
```

## 4. Python 环境

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip tmux htop

cd /home/ecs-user/CBO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn torch gpytorch botorch openpyxl seaborn tqdm
```

安装后检查：

```bash
python -c "import torch, gpytorch, botorch, numpy, pandas; print(torch.__version__)"
python -m new_tr_split --help
python -m py_compile new_tr_split/*.py
```

## 5. 线程环境变量

每个实验进程都设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
```

这样可以避免每个 Python 进程内部再启动大量 BLAS 线程，导致多进程实验互相争抢 CPU。

## 6. 部署后 Smoke Test

```bash
python -m new_tr_split \
  --mode pressure_scan \
  --lambda-values 1.8 \
  --task-probs 30,40,30 \
  --selected-keys reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-iterations 10 \
  --bo-interval 240 \
  --session-duration 2400 \
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
  --output-root /home/ecs-user/CBO/result/smoke_v10_internal6_externalgate_s43
```

## 7. 当前推荐正式实验

先按下面顺序运行：

1. V10 top5 静态场景，seed=43，BO recent=80。
2. BO recent window 消融：120、160、240；CBO 配置保持不变。
3. V10 top5 多 seed：40、41、42、43、44。
4. top5 稳健后再跑 108 静态场景。
5. 静态结论稳定后再跑动态场景和联邦扩展。

当前本地脚本：

```text
run_window240_nogrowth_top5_bo_cbo_v10_internal6_externalgate.ps1
run_window240_nogrowth_top5_bo_cbo_v10_bo_window_sweep.ps1
```

服务器脚本应将每个场景/seed 拆成独立进程。16C/64G 实例初期最多并行 6～8 个，不要在同一个输出目录重复启动相同任务。

## 8. V10 必查配置

每个结果目录都要检查 `refactor_run_config.json`：

```text
selected_keys
bo_history_mode
bo_recent_window
cbo_reference_mode
cbo_shared_reference_policy
cbo_shared_reference_warmup_rounds
cbo_reference_source_method_key
cbo_backlog_growth_penalty_weight
reduced7_energy_scale_bounds
```

同时在 CBO round summary 中确认：

```text
context_mode = internal_pressure6
context_feature_names = [
  start_backlog_norm,
  start_queue_total_norm,
  start_avg_util,
  start_max_util,
  prev_unfinished_rate,
  unfinished_rate_trend
]
external_gate_mode = taskmix_intensity
```

外部门控诊断重点看：

```text
external_gate_raw_count
external_gate_passed_count
external_gate_selected_count
external_gate_fallback_used
external_similarity_mean
```

## 9. 结果完整性检查

top5 单 seed BO+CBO 应有：

```text
5 个 pressure_scan_summary_all.csv
5 个 refactor_run_config.json
10 个 round_summary CSV
```

检查命令：

```bash
find result/<experiment_name> -name "*round_summary*.csv" | wc -l
find result/<experiment_name> -name "pressure_scan_summary_all.csv" | wc -l
find result/<experiment_name> -name "refactor_run_config.json" | wc -l
```

窗口消融如果测试 120、160、240，完整结果应有：

```text
15 个 pressure_scan_summary_all.csv
15 个 refactor_run_config.json
30 个 round_summary CSV
```

## 10. 后台运行

推荐使用 `tmux`：

```bash
tmux new -s cbo_v10
bash run_v10.sh
```

断开：`Ctrl-b d`

恢复：

```bash
tmux attach -t cbo_v10
```

每次正式实验保存：运行命令、代码版本、seed、开始/结束时间、stdout、stderr、`refactor_run_config.json` 和结果摘要。
