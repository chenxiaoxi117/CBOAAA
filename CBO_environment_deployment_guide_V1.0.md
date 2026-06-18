# CBO 环境部署指南 V1.0

## 1. 适用目标

本文档用于在本地或服务器上部署 CBO 实验环境，并保证后续静态扫描、动态实验、消融实验和多 seed 实验可以稳定运行。

推荐代码根目录：

```text
D:\CBOv2\新的代码结构\去掉动态堆积
```

Linux 服务器上建议对应放置为：

```text
/home/ecs-user/CBO
```

核心 Python 包目录：

```text
new_tr_split
```

运行方式：

```bash
python -m new_tr_split
```

## 2. Python 环境

推荐：

```text
Python 3.10 或 3.11
```

建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. 依赖安装

如果项目已有 `requirements.txt`：

```bash
pip install -r requirements.txt
```

如果没有完整 requirements，至少需要确认以下依赖存在：

```bash
pip install numpy pandas matplotlib scipy scikit-learn torch gpytorch botorch
```

可选但建议安装：

```bash
pip install openpyxl seaborn tqdm
```

注意：

- `torch / gpytorch / botorch` 版本需要兼容。
- 如果服务器没有 GPU，安装 CPU 版 torch 即可。
- 本项目实验主要是 CPU 计算，建议限制线程数，避免多进程/多 seed 时抢占资源。

## 4. 运行前环境变量

Linux 推荐每次实验前设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
```

含义：

- `OMP_NUM_THREADS=1`：限制 OpenMP 线程。
- `MKL_NUM_THREADS=1`：限制 MKL 线程。
- `OPENBLAS_NUM_THREADS=1`：限制 OpenBLAS 线程。
- `NUMEXPR_NUM_THREADS=1`：限制 numexpr 线程。
- `MPLBACKEND=Agg`：服务器无图形界面时正常保存图片。

Windows PowerShell：

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
$env:MPLBACKEND="Agg"
```

## 5. 代码完整性检查

进入项目根目录：

```bash
cd /home/ecs-user/CBO
```

或 Windows：

```powershell
cd /d D:\CBOv2\新的代码结构\去掉动态堆积
```

检查包是否可导入：

```bash
python -m new_tr_split --help
```

编译检查：

```bash
python -m py_compile new_tr_split/*.py
```

Windows PowerShell：

```powershell
Get-ChildItem .\new_tr_split -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
```

## 6. 快速 smoke test

建议先跑一个很小的测试，确认代码、依赖、输出目录都正常。

Linux：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

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
  --cbo-objective-mode normalized_tradeoff \
  --cbo-reference-mode calibrate \
  --cbo-shared-reference-policy cbo_first \
  --cbo-shared-reference-warmup-rounds 5 \
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
  --cbo-backlog-growth-penalty-weight 0 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --output-root /home/ecs-user/CBO/result/smoke_test
```

预期：

```text
scenario_experiment_summary_实验汇总.csv
reduced7_bo_greedy_round_summary_轮次汇总.csv
reduced7_cbo_lite_pressure_taskmix_counts_round_summary_轮次汇总.csv
refactor_run_config.json
```

## 7. 推荐目录规范

Linux：

```text
/home/ecs-user/CBO
/home/ecs-user/CBO/result
/home/ecs-user/CBO/result/logs
```

Windows：

```text
D:\CBOv2
D:\CBOv2\results
D:\CBOv2\results\<experiment_name>
```

推荐实验命名：

```text
r7_108_bo_cbo_v8_recent80_s43
r7_108_bo_cbo_v9_statekernel_s43
r7_dynamic_statekernel_s43
window240_nogrowth_top5_static_bo_cbo_v8_fixed5_multiseed
```

命名建议包含：

- 控制维度：`r7`
- 场景类型：`108` / `top5` / `dynamic`
- 方法：`bo_cbo` / `full_baseline`
- 历史口径：`recent80` / `statekernel`
- seed：`s43` / `multiseed`

## 8. 正式实验运行前检查清单

每次正式实验前检查：

```text
1. 是否进入正确代码目录
2. 是否激活虚拟环境
3. 是否设置线程环境变量
4. selected-keys 是否正确
5. energy bounds 是否统一为 0.5,3.0
6. reference 是否使用 cbo_first warmup=5
7. backlog growth penalty 是否为 0
8. history 口径是否符合当前实验目的
9. output-root 是否是新目录
10. seed 是否记录清楚
```

## 9. 常用正式命令模板

### 9.1 V8 recent80 BO+CBO

```bash
python -m new_tr_split \
  --mode pressure_scan \
  --lambda-values 1.8,2.2,2.6,3.0 \
  --selected-keys reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-iterations 500 \
  --bo-interval 240 \
  --session-duration 120000 \
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
  --output-root /home/ecs-user/CBO/result/r7_108_bo_cbo_v8_recent80_s43
```

### 9.2 V9 all-history + state-kernel

```bash
python -m new_tr_split \
  --mode pressure_scan \
  --lambda-values 1.8,2.2,2.6,3.0 \
  --selected-keys reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-iterations 500 \
  --bo-interval 240 \
  --session-duration 120000 \
  --fixed-rng \
  --fixed-seed 43 \
  --reduced7-energy-scale-bounds 0.5,3.0 \
  --feedback-score task_effective_backlog_violation \
  --bo-history-mode all \
  --bo-recent-window 80 \
  --cbo-history-select-mode state_gated_kernel \
  --cbo-state-kernel-topk 100 \
  --cbo-state-kernel-min-rows 20 \
  --cbo-state-kernel-recent-keep 20 \
  --cbo-state-kernel-threshold 0.05 \
  --cbo-state-kernel-fallback recent_context \
  --cbo-state-kernel-rate-gain 1.0 \
  --cbo-state-kernel-rate-power 1.0 \
  --cbo-state-kernel-max-rate-dist 3.0 \
  --cbo-objective-mode normalized_tradeoff \
  --cbo-reference-mode calibrate \
  --cbo-shared-reference-policy cbo_first \
  --cbo-shared-reference-warmup-rounds 5 \
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
  --cbo-backlog-growth-penalty-weight 0 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --output-root /home/ecs-user/CBO/result/r7_108_bo_cbo_v9_statekernel_s43
```

### 9.3 动态实验

```bash
python -m new_tr_split \
  --mode dynamic_scenario \
  --dynamic-schedule "1.8:30,40,30:100;2.6:60,30,10:100;2.6:10,20,70:100;1.8:30,40,30:100;2.6:50,30,20:100;3.0:60,30,10:100;2.6:10,20,70:100;1.8:40,30,30:100" \
  --dynamic-history-mode state_gated_kernel \
  --dynamic-history-window 80 \
  --dynamic-context-topk 50 \
  --selected-keys reduced7_fixed_tuned,reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts \
  --bo-interval 240 \
  --fixed-rng \
  --fixed-seed 43 \
  --reduced7-energy-scale-bounds 0.5,3.0 \
  --feedback-score task_effective_backlog_violation \
  --cbo-objective-mode normalized_tradeoff \
  --cbo-reference-mode calibrate \
  --cbo-shared-reference-policy cbo_first \
  --cbo-shared-reference-warmup-rounds 5 \
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts \
  --cbo-backlog-growth-penalty-weight 0 \
  --scheduler-score-norm-mode candidate_minmax_deadline \
  --task-adaptation \
  --output-root /home/ecs-user/CBO/result/r7_dynamic_statekernel_s43
```

## 10. 后台运行建议

Linux 简单后台：

```bash
nohup bash run_experiment.sh > run.log 2>&1 &
```

查看进程：

```bash
ps aux | grep new_tr_split
```

查看日志：

```bash
tail -f run.log
```

如果使用 `tmux`：

```bash
tmux new -s cbo
bash run_experiment.sh
```

断开：

```text
Ctrl-b d
```

恢复：

```bash
tmux attach -t cbo
```

## 11. 结果完整性检查

静态实验检查：

```bash
find /home/ecs-user/CBO/result/<experiment_name> -name "*round_summary*.csv" | wc -l
find /home/ecs-user/CBO/result/<experiment_name> -name "pressure_scan_summary_all.csv" | wc -l
find /home/ecs-user/CBO/result/<experiment_name> -name "refactor_run_config.json" | wc -l
```

动态实验检查：

```bash
ls /home/ecs-user/CBO/result/<experiment_name>/dynamic_run_config.json
ls /home/ecs-user/CBO/result/<experiment_name>/dynamic_round_summary.csv
ls /home/ecs-user/CBO/result/<experiment_name>/dynamic_phase_summary.csv
ls /home/ecs-user/CBO/result/<experiment_name>/dynamic_transition_summary.csv
```

Windows PowerShell：

```powershell
Get-ChildItem -Path "D:\CBOv2\results\<experiment_name>" -Recurse -Filter "*round_summary*.csv" | Measure-Object
Get-ChildItem -Path "D:\CBOv2\results\<experiment_name>" -Recurse -Filter "pressure_scan_summary_all.csv" | Measure-Object
```

## 12. 必查配置文件

每次实验结束后，先看：

```text
refactor_run_config.json
```

动态实验还要看：

```text
dynamic_run_config.json
```

重点字段：

```text
selected_keys
reduced7_energy_scale_bounds
cbo_objective_mode
cbo_reference_mode
cbo_shared_reference_policy
cbo_shared_reference_warmup_rounds
cbo_reference_source_method_key
cbo_backlog_growth_penalty_weight
bo_history_mode
bo_recent_window
cbo_history_select_mode
cbo_state_kernel_topk
cbo_state_kernel_recent_keep
dynamic_history_mode
dynamic_history_window
dynamic_context_topk
```

如果这些字段和实验目的不一致，不建议直接写结论。

## 13. 常见问题

### 13.1 图形界面报错

症状：

```text
cannot connect to display
```

处理：

```bash
export MPLBACKEND=Agg
```

### 13.2 CPU 占满或速度异常慢

处理：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### 13.3 结果口径混乱

先检查：

```text
refactor_run_config.json
dynamic_run_config.json
```

尤其确认：

```text
recent80 还是 state-kernel
旧 reference 还是 cbo_first
energy bounds 是 0.5,2.0 还是 0.5,3.0
backlog growth penalty 是否为 0
```

### 13.4 stderr 中有 Mean of empty slice

如果结果 CSV 完整，通常是绘图/统计警告，不一定代表实验失败。

仍需确认：

```text
round_summary 文件是否完整
summary 文件是否生成
最后日志是否显示 finished
```

### 13.5 `python -m new_tr_split` 找不到包

确认当前路径是项目根目录，而不是 `new_tr_split` 子目录内部。

正确：

```bash
cd /home/ecs-user/CBO
python -m new_tr_split
```

错误：

```bash
cd /home/ecs-user/CBO/new_tr_split
python -m new_tr_split
```

## 14. 版本记录建议

每次正式实验建议保存：

```text
run command
git commit hash
refactor_run_config.json
dynamic_run_config.json
stdout.log
stderr.log
analysis summary csv
```

建议在结果目录写一个：

```text
README_experiment.md
```

包含：

```text
实验目的
代码版本
运行命令
selected_keys
history 口径
reference 口径
seed
开始/结束时间
主要结论
```

## 15. 最小推荐流程

从零部署后，建议按这个顺序：

1. 创建虚拟环境。
2. 安装依赖。
3. `python -m new_tr_split --help`。
4. `python -m py_compile new_tr_split/*.py`。
5. 跑 10 轮 smoke test。
6. 跑 seed43 小规模正式实验。
7. 检查 config 和 summary。
8. 再跑 5 seed 或 108 场景。

