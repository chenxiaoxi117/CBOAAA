# CBO Sigma A/B/C Top5 实验指南 V1.0

## 实验组

所有组使用相同 seed、相同 Reduced7 新范围、相同 fixed5、相同 recent=80、相同 external gate 和 internal_pressure6。

```text
V11-A_off  : calibration=off, use_in_acq=false
V11-B_diag : calibration=on,  use_in_acq=false
V11-C_soft : calibration=on,  use_in_acq=soft, eta=0.25
```

A/B 都保持 fixed-beta greedy-mean 选点，因此两组唯一有效差别是 B 是否维护 calibration buffer 和输出校准诊断。C 组才允许 soft sigma 改变采集。

每组同时运行 BO+CBO。BO 理论上应在三组中一致，可作为 CRN 和配置一致性的额外检查。

## 上传文件

```text
new_tr_split/
run_top5_v11_sigma_abc_s43.sh
analyze_top5_sigma_abc.py
```

服务器目录：

```text
/home/ecs-user/CBO/new_tr_split/
/home/ecs-user/CBO/run_top5_v11_sigma_abc_s43.sh
/home/ecs-user/CBO/analyze_top5_sigma_abc.py
```

## 启动

```bash
cd /home/ecs-user/CBO
source .venv/bin/activate
python -m py_compile new_tr_split/*.py analyze_top5_sigma_abc.py
bash -n run_top5_v11_sigma_abc_s43.sh
chmod +x run_top5_v11_sigma_abc_s43.sh

tmux new -s cbo_sigma_abc
MAX_JOBS=3 bash run_top5_v11_sigma_abc_s43.sh 43
```

默认输出：

```text
/home/ecs-user/CBO/result/top5_v11_sigma_abc_s43
```

完整结果应包含：

```text
15 个 pressure_scan_summary_all.csv
15 个 refactor_run_config.json
30 个 round_summary CSV
failed_jobs.txt 为空
```

## 分析

```bash
python analyze_top5_sigma_abc.py \
  /home/ecs-user/CBO/result/top5_v11_sigma_abc_s43
```

生成：

```text
analysis_sigma_abc/segment_metrics.csv
analysis_sigma_abc/uncertainty_metrics.csv
analysis_sigma_abc/pairwise_vs_A.csv
```

分段：

```text
001-050
051-100
101-200
201-300
301-400
401-500
451-500
all500
```

## 判断标准

首先检查 B 对 A：

```text
Eval_Cost 和 normalized score 应基本一致
后期性能不应系统性变差
calibration buffer 正常增长
calibrated surprise 和 2-sigma exceed rate 得到改善
```

然后检查 C 对 A/B：

```text
前 100/200 轮 cost 是否下降
451-500 是否不劣于 A/B
positive-step rebound 是否更小
high backlog/high unfinished 次数是否减少
sigma_acq/raw_sigma 的均值和 p95 是否长期超过 2
```

若 C 只改善 surprise 校准，却使实际 cost 变差，则 soft sigma 不进入主方法；calibration 只保留为诊断。`use_in_acq=true` 不参与本轮主实验。
