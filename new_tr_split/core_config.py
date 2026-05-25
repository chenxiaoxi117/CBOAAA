#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 1-819.
# Global imports, config helpers, ExperimentConfig, CFG initialization, control bounds.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
# v6.2 taskmix/runtime patch: task-mix context modes, short export, per-method runtime logging.
# ===============================================================
# 单文件实验版：Reduced6 + BO/EI + BO-greedy + CBO-greedy + CBO-TR-greedy + dual feedback
# 后续实验只需要运行本文件；不再需要 patch 文件或自动生成脚本。
# 说明：本文件已合并 SAFE BO + CBO GREEDY + DUAL FEEDBACK PATCH V3。
# ===============================================================

# ===============================================================
# 4D BO Anchor + Cohort 反馈测试版
# 修改点：recommended anchor、Cloud_Gate 收缩范围、tuned fixed 命名、绘图样式不重复
# 新增：任务批次(cohort)级延迟反馈，用任务完成归因替代窗口即时反馈
# ===============================================================
# 约束 Boltzmann + BO 扩展控制版
# 生成说明：在原“无任务适配 6 维权重版”基础上增加：
# 1) 可行性候选集 F：CPU / hard deadline / 云门控；
# 2) 机会集合 O：Opportunity_Rho 控制近优节点范围；
# 3) BO 扩展控制：W_Queue / W_Risk_Scale / Beta_Control / Opportunity_Rho / Cloud_Gate；
# 4) 所有新增机制均提供 CFG 开关，便于消融实验。
# ===============================================================

# ===============================================================
# Backup no-RoundRobin 500-window runtime defaults (generated 2026-05-21).
# Base: current server experiment_bo_refactor_v6_taskmix_runtime.py.
# Embedded runner defaults from current/run_500_bestcbo_direct36.sh:
# - no RoundRobin in default selected methods
# - BO_ITERATIONS=500, BO_INTERVAL=240, SESSION_DURATION=120000
# - feedback_score=task_effective_backlog_violation
# - BO history mode=recent, BO recent window=80
# - default output family: v6_500_bestcbo_direct36_norr
# ===============================================================

import os
import sys
import random
import time
import collections
import copy
import heapq
import warnings
import json
import re
import numpy as np
import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize, normalize, unnormalize

# ===============================================================
# Refactor v2 notes
# ---------------------------------------------------------------
# 目标：把主实验路径整理干净，而不是继续堆新算法。
# 1) 主线方法固定为 reduced6 fixed candidates + BO-EI + BO-greedy。
# 2) BO 默认冷启动：不再用 anchor_points 前几轮强制部署固定权重。
# 3) Eval_Cost（最终系统评价）与 BO_Training_Cost（BO tell 训练反馈）明确分离。
# 4) feedback 只保留少数清晰模式；dual / cohort 保留为 legacy diagnostic。
# 5) CBO / TR / 旧 anchor 方法继续保留，但默认不作为主实验方法。
# ===============================================================

# ===============================================================
# 代码阅读建议：
# 1) 先看 ExperimentConfig，理解所有全局参数在控制什么。
# 2) 再看 ConnectedFactory.run_continuous，这是单轮窗口推进的主流程。
# 3) 然后看 BoltzmannScheduler.select_node 和 FederatedBOAgent.ask，
#    前者决定任务落到哪个节点，后者决定下一轮 6 维权重怎么调。
# ===============================================================


def resolve_base_seed(seed: int, stream: int = 0) -> int:
    """生成可复现实验随机种子。

    seed:    实验主种子。
    stream:  给不同模块/工厂分配不同随机流，避免彼此串扰。
    """
    if CFG.USE_FIXED_RNG:
        return int(CFG.FIXED_RNG_SEED) + int(seed) * 1009 + int(stream) * 100003
    return random.SystemRandom().randrange(1, 2 ** 31 - 1)

def _get_node_site_from_cfg(node_cfg, default_site=0):
    """读取节点所属车间/云位置。普通车间用 0~4，厂区云/区域云用 5/6。"""
    return int(node_cfg.get("workshop", node_cfg.get("site", default_site)))


def _node_is_cloud(node_cfg):
    role = str(node_cfg.get("role", "")).lower()
    return bool(node_cfg.get("is_cloud", False)) or ("cloud" in role) or ("remote" in role)


def build_topology_matrix(node_count):
    """构造“车间 + 云节点”基础传输时延矩阵。

    设计思想：
    - 任务从车间边缘产生；同车间传输最快；
    - 相邻车间中等；跨车间更慢；
    - 厂区云/区域云算力强，但基础传输时延更高；
    - 节点之间不再按 id 单调排序，而是按所属车间/云位置生成拓扑。
    """
    # 如果配置里没有车间矩阵，则回退到旧的环形拓扑，保证兼容。
    if not hasattr(CFG, "WORKSHOP_BASE_DELAY") or not getattr(CFG, "USE_WORKSHOP_TOPOLOGY", True):
        matrix = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
        cluster_split = node_count // 2
        for i in range(node_count):
            for j in range(node_count):
                if i == j:
                    matrix[i][j] = 0.2
                    continue
                same_cluster = (i < cluster_split and j < cluster_split) or (i >= cluster_split and j >= cluster_split)
                ring_dist = min(abs(i - j), node_count - abs(i - j))
                if ring_dist == 1:
                    matrix[i][j] = 0.8
                elif same_cluster:
                    matrix[i][j] = 1.6 + 0.15 * ring_dist
                else:
                    matrix[i][j] = 3.0 + 0.25 * ring_dist
        return matrix

    site_delay = getattr(CFG, "WORKSHOP_BASE_DELAY")
    local_access = getattr(CFG, "NODE_ACCESS_DELAY", {})
    matrix = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for i in range(node_count):
        src_cfg = CFG.NODES_CFG[i]
        src_site = _get_node_site_from_cfg(src_cfg, default_site=i // 2)
        for j in range(node_count):
            dst_cfg = CFG.NODES_CFG[j]
            dst_site = _get_node_site_from_cfg(dst_cfg, default_site=j // 2)
            base = float(site_delay[src_site][dst_site])
            # 目标节点接入差异：近实时/低功耗边缘稍近，高性能/云节点稍远。
            role = str(dst_cfg.get("role", "normal_edge"))
            access = float(local_access.get(role, 0.0))
            if i == j:
                base = min(base, 0.12)
            matrix[i][j] = max(0.01, base + access)
    return matrix


def build_bandwidth_matrix(node_count):
    """构造节点间带宽矩阵。

    第一版只考虑“不同车间/云位置的带宽差异”，不默认做链路占用。
    如果后续开启 CFG.USE_LINK_QUEUE，会在 run_continuous 中让同一 origin-destination 链路串行排队。
    """
    if not hasattr(CFG, "WORKSHOP_BW") or not getattr(CFG, "USE_WORKSHOP_TOPOLOGY", True):
        return [[float(getattr(CFG, "LINK_BW", 50.0)) for _ in range(node_count)] for _ in range(node_count)]
    site_bw = getattr(CFG, "WORKSHOP_BW")
    matrix = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for i in range(node_count):
        src_site = _get_node_site_from_cfg(CFG.NODES_CFG[i], default_site=i // 2)
        for j in range(node_count):
            dst_site = _get_node_site_from_cfg(CFG.NODES_CFG[j], default_site=j // 2)
            matrix[i][j] = max(1e-6, float(site_bw[src_site][dst_site]))
    return matrix


def get_effective_speed(node_cfg, task_type):
    """返回节点对某类任务的有效速度。默认关闭任务类型适配：三类任务都只使用节点基础 speed。"""
    base_speed = float(node_cfg.get("speed", 1.0)) * 1e9
    if _node_is_cloud(node_cfg):
        base_speed *= float(getattr(CFG, "CLOUD_SPEED_MULT", 1.0))
    try:
        use_adaptation = bool(getattr(CFG, "USE_TASK_TYPE_ADAPTATION", False))
    except NameError:
        use_adaptation = False
    if not use_adaptation:
        return base_speed
    factors = node_cfg.get("type_speed_factor", {}) or {}
    return base_speed * float(factors.get(task_type, 1.0))


def get_transmission_delay(origin_node_idx, target_node_idx, data_size, include_local=True):
    """传输时延 = 本地上传固定时延 + 车间/云基础时延 + data_size / 链路带宽。"""
    origin_node_idx = int(origin_node_idx) if origin_node_idx is not None and origin_node_idx >= 0 else int(target_node_idx)
    target_node_idx = int(target_node_idx)
    base = float(CFG.TRANS_DELAY_MATRIX[origin_node_idx][target_node_idx])
    bw = float(CFG.TRANS_BW_MATRIX[origin_node_idx][target_node_idx]) if hasattr(CFG, "TRANS_BW_MATRIX") else float(CFG.LINK_BW)
    delay = base + float(data_size) / (bw + 1e-9)
    if _node_is_cloud(CFG.NODES_CFG[target_node_idx]):
        delay *= float(getattr(CFG, "CLOUD_DELAY_MULT", 1.0))
    if include_local:
        delay += float(getattr(CFG, "LOCAL_UPLOAD_DELAY", 0.0))
    return max(0.0, delay)


def get_transmission_energy(origin_node_idx, target_node_idx, data_size):
    """传输能耗：数据量 × 单位传输能耗 × 距离/链路类型系数。

    这是第一版轻量模型，不做链路功率积分；好处是简单、稳定、能体现跨车间/上云更耗能。
    """
    origin_node_idx = int(origin_node_idx) if origin_node_idx is not None and origin_node_idx >= 0 else int(target_node_idx)
    target_node_idx = int(target_node_idx)
    src_site = _get_node_site_from_cfg(CFG.NODES_CFG[origin_node_idx], default_site=origin_node_idx // 2)
    dst_site = _get_node_site_from_cfg(CFG.NODES_CFG[target_node_idx], default_site=target_node_idx // 2)
    factor_matrix = getattr(CFG, "WORKSHOP_TRANS_ENERGY_FACTOR", None)
    if factor_matrix is not None:
        factor = float(factor_matrix[src_site][dst_site])
    else:
        factor = 1.0 + 0.25 * float(CFG.TRANS_DELAY_MATRIX[origin_node_idx][target_node_idx])
    energy = float(getattr(CFG, "P_TRANS", 0.5)) * float(data_size) * factor
    if _node_is_cloud(CFG.NODES_CFG[target_node_idx]):
        energy *= float(getattr(CFG, "CLOUD_ENERGY_MULT", 1.0))
    return energy


def _normalize_task_probs(probs):
    """把任务类型概率整理为 RT / Batch / AI 三类，自动归一化。"""
    out = {"RT": 0.0, "Batch": 0.0, "AI": 0.0}
    if isinstance(probs, dict):
        for k, v in probs.items():
            kk = str(k).strip()
            if kk.lower() in {"rt", "real", "realtime", "real_time"}:
                out["RT"] = float(v)
            elif kk.lower() in {"batch", "bat"}:
                out["Batch"] = float(v)
            elif kk.lower() in {"ai", "ml"}:
                out["AI"] = float(v)
    elif isinstance(probs, (list, tuple, np.ndarray)) and len(probs) >= 3:
        out = {"RT": float(probs[0]), "Batch": float(probs[1]), "AI": float(probs[2])}
    total = sum(max(0.0, float(v)) for v in out.values())
    if total <= 0:
        return {"RT": 0.1, "Batch": 0.4, "AI": 0.5}
    return {k: max(0.0, float(v)) / total for k, v in out.items()}


def get_task_type_probs_at_time(current_time=None):
    """读取当前时间对应的任务类型比例。

    默认使用 CFG.TASK_TYPE_PROBS；若设置了 CFG.TASK_TYPE_PROB_SCHEDULE，
    则按 (start, end, probs) 分段使用不同任务比例。
    """
    schedule = getattr(CFG, "TASK_TYPE_PROB_SCHEDULE", None)
    if current_time is not None and schedule:
        t = float(current_time)
        for item in schedule:
            if len(item) != 3:
                continue
            start, end, probs = item
            if float(start) <= t < float(end):
                return _normalize_task_probs(probs)
    return _normalize_task_probs(getattr(CFG, "TASK_TYPE_PROBS", {"RT": 0.1, "Batch": 0.4, "AI": 0.5}))


def sample_task_type(rng, current_time=None):
    """按当前阶段的任务类型概率从 RT / Batch / AI 中采样。"""
    probs = get_task_type_probs_at_time(current_time)
    r = rng.random()
    cum = 0.0
    for t_type in TASK_TYPE_ORDER:
        cum += float(probs.get(t_type, 0.0))
        if r <= cum:
            return t_type
    return "Batch"

TASK_TYPE_ORDER = ["RT", "Batch", "AI"]


def normalize_theta_vector(theta, dim=None, fill=1.0):
    """把权重向量整理成固定维度。

    当前实验默认是 6 维：
    [RT时延, Batch时延, AI时延, RT能耗, Batch能耗, AI能耗]
    """
    dim = CFG.DIM_THETA if dim is None else int(dim)
    if theta is None:
        return [float(fill)] * dim
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().view(-1).tolist()
    elif not isinstance(theta, (list, tuple, np.ndarray)):
        theta = [theta]
    theta = [float(x) for x in theta]
    if len(theta) < dim:
        theta = theta + [float(fill)] * (dim - len(theta))
    return theta[:dim]


def split_task_weights(theta_full):
    """把 theta 前 6 维拆成“分任务类型的时延权重”和“能耗权重”。

    扩展 BO 维度不会改变前 6 维语义：
    [RT时延, Batch时延, AI时延, RT能耗, Batch能耗, AI能耗, ...]
    """
    dim = max(6, int(getattr(CFG, "DIM_THETA", 6)))
    theta = normalize_theta_vector(theta_full, dim=dim)
    latency_weights = {t: float(theta[i]) for i, t in enumerate(TASK_TYPE_ORDER)}
    energy_weights = {t: float(theta[i + 3]) for i, t in enumerate(TASK_TYPE_ORDER)}
    return latency_weights, energy_weights, theta


def theta_to_named_dict(theta):
    """把 theta 转成带名字的字典，便于导出 CSV 和看结果。"""
    theta = normalize_theta_vector(theta)
    return {CFG.FEATURE_NAMES[i]: theta[i] for i in range(min(CFG.DIM_THETA, len(theta)))}


def _extended_control_defaults_by_name():
    """扩展 BO 控制维度的默认值。"""
    return {
        "W_Queue": float(getattr(CFG, "QUEUE_WEIGHT_DEFAULT", 1.0)),
        "W_Risk_Scale": float(getattr(CFG, "RISK_SCALE_DEFAULT", 1.0)),
        "Beta_Control": float(getattr(CFG, "BETA_DEFAULT", getattr(CFG, "BETA_INITIAL", 3.0))),
        "Opportunity_Rho": float(getattr(CFG, "OPPORTUNITY_RHO_DEFAULT", 1.0)),
        "Cloud_Gate": float(getattr(CFG, "CLOUD_GATE_DEFAULT", 0.50)),
    }


def extend_control_point(base_theta):
    """把原始 6 维权重扩展到当前 CFG.DIM_THETA。"""
    vals = list(base_theta)
    if not hasattr(CFG, "FEATURE_NAMES"):
        return vals
    defaults = _extended_control_defaults_by_name()
    for name in list(CFG.FEATURE_NAMES)[len(vals):]:
        vals.append(float(defaults.get(name, 1.0)))
    return vals[:int(CFG.DIM_THETA)]


def default_control_vector(fill=1.5):
    """生成当前控制维度下的默认 theta。"""
    base = [float(fill)] * min(6, int(getattr(CFG, "DIM_THETA", 6)))
    return extend_control_point(base)


def default_scenario_anchor_points():
    """情景 BO 前几轮使用的引导点。开启扩展控制时会自动补齐后 5 维。"""
    base_points = [
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        [2.5, 1.2, 1.4, 0.8, 2.2, 1.8],
        [1.2, 1.4, 1.3, 2.6, 2.4, 2.2],
    ]
    return [extend_control_point(p) for p in base_points]

def infer_arrival_thresholds(lambda_schedule):
    """根据当前 LAMBDA_SCHEDULE 自动推断 arrival state 的 LOW/MID/HIGH 阈值。

    这样切换到不同到达率配置时，不需要手动同步修改 ARRIVAL_THRESHOLDS。
    对三段负载 [a, b, c]，返回 ((a+b)/2, (b+c)/2)。
    """
    lambdas = sorted({float(lam) for _, _, lam in lambda_schedule if float(lam) > 0})
    if len(lambdas) >= 3:
        return ((lambdas[0] + lambdas[1]) / 2.0, (lambdas[1] + lambdas[2]) / 2.0)
    if len(lambdas) == 2:
        low = (lambdas[0] + lambdas[1]) / 2.0
        high = lambdas[1] * 1.05
        return (low, high)
    if len(lambdas) == 1:
        return (0.8 * lambdas[0], 1.2 * lambdas[0])
    return (1.0, 2.0)

# ==========================================
# 1. 配置模块 (Config)
# ==========================================
class ExperimentConfig:
    """全局实验配置。

    你最需要先读懂这一段：这里定义了仿真时长、任务到达过程、
    节点能力、BO 搜索范围、情景信息、代价函数系数等。
    """
    SESSION_DURATION = 120000.0  # 备份版默认：500 * 240 秒；单次连续仿真的总时长（单位：秒）
    BO_ITERATIONS = 500  # 备份版默认：runner 中 ITERATIONS=500；BO 外层优化总轮数
    TASKS_PER_BATCH = 100  # 批模式下每批生成的任务数（当前主流程基本不用批模式）
    BO_INTERVAL = 240.0  # 备份版默认：runner 中 BO_INTERVAL=240；每轮 BO 对应的窗口长度
    SCENARIO_INTERVAL = 20.0  # 多久重新判定一次离散情景状态（LOW/MID/HIGH）
    SCENARIO_WINDOW = 60.0  # 统计情景指标时看的滑动时间窗长度
    SCENARIO_STABLE_K = 3  # 连续多少次判定到相同 state，才认为情景稳定

    # 分段泊松到达率：不同时间段任务到达强度不同，用来制造动态环境
    LAMBDA_SCHEDULE = [
        (0.0, 120000.0, 1.0)]
     #(1000.0, 2000.0, 1.8),
      ##(3000.0, 4000.0, 1.0),
       #(4000.0, 5000.0, 1.8),
       #(5000.0, 6000.0, 2.6),
       #(6000.0, 7000.0, 1.0),
       #(7000.0, 8000.0, 1.8),
       #(8000.0, 9000.0, 2.6),
       #(9000.0, 10000.0, 1.0),
       #(10000.0, 11000.0, 1.8),
       #(11000.0, 12000.0, 2.6)]
    BATCH_POISSON_LAMBDA = 1.5  # 批模式的泊松到达率
    TASK_TYPE_PROBS = {"RT": 0.1, "Batch": 0.40, "AI": 0.50}  # 三类任务的生成概率

    # 三类任务的属性：
    # data:  数据量
    # cpu:   所需 CPU 资源单位
    # dur:   基础执行时长（也决定 cpu_cycles）
    # deadline_factor: 截止期 = create_time + dur * deadline_factor
    TASK_PROPS = {
        "RT": {"data": 10.0, "cpu": 4, "dur": 3.0, "deadline_factor": 2.0},
        "Batch": {"data": 80.0, "cpu": 12, "dur": 15.0, "deadline_factor": 20.0},
        "AI": {"data": 160.0, "cpu": 20, "dur": 20.0, "deadline_factor": 15.0}
    }

    # 2. 增加延迟的惩罚权重：让 BO 不敢随便降频
    # 如果能量是 5000，延迟是 50，我们把 ALPHA_LATENCY 设为 100.0，
    # 这样迟到带来的惩罚 (5000分) 就和能耗 (5000分) 五五开了！
    ALPHA_LATENCY = 100.0  # 代价函数里“平均时延”的惩罚强度

    # 动态维度：每个节点 4 个比例（RT: CPU/BW，Batch: CPU/BW）
    # 在定义完 NODES_CFG 后计算

    # 节点配置：多车间边缘节点 + 两级云节点。
    # workshop/site: 0~4 表示五个车间；5 表示厂区云；6 表示区域云。
    # role: 用于生成接入时延修正、功耗策略和后续诊断。
    # 是否启用“任务类型适配”。
    # False：RT / Batch / AI 都只使用节点基础 speed，避免 AI 加速/RT 加速等人为专用系数。
    # True：启用 type_speed_factor，让部分节点对特定任务更快，作为异构加速器消融实验。
    USE_TASK_TYPE_ADAPTATION = False

    # type_speed_factor: 不同节点对 RT / Batch / AI 的适配差异；例如 AI 加速节点对 AI 更快。
    # 当前默认 USE_TASK_TYPE_ADAPTATION=False，因此这些系数不会生效；保留字段只是方便后续消融对比。
    # 注意：这里故意不做成“0号最强、9号最弱”的单调结构，而是让节点在算力、位置、功耗上有交叉优势。
    NODES_CFG = [
        {"id": 0, "workshop": 0, "role": "rt_edge",     "cpu": 56,  "speed": 3.7, "p_idle": 85,  "p_max": 380,
         "type_speed_factor": {"RT": 1.30, "Batch": 0.90, "AI": 0.65}},
        {"id": 1, "workshop": 0, "role": "low_power",   "cpu": 44,  "speed": 2.9, "p_idle": 45,  "p_max": 210,
         "type_speed_factor": {"RT": 1.00, "Batch": 0.85, "AI": 0.55}},

        {"id": 2, "workshop": 1, "role": "efficient",   "cpu": 72,  "speed": 4.4, "p_idle": 115, "p_max": 500,
         "type_speed_factor": {"RT": 1.00, "Batch": 1.20, "AI": 1.00}},
        {"id": 3, "workshop": 1, "role": "normal_edge", "cpu": 52,  "speed": 3.4, "p_idle": 70,  "p_max": 320,
         "type_speed_factor": {"RT": 1.05, "Batch": 1.00, "AI": 0.85}},

        {"id": 4, "workshop": 2, "role": "ai_accel",    "cpu": 88,  "speed": 5.0, "p_idle": 210, "p_max": 850,
         "type_speed_factor": {"RT": 0.90, "Batch": 1.00, "AI": 1.85}},
        {"id": 5, "workshop": 2, "role": "normal_edge", "cpu": 54,  "speed": 3.5, "p_idle": 78,  "p_max": 340,
         "type_speed_factor": {"RT": 1.00, "Batch": 1.00, "AI": 0.90}},

        {"id": 6, "workshop": 3, "role": "low_power",   "cpu": 40,  "speed": 2.6, "p_idle": 32,  "p_max": 165,
         "type_speed_factor": {"RT": 0.90, "Batch": 0.75, "AI": 0.50}},
        {"id": 7, "workshop": 3, "role": "batch_node",  "cpu": 68,  "speed": 3.9, "p_idle": 105, "p_max": 430,
         "type_speed_factor": {"RT": 0.85, "Batch": 1.45, "AI": 0.90}},

        {"id": 8, "workshop": 4, "role": "high_perf",   "cpu": 96,  "speed": 5.5, "p_idle": 235, "p_max": 940,
         "type_speed_factor": {"RT": 1.05, "Batch": 1.10, "AI": 1.20}},
        {"id": 9, "workshop": 4, "role": "backup_low",  "cpu": 36,  "speed": 2.4, "p_idle": 24,  "p_max": 140,
         "type_speed_factor": {"RT": 0.80, "Batch": 0.65, "AI": 0.45}},

        {"id": 10, "workshop": 5, "role": "factory_cloud", "is_cloud": True, "cpu": 128, "speed": 6.2, "p_idle": 320, "p_max": 1350,
         "type_speed_factor": {"RT": 0.95, "Batch": 1.35, "AI": 1.55}},
        {"id": 11, "workshop": 6, "role": "regional_cloud", "is_cloud": True, "cpu": 192, "speed": 7.0, "p_idle": 480, "p_max": 2100,
         "type_speed_factor": {"RT": 0.85, "Batch": 1.60, "AI": 2.10}},
    ]
    # ===========================================================
    # BO 控制向量设置
    # ===========================================================
    # False：保持原始 6 维控制向量；True：扩展为 11 维。
    USE_EXTENDED_BO_CONTROL = True
    BASE_FEATURE_NAMES = [
        "W_RT_Latency", "W_Batch_Latency", "W_AI_Latency",
        "W_RT_Energy", "W_Batch_Energy", "W_AI_Energy"
    ]
    EXTENDED_FEATURE_NAMES = [
        "W_Queue",          # 队列/拥塞压力权重
        "W_Risk_Scale",     # deadline risk 缩放系数
        "Beta_Control",     # Boltzmann 反温度，越大越贪心
        "Opportunity_Rho",  # 机会窗口，控制近优候选节点范围
        "Cloud_Gate"        # 云卸载门控阈值
    ]
    FEATURE_NAMES = BASE_FEATURE_NAMES + EXTENDED_FEATURE_NAMES if USE_EXTENDED_BO_CONTROL else BASE_FEATURE_NAMES
    DIM_THETA = len(FEATURE_NAMES)
    # deadline risk 作为“软安全项”，不是硬筛选规则。
    # 改成 soft deadline pressure 后，risk 会更频繁出现，因此默认权重调低。
    TASK_RISK_WEIGHTS = {"RT": 3.0, "Batch": 0.3, "AI": 0.8}  # 不同任务类型对违约风险的敏感度
    USE_SCORE_RISK = True  # 是否在节点 score 中加入 deadline risk；False 表示纯时延+能耗调度
    USE_DEADLINE_FILTER = False  # 旧版 hard deadline 开关；新版本建议使用下面的统一可行性筛选

    # ===========================================================
    # 约束 Boltzmann 调度开关：随机性只发生在“可行 + 近优”节点中
    # ===========================================================
    USE_CONSTRAINED_BOLTZMANN = True
    USE_BOLTZMANN_RANDOM = True       # False：机会集合内直接选最小 score，作为贪心消融
    USE_FEASIBILITY_FILTER = True
    USE_HARD_CPU_FILTER = True
    USE_HARD_DEADLINE_FILTER = True
    HARD_DEADLINE_TASKS = {"RT": True, "Batch": False, "AI": False}
    HARD_DEADLINE_FALLBACK = "min_violation"  # min_violation / relax_deadline
    USE_SAFETY_MARGIN_FILTER = True
    SAFETY_MARGIN_FACTOR = {"RT": 0.20, "Batch": 0.05, "AI": 0.05}
    SAFETY_MARGIN_SCALE_DEFAULT = 1.0

    USE_QUEUE_PRESSURE_SCORE = True
    QUEUE_PRESSURE_CLIP = 3.0
    QUEUE_WEIGHT_DEFAULT = 1.0
    USE_BO_QUEUE_WEIGHT = True

    # Scheduler node-score tradeoff. Defaults preserve the old linear score.
    SCHEDULER_TRADEOFF_MODE = "legacy"  # legacy / alpha_fixed / alpha_from_ratio
    SCHEDULER_TRADEOFF_ALPHA = 0.85
    SCHEDULER_ALPHA_MIN = 0.60
    SCHEDULER_ALPHA_MAX = 0.97
    SCHEDULER_SERVICE_LATENCY_WEIGHT = 1.0
    SCHEDULER_SERVICE_RISK_WEIGHT = 1.0
    SCHEDULER_SERVICE_QUEUE_WEIGHT = 1.0
    SCHEDULER_ENERGY_WEIGHT = 1.0

    # Scheduler score normalization. "legacy" keeps norm_mode=fixed/rolling behavior.
    SCHEDULER_SCORE_NORM_MODE = "legacy"  # legacy / candidate_median / candidate_iqr / rolling_ema
    SCHEDULER_NORM_CLIP_MAX = 3.0
    SCHEDULER_NORM_EPS = 1e-6
    SCHEDULER_NORM_EMA_ALPHA = 0.995

    RISK_SCALE_DEFAULT = 1.0
    USE_BO_RISK_SCALE = True
    BETA_DEFAULT = 3.0  # 与 BETA_INITIAL 保持一致；由于类属性定义顺序，不能在此直接引用 BETA_INITIAL
    USE_BO_BETA_CONTROL = True

    OPPORTUNITY_RHO_DEFAULT = 1.0
    USE_BO_OPPORTUNITY_RHO = True
    USE_OPPORTUNITY_WINDOW = True
    OPPORTUNITY_MODE = "std"          # std: min + rho * std(score); absolute: min + rho
    OPPORTUNITY_ABS_FLOOR = 1e-6
    OPPORTUNITY_MIN_CANDIDATES = 2

    USE_CLOUD_GATE = True
    USE_BO_CLOUD_GATE = True
    CLOUD_GATE_DEFAULT = 0.50
    CLOUD_GATE_ALLOW_TASKS = {"RT": False, "Batch": True, "AI": True}
    CLOUD_GATE_ALWAYS_ALLOW_IF_NO_EDGE_FEASIBLE = True
    CLOUD_GATE_PRESSURE_UTIL_WEIGHT = 0.70
    CLOUD_GATE_PRESSURE_BACKLOG_WEIGHT = 0.30
    CLOUD_GATE_BACKLOG_NORM = 24.0

    RISK_CLIP_MAX = 3.0  # soft risk 的截断上限，避免 risk 压过所有其他项
    RISK_MARGIN_FACTOR = {"RT": 1.0, "Batch": 0.3, "AI": 0.3}  # 多接近 deadline 才开始产生 pressure

    K_CPU = 1e-28  # 计算能耗系数，e_comp = K_CPU * speed^2 * cycles
    P_TRANS = 0.5  # 单位数据传输能耗系数
    BETA_INITIAL = 3.0  # Boltzmann 选择温度的反比参数；越大越偏向低分节点
    BETA_TRAINABLE = False  # 是否在线调整 beta；当前默认关闭
    BETA_DELTA = 0.0  # 如果允许训练 beta，每次更新的步长
    # 联邦云端评分权重（用于 mu + beta_cloud * sigma）
    FED_BETA = 3.0  # 联邦聚合时的探索强度，score = mu_fed + FED_BETA * sigma_fed

    # 任务源只从 5 个车间产生，不直接从云节点产生；每个车间内随机落到本地边缘入口节点。
    USE_WORKSHOP_ORIGIN = True
    WORKSHOP_ORIGIN_BIAS = [0.24, 0.22, 0.20, 0.18, 0.16]
    ORIGIN_BIAS = [0.12, 0.12, 0.11, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08]  # 兼容旧入口：仅对应 0~9 车间边缘节点

    # 车间/云拓扑。0~4 是车间，5 是厂区云，6 是区域云。
    USE_WORKSHOP_TOPOLOGY = True
    WORKSHOP_BASE_DELAY = [
        [0.16, 0.90, 1.60, 1.10, 1.90, 2.20, 4.00],
        [0.90, 0.16, 0.90, 1.35, 1.10, 1.90, 3.70],
        [1.60, 0.90, 0.16, 2.00, 1.25, 1.60, 3.40],
        [1.10, 1.35, 2.00, 0.16, 0.90, 2.10, 3.90],
        [1.90, 1.10, 1.25, 0.90, 0.16, 1.70, 3.60],
        [2.20, 1.90, 1.60, 2.10, 1.70, 0.30, 2.40],
        [4.00, 3.70, 3.40, 3.90, 3.60, 2.40, 0.50],
    ]
    WORKSHOP_BW = [
        [120.0, 65.0, 38.0, 58.0, 32.0, 45.0, 24.0],
        [65.0, 120.0, 65.0, 46.0, 58.0, 52.0, 26.0],
        [38.0, 65.0, 120.0, 34.0, 52.0, 60.0, 30.0],
        [58.0, 46.0, 34.0, 120.0, 65.0, 48.0, 25.0],
        [32.0, 58.0, 52.0, 65.0, 120.0, 56.0, 28.0],
        [45.0, 52.0, 60.0, 48.0, 56.0, 160.0, 70.0],
        [24.0, 26.0, 30.0, 25.0, 28.0, 70.0, 220.0],
    ]
    WORKSHOP_TRANS_ENERGY_FACTOR = [
        [1.00, 1.25, 1.55, 1.35, 1.70, 1.95, 2.60],
        [1.25, 1.00, 1.25, 1.45, 1.35, 1.80, 2.45],
        [1.55, 1.25, 1.00, 1.75, 1.40, 1.65, 2.35],
        [1.35, 1.45, 1.75, 1.00, 1.25, 1.90, 2.55],
        [1.70, 1.35, 1.40, 1.25, 1.00, 1.75, 2.40],
        [1.95, 1.80, 1.65, 1.90, 1.75, 1.00, 1.70],
        [2.60, 2.45, 2.35, 2.55, 2.40, 1.70, 1.00],
    ]
    NODE_ACCESS_DELAY = {"rt_edge": -0.04, "low_power": -0.02, "normal_edge": 0.00, "efficient": 0.02,
                         "batch_node": 0.03, "ai_accel": 0.08, "high_perf": 0.10, "backup_low": -0.01,
                         "factory_cloud": 0.20, "regional_cloud": 0.35}
    LINK_BW = 50.0  # 兼容旧逻辑的默认带宽；新模型实际使用 TRANS_BW_MATRIX
    LOCAL_UPLOAD_DELAY = 0.20  # 本地接入固定附加时延
    # ===========================================================
    # Trade-off 场景倍率：用于把“云不是永远最优”的冲突显式化。
    # 1.0 表示保持原始设定；>1 表示上云传输更慢/更耗能；<1 表示云算力被削弱。
    # 这些参数只作用于云目标节点，不改变边缘节点。
    # ===========================================================
    CLOUD_DELAY_MULT = 1.0
    CLOUD_ENERGY_MULT = 1.0
    CLOUD_SPEED_MULT = 1.0
    USE_LINK_QUEUE = False  # 第一版默认不做链路占用；如需链路串行排队，可改 True
    TRANS_DELAY_MATRIX = []  # 节点对之间的基础拓扑时延矩阵，后面会自动生成
    TRANS_BW_MATRIX = []  # 节点对之间的带宽矩阵，后面会自动生成

    # 利用率功耗模型：P(u)=P_idle+(P_max-P_idle)*u^alpha。
    # objective 默认采用 active_idle：节点有运行/排队任务才计 idle power，空闲时视为可睡眠。
    # real_energy 额外记录所有节点常开时的真实背景能耗，便于论文解释，但默认不放入 cost。
    USE_POWER_ENERGY_MODEL = True
    UTIL_POWER_ALPHA = 1.0
    OBJECTIVE_IDLE_MODE = "active_only"  # active_only / always_on / none
    REAL_IDLE_MODE = "always_on"  # always_on / active_only / none
    SLEEP_POWER_RATIO = 0.05

    MAX_HISTORY = 120  # GP 训练时保留的额外历史上限
    RECENT_HISTORY = 200  # 最近样本缓存长度
    ARCHIVE_PER_STATE = 12  # 每个情景/状态下额外保留的历史代表样本数

    ARRIVAL_THRESHOLDS = (1.25, 1.75)  # 到达率划分 LOW/MID/HIGH 的阈值；运行时会按 LAMBDA_SCHEDULE 自动重算
    DELAY_THRESHOLDS = (6.0, 12.0)  # 平均时延划分 LOW/MID/HIGH 的阈值
    UTIL_THRESHOLDS = (0.5, 0.85)  # 平均利用率划分 LOW/MID/HIGH 的阈值

    # 7 维连续情景向量：既包含系统压力，也显式包含任务结构
    # 说明：去掉了“平均时延”这一偏滞后的结果量，加入三类任务到达占比，
    # 让情景表示更偏“原因 + 压力”，更适合做相似场景检索。
    CONTEXT_FEATURE_NAMES = [
        "arrival_rate",         # 总到达强度
        "avg_util",             # 平均资源利用率
        "backlog",              # 系统积压量
        "vio_rate",             # 违约率
        "rt_arrival_ratio",     # 最近窗口内 RT 到达占比
        "batch_arrival_ratio",  # 最近窗口内 Batch 到达占比
        "ai_arrival_ratio",     # 最近窗口内 AI 到达占比
    ]
    CONTEXT_BOUNDS = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, 1.0, 500.0, 1.0, 1.0, 1.0, 1.0],
    ]  # 情景向量归一化边界
    CONTEXT_KNN = 8  # 情景 BO 中找相似上下文历史样本时使用的近邻数
    CONTEXT_SIMILARITY_MODE = "gaussian"  # 默认用高斯核；可切到 inverse_distance 做退化/轻量相似度
    CONTEXT_KERNEL_LENGTHS = [0.28, 0.20, 0.25, 0.18, 0.16, 0.16, 0.16]  # 归一化空间中的核长度尺度
    TRUST_RADIUS_INIT = 0.10  # Trust Region 初始半径（相对于归一化空间）
    TRUST_RADIUS_MIN = 0.04  # Trust Region 最小半径
    TRUST_RADIUS_MAX = 0.35  # Trust Region 最大半径
    TRUST_RADIUS_GROWTH = 1.15  # 如果本轮更优，TR 半径扩大倍数
    TRUST_RADIUS_SHRINK = 0.92  # 如果本轮没变好，TR 半径缩小倍数

    DELAY_NORM = 10.0  # 节点评分时对时延的归一化尺度；车间/云拓扑下传输差异更明显
    ENERGY_NORM = 4000.0  # 节点评分时对能耗的归一化尺度；功率×时间模型下能耗量级更大
    DEADLINE_RISK_NORM = 10.0  # 节点评分时对违约风险的归一化尺度
    ENERGY_WEIGHT = 1.0  # 预留参数；当前主要通过 theta 中后三维控制能耗权重
    DEADLINE_WEIGHT = 2.0  # 默认违约风险权重（若任务类型未单独指定）
    COMPLETE_PENALTY = 2.0  # 预留参数，当前主 cost 未直接使用
    VIO_PENALTY = 3.0  # 预留参数，当前主 cost 未直接使用
    SLA_PENALTY_WEIGHT = 1500.0  # 违反 SLA 的惩罚强度
    EARLY_BONUS_WEIGHT = 200.0  # 提前完成的奖励强度
    USE_EARLY_BONUS = False  # 主实验先关闭提前奖励，避免长 deadline 任务主导 reward
    EARLY_BONUS_CAP = 5.0  # 若开启提前奖励，只奖励有限提前量，避免 reward 被 slack 无限放大
    LATE_PENALTY_WEIGHT = 300.0  # 超期完成的惩罚强度
    BACKLOG_WEIGHT = 200.0  # 系统积压任务数的惩罚强度
    ZERO_COMPLETION_PENALTY = 5000.0  # 本窗口有到达但一个都没完成时的重罚

    # ===========================================================
    # BO 反馈模式
    # ===========================================================
    # window：旧版窗口级即时反馈，窗口结束后把该窗口 cost 直接 tell 给 BO。
    # cohort_complete：任务批次级延迟反馈。每个 BO 窗口新到任务绑定当前 theta；
    # 只有该批任务全部完成，或者实验结束强制结算时，才把 cohort_cost 反馈给 BO。
    # 注意：即使使用 cohort_complete，窗口级指标仍然会保留用于画图和系统监控。
    FEEDBACK_MODE = "window"
    COHORT_UNFINISHED_PENALTY = 1000.0
    COHORT_PENDING_AREA_WEIGHT = 5.0
    COHORT_FORCE_FINALIZE_AT_RUN_END = True

    HARD_DEADLINE = False  # 预留开关；当前不再默认硬筛选 RT 可行节点

    REPEAT_RUNS = 1  # 整体实验重复次数
    BASE_SEED = 42  # 默认实验主种子
    # 服务器当前 no-RoundRobin runner 默认值，写入脚本方便备份和复现实验。
    DEFAULT_NO_RR_OUTPUT_FAMILY = "v6_500_bestcbo_direct36_norr"
    DEFAULT_SCENARIO_FEEDBACK_SCORE = "task_effective_backlog_violation"
    DEFAULT_BO_HISTORY_MODE = "recent"
    DEFAULT_BO_RECENT_WINDOW = 80
    # CBO stability extensions. Defaults intentionally preserve existing behavior.
    DEFAULT_CBO_HISTORY_SELECT_MODE = "recent"
    DEFAULT_CBO_CONTEXT_K = 50
    DEFAULT_CBO_ELITE_K = 20
    DEFAULT_CBO_DIVERSE_K = 20
    DEFAULT_CBO_ROBUST_SCORE_MODE = "none"
    DEFAULT_CBO_ROBUST_STD_WEIGHT = 0.5
    DEFAULT_CBO_THETA_MERGE_EPS = 0.05
    DEFAULT_CBO_CONTEXT_SIM_THRESHOLD = 0.0
    DEFAULT_CBO_TR_MODE = "off"
    DEFAULT_CBO_TR_ANCHOR_MODE = "posterior_mean"
    DEFAULT_CBO_ROBUST_INCUMBENT_MODE = "off"
    DEFAULT_CBO_MACRO_GATE_MODE = "off"
    DEFAULT_CBO_MACRO_K = 100
    DEFAULT_CBO_MACRO_TOTAL_SCALE = "auto"
    DEFAULT_CBO_MACRO_LENGTHSCALE_TOTAL = 1.0
    DEFAULT_CBO_MACRO_LENGTHSCALE_RT = 0.15
    DEFAULT_CBO_MACRO_LENGTHSCALE_BATCH = 0.15
    DEFAULT_CBO_MACRO_ALPHA = 1.0
    DEFAULT_CBO_DUMP_CANDIDATES = False
    DEFAULT_CBO_DUMP_CANDIDATES_EVERY = 20
    DEFAULT_CBO_DUMP_CANDIDATES_TOPN = 30
    # CBO TR / selection experimental extensions. Defaults preserve old behavior.
    DEFAULT_CBO_SELECT_MODE = "greedy"  # greedy / topk_stochastic / epsilon_greedy / randomized_ucb
    DEFAULT_CBO_TOPK = 5
    DEFAULT_CBO_SELECT_TEMPERATURE = 0.20
    DEFAULT_CBO_EPSILON = 0.10
    DEFAULT_CBO_ACQ_BETA = 3.0
    DEFAULT_CBO_SURPRISE_WINDOW = 10
    DEFAULT_CBO_SURPRISE_Z_THRESHOLD = 2.0
    DEFAULT_CBO_SURPRISE_COST_GAP_PCT = 0.03
    DEFAULT_CBO_SIGMA_FLOOR = 1e-6
    DEFAULT_CBO_RADIUS_RESET = 0.12
    DEFAULT_CBO_RADIUS_MIN_STUCK_ROUNDS = 10
    DEFAULT_CBO_REBOUND_WINDOW = 20
    DEFAULT_CBO_REBOUND_THRESHOLD_PCT = 0.03
    DEFAULT_CBO_SELECTION_COOLDOWN = 5
    DEFAULT_CBO_CONDITION_ANCHOR_SWITCH = "context_best"  # off / recent_best / context_best / robust_elite
    DEFAULT_SELECTED_KEYS_NO_RR = [
        "reduced6_fixed_mid",
        "reduced6_fixed_tuned",
        "reduced6_bo_greedy",
        "reduced6_cbo_lite_pressure_taskmix_counts",
        "direct_greedy_cost",
        "direct_least_load",
        "direct_queue_aware_greedy",
    ]

    REWARD_TARGET = -0.5  # 若训练 beta，可把它视为目标 reward/cost 参考值
    USE_FIXED_RNG = True  # 是否固定随机数，便于复现实验
    FIXED_RNG_SEED = 42  # 固定随机种子值
    CONTROL_WEIGHT_BOUNDS = (0.1, 5.0)
    CONTROL_QUEUE_BOUNDS = (0.0, 5.0)
    CONTROL_RISK_SCALE_BOUNDS = (0.0, 5.0)
    CONTROL_BETA_BOUNDS = (0.5, 8.0)
    CONTROL_OPPORTUNITY_RHO_BOUNDS = (0.0, 3.0)
    CONTROL_CLOUD_GATE_BOUNDS = (0.05, 0.95)

CFG = ExperimentConfig()
CFG.ARRIVAL_THRESHOLDS = infer_arrival_thresholds(CFG.LAMBDA_SCHEDULE)
CFG.TRANS_DELAY_MATRIX = build_topology_matrix(len(CFG.NODES_CFG))
CFG.TRANS_BW_MATRIX = build_bandwidth_matrix(len(CFG.NODES_CFG))

if not getattr(CFG, "USE_EXTENDED_BO_CONTROL", True):
    CFG.FEATURE_NAMES = list(CFG.BASE_FEATURE_NAMES)
    CFG.DIM_THETA = len(CFG.FEATURE_NAMES)


def get_control_bounds(dim=None):
    """返回当前 BO 控制变量的逐维搜索边界。"""
    dim = int(CFG.DIM_THETA if dim is None else dim)
    lows, highs = [], []
    for name in list(CFG.FEATURE_NAMES)[:dim]:
        if name == "W_Queue":
            lo, hi = CFG.CONTROL_QUEUE_BOUNDS
        elif name == "W_Risk_Scale":
            lo, hi = CFG.CONTROL_RISK_SCALE_BOUNDS
        elif name == "Beta_Control":
            lo, hi = CFG.CONTROL_BETA_BOUNDS
        elif name == "Opportunity_Rho":
            lo, hi = CFG.CONTROL_OPPORTUNITY_RHO_BOUNDS
        elif name == "Cloud_Gate":
            lo, hi = CFG.CONTROL_CLOUD_GATE_BOUNDS
        else:
            lo, hi = CFG.CONTROL_WEIGHT_BOUNDS
        lows.append(float(lo))
        highs.append(float(hi))
    return [lows, highs]


def get_theta_value(theta_full, name, default):
    try:
        names = list(CFG.FEATURE_NAMES)
        if name not in names:
            return float(default)
        idx = names.index(name)
        theta = normalize_theta_vector(theta_full, dim=CFG.DIM_THETA, fill=default)
        if idx >= len(theta):
            return float(default)
        val = float(theta[idx])
        if not np.isfinite(val):
            return float(default)
        return val
    except Exception:
        return float(default)


def extract_scheduler_controls(theta_full):
    """从 BO 输出 theta 中解析调度器使用的扩展控制量。"""
    beta = get_theta_value(theta_full, "Beta_Control", getattr(CFG, "BETA_DEFAULT", CFG.BETA_INITIAL))
    beta = float(np.clip(beta, *CFG.CONTROL_BETA_BOUNDS))
    rho = get_theta_value(theta_full, "Opportunity_Rho", getattr(CFG, "OPPORTUNITY_RHO_DEFAULT", 1.0))
    rho = float(np.clip(rho, *CFG.CONTROL_OPPORTUNITY_RHO_BOUNDS))
    cloud_gate = get_theta_value(theta_full, "Cloud_Gate", getattr(CFG, "CLOUD_GATE_DEFAULT", 0.50))
    cloud_gate = float(np.clip(cloud_gate, *CFG.CONTROL_CLOUD_GATE_BOUNDS))
    queue_w = get_theta_value(theta_full, "W_Queue", getattr(CFG, "QUEUE_WEIGHT_DEFAULT", 1.0))
    queue_w = float(np.clip(queue_w, *CFG.CONTROL_QUEUE_BOUNDS))
    risk_scale = get_theta_value(theta_full, "W_Risk_Scale", getattr(CFG, "RISK_SCALE_DEFAULT", 1.0))
    risk_scale = float(np.clip(risk_scale, *CFG.CONTROL_RISK_SCALE_BOUNDS))
    return {
        "beta": beta if getattr(CFG, "USE_BO_BETA_CONTROL", True) else float(getattr(CFG, "BETA_INITIAL", 3.0)),
        "rho": rho if getattr(CFG, "USE_BO_OPPORTUNITY_RHO", True) else float(getattr(CFG, "OPPORTUNITY_RHO_DEFAULT", 1.0)),
        "cloud_gate": cloud_gate if getattr(CFG, "USE_BO_CLOUD_GATE", True) else float(getattr(CFG, "CLOUD_GATE_DEFAULT", 0.50)),
        "queue_w": queue_w if getattr(CFG, "USE_BO_QUEUE_WEIGHT", True) else float(getattr(CFG, "QUEUE_WEIGHT_DEFAULT", 1.0)),
        "risk_scale": risk_scale if getattr(CFG, "USE_BO_RISK_SCALE", True) else float(getattr(CFG, "RISK_SCALE_DEFAULT", 1.0)),
        "safety_margin_scale": float(getattr(CFG, "SAFETY_MARGIN_SCALE_DEFAULT", 1.0)),
    }

SAVE_DIR = os.path.abspath("pic_core_v2")
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
SCENARIO_SAVE_DIR = os.path.abspath("pic_scenario_trust_region")
if not os.path.exists(SCENARIO_SAVE_DIR): os.makedirs(SCENARIO_SAVE_DIR)

# 字体设置
plt.rcParams['axes.unicode_minus'] = False
for font in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']:
    try:
        matplotlib.font_manager.fontManager.findfont(font, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font]
        break
    except:
        continue

# ==========================================
# 2. 仿真核心 (Simulation)
# ==========================================
