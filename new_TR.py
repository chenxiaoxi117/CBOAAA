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
class EventType(Enum):
    """事件驱动仿真的三类事件。"""
    TASK_ARRIVAL = 1
    TRANS_FINISH = 2
    TASK_FINISH = 3

@dataclass(order=True)
class Event:
    """事件队列中的元素。time 决定弹出顺序。"""
    time: float
    type: EventType = field(compare=False)
    payload: Any = field(compare=False)

@dataclass
class Task:
    """任务对象。

    create_time:   创建时间
    data_size:     需要传输的数据量
    cpu_req:       需要占用的 CPU 资源单位
    duration_base: 基础执行时长
    deadline:      自动根据 duration_base * deadline_factor 生成
    """
    id: str
    create_time: float
    data_size: float
    cpu_req: int
    duration_base: float
    task_type: str
    deadline_factor: float
    cpu_cycles: float = 0.0
    deadline: float = 0.0
    arrival_node_idx: int = -1
    origin_node_id: int = -1
    finish_time: float = -1.0
    energy_consumed: float = 0.0
    transmission_energy: float = 0.0

    def __post_init__(self):
        self.deadline = self.create_time + self.duration_base * self.deadline_factor
        self.cpu_cycles = self.duration_base * 4.5e9

class Node:
    """计算节点。

    增强版节点模型：
    1) 每个节点属于某个车间/云位置，传输时延由车间拓扑决定；
    2) 不同节点对 RT / Batch / AI 有不同加速系数；
    3) 能耗不再只按 K_CPU * speed^2 估算，而是使用 p_idle / p_max / CPU 利用率功耗模型。

    仍保留“固定服务时间 + CPU 资源准入 + 队列等待估计”的事件驱动框架，避免一次性推翻原仿真。
    """
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.id = cfg["id"]
        self.workshop = _get_node_site_from_cfg(cfg, default_site=int(self.id) // 2)
        self.role = str(cfg.get("role", "normal_edge"))
        self.is_cloud = _node_is_cloud(cfg)
        self.cpu_total = int(cfg["cpu"])
        self.base_speed = float(cfg["speed"]) * 1e9
        self.p_idle = float(cfg.get("p_idle", 0.0))
        self.p_max = float(cfg.get("p_max", self.p_idle))
        self.base_bw = CFG.LINK_BW
        self.cpu_free = self.cpu_total
        self.ready_queue = []
        self.running_tasks = []

    def effective_speed(self, task_or_type):
        task_type = task_or_type.task_type if hasattr(task_or_type, "task_type") else str(task_or_type)
        return get_effective_speed(self.cfg, task_type)

    def utilization(self):
        return float(np.clip(1.0 - self.cpu_free / max(1.0, float(self.cpu_total)), 0.0, 1.0))

    def has_work(self):
        return bool(self.ready_queue or self.running_tasks)

    def power_components(self, objective=True):
        """返回当前节点功率分量。单位可理解为 W。

        objective=True 用于优化目标；默认 active_only：节点有运行/排队任务才计 idle，空闲视为可睡眠。
        objective=False 用于真实能耗诊断；默认 always_on：所有节点空闲也有 idle power。
        """
        u = self.utilization()
        alpha = float(getattr(CFG, "UTIL_POWER_ALPHA", 1.0))
        dynamic_power = max(0.0, self.p_max - self.p_idle) * (u ** alpha)
        mode = getattr(CFG, "OBJECTIVE_IDLE_MODE", "active_only") if objective else getattr(CFG, "REAL_IDLE_MODE", "always_on")
        if mode == "always_on":
            idle_power = self.p_idle
        elif mode == "active_only":
            idle_power = self.p_idle if self.has_work() else self.p_idle * float(getattr(CFG, "SLEEP_POWER_RATIO", 0.05))
        elif mode == "none":
            idle_power = 0.0
        else:
            idle_power = self.p_idle if self.has_work() else 0.0
        return float(idle_power), float(dynamic_power), float(idle_power + dynamic_power)

    def _service_time(self, task):
        return task.cpu_cycles / (self.effective_speed(task) + 1e-9)

    def _estimate_earliest_start_time(self, task, current_time):
        if task.cpu_req > self.cpu_total:
            return float("inf")

        now = float(current_time)
        cpu_free = int(self.cpu_free)
        pending = list(self.ready_queue) + [task]

        finish_events = []
        counter = 0
        for rt in self.running_tasks:
            finish_t = float(getattr(rt, "finish_time", now))
            if finish_t < now:
                finish_t = now
            finish_events.append((finish_t, counter, int(rt.cpu_req)))
            counter += 1
        finish_events.sort()

        def try_start_once(at_time, cpu_free_now, pending_now, finish_events_now, counter_now):
            new_pending = []
            for q in pending_now:
                if q.cpu_req <= cpu_free_now:
                    if q is task:
                        return at_time, cpu_free_now, new_pending, finish_events_now, counter_now, True
                    cpu_free_now -= q.cpu_req
                    finish_t = at_time + self._service_time(q)
                    finish_events_now.append((finish_t, counter_now, int(q.cpu_req)))
                    counter_now += 1
                else:
                    new_pending.append(q)
            finish_events_now.sort()
            return at_time, cpu_free_now, new_pending, finish_events_now, counter_now, False

        while True:
            _, cpu_free, pending, finish_events, counter, started = try_start_once(
                now, cpu_free, pending, finish_events, counter
            )
            if started:
                return now

            if not finish_events:
                return float("inf")

            next_time = finish_events[0][0]
            now = next_time
            remain_events = []
            for finish_t, idx, req in finish_events:
                if abs(finish_t - next_time) <= 1e-9:
                    cpu_free += int(req)
                else:
                    remain_events.append((finish_t, idx, req))
            finish_events = remain_events

    def estimate_compute_energy_for_task(self, task):
        """估计把该任务放到当前节点的增量计算能耗。

        这是调度打分用的近似值，不等于最终窗口能耗。
        最终评估在 run_continuous 中按节点利用率对功率积分。
        """
        service_time = self._service_time(task)
        util_after = min(1.0, (self.cpu_total - self.cpu_free + task.cpu_req) / max(1.0, float(self.cpu_total)))
        alpha = float(getattr(CFG, "UTIL_POWER_ALPHA", 1.0))
        power_after = self.p_idle + max(0.0, self.p_max - self.p_idle) * (util_after ** alpha)
        # 只给该任务分摊一部分节点功率，避免多任务并发时每个任务都承担整机功率。
        share = min(1.0, max(0.05, task.cpu_req / max(1.0, float(self.cpu_total))))
        return power_after * service_time * share

    def estimate_metrics(self, task, current_time, origin_node_idx=None):
        origin = origin_node_idx if origin_node_idx is not None and origin_node_idx >= 0 else self.id
        t_trans = get_transmission_delay(origin, self.id, task.data_size, include_local=True)

        earliest_start = self._estimate_earliest_start_time(task, current_time)
        if earliest_start == float("inf"):
            wait_time = 1e12
        else:
            wait_time = max(0.0, earliest_start - float(current_time))

        t_comp = self._service_time(task)
        expected_latency = t_trans + wait_time + t_comp

        e_comp = self.estimate_compute_energy_for_task(task)
        e_trans = get_transmission_energy(origin, self.id, task.data_size)
        expected_energy = e_comp + e_trans
        return expected_energy, expected_latency

    def enqueue_task(self, t):
        self.ready_queue.append(t)

    def allocate(self, t):
        if self.cpu_free >= t.cpu_req:
            self.cpu_free -= t.cpu_req
            self.running_tasks.append(t)
            return True
        return False

    def release(self, t):
        self.cpu_free += t.cpu_req
        self.cpu_free = min(self.cpu_free, self.cpu_total)
        if t in self.running_tasks:
            self.running_tasks.remove(t)


def sample_origin_node(rng):
    """按车间采样任务源，然后在该车间的边缘节点中选一个入口节点。

    云节点不作为任务源；它们只是可选执行节点。
    """
    if getattr(CFG, "USE_WORKSHOP_ORIGIN", False):
        probs = list(getattr(CFG, "WORKSHOP_ORIGIN_BIAS", []))
        r = rng.random()
        cum = 0.0
        workshop = 0
        for idx, p in enumerate(probs):
            cum += float(p)
            if r <= cum:
                workshop = idx
                break
        candidates = [cfg["id"] for cfg in CFG.NODES_CFG if _get_node_site_from_cfg(cfg, 999) == workshop and not _node_is_cloud(cfg)]
        if candidates:
            return int(rng.choice(candidates)) if hasattr(rng, "choice") else int(candidates[int(rng.random() * len(candidates)) % len(candidates)])
    # 兼容旧逻辑：按 ORIGIN_BIAS 从前若干个节点里采样。
    r = rng.random()
    cum = 0.0
    for idx, p in enumerate(CFG.ORIGIN_BIAS):
        cum += p
        if r <= cum:
            return idx
    return len(CFG.ORIGIN_BIAS) - 1

class BatchWorkloadGenerator:
    """批模式任务生成器。当前主流程主要不用它，保留作兼容。"""
    def __init__(self, factory_id, seed):
        self.fid = factory_id
        self.rng = random.Random(seed)
        self.task_counter = 0

    def get_batch(self, current_time, batch_size):
        tasks = []
        t = current_time
        lam = CFG.BATCH_POISSON_LAMBDA
        for _ in range(batch_size):
            dt = self.rng.expovariate(lam)
            t += dt
            t_type = sample_task_type(self.rng, current_time=t)
            props = CFG.TASK_PROPS[t_type]
            origin = self._sample_origin()
            task = Task(
                id=f"{self.fid}-{self.task_counter}",
                create_time=t,
                data_size=props["data"],
                cpu_req=props["cpu"],
                duration_base=props["dur"],
                task_type=t_type,
                deadline_factor=props["deadline_factor"],
                origin_node_id=origin
            )
            tasks.append(task)
            self.task_counter += 1
        return tasks, t

    def _sample_origin(self):
        return sample_origin_node(self.rng)

class PiecewisePoissonWorkloadGenerator:
    """连续模式任务生成器。

    根据 LAMBDA_SCHEDULE 分段改变到达率，模拟动态工作负载。
    """
    def __init__(self, factory_id, seed):
        self.fid = factory_id
        self.rng = random.Random(seed)
        self.task_counter = 0

    def _get_lambda(self, t):
        for start, end, lam in CFG.LAMBDA_SCHEDULE:
            if start <= t < end: return lam, end
        return 0.0, CFG.SESSION_DURATION * 2.0

    def _next_arrival(self, current_time):
        t = current_time
        while True:
            lam, seg_end = self._get_lambda(t)
            if lam <= 0: return CFG.SESSION_DURATION * 2.0
            dt = self.rng.expovariate(lam)
            if t + dt <= seg_end: return t + dt
            t = seg_end

    def get_next_task(self, current_time):
        arrival = self._next_arrival(current_time)
        if arrival > CFG.SESSION_DURATION: return None, arrival
        t_type = sample_task_type(self.rng, current_time=arrival)
        props = CFG.TASK_PROPS[t_type]
        origin = self._sample_origin()
        task = Task(
            id=f"{self.fid}-{self.task_counter}",
            create_time=arrival,
            data_size=props["data"],
            cpu_req=props["cpu"],
            duration_base=props["dur"],
            task_type=t_type,
            deadline_factor=props["deadline_factor"],
            origin_node_id=origin
        )
        self.task_counter += 1
        return task, arrival

    def _sample_origin(self):
        return sample_origin_node(self.rng)

# ==========================================
# 3. 调度算法 (Scheduler)
# ==========================================
class BoltzmannScheduler:
    """基于 Boltzmann/Softmax 的节点选择器。

    norm_mode 支持两种归一化：
    1) fixed：旧版固定归一化，使用 CFG.ENERGY_NORM / DELAY_NORM / DEADLINE_RISK_NORM；
    2) rolling：方案三滚动归一化，energy / latency 按 RT/Batch/AI 分别维护 EMA 参考尺度，risk 保持绝对语义。
    """

    def __init__(self, np_rng=None, norm_mode="rolling"):
        self.beta = float(CFG.BETA_INITIAL)
        self.delta = float(CFG.BETA_DELTA)
        self.last_total_energy = None
        self.alpha_latency = CFG.ALPHA_LATENCY
        self.current_time = 0.0
        self.ma_baseline = None
        self.ma_alpha = 0.3
        self.deadband_ratio = 0.005
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng()

        self.norm_mode = str(norm_mode or "rolling").lower()
        if self.norm_mode not in {"fixed", "rolling"}:
            raise ValueError(f"Unknown norm_mode={norm_mode}. Expected 'fixed' or 'rolling'.")

        self.scheduler_score_norm_mode = str(getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy") or "legacy").lower()
        if self.scheduler_score_norm_mode not in {"legacy", "candidate_median", "candidate_iqr", "rolling_ema"}:
            raise ValueError(
                f"Unknown scheduler score norm mode={self.scheduler_score_norm_mode}. "
                "Expected legacy/candidate_median/candidate_iqr/rolling_ema."
            )

        # Rolling EMA normalization parameters are command-line configurable.
        self.norm_alpha = float(getattr(CFG, "SCHEDULER_NORM_EMA_ALPHA", 0.995))
        self.norm_clip_max = float(getattr(CFG, "SCHEDULER_NORM_CLIP_MAX", 3.0))
        self.norm_floor = float(getattr(CFG, "SCHEDULER_NORM_EPS", 1e-6))
        self.norm_refs = self._init_task_type_norm_refs()
        self.last_score_debug = {}

    def _init_task_type_norm_refs(self):
        """根据任务属性和节点配置，初始化每类任务的 energy / latency 参考尺度。"""
        refs = {}
        for task_type in TASK_TYPE_ORDER:
            props = CFG.TASK_PROPS[task_type]
            data_size = float(props["data"])
            duration_base = float(props["dur"])
            cpu_cycles = duration_base * 4.5e9

            energy_vals = []
            latency_vals = []
            source_nodes = [cfg["id"] for cfg in CFG.NODES_CFG if not _node_is_cloud(cfg)] or [0]
            for node_cfg in CFG.NODES_CFG:
                speed = get_effective_speed(node_cfg, task_type)
                t_comp = cpu_cycles / (speed + 1e-9)
                util_guess = min(1.0, float(props["cpu"]) / max(1.0, float(node_cfg["cpu"])))
                p_idle = float(node_cfg.get("p_idle", 0.0))
                p_max = float(node_cfg.get("p_max", p_idle))
                power_guess = p_idle + max(0.0, p_max - p_idle) * (util_guess ** float(getattr(CFG, "UTIL_POWER_ALPHA", 1.0)))
                e_comp = power_guess * t_comp * max(0.05, util_guess)
                trans_delays = []
                trans_energies = []
                for origin_id in source_nodes:
                    trans_delays.append(get_transmission_delay(origin_id, int(node_cfg["id"]), data_size, include_local=True))
                    trans_energies.append(get_transmission_energy(origin_id, int(node_cfg["id"]), data_size))
                expected_energy = e_comp + float(np.median(trans_energies))
                expected_latency = float(np.median(trans_delays)) + t_comp

                energy_vals.append(expected_energy)
                latency_vals.append(expected_latency)

            refs[task_type] = {
                "energy": max(float(np.median(energy_vals)), self.norm_floor),
                "latency": max(float(np.median(latency_vals)), self.norm_floor),
            }
        return refs

    def _ema_update_ref(self, task_type, key, current_value):
        """用 EMA 慢速更新某类任务的归一化参考值。"""
        current_value = max(float(current_value), self.norm_floor)
        old_value = float(self.norm_refs[task_type][key])
        new_value = self.norm_alpha * old_value + (1.0 - self.norm_alpha) * current_value
        self.norm_refs[task_type][key] = max(float(new_value), self.norm_floor)

    def update_beta(self, normalized_metric):
        """可选的 beta 在线更新逻辑；默认关闭。"""
        if not getattr(CFG, "BETA_TRAINABLE", False):
            return
        target = float(getattr(CFG, "REWARD_TARGET", -0.5))
        metric = float(normalized_metric)
        gap = metric - target
        if abs(gap) <= abs(target) * self.deadband_ratio:
            return
        self.beta = float(np.clip(self.beta + self.delta * np.sign(gap), *CFG.CONTROL_BETA_BOUNDS))

    def _node_score(self, task, node_idx, node, latency_w, energy_w, risk_w):
        """先只计算该节点对当前任务的原始指标，真正的归一化和 score 在 select_node() 里做。

        deadline_risk 使用 soft deadline pressure：
        - slack 足够大时 risk=0；
        - 接近 deadline 时 risk 逐渐增大；
        - 预测超期时 risk > 1，并做上限截断。
        这样不再通过硬规则筛掉节点，而是把安全性转成连续可优化的 score 项。
        """
        origin = task.origin_node_id if task.origin_node_id >= 0 else node_idx
        energy_raw, latency_total = node.estimate_metrics(task, self.current_time, origin_node_idx=origin)
        predicted_finish = self.current_time + latency_total
        deadline_overrun = max(0.0, predicted_finish - task.deadline)

        slack = task.deadline - predicted_finish
        margin_factor = getattr(CFG, "RISK_MARGIN_FACTOR", {"RT": 1.0, "Batch": 0.3, "AI": 0.3}).get(task.task_type, 0.3)
        safe_margin = max(1.0, float(margin_factor) * float(task.duration_base))
        deadline_risk = max(0.0, (safe_margin - slack) / safe_margin)
        deadline_risk = min(deadline_risk, float(getattr(CFG, "RISK_CLIP_MAX", 3.0)))

        return {
            "node_idx": int(node_idx),
            "energy_raw": float(energy_raw),
            "latency_total": float(latency_total),
            "predicted_finish": float(predicted_finish),
            "deadline_overrun": float(deadline_overrun),
            "deadline_risk": float(deadline_risk),
            "slack": float(slack),
        }

    def _compute_fixed_norms(self, candidates):
        """旧版固定归一化：所有方法共用固定常数尺度。

        risk 不再使用 deadline_overrun/DEADLINE_RISK_NORM，
        而是统一使用 _node_score 里构造好的 deadline_risk，保证 fixed/rolling 下 risk 含义一致。
        """
        e_vals = np.array([c["energy_raw"] for c in candidates], dtype=float)
        l_vals = np.array([c["latency_total"] for c in candidates], dtype=float)
        norm_e = e_vals / max(float(CFG.ENERGY_NORM), self.norm_floor)
        norm_l = l_vals / max(float(CFG.DELAY_NORM), self.norm_floor)
        norm_r = self._compute_risk_norms(candidates)
        return norm_e, norm_l, norm_r, {
            "energy_ref": float(CFG.ENERGY_NORM),
            "latency_ref": float(CFG.DELAY_NORM),
            "energy_norm_mode": "fixed_legacy",
            "latency_norm_mode": "fixed_legacy",
            "risk_ref": "soft_deadline_pressure" if getattr(CFG, "USE_SCORE_RISK", True) else "disabled",
        }

    def _compute_risk_norms(self, candidates):
        if getattr(CFG, "USE_SCORE_RISK", True):
            norm_r = np.array([c.get("deadline_risk", 0.0) for c in candidates], dtype=float)
            return np.clip(norm_r, 0.0, float(getattr(CFG, "RISK_CLIP_MAX", 3.0)))
        return np.zeros(len(candidates), dtype=float)

    def _compute_rolling_norms(self, task, candidates):
        """方案三滚动归一化：energy/latency 按任务类型 EMA 更新，risk 绝对归一化。"""
        e_vals = np.array([c["energy_raw"] for c in candidates], dtype=float)
        l_vals = np.array([c["latency_total"] for c in candidates], dtype=float)
        task_type = task.task_type
        cur_e_med = float(np.median(e_vals))
        cur_l_med = float(np.median(l_vals))
        self._ema_update_ref(task_type, "energy", cur_e_med)
        self._ema_update_ref(task_type, "latency", cur_l_med)
        e_ref = max(float(self.norm_refs[task_type]["energy"]), self.norm_floor)
        l_ref = max(float(self.norm_refs[task_type]["latency"]), self.norm_floor)
        norm_e = np.clip(e_vals / e_ref, 0.0, self.norm_clip_max)
        norm_l = np.clip(l_vals / l_ref, 0.0, self.norm_clip_max)
        norm_r = self._compute_risk_norms(candidates)
        return norm_e, norm_l, norm_r, {
            "energy_ref": float(e_ref),
            "latency_ref": float(l_ref),
            "energy_median_current": float(cur_e_med),
            "latency_median_current": float(cur_l_med),
            "energy_norm_mode": "rolling_ema",
            "latency_norm_mode": "rolling_ema",
            "risk_ref": "soft_deadline_pressure" if getattr(CFG, "USE_SCORE_RISK", True) else "disabled",
        }

    def _compute_candidate_median_norms(self, task, candidates, mode="candidate_median"):
        e_vals = np.array([c["energy_raw"] for c in candidates], dtype=float)
        l_vals = np.array([c["latency_total"] for c in candidates], dtype=float)
        e_ref = max(float(np.median(e_vals)), self.norm_floor)
        l_ref = max(float(np.median(l_vals)), self.norm_floor)
        norm_e = np.clip(e_vals / e_ref, 0.0, self.norm_clip_max)
        norm_l = np.clip(l_vals / l_ref, 0.0, self.norm_clip_max)
        norm_r = self._compute_risk_norms(candidates)
        debug = {
            "energy_ref": float(e_ref),
            "latency_ref": float(l_ref),
            "energy_norm_mode": mode,
            "latency_norm_mode": mode,
            "risk_ref": "soft_deadline_pressure" if getattr(CFG, "USE_SCORE_RISK", True) else "disabled",
        }
        if mode == "candidate_iqr":
            e_q75, e_q25 = np.percentile(e_vals, [75, 25]) if len(e_vals) else (0.0, 0.0)
            l_q75, l_q25 = np.percentile(l_vals, [75, 25]) if len(l_vals) else (0.0, 0.0)
            debug.update({
                "energy_iqr": float(max(0.0, e_q75 - e_q25)),
                "latency_iqr": float(max(0.0, l_q75 - l_q25)),
                "energy_center": float(e_ref),
                "latency_center": float(l_ref),
                "iqr_fallback": bool((e_q75 - e_q25) <= self.norm_floor or (l_q75 - l_q25) <= self.norm_floor),
            })
        return norm_e, norm_l, norm_r, debug

    def _compute_scheduler_norms(self, task, candidates):
        """Node-score normalization used only for scheduler candidate comparison."""
        mode = str(getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", self.scheduler_score_norm_mode) or "legacy").lower()
        if mode == "legacy":
            if self.norm_mode == "fixed":
                norm_e, norm_l, norm_r, debug = self._compute_fixed_norms(candidates)
            else:
                norm_e, norm_l, norm_r, debug = self._compute_rolling_norms(task, candidates)
        elif mode == "rolling_ema":
            norm_e, norm_l, norm_r, debug = self._compute_rolling_norms(task, candidates)
        elif mode in {"candidate_median", "candidate_iqr"}:
            norm_e, norm_l, norm_r, debug = self._compute_candidate_median_norms(task, candidates, mode=mode)
        else:
            raise ValueError(f"Unknown scheduler score norm mode={mode}")
        debug["scheduler_score_norm_mode"] = mode
        debug["legacy_norm_mode"] = self.norm_mode
        return norm_e, norm_l, norm_r, debug

    def _resolve_scheduler_alpha(self, task_type, latency_w, energy_w):
        mode = str(getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy") or "legacy").lower()
        alpha_min = float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60))
        alpha_max = float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97))
        if alpha_min > alpha_max:
            alpha_min, alpha_max = alpha_max, alpha_min
        eps = max(float(getattr(CFG, "SCHEDULER_NORM_EPS", 1e-6)), 1e-12)
        if mode == "alpha_fixed":
            alpha_raw = float(getattr(CFG, "SCHEDULER_TRADEOFF_ALPHA", 0.85))
            return float(np.clip(alpha_raw, alpha_min, alpha_max)), "scheduler_tradeoff_alpha", mode
        if mode == "alpha_from_ratio":
            denom = max(float(latency_w) + float(energy_w), eps)
            alpha_raw = float(latency_w) / denom
            return float(np.clip(alpha_raw, alpha_min, alpha_max)), f"{task_type}_latency_energy_ratio", mode
        if mode != "legacy":
            raise ValueError(f"Unknown scheduler tradeoff mode={mode}")
        return None, "legacy_linear_weights", mode

    def _score_candidate_components(self, c, latency_w, energy_w, risk_w, queue_w=0.0, alpha=None, tradeoff_mode="legacy"):
        norm_l = float(c.get("norm_l", 0.0))
        norm_e = float(c.get("norm_e", 0.0))
        norm_r = float(c.get("norm_risk", 0.0)) if getattr(CFG, "USE_SCORE_RISK", True) else 0.0
        norm_q = float(c.get("norm_queue", 0.0)) if getattr(CFG, "USE_QUEUE_PRESSURE_SCORE", True) else 0.0
        if str(tradeoff_mode).lower() == "legacy":
            service_component = float(latency_w) * norm_l + float(risk_w) * norm_r + float(queue_w) * norm_q
            energy_component = float(energy_w) * norm_e
            score = energy_component + service_component
            latency_energy_component = energy_component + float(latency_w) * norm_l
        else:
            a = float(alpha if alpha is not None else getattr(CFG, "SCHEDULER_TRADEOFF_ALPHA", 0.85))
            latency_component = norm_l
            energy_component = norm_e
            risk_penalty = float(risk_w) * norm_r
            queue_penalty = float(queue_w) * norm_q
            latency_energy_component = a * latency_component + (1.0 - a) * energy_component
            service_component = latency_energy_component  # Deprecated alias kept for old analysis scripts.
            score = latency_energy_component + risk_penalty + queue_penalty
            c["latency_component"] = float(latency_component)
            c["latency_energy_component"] = float(latency_energy_component)
            c["base_latency_energy_score"] = float(latency_energy_component)
            c["risk_penalty"] = float(risk_penalty)
            c["queue_penalty"] = float(queue_penalty)
            c["service_component_deprecated"] = float(service_component)
        c["latency_energy_component"] = float(latency_energy_component)
        c["base_latency_energy_score"] = float(latency_energy_component)
        return float(score), float(service_component), float(energy_component)

    def select_node(self, task, nodes, theta_full):
        """为当前任务选择节点。"""
        latency_weights, energy_weights, theta = split_task_weights(theta_full)
        latency_w = latency_weights.get(task.task_type, 1.0)
        energy_w = energy_weights.get(task.task_type, 1.0)
        risk_w = float(CFG.TASK_RISK_WEIGHTS.get(task.task_type, CFG.DEADLINE_WEIGHT))
        queue_w = 0.0
        alpha, alpha_source, tradeoff_mode = self._resolve_scheduler_alpha(task.task_type, latency_w, energy_w)
        raw_infos = []
        for idx, node in enumerate(nodes):
            raw_infos.append(self._node_score(task, idx, node, latency_w, energy_w, risk_w))
        if getattr(CFG, "USE_DEADLINE_FILTER", False) and task.task_type == "RT":
            feasible = [x for x in raw_infos if x["predicted_finish"] <= task.deadline]
            candidates = feasible if feasible else raw_infos
        else:
            candidates = raw_infos
        if not candidates:
            return 0, [1.0], 0.0
        norm_e, norm_l, norm_r, norm_debug = self._compute_scheduler_norms(task, candidates)
        norm_q = np.zeros(len(candidates), dtype=float)
        for i, c in enumerate(candidates):
            c["norm_e"] = float(norm_e[i])
            c["norm_l"] = float(norm_l[i])
            c["norm_risk"] = float(norm_r[i])
            c["norm_queue"] = float(norm_q[i])
            score, latency_energy_component, energy_component = self._score_candidate_components(
                c, latency_w, energy_w, risk_w, queue_w=queue_w, alpha=alpha, tradeoff_mode=tradeoff_mode
            )
            c["latency_energy_component"] = latency_energy_component
            c["base_latency_energy_score"] = latency_energy_component
            c["service_component"] = latency_energy_component  # Deprecated alias.
            c["energy_component"] = energy_component
            c["score"] = score
        candidate_scores = np.array([c["score"] for c in candidates], dtype=float)
        if len(candidate_scores) == 0 or not np.all(np.isfinite(candidate_scores)):
            choice = 0
            probs = np.ones(max(1, len(candidates)), dtype=float) / max(1, len(candidates))
        else:
            shifted = candidate_scores - np.min(candidate_scores)
            logits = -float(self.beta) * shifted
            logits = np.clip(logits, -60.0, 60.0)
            weights = np.exp(logits)
            if not np.isfinite(weights).all() or np.sum(weights) <= 0:
                choice = int(np.argmin(candidate_scores))
                probs = np.zeros(len(candidates), dtype=float)
                probs[choice] = 1.0
            else:
                probs = weights / np.sum(weights)
                choice = int(self.np_rng.choice(len(candidates), p=probs))
        selected = candidates[choice]
        selected_idx = int(selected["node_idx"])
        self.last_score_debug = {
            "norm_mode": self.norm_mode,
            "task_type": task.task_type,
            "scheduler_tradeoff_mode": tradeoff_mode,
            "scheduler_score_norm_mode": norm_debug.get("scheduler_score_norm_mode", getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")),
            "scheduler_alpha": float(alpha) if alpha is not None else None,
            "scheduler_alpha_source": alpha_source,
            "scheduler_alpha_min": float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60)),
            "scheduler_alpha_max": float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97)),
            "latency_w": float(latency_w),
            "energy_w": float(energy_w),
            "risk_w": float(risk_w),
            "queue_w": float(queue_w),
            **norm_debug,
            "energy_norm_min": float(np.min(norm_e)),
            "energy_norm_max": float(np.max(norm_e)),
            "latency_norm_min": float(np.min(norm_l)),
            "latency_norm_max": float(np.max(norm_l)),
            "risk_norm_min": float(np.min(norm_r)),
            "risk_norm_max": float(np.max(norm_r)),
            "candidate_count": int(len(candidates)),
            "selected_node": int(selected_idx),
            "selected_norm_e": float(selected.get("norm_e", 0.0)),
            "selected_norm_l": float(selected.get("norm_l", 0.0)),
            "selected_norm_risk": float(selected.get("norm_risk", 0.0)),
            "selected_norm_queue": float(selected.get("norm_queue", 0.0)),
            "selected_latency_component": float(selected.get("latency_component", np.nan)),
            "selected_risk_penalty": float(selected.get("risk_penalty", 0.0)),
            "selected_queue_penalty": float(selected.get("queue_penalty", 0.0)),
            "selected_latency_energy_component": float(selected.get("latency_energy_component", np.nan)),
            "selected_base_latency_energy_score": float(selected.get("base_latency_energy_score", np.nan)),
            "selected_service_component": float(selected.get("service_component", np.nan)),
            "selected_service_component_deprecated": float(selected.get("service_component", np.nan)),
            "selected_energy_component": float(selected.get("energy_component", np.nan)),
            "selected_score": float(selected.get("score", np.nan)),
        }
        return selected_idx, probs.tolist(), float(selected.get("score", 0.0))


class ConstrainedBoltzmannScheduler(BoltzmannScheduler):
    """约束 Boltzmann 调度器。

    核心逻辑：
    1) 先筛可行候选集 F：CPU、deadline、云卸载门控；
    2) 再筛机会集合 O：只保留 score 接近最优的节点；
    3) 只在 O 内进行 Boltzmann 随机选择。
    """

    def _queue_pressure(self, task, node):
        capacity_in_task_units = max(1.0, float(node.cpu_total) / max(1.0, float(task.cpu_req)))
        pressure = (len(node.ready_queue) + len(node.running_tasks)) / capacity_in_task_units
        return float(np.clip(pressure, 0.0, float(getattr(CFG, "QUEUE_PRESSURE_CLIP", 3.0))))

    def _local_cloud_pressure(self, nodes):
        edge_nodes = [n for n in nodes if not getattr(n, "is_cloud", False)] or list(nodes)
        avg_util = float(np.mean([n.utilization() for n in edge_nodes])) if edge_nodes else 0.0
        backlog = float(sum(len(n.ready_queue) + len(n.running_tasks) for n in edge_nodes))
        backlog_norm = backlog / max(1.0, float(getattr(CFG, "CLOUD_GATE_BACKLOG_NORM", 24.0)))
        pressure = (
            float(getattr(CFG, "CLOUD_GATE_PRESSURE_UTIL_WEIGHT", 0.7)) * avg_util
            + float(getattr(CFG, "CLOUD_GATE_PRESSURE_BACKLOG_WEIGHT", 0.3)) * backlog_norm
        )
        return float(np.clip(pressure, 0.0, 1.0))

    def _node_score(self, task, node_idx, node, latency_w=None, energy_w=None, risk_w=None):
        # 保留父类原有 energy / latency / deadline_risk 计算。
        info = super()._node_score(task, node_idx, node, latency_w, energy_w, risk_w)
        hard_margin_factor = getattr(CFG, "SAFETY_MARGIN_FACTOR", {"RT": 0.2, "Batch": 0.05, "AI": 0.05}).get(task.task_type, 0.0)
        hard_margin = float(hard_margin_factor) * float(task.duration_base) if getattr(CFG, "USE_SAFETY_MARGIN_FILTER", True) else 0.0
        info["hard_margin"] = float(hard_margin)
        info["queue_pressure"] = self._queue_pressure(task, node)
        info["utilization"] = float(node.utilization())
        info["is_cloud"] = bool(getattr(node, "is_cloud", False))
        info["cpu_feasible"] = bool(task.cpu_req <= node.cpu_total)
        return info

    def _apply_feasibility_filter(self, task, nodes, raw_infos, controls):
        debug = {
            "feasibility_enabled": bool(getattr(CFG, "USE_FEASIBILITY_FILTER", True)),
            "raw_candidate_count": int(len(raw_infos)),
            "after_cpu_count": int(len(raw_infos)),
            "after_cloud_gate_count": int(len(raw_infos)),
            "after_deadline_count": int(len(raw_infos)),
            "fallback_reason": None,
            "cloud_pressure": None,
            "cloud_gate": float(controls.get("cloud_gate", getattr(CFG, "CLOUD_GATE_DEFAULT", 0.5))),
        }
        if not getattr(CFG, "USE_FEASIBILITY_FILTER", True):
            return list(raw_infos), debug

        candidates = list(raw_infos)

        if getattr(CFG, "USE_HARD_CPU_FILTER", True):
            candidates = [c for c in candidates if c.get("cpu_feasible", True)]
            debug["after_cpu_count"] = int(len(candidates))
            if not candidates:
                best = min(raw_infos, key=lambda c: c.get("latency_total", 1e18))
                debug["fallback_reason"] = "no_cpu_feasible"
                return [best], debug

        if getattr(CFG, "USE_CLOUD_GATE", True):
            pressure = self._local_cloud_pressure(nodes)
            debug["cloud_pressure"] = float(pressure)
            gate = float(controls.get("cloud_gate", getattr(CFG, "CLOUD_GATE_DEFAULT", 0.50)))
            allow_task = bool(getattr(CFG, "CLOUD_GATE_ALLOW_TASKS", {}).get(task.task_type, True))
            edge_candidates = [c for c in candidates if not c.get("is_cloud", False)]
            cloud_candidates = [c for c in candidates if c.get("is_cloud", False)]
            allow_cloud = allow_task and (pressure >= gate)
            candidates = edge_candidates + cloud_candidates if allow_cloud else edge_candidates
            if not candidates and getattr(CFG, "CLOUD_GATE_ALWAYS_ALLOW_IF_NO_EDGE_FEASIBLE", True):
                candidates = edge_candidates + cloud_candidates
                debug["fallback_reason"] = "cloud_gate_relaxed_no_edge_candidate"
            debug["after_cloud_gate_count"] = int(len(candidates))

        hard_deadline_map = getattr(CFG, "HARD_DEADLINE_TASKS", {"RT": True})
        use_hard_deadline = bool(getattr(CFG, "USE_HARD_DEADLINE_FILTER", True)) and bool(hard_deadline_map.get(task.task_type, False))
        use_hard_deadline = use_hard_deadline or (getattr(CFG, "USE_DEADLINE_FILTER", False) and task.task_type == "RT")
        if use_hard_deadline:
            before_deadline = list(candidates)
            feasible = [
                c for c in candidates
                if c["predicted_finish"] <= task.deadline - float(c.get("hard_margin", 0.0)) * float(controls.get("safety_margin_scale", 1.0))
            ]
            if feasible:
                candidates = feasible
            else:
                mode = str(getattr(CFG, "HARD_DEADLINE_FALLBACK", "min_violation")).lower()
                if mode == "relax_deadline":
                    candidates = before_deadline
                    debug["fallback_reason"] = "deadline_relaxed"
                else:
                    # 关键：RT 无可行节点时不随机，直接选违约最小节点。
                    best = min(before_deadline, key=lambda c: (max(0.0, c["predicted_finish"] - task.deadline), c["latency_total"]))
                    candidates = [best]
                    debug["fallback_reason"] = "deadline_min_violation"
            debug["after_deadline_count"] = int(len(candidates))
        else:
            debug["after_deadline_count"] = int(len(candidates))

        return candidates, debug

    def _compute_norms_with_queue(self, task, candidates):
        norm_e, norm_l, norm_r, norm_debug = self._compute_scheduler_norms(task, candidates)
        if getattr(CFG, "USE_QUEUE_PRESSURE_SCORE", True):
            q_vals = np.array([c.get("queue_pressure", 0.0) for c in candidates], dtype=float)
            norm_q = np.clip(q_vals, 0.0, float(getattr(CFG, "QUEUE_PRESSURE_CLIP", 3.0)))
            norm_debug["queue_ref"] = "task_scaled_queue_pressure"
        else:
            norm_q = np.zeros(len(candidates), dtype=float)
            norm_debug["queue_ref"] = "disabled"
        return norm_e, norm_l, norm_r, norm_q, norm_debug

    def _apply_opportunity_window(self, candidates, controls):
        debug = {
            "opportunity_enabled": bool(getattr(CFG, "USE_OPPORTUNITY_WINDOW", True)),
            "rho": float(controls.get("rho", getattr(CFG, "OPPORTUNITY_RHO_DEFAULT", 1.0))),
            "before_opportunity_count": int(len(candidates)),
            "after_opportunity_count": int(len(candidates)),
            "opportunity_threshold": None,
        }
        if not getattr(CFG, "USE_OPPORTUNITY_WINDOW", True) or len(candidates) <= 1:
            return list(candidates), debug

        scores = np.array([c["score"] for c in candidates], dtype=float)
        min_score = float(np.min(scores))
        rho = max(0.0, float(controls.get("rho", getattr(CFG, "OPPORTUNITY_RHO_DEFAULT", 1.0))))
        mode = str(getattr(CFG, "OPPORTUNITY_MODE", "std")).lower()
        if mode == "absolute":
            threshold = min_score + rho
        else:
            scale = max(float(np.std(scores)), float(getattr(CFG, "OPPORTUNITY_ABS_FLOOR", 1e-6)))
            threshold = min_score + rho * scale

        opp = [c for c in candidates if c["score"] <= threshold]
        min_keep = max(1, int(getattr(CFG, "OPPORTUNITY_MIN_CANDIDATES", 2)))
        if len(opp) < min_keep:
            opp = sorted(candidates, key=lambda c: c["score"])[:min(min_keep, len(candidates))]
        debug["after_opportunity_count"] = int(len(opp))
        debug["opportunity_threshold"] = float(threshold)
        return opp, debug

    def select_node(self, task, nodes, theta_full):
        latency_weights, energy_weights, theta = split_task_weights(theta_full)
        controls = extract_scheduler_controls(theta_full)
        self.beta = float(controls.get("beta", self.beta))

        latency_w = latency_weights.get(task.task_type, 1.0)
        energy_w = energy_weights.get(task.task_type, 1.0)
        base_risk_w = float(CFG.TASK_RISK_WEIGHTS.get(task.task_type, CFG.DEADLINE_WEIGHT))
        risk_w = base_risk_w * float(controls.get("risk_scale", getattr(CFG, "RISK_SCALE_DEFAULT", 1.0)))
        queue_w = float(controls.get("queue_w", getattr(CFG, "QUEUE_WEIGHT_DEFAULT", 1.0))) if getattr(CFG, "USE_QUEUE_PRESSURE_SCORE", True) else 0.0
        alpha, alpha_source, tradeoff_mode = self._resolve_scheduler_alpha(task.task_type, latency_w, energy_w)

        raw_infos = [self._node_score(task, idx, node, latency_w, energy_w, risk_w) for idx, node in enumerate(nodes)]
        candidates, feasibility_debug = self._apply_feasibility_filter(task, nodes, raw_infos, controls)
        if not candidates:
            candidates = [min(raw_infos, key=lambda c: c.get("latency_total", 1e18))]
            feasibility_debug["fallback_reason"] = "empty_after_all_filters"

        norm_e, norm_l, norm_r, norm_q, norm_debug = self._compute_norms_with_queue(task, candidates)
        for i, c in enumerate(candidates):
            c["norm_e"] = float(norm_e[i])
            c["norm_l"] = float(norm_l[i])
            c["norm_risk"] = float(norm_r[i])
            c["norm_queue"] = float(norm_q[i])
            score, latency_energy_component, energy_component = self._score_candidate_components(
                c, latency_w, energy_w, risk_w, queue_w=queue_w, alpha=alpha, tradeoff_mode=tradeoff_mode
            )
            c["latency_energy_component"] = latency_energy_component
            c["base_latency_energy_score"] = latency_energy_component
            c["service_component"] = latency_energy_component  # Deprecated alias.
            c["energy_component"] = energy_component
            c["score"] = score

        opportunity_candidates, opportunity_debug = self._apply_opportunity_window(candidates, controls)
        candidate_scores = np.array([c["score"] for c in opportunity_candidates], dtype=float)

        if len(candidate_scores) == 0 or not np.all(np.isfinite(candidate_scores)):
            choice = 0
            probs = np.ones(max(1, len(opportunity_candidates)), dtype=float) / max(1, len(opportunity_candidates))
        elif not getattr(CFG, "USE_BOLTZMANN_RANDOM", True):
            choice = int(np.argmin(candidate_scores))
            probs = np.zeros(len(opportunity_candidates), dtype=float)
            probs[choice] = 1.0
        else:
            shifted = candidate_scores - np.min(candidate_scores)
            logits = -float(self.beta) * shifted
            logits = np.clip(logits, -60.0, 60.0)
            weights = np.exp(logits)
            if not np.isfinite(weights).all() or np.sum(weights) <= 0:
                choice = int(np.argmin(candidate_scores))
                probs = np.zeros(len(opportunity_candidates), dtype=float)
                probs[choice] = 1.0
            else:
                probs = weights / np.sum(weights)
                choice = int(self.np_rng.choice(len(opportunity_candidates), p=probs))

        selected = opportunity_candidates[choice]
        selected_idx = int(selected["node_idx"])
        full_probs = [0.0 for _ in nodes]
        for i, c in enumerate(opportunity_candidates):
            full_probs[int(c["node_idx"])] = float(probs[i])

        all_scores = np.array([c.get("score", np.nan) for c in candidates], dtype=float)
        self.last_score_debug = {
            "norm_mode": self.norm_mode,
            "task_type": task.task_type,
            "beta": float(self.beta),
            "latency_w": float(latency_w),
            "energy_w": float(energy_w),
            "risk_w": float(risk_w),
            "queue_w": float(queue_w),
            "scheduler_tradeoff_mode": tradeoff_mode,
            "scheduler_score_norm_mode": norm_debug.get("scheduler_score_norm_mode", getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")),
            "scheduler_alpha": float(alpha) if alpha is not None else None,
            "scheduler_alpha_source": alpha_source,
            "scheduler_alpha_min": float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60)),
            "scheduler_alpha_max": float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97)),
            **norm_debug,
            **feasibility_debug,
            **opportunity_debug,
            "candidate_count": int(len(candidates)),
            "opportunity_candidate_count": int(len(opportunity_candidates)),
            "selected_node": int(selected_idx),
            "selected_score": float(selected.get("score", np.nan)),
            "selected_deadline_risk": float(selected.get("deadline_risk", 0.0)),
            "selected_slack": float(selected.get("slack", 0.0)),
            "selected_is_cloud": bool(selected.get("is_cloud", False)),
            "selected_norm_e": float(selected.get("norm_e", 0.0)),
            "selected_norm_l": float(selected.get("norm_l", 0.0)),
            "selected_norm_risk": float(selected.get("norm_risk", 0.0)),
            "selected_norm_queue": float(selected.get("norm_queue", 0.0)),
            "selected_latency_component": float(selected.get("latency_component", np.nan)),
            "selected_risk_penalty": float(selected.get("risk_penalty", 0.0)),
            "selected_queue_penalty": float(selected.get("queue_penalty", 0.0)),
            "selected_latency_energy_component": float(selected.get("latency_energy_component", np.nan)),
            "selected_base_latency_energy_score": float(selected.get("base_latency_energy_score", np.nan)),
            "selected_service_component": float(selected.get("service_component", np.nan)),
            "selected_service_component_deprecated": float(selected.get("service_component", np.nan)),
            "selected_energy_component": float(selected.get("energy_component", np.nan)),
            "score_min": float(np.nanmin(all_scores)) if len(all_scores) else 0.0,
            "score_max": float(np.nanmax(all_scores)) if len(all_scores) else 0.0,
        }
        return selected_idx, full_probs, float(selected.get("score", 0.0))


class DirectHeuristicScheduler:
    """Direct non-BO scheduling baselines.

    Modes:
    - round_robin_direct: cyclic node assignment with feasibility scan.
    - greedy_direct_cost: immediate cost greedy.
    - least_load_direct: projected finish / queue load greedy.
    - queue_aware_greedy_direct: drift-plus-penalty style queue-aware greedy.

    These baselines intentionally do not call BO/CBO. They reuse the same
    event-driven simulator, node energy/latency estimates, and output pipeline
    as the existing fixed/BO/CBO methods.
    """

    def __init__(self, mode="round_robin_direct", np_rng=None):
        self.mode = str(mode or "round_robin_direct").strip().lower().replace("-", "_")
        self.idx = 0
        self.current_time = 0.0
        self.last_score_debug = {}
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng()

    def update_beta(self, e):
        return

    def _candidate_info(self, task, nodes, node_idx):
        node = nodes[int(node_idx)]
        origin = task.origin_node_id if getattr(task, "origin_node_id", -1) >= 0 else int(node_idx)
        try:
            energy_raw, latency_total = node.estimate_metrics(task, self.current_time, origin_node_idx=origin)
        except Exception:
            energy_raw, latency_total = float("inf"), float("inf")

        predicted_finish = float(self.current_time) + float(latency_total)
        lateness = max(0.0, predicted_finish - float(task.deadline))
        deadline_risk = 1.0 if predicted_finish > float(task.deadline) else 0.0
        # Smooth pressure near deadline; consistent with existing scheduler logic.
        try:
            slack = float(task.deadline) - predicted_finish
            margin_factor = getattr(CFG, "RISK_MARGIN_FACTOR", {"RT": 1.0, "Batch": 0.3, "AI": 0.3}).get(task.task_type, 0.3)
            safe_margin = max(1.0, float(margin_factor) * float(task.duration_base))
            soft_risk = max(0.0, (safe_margin - slack) / safe_margin)
            soft_risk = min(soft_risk, float(getattr(CFG, "RISK_CLIP_MAX", 3.0)))
        except Exception:
            soft_risk = deadline_risk

        queue_len = len(getattr(node, "ready_queue", [])) + len(getattr(node, "running_tasks", []))
        capacity_in_task_units = max(1.0, float(getattr(node, "cpu_total", 1.0)) / max(1.0, float(getattr(task, "cpu_req", 1.0))))
        queue_pressure = float(np.clip(queue_len / capacity_in_task_units, 0.0, float(getattr(CFG, "QUEUE_PRESSURE_CLIP", 3.0))))
        projected_backlog = float(queue_len + 1)
        cpu_feasible = bool(float(getattr(task, "cpu_req", 1.0)) <= float(getattr(node, "cpu_total", 1.0)))

        return {
            "node_idx": int(node_idx),
            "energy_raw": float(energy_raw),
            "latency_total": float(latency_total),
            "predicted_finish": float(predicted_finish),
            "lateness": float(lateness),
            "deadline_risk": float(deadline_risk),
            "soft_deadline_risk": float(soft_risk),
            "queue_len": int(queue_len),
            "queue_pressure": float(queue_pressure),
            "projected_backlog": float(projected_backlog),
            "cpu_feasible": bool(cpu_feasible),
            "is_cloud": bool(getattr(node, "is_cloud", False)),
        }

    def _score(self, c):
        mode = self.mode
        if mode in {"greedy_direct_cost", "greedy_cost", "direct_greedy_cost"}:
            return (
                float(c["energy_raw"])
                + float(getattr(CFG, "ALPHA_LATENCY", 100.0)) * float(c["latency_total"])
                + float(getattr(CFG, "LATE_PENALTY_WEIGHT", 300.0)) * float(c.get("lateness", 0.0))
                + float(getattr(CFG, "SLA_PENALTY_WEIGHT", 1500.0)) * float(c.get("deadline_risk", 0.0))
            )

        if mode in {"least_load_direct", "least_load", "direct_least_load"}:
            # Primary objective is projected finish/load. Energy only breaks ties.
            return (
                float(c["predicted_finish"])
                + 0.01 * float(c.get("queue_len", 0.0))
                + 1e-6 * float(c.get("energy_raw", 0.0))
            )

        if mode in {"queue_aware_greedy_direct", "queue_aware_greedy", "direct_queue_aware_greedy", "dpp_greedy"}:
            return (
                float(c["energy_raw"])
                + float(getattr(CFG, "ALPHA_LATENCY", 100.0)) * float(c["latency_total"])
                + float(getattr(CFG, "BACKLOG_WEIGHT", 200.0)) * float(c.get("queue_pressure", c.get("projected_backlog", 0.0)))
                + float(getattr(CFG, "LATE_PENALTY_WEIGHT", 300.0)) * float(c.get("lateness", 0.0))
                + float(getattr(CFG, "SLA_PENALTY_WEIGHT", 1500.0)) * float(c.get("soft_deadline_risk", c.get("deadline_risk", 0.0)))
            )

        # Default fallback behaves like immediate cost greedy.
        return float(c["energy_raw"]) + float(getattr(CFG, "ALPHA_LATENCY", 100.0)) * float(c["latency_total"])

    def select_node(self, task, nodes, theta_full=None):
        if not nodes:
            return 0, [1.0], 0.0

        raw_infos = [self._candidate_info(task, nodes, idx) for idx in range(len(nodes))]
        feasible = [c for c in raw_infos if c.get("cpu_feasible", True)]
        candidates = feasible if feasible else raw_infos

        if self.mode in {"round_robin_direct", "round_robin", "rr", "roundrobin"}:
            n = len(raw_infos)
            selected = None
            start = int(self.idx) % max(1, n)
            for k in range(n):
                cand = raw_infos[(start + k) % n]
                if cand.get("cpu_feasible", True):
                    selected = cand
                    self.idx = (int(cand["node_idx"]) + 1) % n
                    break
            if selected is None:
                selected = raw_infos[start]
                self.idx = (start + 1) % n
            score = 0.0
        else:
            for c in candidates:
                c["score"] = float(self._score(c))
            selected = min(
                candidates,
                key=lambda c: (
                    float(c.get("score", float("inf"))),
                    float(c.get("latency_total", float("inf"))),
                    float(c.get("energy_raw", float("inf"))),
                    int(c.get("node_idx", 0)),
                ),
            )
            score = float(selected.get("score", 0.0))

        selected_idx = int(selected.get("node_idx", 0))
        probs = [0.0 for _ in nodes]
        if 0 <= selected_idx < len(probs):
            probs[selected_idx] = 1.0

        score_vals = [float(c.get("score", np.nan)) for c in candidates]
        self.last_score_debug = {
            "scheduler_mode": str(self.mode),
            "direct_baseline": True,
            "task_type": getattr(task, "task_type", ""),
            "candidate_count": int(len(candidates)),
            "raw_candidate_count": int(len(raw_infos)),
            "selected_node": int(selected_idx),
            "selected_score": float(score),
            "selected_latency": float(selected.get("latency_total", np.nan)),
            "selected_energy": float(selected.get("energy_raw", np.nan)),
            "selected_queue_pressure": float(selected.get("queue_pressure", 0.0)),
            "selected_deadline_risk": float(selected.get("soft_deadline_risk", selected.get("deadline_risk", 0.0))),
            "score_min": float(np.nanmin(score_vals)) if score_vals else float(score),
            "score_max": float(np.nanmax(score_vals)) if score_vals else float(score),
        }
        return selected_idx, probs, float(score)


class RoundRobinScheduler(DirectHeuristicScheduler):
    def __init__(self):
        super().__init__(mode="round_robin_direct")

# ==========================================
# 4. 代理与联邦逻辑 (Agent & Federated)
# ==========================================
class StateLevel(Enum):
    """离散情景等级。"""
    LOW = 0
    MID = 1
    HIGH = 2

@dataclass(frozen=True)
class StateSignature:
    """离散情景签名 = [负载等级, 时延等级, 到达等级]。"""
    load_level: StateLevel
    delay_level: StateLevel
    arrival_level: StateLevel
    def __repr__(self):
        return f"S({self.load_level.name},{self.delay_level.name},{self.arrival_level.name})"
    def __hash__(self):
        return hash((self.load_level, self.delay_level, self.arrival_level))

@dataclass
class WindowSnapshot:
    """单个 BO 窗口内的统计快照。

    每个窗口结束后，会把这些统计量转成 metrics/cost/reward，
    再反馈给 BO。
    """
    start_time: float
    end_time: float

    arrivals: int = 0
    completed: int = 0

    rt_arrivals: int = 0
    batch_arrivals: int = 0
    ai_arrivals: int = 0

    # total_energy：用于优化目标的窗口能耗，默认 = active idle + dynamic compute + transmission。
    # total_energy_real：诊断用真实能耗，默认按所有节点常开 idle + dynamic compute + transmission。
    total_energy: float = 0.0
    total_energy_real: float = 0.0
    compute_dynamic_energy: float = 0.0
    compute_idle_energy: float = 0.0
    compute_energy_real: float = 0.0
    transmission_energy: float = 0.0
    total_delay: float = 0.0
    total_vio: int = 0

    early_sum: float = 0.0
    late_sum: float = 0.0

    rt_delay_sum: float = 0.0
    rt_completed: int = 0

    batch_delay_sum: float = 0.0
    batch_completed: int = 0

    ai_delay_sum: float = 0.0
    ai_completed: int = 0

    # Per-class window diagnostics. These are not separate final objectives;
    # they are exported to explain which task type improves/degrades.
    rt_late_sum: float = 0.0
    batch_late_sum: float = 0.0
    ai_late_sum: float = 0.0
    rt_vio_count: int = 0
    batch_vio_count: int = 0
    ai_vio_count: int = 0
    rt_energy_sum: float = 0.0
    batch_energy_sum: float = 0.0
    ai_energy_sum: float = 0.0

    cpu_demand: float = 0.0
    data_demand: float = 0.0

    alloc_counts: List[int] = field(default_factory=list)

    def duration(self) -> float:
        return max(1e-9, self.end_time - self.start_time)

    def avg_delay(self) -> float:
        return self.total_delay / self.completed if self.completed > 0 else 0.0

    def avg_delay_rt(self) -> float:
        return self.rt_delay_sum / self.rt_completed if self.rt_completed > 0 else 0.0

    def avg_delay_batch(self) -> float:
        return self.batch_delay_sum / self.batch_completed if self.batch_completed > 0 else 0.0

    def avg_delay_ai(self) -> float:
        return self.ai_delay_sum / self.ai_completed if self.ai_completed > 0 else 0.0

    def avg_lateness_rt(self) -> float:
        return self.rt_late_sum / self.rt_completed if self.rt_completed > 0 else 0.0

    def avg_lateness_batch(self) -> float:
        return self.batch_late_sum / self.batch_completed if self.batch_completed > 0 else 0.0

    def avg_lateness_ai(self) -> float:
        return self.ai_late_sum / self.ai_completed if self.ai_completed > 0 else 0.0

    def vio_rate_rt(self) -> float:
        return self.rt_vio_count / self.rt_completed if self.rt_completed > 0 else 0.0

    def vio_rate_batch(self) -> float:
        return self.batch_vio_count / self.batch_completed if self.batch_completed > 0 else 0.0

    def vio_rate_ai(self) -> float:
        return self.ai_vio_count / self.ai_completed if self.ai_completed > 0 else 0.0

    def completion_ratio_rt(self) -> float:
        return self.rt_completed / max(1, self.rt_arrivals)

    def completion_ratio_batch(self) -> float:
        return self.batch_completed / max(1, self.batch_arrivals)

    def completion_ratio_ai(self) -> float:
        return self.ai_completed / max(1, self.ai_arrivals)

    def avg_energy_rt(self) -> float:
        return self.rt_energy_sum / self.rt_completed if self.rt_completed > 0 else 0.0

    def avg_energy_batch(self) -> float:
        return self.batch_energy_sum / self.batch_completed if self.batch_completed > 0 else 0.0

    def avg_energy_ai(self) -> float:
        return self.ai_energy_sum / self.ai_completed if self.ai_completed > 0 else 0.0

    def _window_class_cost(self, task_type: str) -> Optional[float]:
        prefix = "rt" if task_type == "RT" else ("ai" if task_type == "AI" else "batch")
        arrivals = int(getattr(self, f"{prefix}_arrivals", 0))
        completed = int(getattr(self, f"{prefix}_completed", 0))
        if arrivals <= 0:
            return None
        avg_delay = float(getattr(self, f"avg_delay_{prefix}")())
        avg_late = float(getattr(self, f"avg_lateness_{prefix}")())
        vio_rate = float(getattr(self, f"vio_rate_{prefix}")())
        comp_ratio = completed / max(1, arrivals)
        unfinished_weight = float(getattr(CFG, "DUAL_CLASS_UNFINISHED_WEIGHT", getattr(CFG, "COHORT_UNFINISHED_PENALTY", 1000.0)))
        energy_term = float(getattr(self, f"avg_energy_{prefix}")())
        return (
            energy_term
            + float(CFG.ALPHA_LATENCY) * avg_delay
            + float(CFG.SLA_PENALTY_WEIGHT) * vio_rate
            + float(CFG.LATE_PENALTY_WEIGHT) * avg_late
            + unfinished_weight * (1.0 - comp_ratio)
        )

    def vio_rate(self) -> float:
        return self.total_vio / self.completed if self.completed > 0 else 0.0

    def avg_earliness(self) -> float:
        return self.early_sum / self.completed if self.completed > 0 else 0.0

    def avg_lateness(self) -> float:
        return self.late_sum / self.completed if self.completed > 0 else 0.0

    def avg_energy(self) -> float:
        return self.total_energy / self.completed if self.completed > 0 else 0.0

    def arrival_rate(self) -> float:
        return self.arrivals / self.duration()

    def completion_rate(self) -> float:
        return self.completed / self.duration()

    def sla_success_rate(self) -> float:
        return 1.0 - self.vio_rate()

    def to_metrics(self, cumulative_energy: float, backlog: int = 0, cumulative_energy_real: Optional[float] = None) -> Dict[str, Any]:
        """把窗口统计转换成 cost / reward 和各种监控指标。"""
        n = max(1, self.completed)
        backlog = max(0, int(backlog))
        avg_delay = self.avg_delay()
        vio_rate = self.vio_rate()
        avg_early = self.avg_earliness()
        avg_late = self.avg_lateness()
        zero_completion_penalty = CFG.ZERO_COMPLETION_PENALTY if self.arrivals > 0 and self.completed == 0 else 0.0

        early_bonus = 0.0
        if getattr(CFG, "USE_EARLY_BONUS", False):
            # 提前完成奖励封顶，避免 Batch/AI 的长 deadline slack 无限放大 reward。
            early_bonus = CFG.EARLY_BONUS_WEIGHT * min(avg_early, float(getattr(CFG, "EARLY_BONUS_CAP", 5.0)))

        cost = (
            self.total_energy / n
            + CFG.ALPHA_LATENCY * avg_delay
            + CFG.SLA_PENALTY_WEIGHT * vio_rate
            + CFG.LATE_PENALTY_WEIGHT * avg_late
            - early_bonus
            + CFG.BACKLOG_WEIGHT * backlog
            + zero_completion_penalty
        )

        return {
            "cost": cost,
            "reward": -cost,
            "total_energy": self.total_energy,
            "total_energy_real": self.total_energy_real,
            "compute_dynamic_energy": self.compute_dynamic_energy,
            "compute_idle_energy": self.compute_idle_energy,
            "compute_energy_real": self.compute_energy_real,
            "transmission_energy": self.transmission_energy,
            "cumulative_energy": cumulative_energy,
            "cumulative_energy_real": cumulative_energy if cumulative_energy_real is None else cumulative_energy_real,
            "avg_energy": self.avg_energy(),
            "avg_delay": avg_delay,
            "avg_delay_rt": self.avg_delay_rt(),
            "avg_delay_batch": self.avg_delay_batch(),
            "avg_delay_ai": self.avg_delay_ai(),

            # Per-class window diagnostics for explanation and debugging.
            "avg_lateness_rt": self.avg_lateness_rt(),
            "avg_lateness_batch": self.avg_lateness_batch(),
            "avg_lateness_ai": self.avg_lateness_ai(),
            "vio_rate_rt": self.vio_rate_rt(),
            "vio_rate_batch": self.vio_rate_batch(),
            "vio_rate_ai": self.vio_rate_ai(),
            "completion_ratio_rt": self.completion_ratio_rt(),
            "completion_ratio_batch": self.completion_ratio_batch(),
            "completion_ratio_ai": self.completion_ratio_ai(),
            "avg_energy_rt": self.avg_energy_rt(),
            "avg_energy_batch": self.avg_energy_batch(),
            "avg_energy_ai": self.avg_energy_ai(),
            "window_rt_class_cost": self._window_class_cost("RT"),
            "window_batch_class_cost": self._window_class_cost("Batch"),
            "window_ai_class_cost": self._window_class_cost("AI"),
            "vio_rate": vio_rate,
            "sla_success_rate": self.sla_success_rate(),
            "avg_earliness": avg_early,
            "avg_lateness": avg_late,
            "arrival_rate": self.arrival_rate(),
            "completion_rate": self.completion_rate(),
            "rt_arrival_ratio": self.rt_arrivals / max(1, self.arrivals),
            "batch_arrival_ratio": self.batch_arrivals / max(1, self.arrivals),
            "ai_arrival_ratio": self.ai_arrivals / max(1, self.arrivals),
            "task_count": self.completed,
            "arrivals_total": self.arrivals,
            "arrivals_rt": self.rt_arrivals,
            "arrivals_batch": self.batch_arrivals,
            "arrivals_ai": self.ai_arrivals,
            "completed_total": self.completed,
            "completed_rt": self.rt_completed,
            "completed_batch": self.batch_completed,
            "completed_ai": self.ai_completed,
            "backlog": backlog,
            "unfinished_end": backlog,
            "zero_completion_penalty": zero_completion_penalty,
            "early_bonus_applied": early_bonus,
            "cpu_demand": self.cpu_demand,
            "data_demand": self.data_demand,
            "alloc": list(self.alloc_counts),
            "alloc_by_type": getattr(self, "alloc_by_type", {}),
        }



@dataclass
class CohortRecord:
    """任务批次级反馈记录。

    一个 cohort 对应一个 BO 窗口中“新到达并由同一组 theta 调度”的任务集合。
    任务可以跨窗口完成，但最终完成结果会回填到该 cohort，而不是回填到完成时所在窗口。
    """
    cohort_id: int
    theta_control: List[float]
    theta_full: List[float]
    context: Optional[List[float]]
    state: Any
    start_time: float
    window_index: int
    label: str = ""
    total_tasks: int = 0
    completed_tasks: int = 0
    assigned_energy_sum: float = 0.0
    completed_delay_sum: float = 0.0
    completed_lateness_sum: float = 0.0
    completed_violation_count: int = 0
    rt_arrivals: int = 0
    batch_arrivals: int = 0
    ai_arrivals: int = 0
    rt_completed: int = 0
    batch_completed: int = 0
    ai_completed: int = 0

    # Per-class cohort diagnostics for refined feedback and explanation.
    rt_delay_sum: float = 0.0
    batch_delay_sum: float = 0.0
    ai_delay_sum: float = 0.0
    rt_lateness_sum: float = 0.0
    batch_lateness_sum: float = 0.0
    ai_lateness_sum: float = 0.0
    rt_vio_count: int = 0
    batch_vio_count: int = 0
    ai_vio_count: int = 0
    rt_assigned_energy_sum: float = 0.0
    batch_assigned_energy_sum: float = 0.0
    ai_assigned_energy_sum: float = 0.0
    rt_estimated_latency_sum: float = 0.0
    batch_estimated_latency_sum: float = 0.0
    ai_estimated_latency_sum: float = 0.0

    unfinished_tasks: Dict[str, Any] = field(default_factory=dict)
    pending_task_time_area: float = 0.0
    last_area_update_time: float = 0.0
    finalized: bool = False
    finalize_time: Optional[float] = None
    finalize_reason: str = ""
    feedback_cost: Optional[float] = None
    confidence: float = 0.0

class FederatedBOAgent:
    """BO 代理。

    作用：
    1) 记住历史样本 (theta, cost, state/context)；
    2) 拟合 GP；
    3) 给下一轮提出新的 theta。

    当 use_context=True 时，输入特征 = theta + context。
    当 use_trust_region=True 时，会围绕历史较优点附近加强局部搜索。
    """
    def __init__(self, dim=None, beta=3.0, py_rng=None, torch_gen=None, bounds=None, feature_names=None,
                 use_context=False, use_trust_region=False, context_dim=0, context_bounds=None,
                 anchor_points=None, neighbor_k=None, use_state_partition=False):
        self.dim = dim if dim is not None else CFG.DIM_THETA
        self.beta_init = beta
        self.local_recent = collections.deque(maxlen=CFG.RECENT_HISTORY)
        self.local_archive = collections.defaultdict(list)
        self.py_rng = py_rng if py_rng is not None else random.Random()
        self.torch_gen = torch_gen if torch_gen is not None else torch.Generator().manual_seed(
            self.py_rng.randrange(1, 2 ** 31 - 1)
        )
        if bounds is None:
            bounds = get_control_bounds(self.dim)
        self.bounds = torch.tensor(bounds, dtype=torch.double)
        self.feature_names = list(feature_names) if feature_names is not None else list(CFG.FEATURE_NAMES[:self.dim])
        self.use_context = bool(use_context)
        # use_state_partition=True 表示：即使不用连续 context，也仍按离散 state 分桶记忆。
        # 这样适合做“弱情景”版本；若要真正无情景基线，应设为 False。
        self.use_state_partition = bool(use_state_partition)
        self.use_trust_region = bool(use_trust_region)
        self.context_dim = int(context_dim) if use_context else 0
        if self.use_context:
            if context_bounds is None:
                context_bounds = CFG.CONTEXT_BOUNDS
            self.context_bounds = torch.tensor(context_bounds, dtype=torch.double)
            if self.context_bounds.shape[-1] != self.context_dim:
                self.context_dim = int(self.context_bounds.shape[-1])
        else:
            self.context_bounds = None
        self.anchor_points = [self._normalize_theta(p) for p in (anchor_points or [])]
        self.neighbor_k = int(neighbor_k if neighbor_k is not None else CFG.CONTEXT_KNN)
        self.step_count = 0
        self.acq_history = []
        self.prev_best = None
        self.prev_best_value = None
        self.prev_best_iter = None
        self.trust_radius = float(CFG.TRUST_RADIUS_INIT)
        self.last_theta = None
        self.theta_momentum = 0.0
        self.last_debug_info = {}
        self._last_similarity_pool = []

    def _sample_in_bounds(self, low, high):
        return [low[d] + (high[d] - low[d]) * self.py_rng.random() for d in range(self.dim)]

    def _next_torch_seed(self) -> int:
        return self.py_rng.randrange(1, 2 ** 31 - 1)

    def _normalize_theta(self, theta):
        if isinstance(theta, (list, tuple, np.ndarray)):
            t = list(theta)
            if len(t) < self.dim:
                t = t + [t[-1]] * (self.dim - len(t))
            else:
                t = t[:self.dim]
            return [float(v) for v in t]
        return theta

    def _normalize_context(self, context):
        if not self.use_context:
            return []
        if context is None:
            return [0.0] * self.context_dim
        c = list(context)
        if len(c) < self.context_dim:
            c = c + [0.0] * (self.context_dim - len(c))
        else:
            c = c[:self.context_dim]
        lows = self.context_bounds[0].tolist()
        highs = self.context_bounds[1].tolist()
        out = []
        for i, v in enumerate(c):
            out.append(float(min(max(v, lows[i]), highs[i])))
        return out

    def _context_similarity(self, context_a, context_b):
        """计算两个连续情景之间的相似度。

        默认使用高斯核（RBF）：
            sim = exp(-0.5 * sum(((a-b)/l)^2))

        之所以默认选高斯核，而不是更“退化”的距离倒数：
        1) 当前情景是连续变量，且已做归一化；
        2) 高斯核能更平滑地区分“很像 / 有点像 / 完全不像”；
        3) 在 7 维情景下，用高斯核做权重，比硬阈值或简单倒数更稳。

        如果后续更在意速度，可以把 CFG.CONTEXT_SIMILARITY_MODE 改成 "inverse_distance"。
        """
        if not self.use_context:
            return 1.0

        a = np.array(self._normalize_context(context_a), dtype=float)
        b = np.array(self._normalize_context(context_b), dtype=float)
        if a.size == 0 or b.size == 0:
            return 0.0

        mode = str(getattr(CFG, "CONTEXT_SIMILARITY_MODE", "gaussian")).lower()
        diff = a - b

        if mode == "inverse_distance":
            dist = float(np.linalg.norm(diff))
            return 1.0 / (1.0 + dist)

        lengths = np.array(getattr(CFG, "CONTEXT_KERNEL_LENGTHS", [0.25] * len(a)), dtype=float)
        if lengths.size != a.size:
            lengths = np.full(a.shape, 0.25, dtype=float)
        lengths = np.maximum(lengths, 1e-6)
        scaled = diff / lengths
        sq_norm = float(np.dot(scaled, scaled))
        return float(np.exp(-0.5 * sq_norm))

    def _pack_sample(self, theta, y_val, state=None, context=None):
        return {
            "theta": self._normalize_theta(theta),
            "y": float(y_val),
            "state": state,
            "context": self._normalize_context(context) if self.use_context else None,
        }

    def _unpack_sample(self, sample):
        if isinstance(sample, dict):
            return sample
        if isinstance(sample, tuple):
            if len(sample) == 4:
                theta, y, state, context = sample
            elif len(sample) == 3:
                theta, y, state = sample
                context = None
            else:
                theta, y = sample[:2]
                state, context = None, None
            return self._pack_sample(theta, y, state=state, context=context)
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    def _archive_state_sample(self, state, sample):
        rec = self._unpack_sample(sample)
        key = state if state is not None else "GLOBAL"
        bucket = self.local_archive[key]
        bucket.append(rec)
        bucket.sort(key=lambda x: x["y"], reverse=True)
        if len(bucket) > CFG.ARCHIVE_PER_STATE:
            best_keep = max(1, CFG.ARCHIVE_PER_STATE // 2)
            keep = bucket[:best_keep] + bucket[-(CFG.ARCHIVE_PER_STATE - best_keep):]
            bucket[:] = keep

    def tell(self, theta, cost, state=None, context=None):
        """把本轮真实反馈喂给 BO。cost 越小越好，所以内部会转成 y=-cost。"""
        theta = self._normalize_theta(theta)
        y_val = -float(cost)
        sample = self._pack_sample(theta, y_val, state=state, context=context)
        if len(self.local_recent) == self.local_recent.maxlen and self.local_recent:
            old = self.local_recent[0]
            old_rec = self._unpack_sample(old)
            self._archive_state_sample(old_rec.get("state"), old_rec)
        self.local_recent.append(sample)
        if self.prev_best_value is None or y_val >= self.prev_best_value:
            self.prev_best_value = y_val
            self.prev_best = list(theta)
            if self.use_trust_region:
                self.trust_radius = min(CFG.TRUST_RADIUS_MAX, self.trust_radius * CFG.TRUST_RADIUS_GROWTH)
        else:
            if self.use_trust_region:
                self.trust_radius = max(CFG.TRUST_RADIUS_MIN, self.trust_radius * CFG.TRUST_RADIUS_SHRINK)

    def ingest_external(self, samples: List[Tuple[List[float], float]]):
        return

    def _compose_features(self, theta, context=None):
        theta = self._normalize_theta(theta)
        if self.use_context:
            return theta + self._normalize_context(context)
        return theta

    def _combined_bounds(self):
        if self.use_context:
            return torch.cat([self.bounds, self.context_bounds], dim=1)
        return self.bounds

    def _collect_samples(self, state=None):
        """收集用于拟合 GP 的样本。

        规则：
        1) use_context=True 且 use_state_partition=True：优先使用当前 state 的历史，再补最近样本；
        2) use_context=True 且 use_state_partition=False：使用全局历史；
        3) use_context=False 且 use_state_partition=True：按离散 state 分桶；
        4) use_context=False 且 use_state_partition=False：真正的无情景 BO，直接混合所有历史。
        """
        local_samples = [self._unpack_sample(s) for s in self.local_recent]
        archive_samples = []
        if self.use_context and self.use_state_partition and state is not None:
            archive_samples = [self._unpack_sample(s) for s in self.local_archive.get(state, [])]
            min_needed = max(CFG.ARCHIVE_PER_STATE, self.neighbor_k * 2)
            if len(archive_samples) + len(local_samples) < min_needed:
                for key, bucket in self.local_archive.items():
                    if key == state:
                        continue
                    archive_samples.extend([self._unpack_sample(s) for s in bucket[:2]])
        elif self.use_context:
            for bucket in self.local_archive.values():
                archive_samples.extend([self._unpack_sample(s) for s in bucket])
        elif self.use_state_partition and state is not None:
            archive_samples = [self._unpack_sample(s) for s in self.local_archive.get(state, [])]
        else:
            for bucket in self.local_archive.values():
                archive_samples.extend([self._unpack_sample(s) for s in bucket])

        merged = archive_samples + local_samples
        limit = CFG.RECENT_HISTORY + CFG.ARCHIVE_PER_STATE + CFG.MAX_HISTORY
        if len(merged) > limit:
            merged = merged[-limit:]
        return merged

    def _training_data(self, state=None):
        records = self._collect_samples(state=state)
        if records:
            x = torch.tensor([self._compose_features(r["theta"], r.get("context")) for r in records], dtype=torch.double)
            y = torch.tensor([[r["y"]] for r in records], dtype=torch.double)
        else:
            feat_dim = self.dim + (self.context_dim if self.use_context else 0)
            x = torch.empty(0, feat_dim, dtype=torch.double)
            y = torch.empty(0, 1, dtype=torch.double)
        return x, y, records

    def fit_local_gp(self, state=None):
        """用当前收集到的样本拟合一个本地 GP 代理模型。"""
        x, y, records = self._training_data(state=state)
        if len(x) < 2:
            return None
        try:
            y_mean = y.mean(dim=0)
            y_std = y.std(dim=0, unbiased=False)
            y_std = torch.where(y_std == 0, torch.tensor(1.0, dtype=y_std.dtype), y_std)
            y_std_vals = (y - y_mean) / y_std
            bounds_full = self._combined_bounds()
            x_norm = torch.clamp(normalize(x, bounds_full), 0.0, 1.0)
            gp = SingleTaskGP(x_norm, y_std_vals)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_gpytorch_mll(mll)
            gp.eval()
            return {
                "gp": gp,
                "y_mean": y_mean.detach(),
                "y_std": y_std.detach(),
                "bounds": bounds_full.clone(),
                "records": records,
            }
        except Exception as e:
            print(f"fit_local_gp failed: {e}")
            return None

    def predict_candidates(self, candidate_thetas: List[List[float]], state=None, context=None):
        """对一组候选 theta 给出 GP 预测均值 mu 和不确定性 sigma。"""
        model_pack = self.fit_local_gp(state=state)
        if model_pack is None:
            return []
        gp = model_pack["gp"]
        y_mean = model_pack["y_mean"]
        y_std = model_pack["y_std"]
        bounds = model_pack["bounds"]
        try:
            xs = torch.tensor([self._compose_features(t, context=context) for t in candidate_thetas], dtype=torch.double)
            xs_norm = torch.clamp(normalize(xs, bounds), 0.0, 1.0)
            posterior = gp.posterior(xs_norm)
            mu_std = posterior.mean.detach().squeeze(-1)
            var_std = posterior.variance.detach().squeeze(-1)
            mu = (mu_std * y_std.squeeze(-1)) + y_mean.squeeze(-1)
            var = var_std * (y_std.squeeze(-1) ** 2)
            sigma = torch.sqrt(torch.clamp(var, min=0.0))
            out = []
            for i in range(len(candidate_thetas)):
                out.append({
                    "theta": self._normalize_theta(candidate_thetas[i]),
                    "mu": float(mu[i].item()),
                    "sigma": float(sigma[i].item())
                })
            return out
        except Exception as e:
            print(f"predict_candidates failed: {e}")
            return []

    def _select_pivot_theta(self, context, records):
        """在情景 BO 中，用核相似度从历史中构造局部搜索中心。"""
        self._last_similarity_pool = []
        if not self.use_context or context is None or not records:
            return self.prev_best

        ctx_records = [r for r in records if r.get("context") is not None]
        if not ctx_records:
            return self.prev_best

        scored = []
        for rec in ctx_records:
            sim = self._context_similarity(context, rec.get("context"))
            scored.append((sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        pool = scored[:max(1, self.neighbor_k)]
        self._last_similarity_pool = [
            {
                "rank": idx + 1,
                "similarity": float(sim),
                "y": float(rec.get("y", np.nan)),
                "theta": list(rec.get("theta", [])),
                "context": list(rec.get("context", [])) if rec.get("context") is not None else None,
                "state": str(rec.get("state")) if rec.get("state") is not None else None,
            }
            for idx, (sim, rec) in enumerate(pool)
        ]
        if not pool:
            return self.prev_best

        y_vals = np.array([float(rec["y"]) for _, rec in pool], dtype=float)
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        y_span = max(1e-9, y_max - y_min)

        # 高斯核相似度负责“像不像”，历史 y 负责“好不好”；
        # 最终用相似度加权的局部平均来构造 pivot，比只拿单点更稳。
        weighted_thetas = []
        weights = []
        for sim, rec in pool:
            y_norm = (float(rec["y"]) - y_min) / y_span
            weight = float(sim) * (0.35 + 0.65 * y_norm)
            if weight <= 0:
                continue
            weighted_thetas.append(np.array(rec["theta"], dtype=float))
            weights.append(weight)

        if not weights:
            best = max((rec for _, rec in pool), key=lambda r: r["y"])
            return list(best["theta"])

        w = np.array(weights, dtype=float)
        theta_stack = np.vstack(weighted_thetas)
        pivot = np.average(theta_stack, axis=0, weights=w)
        low = np.array(self.bounds[0].tolist(), dtype=float)
        high = np.array(self.bounds[1].tolist(), dtype=float)
        pivot = np.clip(pivot, low, high)
        return pivot.tolist()

    def _random_candidates(self, low, high, n):
        return [self._sample_in_bounds(low, high) for _ in range(n)]

    def _contextual_scores(self, gp, y_mean, y_std, bounds_full, candidate_thetas, context):
        xs = torch.tensor([self._compose_features(t, context=context) for t in candidate_thetas], dtype=torch.double)
        xs_norm = torch.clamp(normalize(xs, bounds_full), 0.0, 1.0)
        posterior = gp.posterior(xs_norm)
        mu_std = posterior.mean.detach().squeeze(-1)
        var_std = posterior.variance.detach().squeeze(-1)
        mu = (mu_std * y_std.squeeze(-1)) + y_mean.squeeze(-1)
        sigma = torch.sqrt(torch.clamp(var_std * (y_std.squeeze(-1) ** 2), min=0.0))
        score = mu + self.beta_init * sigma
        return mu.cpu().numpy(), sigma.cpu().numpy(), score.cpu().numpy()

    def _ask_contextual(self, state=None, context=None):
        """情景 BO / 情景+TR 的选点逻辑。默认使用 7 维连续情景 + 高斯核相似度。"""
        self.step_count += 1
        step_acq_data = {"step": self.step_count, "candidates": [], "acq_values": [], "best_selected": None, "model_state_dict": None}
        low = self.bounds[0].tolist()
        high = self.bounds[1].tolist()
        archive_sample_count = int(sum(len(v) for v in self.local_archive.values()))
        base_debug = {
            "step": int(self.step_count),
            "state": str(state) if state is not None else None,
            "context": self._normalize_context(context) if context is not None else ([0.0] * self.context_dim if self.use_context else []),
            "training_sample_count": 0,
            "recent_sample_count": int(len(self.local_recent)),
            "archive_sample_count": archive_sample_count,
            "neighbor_k": int(self.neighbor_k),
            "topk_history": [],
            "topk_similarity": [],
            "pivot": None,
            "trust_radius": float(self.trust_radius),
            "best_selected": None,
            "candidate_count": 0,
        }
        if self.anchor_points and self.step_count <= len(self.anchor_points):
            theta = list(self.anchor_points[self.step_count - 1])
            self.acq_history.append({**step_acq_data, "best_selected": theta})
            self.last_theta = theta
            self.last_debug_info = {**base_debug, "best_selected": list(theta), "candidate_count": 1}
            return theta
        x, y, records = self._training_data(state=state)
        base_debug["training_sample_count"] = int(len(records))
        if len(x) < 2:
            theta = self._sample_in_bounds(low, high)
            self.acq_history.append({**step_acq_data, "best_selected": theta})
            self.last_theta = theta
            self.last_debug_info = {**base_debug, "best_selected": list(theta), "candidate_count": 1}
            return theta
        model_pack = self.fit_local_gp(state=state)
        if model_pack is None:
            theta = self._sample_in_bounds(low, high)
            self.acq_history.append({**step_acq_data, "best_selected": theta})
            self.last_theta = theta
            self.last_debug_info = {**base_debug, "best_selected": list(theta), "candidate_count": 1}
            return theta
        gp = model_pack["gp"]
        y_mean = model_pack["y_mean"]
        y_std = model_pack["y_std"]
        bounds_full = model_pack["bounds"]
        step_acq_data["model_state_dict"] = gp.state_dict()
        candidates = []
        candidates.extend(self._random_candidates(low, high, 48))
        pivot = self._select_pivot_theta(context, records)
        base_debug["pivot"] = list(pivot) if pivot is not None else None
        base_debug["topk_history"] = list(self._last_similarity_pool)
        base_debug["topk_similarity"] = [float(item.get("similarity", 0.0)) for item in self._last_similarity_pool]
        if self.use_trust_region and pivot is not None:
            for _ in range(64):
                cand = []
                for d in range(self.dim):
                    span = (high[d] - low[d]) * self.trust_radius
                    cand.append(min(max(pivot[d] + span * (2.0 * self.py_rng.random() - 1.0), low[d]), high[d]))
                candidates.append(cand)
        elif self.prev_best is not None:
            for _ in range(24):
                cand = []
                for d in range(self.dim):
                    span = (high[d] - low[d]) * max(self.trust_radius, 0.08)
                    cand.append(min(max(self.prev_best[d] + span * (2.0 * self.py_rng.random() - 1.0), low[d]), high[d]))
                candidates.append(cand)
        # deduplicate
        unique = []
        seen = set()
        for c in candidates:
            key = tuple(round(float(v), 6) for v in c)
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
        candidates = unique[:128]
        if not candidates:
            candidates = [self._sample_in_bounds(low, high)]
        mu, sigma, score = self._contextual_scores(gp, y_mean, y_std, bounds_full, candidates, context)
        best_idx = int(np.argmax(score))
        best = candidates[best_idx]
        if self.last_theta is not None and self.theta_momentum > 0.0:
            best = [self.theta_momentum * self.last_theta[d] + (1.0 - self.theta_momentum) * best[d] for d in range(self.dim)]
        best = [min(max(best[d], low[d]), high[d]) for d in range(self.dim)]
        step_acq_data["candidates"] = [list(c) for c in candidates]
        step_acq_data["acq_values"] = [float(v) for v in score.tolist()]
        step_acq_data["best_selected"] = list(best)
        self.acq_history.append(step_acq_data)
        self.last_theta = list(best)
        self.last_debug_info = {
            **base_debug,
            "best_selected": list(best),
            "candidate_count": int(len(candidates)),
        }
        return list(best)

    def ask(self, state=None, context=None, model_prior: Optional[Dict[str, torch.Tensor]] = None):
        """对外统一接口：返回下一轮要试验的 theta。"""
        if self.use_context:
            return self._ask_contextual(state=state, context=context)
        self.step_count += 1
        raw_candidates = []
        train_x, train_y, _ = self._training_data(state=state)
        step_acq_data = {"step": self.step_count, "candidates": [], "acq_values": [], "best_selected": None, "model_state_dict": None}
        low = self.bounds[0].tolist()
        high = self.bounds[1].tolist()
        def smooth_and_clip(theta):
            if self.last_theta is not None:
                theta = [self.theta_momentum * self.last_theta[d] + (1.0 - self.theta_momentum) * theta[d] for d in range(self.dim)]
            theta = [min(max(theta[d], low[d]), high[d]) for d in range(self.dim)]
            return theta
        if self.anchor_points and self.step_count <= len(self.anchor_points):
            final_c = list(self.anchor_points[self.step_count - 1])
            step_acq_data["best_selected"] = final_c
            self.acq_history.append(step_acq_data)
            self.last_theta = final_c
            return final_c
        if len(train_x) < 2:
            for _ in range(20):
                c = self._sample_in_bounds(low, high)
                raw_candidates.append(c)
                step_acq_data["candidates"].append(c)
                step_acq_data["acq_values"].append(self.py_rng.random())
        else:
            try:
                with torch.random.fork_rng():
                    torch.manual_seed(self._next_torch_seed())
                    train_y_std = standardize(train_y)
                    train_x_norm = torch.clamp(normalize(train_x, self.bounds), 0.0, 1.0)
                    gp = SingleTaskGP(train_x_norm, train_y_std)
                    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit_gpytorch_mll(mll)
                    step_acq_data["model_state_dict"] = gp.state_dict()
                    best_f = torch.max(train_y_std).item() if hasattr(train_y_std, "numel") else float(train_y_std)
                    acq = LogExpectedImprovement(gp, best_f=best_f)
                    cands, acq_vals = optimize_acqf(
                        acq,
                        bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]).double(),
                        q=1, num_restarts=10, raw_samples=256
                    )
                    cands_unnorm = unnormalize(cands, self.bounds).detach()
                    with torch.no_grad():
                        rand_pts_std = torch.rand(25, self.dim, dtype=torch.double, generator=self.torch_gen)
                        rand_acq = acq(rand_pts_std.unsqueeze(1))
                        rand_pts_unnorm = unnormalize(rand_pts_std, self.bounds)
                        for i in range(len(rand_pts_unnorm)):
                            step_acq_data["candidates"].append(rand_pts_unnorm[i].tolist())
                            step_acq_data["acq_values"].append(rand_acq[i].item())
                        if self.prev_best is not None and self.use_trust_region:
                            prev_norm = normalize(torch.tensor([self.prev_best], dtype=torch.double), self.bounds)[0]
                            tr_low = torch.clamp(prev_norm - self.trust_radius, 0.0, 1.0)
                            tr_high = torch.clamp(prev_norm + self.trust_radius, 0.0, 1.0)
                            tr_samples = tr_low + (tr_high - tr_low) * torch.rand(
                                40, self.dim, dtype=torch.double, generator=self.torch_gen
                            )
                            tr_vals = acq(tr_samples.unsqueeze(1)).detach().squeeze()
                            tr_unnorm = unnormalize(tr_samples, self.bounds)
                            for i in range(len(tr_unnorm)):
                                step_acq_data["candidates"].append(tr_unnorm[i].tolist())
                                step_acq_data["acq_values"].append(tr_vals[i].item())
                            all_cands = torch.cat([cands, tr_samples], dim=0)
                            all_vals = torch.cat([acq_vals, tr_vals], dim=0)
                            best_idx = torch.argmax(all_vals)
                            cands_unnorm = unnormalize(all_cands[best_idx:best_idx+1], self.bounds).detach()
                for i in range(len(cands_unnorm)):
                    c_list = cands_unnorm[i].tolist()
                    raw_candidates.append(c_list)
                    step_acq_data["candidates"].append(c_list)
                    val = acq_vals[i].item() if acq_vals.ndim > 0 else acq_vals.item()
                    step_acq_data["acq_values"].append(val)
            except Exception:
                for _ in range(5):
                    c = self._sample_in_bounds(low, high)
                    raw_candidates.append(c)
                    step_acq_data["candidates"].append(c)
                    step_acq_data["acq_values"].append(0.0)
        if not raw_candidates:
            final_c = self._sample_in_bounds(low, high)
        else:
            final_c = raw_candidates[0]
        final_c = smooth_and_clip(final_c)
        step_acq_data["best_selected"] = final_c
        self.acq_history.append(step_acq_data)
        self.last_theta = final_c
        return final_c

class FederatedAggregator:
    """联邦聚合器。

    不上传原始数据，只聚合各工厂对候选点的预测输出(mu, sigma)。
    """
    def __init__(self, max_samples=50):
        self.max_samples = max_samples

    def aggregate_predictions(self, prediction_packets: List[Dict], weights: Optional[List[float]] = None, beta_cloud: Optional[float] = None):
        """Aggregate predictive outputs from multiple factories.
        prediction_packets: list of {"factory_id": id, "state": state, "predictions": [{"theta":..., "mu":..., "sigma":...}, ...]}
        Returns list of aggregated predictions: [{"theta":..., "mu_fed":..., "sigma_fed":..., "score_fed":...}, ...]
        """
        if not prediction_packets:
            return []
        # determine candidate count from first packet
        first_preds = prediction_packets[0].get("predictions", [])
        if not first_preds:
            return []
        m = len(first_preds)
        K = len(prediction_packets)
        if weights is None:
            weights = [1.0 / K] * K
        else:
            # normalize
            wsum = float(sum(weights)) if sum(weights) != 0 else 1.0
            weights = [float(w) / wsum for w in weights]
        beta = CFG.FED_BETA if beta_cloud is None else beta_cloud
        aggregated = []
        for idx in range(m):
            theta = None
            mus = []
            sigs = []
            for k, packet in enumerate(prediction_packets):
                preds = packet.get("predictions", [])
                if idx < len(preds):
                    p = preds[idx]
                    theta = p.get("theta", theta)
                    mus.append(weights[k] * float(p.get("mu", 0.0)))
                    sigs.append(float(p.get("sigma", 0.0)))
            if not mus:
                continue
            mu_fed = float(sum(mus))
            # combine variances: var_fed = sum_k w_k * sigma_k^2 ; sigma_fed = sqrt(var_fed)
            var_fed = 0.0
            for k, packet in enumerate(prediction_packets):
                preds = packet.get("predictions", [])
                if idx < len(preds):
                    s = float(preds[idx].get("sigma", 0.0))
                    var_fed += weights[k] * (s ** 2)
            sigma_fed = float(math.sqrt(max(0.0, var_fed)))
            score = mu_fed + beta * sigma_fed
            aggregated.append({"theta": theta, "mu_fed": mu_fed, "sigma_fed": sigma_fed, "score_fed": score})
        return aggregated

# ==========================================
# 5. 工厂模型 (Factory)
# ==========================================
class ScenarioMonitor:
    """情景监测器。

    连续情景向量：arrival_rate, avg_delay, avg_util
    离散情景状态：StateSignature(load, delay, arrival)
    """
    def __init__(self):
        self.arrivals = collections.deque()
        self.completions = collections.deque()
        self.util_samples = collections.deque()
        self.last_state = None
        self.stable_count = 0
        self.last_eval_time = None
        self.window_feedback = {
            "backlog": 0.0,
            "vio_rate": 0.0,
            "completion_rate": 0.0,
            "sla_success_rate": 1.0,
            "rt_arrival_ratio": 0.0,
            "batch_arrival_ratio": 0.0,
            "ai_arrival_ratio": 0.0,
        }
        self.last_metrics = {"arrival_rate": 0.0, "avg_delay": 0.0, "avg_util": 0.0, **self.window_feedback}

    def _prune(self, now):
        window = CFG.SCENARIO_WINDOW
        while self.arrivals and self.arrivals[0] < now - window: self.arrivals.popleft()
        while self.completions and self.completions[0][0] < now - window: self.completions.popleft()
        while self.util_samples and self.util_samples[0][0] < now - window: self.util_samples.popleft()

    def record_arrival(self, t): self.arrivals.append(t)
    def record_completion(self, t, delay): self.completions.append((t, delay))
    def record_utilization(self, t, util): self.util_samples.append((t, util))

    def _level(self, value, thresholds):
        low, high = thresholds
        if value > high: return StateLevel.HIGH
        if value > low: return StateLevel.MID
        return StateLevel.LOW

    def compute_metrics(self, now):
        """统计当前滑动窗口内的情景指标。"""
        self._prune(now)
        arrival_rate = len(self.arrivals) / max(CFG.SCENARIO_WINDOW, 1.0)
        avg_delay = sum(d for _, d in self.completions) / len(self.completions) if self.completions else 0.0
        avg_util = sum(u for _, u in self.util_samples) / len(self.util_samples) if self.util_samples else 0.0
        return {
            "arrival_rate": arrival_rate,
            "avg_delay": avg_delay,
            "avg_util": avg_util,
            "backlog": float(self.window_feedback.get("backlog", 0.0)),
            "vio_rate": float(self.window_feedback.get("vio_rate", 0.0)),
            "completion_rate": float(self.window_feedback.get("completion_rate", 0.0)),
            "sla_success_rate": float(self.window_feedback.get("sla_success_rate", 1.0)),
            "rt_arrival_ratio": float(self.window_feedback.get("rt_arrival_ratio", 0.0)),
            "batch_arrival_ratio": float(self.window_feedback.get("batch_arrival_ratio", 0.0)),
            "ai_arrival_ratio": float(self.window_feedback.get("ai_arrival_ratio", 0.0)),
        }

    def update_window_feedback(self, metrics):
        self.window_feedback = {
            "backlog": float(metrics.get("backlog", 0.0)),
            "vio_rate": float(metrics.get("vio_rate", 0.0)),
            "completion_rate": float(metrics.get("completion_rate", 0.0)),
            "sla_success_rate": float(metrics.get("sla_success_rate", 1.0)),
            "rt_arrival_ratio": float(metrics.get("rt_arrival_ratio", 0.0)),
            "batch_arrival_ratio": float(metrics.get("batch_arrival_ratio", 0.0)),
            "ai_arrival_ratio": float(metrics.get("ai_arrival_ratio", 0.0)),
        }
        self.last_metrics = {**self.last_metrics, **self.window_feedback}

    def build_context_vector(self, metrics=None):
        metrics = metrics if metrics is not None else self.last_metrics
        names = getattr(CFG, "CONTEXT_FEATURE_NAMES", ["arrival_rate", "avg_util", "backlog", "vio_rate", "rt_arrival_ratio", "batch_arrival_ratio", "ai_arrival_ratio"])
        vec = []
        for name in names:
            vec.append(float(metrics.get(name, 0.0)))
        return vec

    def get_context_vector(self, now, metrics=None):
        if metrics is None:
            metrics = self.compute_metrics(now)
        return self.build_context_vector(metrics)

    def get_state(self, now):
        """把连续指标离散化成 LOW/MID/HIGH 状态，并判断是否稳定。"""
        if self.last_eval_time is not None and now - self.last_eval_time < CFG.SCENARIO_INTERVAL:
            stable = self.stable_count >= CFG.SCENARIO_STABLE_K
            return self.last_state, stable, self.last_metrics
        metrics = self.compute_metrics(now)
        load_level = self._level(metrics["avg_util"], CFG.UTIL_THRESHOLDS)
        delay_level = self._level(metrics["avg_delay"], CFG.DELAY_THRESHOLDS)
        arrival_level = self._level(metrics["arrival_rate"], CFG.ARRIVAL_THRESHOLDS)
        state = StateSignature(load_level, delay_level, arrival_level)
        if self.last_state == state: self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_state = state
        self.last_eval_time = now
        self.last_metrics = metrics
        stable = self.stable_count >= CFG.SCENARIO_STABLE_K
        return state, stable, metrics

class ConnectedFactory:
    """一个工厂/一个仿真实例。

    它把以下模块串起来：
    工作负载生成器 -> 调度器 -> 事件驱动仿真 -> 窗口统计 -> BO 反馈
    """
    def __init__(self, fid, name, seed, node_config, scheduler_type="Boltzmann", norm_mode="rolling"):
        self.id = fid
        self.name = name
        self.seed = seed
        self.node_config = node_config
        self.scheduler_type = scheduler_type
        self.norm_mode = norm_mode
        self.reset()

    def reset(self, use_batch=False):
        """重置工厂状态，重新初始化节点、调度器、BO 代理和事件队列。"""
        self.base_seed = resolve_base_seed(self.seed, stream=self.id)
        scheduler_seed = self.base_seed + 1
        agent_py_seed = self.base_seed + 2
        agent_torch_seed = self.base_seed + 3
        workload_seed = self.base_seed + 4
        self.scenario_monitor = ScenarioMonitor()
        self.workload = BatchWorkloadGenerator(self.id, workload_seed) if use_batch else PiecewisePoissonWorkloadGenerator(self.id, workload_seed)
        self.nodes = [Node(cfg) for cfg in self.node_config]
        scheduler_type_norm = str(self.scheduler_type or "Boltzmann").strip().lower().replace("-", "_")
        if scheduler_type_norm == "boltzmann":
            scheduler_cls = ConstrainedBoltzmannScheduler if getattr(CFG, "USE_CONSTRAINED_BOLTZMANN", True) else BoltzmannScheduler
            self.scheduler = scheduler_cls(np_rng=np.random.default_rng(scheduler_seed), norm_mode=self.norm_mode)
        elif scheduler_type_norm in {"roundrobin", "round_robin", "round_robin_direct", "rr"}:
            self.scheduler = RoundRobinScheduler()
        elif scheduler_type_norm in {
            "greedy_direct_cost", "direct_greedy_cost",
            "least_load_direct", "direct_least_load", "least_load",
            "queue_aware_greedy_direct", "direct_queue_aware_greedy", "queue_aware_greedy", "dpp_greedy",
        }:
            self.scheduler = DirectHeuristicScheduler(mode=scheduler_type_norm, np_rng=np.random.default_rng(scheduler_seed))
        else:
            print(f"[WARN] unknown scheduler_type={self.scheduler_type}, fallback to RoundRobinScheduler")
            self.scheduler = RoundRobinScheduler()
        self.agent = FederatedBOAgent(
            dim=CFG.DIM_THETA,
            py_rng=random.Random(agent_py_seed),
            torch_gen=torch.Generator().manual_seed(agent_torch_seed),
            bounds=get_control_bounds(CFG.DIM_THETA),
            feature_names=list(CFG.FEATURE_NAMES),
            use_context=True,
            use_trust_region=True,
            context_dim=len(CFG.CONTEXT_FEATURE_NAMES),
            context_bounds=CFG.CONTEXT_BOUNDS,
            anchor_points=default_scenario_anchor_points(),
        )
        self.current_time = 0.0
        self.events = []
        self.cumulative_energy = 0.0
        self.cumulative_energy_real = 0.0
        self.link_busy_until = {}
        self.current_theta = default_control_vector(fill=1.5)
        self.current_control_vector = list(self.current_theta)
        self.current_control_label = "weights_only"
        self.batch_counter = 0
        self.bo_step = 0
        self.next_bo_time = CFG.BO_INTERVAL
        self.perf_log = {
            "time": [], "reward": [], "vio": [], "ene": [], "avg_delay": [],
            "completion_rate": [], "arrival_rate": [], "avg_util": [],
            "stable": [], "state": [], "alpha": [], "beta": [],
            "total_energy": [], "total_energy_real": [], "compute_dynamic_energy": [], "compute_idle_energy": [],
            "compute_energy_real": [], "transmission_energy": [], "cumulative_energy": [], "cumulative_energy_real": [], "sla_success_rate": [],
            "task_count": [], "cpu_demand": [], "data_demand": [], "avg_energy": [],
            "avg_delay_rt": [], "avg_delay_batch": [], "avg_delay_ai": [], "vio_rate": [],
            "backlog": [], "zero_completion_penalty": [], "alloc": [],
            "control_vector": [], "control_label": [],
            "arrivals_total": [], "arrivals_rt": [], "arrivals_batch": [], "arrivals_ai": [],
            "completed_total": [], "completed_rt": [], "completed_batch": [], "completed_ai": [],
            "unfinished_end": [],
            "rt_arrival_ratio": [], "batch_arrival_ratio": [], "ai_arrival_ratio": [],
            "context_vector": [], "context_label": [], "context_vector_after": [], "feedback_state_after": [],
            "training_sample_count": [], "recent_sample_count": [], "archive_sample_count": [],
            "neighbor_k": [], "topk_history": [], "topk_similarity": [],
            "pivot_theta": [], "trust_radius": [], "best_selected_theta": [], "candidate_count": [],
            "feedback_mode": [], "cohort_id": [], "cohort_arrivals": [],
            "cohort_feedback_count": [], "cohort_feedback_cost_mean": [],
            "cohort_active_count": [], "cohort_finalized_total": [], "cohort_pending_tasks": [],
            "scheduler_tradeoff_mode": [], "scheduler_score_norm_mode": [],
            "scheduler_alpha_last": [], "scheduler_alpha_mean": [],
            "selected_service_component_last": [], "selected_energy_component_last": [],
            "selected_latency_component_last": [], "selected_risk_penalty_last": [],
            "selected_queue_penalty_last": [], "selected_latency_energy_component_last": [],
            "selected_norm_e_last": [], "selected_norm_l_last": [],
            "selected_norm_risk_last": [], "selected_norm_queue_last": [],
            "selected_score_last": []
        }
        self.window_history = []
        # cohort 反馈诊断数据。窗口级指标仍正常记录，BO 反馈可切换为 cohort_complete。
        self.cohorts = {}
        self.current_cohort_id = None
        self.cohort_counter = 0
        self.cohort_feedback_rows = []
        self.cohort_finalized_total = 0
        if not use_batch:
            self._schedule_next()

    def run_batch(self, theta, batch_size=200, external_samples=None, model_prior=None):
        horizon = max(CFG.BO_INTERVAL, float(batch_size) / max(1e-9, CFG.BATCH_POISSON_LAMBDA))
        return self.run_continuous(theta, external_samples=external_samples, model_prior=model_prior, window_end=self.current_time + horizon)

    def _new_window_snapshot(self, start_time, end_time):
        snapshot = WindowSnapshot(start_time=start_time, end_time=end_time, alloc_counts=[0 for _ in self.nodes])
        # 诊断用：记录每类任务实际被分配到各节点的次数。
        # 这不是 BO 训练目标，只用于后续判断“参数变化是否真的改变任务-节点偏好”。
        snapshot.alloc_by_type = {t: [0 for _ in self.nodes] for t in TASK_TYPE_ORDER}
        return snapshot

    def _record_window_log(self, metrics, state, stable, theta, context_vec=None):
        self.perf_log["time"].append(self.current_time)
        self.perf_log["reward"].append(metrics["reward"])
        self.perf_log["ene"].append(metrics["total_energy"])
        self.perf_log["total_energy"].append(metrics["total_energy"])
        self.perf_log.setdefault("total_energy_real", []).append(metrics.get("total_energy_real", metrics["total_energy"]))
        self.perf_log.setdefault("compute_dynamic_energy", []).append(metrics.get("compute_dynamic_energy", 0.0))
        self.perf_log.setdefault("compute_idle_energy", []).append(metrics.get("compute_idle_energy", 0.0))
        self.perf_log.setdefault("compute_energy_real", []).append(metrics.get("compute_energy_real", 0.0))
        self.perf_log.setdefault("transmission_energy", []).append(metrics.get("transmission_energy", 0.0))
        self.perf_log["avg_delay"].append(metrics["avg_delay"])
        self.perf_log["avg_delay_rt"].append(metrics["avg_delay_rt"])
        self.perf_log["avg_delay_batch"].append(metrics["avg_delay_batch"])
        self.perf_log["avg_delay_ai"].append(metrics["avg_delay_ai"])
        self.perf_log["vio_rate"].append(metrics["vio_rate"])
        self.perf_log["cumulative_energy"].append(metrics["cumulative_energy"])
        self.perf_log.setdefault("cumulative_energy_real", []).append(metrics.get("cumulative_energy_real", metrics["cumulative_energy"]))
        self.perf_log["sla_success_rate"].append(metrics["sla_success_rate"])
        self.perf_log["avg_energy"].append(metrics["avg_energy"])
        self.perf_log.setdefault("avg_earliness", []).append(metrics["avg_earliness"])
        self.perf_log.setdefault("avg_lateness", []).append(metrics["avg_lateness"])
        self.perf_log["alpha"].append(theta)
        self.perf_log.setdefault("alloc", []).append(metrics["alloc"])
        self.perf_log.setdefault("alloc_by_type", []).append(metrics.get("alloc_by_type", {}))
        self.perf_log["task_count"].append(metrics["task_count"])
        self.perf_log["cpu_demand"].append(metrics["cpu_demand"])
        self.perf_log["data_demand"].append(metrics["data_demand"])
        self.perf_log["backlog"].append(metrics.get("backlog", 0))
        self.perf_log["zero_completion_penalty"].append(metrics.get("zero_completion_penalty", 0.0))
        _cbo_log_reference_fields(self.perf_log, metrics)
        self.perf_log.setdefault("early_bonus_applied", []).append(metrics.get("early_bonus_applied", 0.0))
        self.perf_log["arrival_rate"].append(self.scenario_monitor.last_metrics.get("arrival_rate", 0.0))
        self.perf_log["avg_util"].append(self.scenario_monitor.last_metrics.get("avg_util", 0.0))
        self.perf_log["stable"].append(stable)
        self.perf_log["state"].append(str(state) if state is not None else "None")
        self.perf_log["beta"].append(getattr(self.scheduler, "beta", None))
        self.perf_log["control_vector"].append(list(getattr(self, "current_control_vector", theta)))
        self.perf_log["control_label"].append(getattr(self, "current_control_label", "weights_only"))
        sched_debug = dict(getattr(self, "_last_window_scheduler_debug", {}) or {})
        self.perf_log.setdefault("scheduler_tradeoff_mode", []).append(sched_debug.get("scheduler_tradeoff_mode", getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy")))
        self.perf_log.setdefault("scheduler_score_norm_mode", []).append(sched_debug.get("scheduler_score_norm_mode", getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")))
        self.perf_log.setdefault("scheduler_alpha_last", []).append(sched_debug.get("scheduler_alpha"))
        self.perf_log.setdefault("scheduler_alpha_mean", []).append(sched_debug.get("scheduler_alpha_mean", sched_debug.get("scheduler_alpha")))
        self.perf_log.setdefault("selected_latency_component_last", []).append(sched_debug.get("selected_latency_component"))
        self.perf_log.setdefault("selected_risk_penalty_last", []).append(sched_debug.get("selected_risk_penalty"))
        self.perf_log.setdefault("selected_queue_penalty_last", []).append(sched_debug.get("selected_queue_penalty"))
        self.perf_log.setdefault("selected_latency_energy_component_last", []).append(sched_debug.get("selected_latency_energy_component"))
        self.perf_log.setdefault("selected_service_component_last", []).append(sched_debug.get("selected_service_component"))  # Deprecated alias.
        self.perf_log.setdefault("selected_energy_component_last", []).append(sched_debug.get("selected_energy_component"))
        self.perf_log.setdefault("selected_norm_e_last", []).append(sched_debug.get("selected_norm_e"))
        self.perf_log.setdefault("selected_norm_l_last", []).append(sched_debug.get("selected_norm_l"))
        self.perf_log.setdefault("selected_norm_risk_last", []).append(sched_debug.get("selected_norm_risk"))
        self.perf_log.setdefault("selected_norm_queue_last", []).append(sched_debug.get("selected_norm_queue"))
        self.perf_log.setdefault("selected_score_last", []).append(sched_debug.get("selected_score"))
        self.perf_log.setdefault("arrivals_total", []).append(metrics.get("arrivals_total", 0))
        self.perf_log.setdefault("arrivals_rt", []).append(metrics.get("arrivals_rt", 0))
        self.perf_log.setdefault("arrivals_batch", []).append(metrics.get("arrivals_batch", 0))
        self.perf_log.setdefault("arrivals_ai", []).append(metrics.get("arrivals_ai", 0))
        self.perf_log.setdefault("completed_total", []).append(metrics.get("completed_total", 0))
        self.perf_log.setdefault("completed_rt", []).append(metrics.get("completed_rt", 0))
        self.perf_log.setdefault("completed_batch", []).append(metrics.get("completed_batch", 0))
        self.perf_log.setdefault("completed_ai", []).append(metrics.get("completed_ai", 0))
        self.perf_log.setdefault("unfinished_end", []).append(metrics.get("unfinished_end", metrics.get("backlog", 0)))
        self.perf_log.setdefault("rt_arrival_ratio", []).append(metrics.get("rt_arrival_ratio", 0.0))
        self.perf_log.setdefault("batch_arrival_ratio", []).append(metrics.get("batch_arrival_ratio", 0.0))
        self.perf_log.setdefault("ai_arrival_ratio", []).append(metrics.get("ai_arrival_ratio", 0.0))
        self.perf_log.setdefault("avg_lateness_rt", []).append(metrics.get("avg_lateness_rt"))
        self.perf_log.setdefault("avg_lateness_batch", []).append(metrics.get("avg_lateness_batch"))
        self.perf_log.setdefault("avg_lateness_ai", []).append(metrics.get("avg_lateness_ai"))
        self.perf_log.setdefault("vio_rate_rt", []).append(metrics.get("vio_rate_rt"))
        self.perf_log.setdefault("vio_rate_batch", []).append(metrics.get("vio_rate_batch"))
        self.perf_log.setdefault("vio_rate_ai", []).append(metrics.get("vio_rate_ai"))
        self.perf_log.setdefault("completion_ratio_rt", []).append(metrics.get("completion_ratio_rt"))
        self.perf_log.setdefault("completion_ratio_batch", []).append(metrics.get("completion_ratio_batch"))
        self.perf_log.setdefault("completion_ratio_ai", []).append(metrics.get("completion_ratio_ai"))
        self.perf_log.setdefault("avg_energy_rt", []).append(metrics.get("avg_energy_rt"))
        self.perf_log.setdefault("avg_energy_batch", []).append(metrics.get("avg_energy_batch"))
        self.perf_log.setdefault("avg_energy_ai", []).append(metrics.get("avg_energy_ai"))
        self.perf_log.setdefault("window_rt_class_cost", []).append(metrics.get("window_rt_class_cost"))
        self.perf_log.setdefault("window_batch_class_cost", []).append(metrics.get("window_batch_class_cost"))
        self.perf_log.setdefault("window_ai_class_cost", []).append(metrics.get("window_ai_class_cost"))
        self.perf_log.setdefault("context_vector", []).append(list(context_vec) if context_vec is not None else [])
        self.perf_log.setdefault("context_label", []).append(str(state) if state is not None else "None")
        self.perf_log.setdefault("feedback_mode", []).append(metrics.get("feedback_mode", getattr(CFG, "FEEDBACK_MODE", "window")))
        self.perf_log.setdefault("cohort_id", []).append(metrics.get("cohort_id"))
        self.perf_log.setdefault("cohort_arrivals", []).append(metrics.get("cohort_arrivals", 0))
        self.perf_log.setdefault("cohort_feedback_count", []).append(metrics.get("cohort_feedback_count", 0))
        self.perf_log.setdefault("cohort_feedback_cost_mean", []).append(metrics.get("cohort_feedback_cost_mean", np.nan))
        self.perf_log.setdefault("cohort_active_count", []).append(metrics.get("cohort_active_count", 0))
        self.perf_log.setdefault("cohort_finalized_total", []).append(metrics.get("cohort_finalized_total", 0))
        self.perf_log.setdefault("cohort_pending_tasks", []).append(metrics.get("cohort_pending_tasks", 0))
        agent_debug = getattr(self.agent, "last_debug_info", {}) if self.agent is not None else {}
        self.perf_log.setdefault("training_sample_count", []).append(agent_debug.get("training_sample_count"))
        self.perf_log.setdefault("recent_sample_count", []).append(agent_debug.get("recent_sample_count"))
        self.perf_log.setdefault("archive_sample_count", []).append(agent_debug.get("archive_sample_count"))
        self.perf_log.setdefault("neighbor_k", []).append(agent_debug.get("neighbor_k"))
        self.perf_log.setdefault("topk_history", []).append(agent_debug.get("topk_history", []))
        self.perf_log.setdefault("topk_similarity", []).append(agent_debug.get("topk_similarity", []))
        self.perf_log.setdefault("pivot_theta", []).append(agent_debug.get("pivot"))
        self.perf_log.setdefault("trust_radius", []).append(agent_debug.get("trust_radius"))
        self.perf_log.setdefault("best_selected_theta", []).append(agent_debug.get("best_selected"))
        self.perf_log.setdefault("candidate_count", []).append(agent_debug.get("candidate_count"))

    def _accumulate_power_energy(self, snapshot, start_time, end_time):
        """按事件间隔对节点功率积分。

        这是增强能耗模型的核心：
        - 在两个事件之间，节点 running_tasks / ready_queue 状态不变；
        - 根据每个节点当前 CPU 利用率计算功率；
        - 把功率乘以时间间隔，累计到窗口能耗。
        """
        dt = float(end_time) - float(start_time)
        if dt <= 1e-12:
            return
        obj_energy = 0.0
        real_energy = 0.0
        dyn_energy = 0.0
        idle_obj_energy = 0.0
        for node in self.nodes:
            idle_obj, dyn_power, total_obj_power = node.power_components(objective=True)
            idle_real, dyn_power_real, total_real_power = node.power_components(objective=False)
            obj_energy += total_obj_power * dt
            real_energy += total_real_power * dt
            dyn_energy += dyn_power * dt
            idle_obj_energy += idle_obj * dt
        snapshot.total_energy += obj_energy
        snapshot.total_energy_real += real_energy
        snapshot.compute_dynamic_energy += dyn_energy
        snapshot.compute_idle_energy += idle_obj_energy
        snapshot.compute_energy_real += real_energy
        self.cumulative_energy += obj_energy
        self.cumulative_energy_real += real_energy
        self._accumulate_cohort_pending_area(start_time, end_time)

    def _system_backlog_count(self):
        """统计系统真实积压量 = 队列中 + 运行中 + 传输中任务数。"""
        in_service = sum(len(node.ready_queue) + len(node.running_tasks) for node in self.nodes)
        in_transit = sum(1 for ev in self.events if ev.type == EventType.TRANS_FINISH)
        return int(in_service + in_transit)

    def _use_cohort_feedback(self):
        return str(getattr(CFG, "FEEDBACK_MODE", "window")).lower() in {"cohort", "cohort_complete", "cohort_final"}

    def _create_cohort(self, theta_full, feedback_control, state, context):
        """创建本窗口任务 cohort。"""
        cid = int(self.cohort_counter)
        self.cohort_counter += 1
        record = CohortRecord(
            cohort_id=cid,
            theta_control=list(feedback_control if feedback_control is not None else theta_full),
            theta_full=list(theta_full),
            context=list(context) if context is not None else None,
            state=state,
            start_time=float(self.current_time),
            window_index=int(self.bo_step),
            label=str(getattr(self, "current_control_label", "")),
            last_area_update_time=float(self.current_time),
        )
        self.cohorts[cid] = record
        self.current_cohort_id = cid
        return record

    def _register_task_to_cohort(self, task, node_idx, estimated_energy=None, estimated_latency=None):
        """任务到达时，把任务归入当前窗口 cohort。"""
        if not self._use_cohort_feedback() or self.current_cohort_id is None:
            return
        cohort = self.cohorts.get(self.current_cohort_id)
        if cohort is None or cohort.finalized:
            return
        task.cohort_id = cohort.cohort_id
        task.cohort_node_idx = int(node_idx)
        task.cohort_estimated_energy = float(estimated_energy) if estimated_energy is not None and np.isfinite(estimated_energy) else 0.0
        task.cohort_estimated_latency = float(estimated_latency) if estimated_latency is not None and np.isfinite(estimated_latency) else 0.0
        cohort.total_tasks += 1
        cohort.assigned_energy_sum += float(task.cohort_estimated_energy)
        cohort.unfinished_tasks[task.id] = task
        if task.task_type == "RT":
            cohort.rt_arrivals += 1
            cohort.rt_assigned_energy_sum += float(task.cohort_estimated_energy)
            cohort.rt_estimated_latency_sum += float(task.cohort_estimated_latency)
        elif task.task_type == "AI":
            cohort.ai_arrivals += 1
            cohort.ai_assigned_energy_sum += float(task.cohort_estimated_energy)
            cohort.ai_estimated_latency_sum += float(task.cohort_estimated_latency)
        else:
            cohort.batch_arrivals += 1
            cohort.batch_assigned_energy_sum += float(task.cohort_estimated_energy)
            cohort.batch_estimated_latency_sum += float(task.cohort_estimated_latency)

    def _accumulate_cohort_pending_area(self, start_time, end_time):
        """对所有未完成 cohort 记录 pending task-time area。"""
        if not self._use_cohort_feedback():
            return
        dt = float(end_time) - float(start_time)
        if dt <= 1e-12:
            return
        for cohort in self.cohorts.values():
            if cohort.finalized:
                continue
            pending = len(cohort.unfinished_tasks)
            if pending > 0:
                cohort.pending_task_time_area += float(pending) * dt
                cohort.last_area_update_time = float(end_time)

    def _on_task_finished_cohort(self, task, delay):
        """任务完成时，把结果回填到其创建时绑定的 cohort。"""
        if not self._use_cohort_feedback():
            return []
        cid = getattr(task, "cohort_id", None)
        if cid is None or cid not in self.cohorts:
            return []
        cohort = self.cohorts[cid]
        if cohort.finalized:
            return []
        cohort.completed_tasks += 1
        cohort.completed_delay_sum += float(delay)
        late = max(0.0, float(task.finish_time) - float(task.deadline))
        cohort.completed_lateness_sum += late
        if task.finish_time > task.deadline:
            cohort.completed_violation_count += 1
        if task.task_type == "RT":
            cohort.rt_completed += 1
        elif task.task_type == "AI":
            cohort.ai_completed += 1
        else:
            cohort.batch_completed += 1
        cohort.unfinished_tasks.pop(task.id, None)
        return self._finalize_ready_cohorts(self.current_time, force=False, reason="all_completed")

    def _cohort_metrics(self, cohort, now, reason=""):
        """计算 cohort 级 cost。未完成任务会以截尾等待时间和未完成比例计入。"""
        n = max(1, int(cohort.total_tasks))
        unfinished_tasks = list(cohort.unfinished_tasks.values())
        unfinished = len(unfinished_tasks)
        unfinished_elapsed_sum = 0.0
        unfinished_over_deadline = 0
        unfinished_lateness_sum = 0.0
        for task in unfinished_tasks:
            elapsed = max(0.0, float(now) - float(task.create_time))
            unfinished_elapsed_sum += elapsed
            late_now = max(0.0, float(now) - float(task.deadline))
            unfinished_lateness_sum += late_now
            if float(now) > float(task.deadline):
                unfinished_over_deadline += 1

        censored_avg_delay = (cohort.completed_delay_sum + unfinished_elapsed_sum) / n
        completed_avg_delay = cohort.completed_delay_sum / max(1, int(cohort.completed_tasks)) if cohort.completed_tasks > 0 else 0.0
        avg_lateness = (cohort.completed_lateness_sum + unfinished_lateness_sum) / n
        effective_vio_rate = (cohort.completed_violation_count + unfinished_over_deadline) / n
        unfinished_ratio = unfinished / n
        pending_area_per_task = cohort.pending_task_time_area / n
        avg_energy = cohort.assigned_energy_sum / n
        confidence = float(cohort.completed_tasks) / n
        cost = (
            avg_energy
            + CFG.ALPHA_LATENCY * censored_avg_delay
            + CFG.SLA_PENALTY_WEIGHT * effective_vio_rate
            + CFG.LATE_PENALTY_WEIGHT * avg_lateness
            + float(getattr(CFG, "COHORT_UNFINISHED_PENALTY", 1000.0)) * unfinished_ratio
            + float(getattr(CFG, "COHORT_PENDING_AREA_WEIGHT", 5.0)) * pending_area_per_task
        )

        # Per-class cohort diagnostics, including completed-only and censored/effective variants.
        class_metrics = {}
        unfinished_by_prefix = {"rt": [], "batch": [], "ai": []}
        for task in unfinished_tasks:
            pfx = "rt" if getattr(task, "task_type", "Batch") == "RT" else ("ai" if getattr(task, "task_type", "Batch") == "AI" else "batch")
            unfinished_by_prefix[pfx].append(task)
        for name, pfx in [("RT", "rt"), ("Batch", "batch"), ("AI", "ai")]:
            arrivals_k = int(getattr(cohort, f"{pfx}_arrivals", 0))
            completed_k = int(getattr(cohort, f"{pfx}_completed", 0))
            unfinished_k_tasks = unfinished_by_prefix[pfx]
            unfinished_k = len(unfinished_k_tasks)
            delay_sum_k = float(getattr(cohort, f"{pfx}_delay_sum", 0.0))
            late_sum_k = float(getattr(cohort, f"{pfx}_lateness_sum", 0.0))
            vio_count_k = int(getattr(cohort, f"{pfx}_vio_count", 0))
            unfinished_elapsed_k = 0.0
            unfinished_late_k = 0.0
            unfinished_vio_k = 0
            for task in unfinished_k_tasks:
                unfinished_elapsed_k += max(0.0, float(now) - float(task.create_time))
                late_now = max(0.0, float(now) - float(task.deadline))
                unfinished_late_k += late_now
                if float(now) > float(task.deadline):
                    unfinished_vio_k += 1
            n_k = max(1, arrivals_k)
            completed_avg_delay_k = delay_sum_k / max(1, completed_k) if completed_k > 0 else 0.0
            completed_avg_late_k = late_sum_k / max(1, completed_k) if completed_k > 0 else 0.0
            completed_vio_rate_k = vio_count_k / max(1, completed_k) if completed_k > 0 else 0.0
            censored_avg_delay_k = (delay_sum_k + unfinished_elapsed_k) / n_k if arrivals_k > 0 else 0.0
            effective_avg_late_k = (late_sum_k + unfinished_late_k) / n_k if arrivals_k > 0 else 0.0
            effective_vio_rate_k = (vio_count_k + unfinished_vio_k) / n_k if arrivals_k > 0 else 0.0
            completion_ratio_k = completed_k / n_k if arrivals_k > 0 else 0.0
            avg_energy_est_k = float(getattr(cohort, f"{pfx}_assigned_energy_sum", 0.0)) / n_k if arrivals_k > 0 else 0.0
            avg_est_latency_k = float(getattr(cohort, f"{pfx}_estimated_latency_sum", 0.0)) / n_k if arrivals_k > 0 else 0.0
            unfinished_ratio_k = unfinished_k / n_k if arrivals_k > 0 else 0.0
            class_metrics.update({
                f"{pfx}_unfinished": int(unfinished_k),
                f"{pfx}_completion_ratio": float(completion_ratio_k),
                f"{pfx}_completed_avg_delay": float(completed_avg_delay_k),
                f"{pfx}_completed_avg_lateness": float(completed_avg_late_k),
                f"{pfx}_completed_vio_rate": float(completed_vio_rate_k),
                f"{pfx}_censored_avg_delay": float(censored_avg_delay_k),
                f"{pfx}_effective_avg_lateness": float(effective_avg_late_k),
                f"{pfx}_effective_vio_rate": float(effective_vio_rate_k),
                f"{pfx}_unfinished_ratio": float(unfinished_ratio_k),
                f"{pfx}_avg_energy_est": float(avg_energy_est_k),
                f"{pfx}_avg_estimated_latency": float(avg_est_latency_k),
            })
        return {
            "cohort_id": int(cohort.cohort_id),
            "window_index": int(cohort.window_index),
            "label": cohort.label,
            "start_time": float(cohort.start_time),
            "finalize_time": float(now),
            "age_windows": float(max(0.0, (float(now) - float(cohort.start_time)) / max(1e-9, float(CFG.BO_INTERVAL)))),
            "reason": str(reason),
            "theta_control": list(cohort.theta_control),
            "theta_full": list(cohort.theta_full),
            "context": list(cohort.context) if cohort.context is not None else None,
            "state": str(cohort.state) if cohort.state is not None else None,
            "total_tasks": int(cohort.total_tasks),
            "completed_tasks": int(cohort.completed_tasks),
            "unfinished_tasks": int(unfinished),
            "completion_ratio": float(confidence),
            "confidence": float(confidence),
            "cohort_cost": float(cost),
            "cohort_reward": float(-cost),
            "avg_energy_est": float(avg_energy),
            "censored_avg_delay": float(censored_avg_delay),
            "completed_avg_delay": float(completed_avg_delay),
            "effective_violation_rate": float(effective_vio_rate),
            "avg_lateness_effective": float(avg_lateness),
            "unfinished_ratio": float(unfinished_ratio),
            "pending_area_per_task": float(pending_area_per_task),
            "rt_arrivals": int(cohort.rt_arrivals),
            "batch_arrivals": int(cohort.batch_arrivals),
            "ai_arrivals": int(cohort.ai_arrivals),
            "rt_completed": int(cohort.rt_completed),
            "batch_completed": int(cohort.batch_completed),
            "ai_completed": int(cohort.ai_completed),
            **class_metrics,
        }

    def _finalize_ready_cohorts(self, now, force=False, reason="all_completed"):
        """完成的 cohort 才反馈给 BO；实验结束时可强制截尾反馈。"""
        if not self._use_cohort_feedback():
            return []
        finalized_rows = []
        for cohort in list(self.cohorts.values()):
            if cohort.finalized or cohort.total_tasks <= 0:
                continue
            ready = (cohort.completed_tasks >= cohort.total_tasks)
            if not (ready or force):
                continue
            final_reason = reason if ready else str(reason or "forced")
            row = self._cohort_metrics(cohort, now, reason=final_reason)
            cohort.finalized = True
            cohort.finalize_time = float(now)
            cohort.finalize_reason = final_reason
            cohort.feedback_cost = float(row["cohort_cost"])
            cohort.confidence = float(row["confidence"])
            self.cohort_finalized_total += 1
            self.cohort_feedback_rows.append(row)
            finalized_rows.append(row)

            # 只有有 agent 的 BO 方法才反馈；fixed 方法只记录 debug。
            if self.scheduler_type == "Boltzmann" and self.agent is not None:
                state_arg = cohort.state if getattr(self.agent, "use_state_partition", False) else None
                context_arg = cohort.context if getattr(self.agent, "use_context", False) else None
                self.agent.tell(cohort.theta_control, row["cohort_cost"], state=state_arg, context=context_arg)
                self.scheduler.update_beta(row["cohort_cost"])
        return finalized_rows

    def run_continuous(self, theta=None, external_samples=None, model_prior=None, window_end=None, eval_state=None, eval_context=None, feedback_control=None):
        """推进一个 BO 时间窗口，是全代码最核心的主循环。

        主要步骤：
        1) 在窗口开始前固定“本轮用于 ask/tell 的情景”；
        2) 在窗口内按事件时间推进；
        3) 处理任务到达、传输完成、执行完成；
        4) 窗口结束后计算 cost/reward；
        5) 用窗口开始前的 (state/context, theta) 和窗口反馈更新 agent。

        eval_state/eval_context 用于解决情景 BO 的时间对齐问题：
        ask 用的是哪个窗口开始前 context，tell 就必须存哪个 context，不能用窗口结束后的结果态。
        """
        t0 = time.time()
        if theta is not None:
            self.current_theta = list(theta)
            self.current_control_vector = list(theta)

        # 固定本轮反馈要绑定的“执行前情景”。如果调用者没有显式传入，就在窗口开始时现场取一次。
        if eval_state is None or eval_context is None:
            pre_state, _, _ = self.scenario_monitor.get_state(self.current_time)
            pre_context = self.scenario_monitor.get_context_vector(self.current_time)
            if eval_state is None:
                eval_state = pre_state
            if eval_context is None:
                eval_context = pre_context
        else:
            eval_context = list(eval_context)

        # external_samples and model_prior are ignored in output-sharing FBO mode
        target_end = self.next_bo_time if window_end is None else window_end
        window_scheduler_debugs = []
        if self._use_cohort_feedback():
            fb_control = list(feedback_control) if feedback_control is not None else list(getattr(self, "current_control_vector", self.current_theta))
            self._create_cohort(self.current_theta, fb_control, eval_state, eval_context)
        snapshot = self._new_window_snapshot(self.current_time, target_end)
        while True:
            next_event_time = self.events[0].time if self.events else float('inf')
            if next_event_time > target_end:
                self._accumulate_power_energy(snapshot, self.current_time, min(target_end, CFG.SESSION_DURATION))
                self.current_time = min(target_end, CFG.SESSION_DURATION)
                break
            ev = heapq.heappop(self.events)
            self._accumulate_power_energy(snapshot, self.current_time, ev.time)
            self.current_time = ev.time
            if ev.type == EventType.TASK_ARRIVAL:
                task = ev.payload
                try:
                    self.scheduler.current_time = self.current_time
                except Exception:
                    pass
                node_idx, _, _ = self.scheduler.select_node(task, self.nodes, self.current_theta)
                sched_debug = getattr(self.scheduler, "last_score_debug", {})
                if sched_debug:
                    sched_debug = dict(sched_debug)
                    self.perf_log.setdefault("scheduler_debug_last", []).append(sched_debug)
                    window_scheduler_debugs.append(sched_debug)
                task.arrival_node_idx = node_idx
                snapshot.alloc_counts[node_idx] += 1
                if hasattr(snapshot, "alloc_by_type") and task.task_type in snapshot.alloc_by_type:
                    snapshot.alloc_by_type[task.task_type][node_idx] += 1
                snapshot.arrivals += 1
                if task.task_type == "RT":
                    snapshot.rt_arrivals += 1
                elif task.task_type == "AI":
                    snapshot.ai_arrivals += 1
                else:
                    snapshot.batch_arrivals += 1
                snapshot.cpu_demand += task.cpu_cycles
                snapshot.data_demand += task.data_size
                node = self.nodes[node_idx]
                origin = task.origin_node_id if task.origin_node_id >= 0 else node_idx
                try:
                    est_energy_for_cohort, est_latency_for_cohort = node.estimate_metrics(task, self.current_time, origin_node_idx=origin)
                except Exception:
                    est_energy_for_cohort, est_latency_for_cohort = 0.0, 0.0
                self._register_task_to_cohort(task, node_idx, estimated_energy=est_energy_for_cohort, estimated_latency=est_latency_for_cohort)
                trans_delay = get_transmission_delay(origin, node_idx, task.data_size, include_local=True)
                if getattr(CFG, "USE_LINK_QUEUE", False):
                    link_key = (int(origin), int(node_idx))
                    trans_start = max(self.current_time, float(self.link_busy_until.get(link_key, self.current_time)))
                    trans_finish = trans_start + trans_delay
                    self.link_busy_until[link_key] = trans_finish
                else:
                    trans_finish = self.current_time + trans_delay
                task.transmission_energy = get_transmission_energy(origin, node_idx, task.data_size)
                snapshot.transmission_energy += task.transmission_energy
                snapshot.total_energy += task.transmission_energy
                snapshot.total_energy_real += task.transmission_energy
                self.cumulative_energy += task.transmission_energy
                self.cumulative_energy_real += task.transmission_energy
                heapq.heappush(self.events, Event(trans_finish, EventType.TRANS_FINISH, task))
                self.scenario_monitor.record_arrival(self.current_time)
                avg_util = np.mean([1.0 - n.cpu_free / max(1, n.cpu_total) for n in self.nodes])
                self.scenario_monitor.record_utilization(self.current_time, float(avg_util))
                self._schedule_next()
            elif ev.type == EventType.TRANS_FINISH:
                node = self.nodes[ev.payload.arrival_node_idx]
                node.enqueue_task(ev.payload)
                self._try_start(node)
            elif ev.type == EventType.TASK_FINISH:
                task = ev.payload
                node = self.nodes[task.arrival_node_idx]
                node.release(task)
                self._try_start(node)
                # 计算能耗已在事件间隔中按节点功率积分；任务完成时不再按任务重复累加 e_comp。
                task.energy_consumed = float(getattr(task, "transmission_energy", 0.0))
                delay = task.finish_time - task.create_time
                snapshot.completed += 1
                snapshot.total_delay += delay
                if task.task_type == "RT":
                    snapshot.rt_delay_sum += delay
                    snapshot.rt_completed += 1
                elif task.task_type == "AI":
                    snapshot.ai_delay_sum += delay
                    snapshot.ai_completed += 1
                else:
                    snapshot.batch_delay_sum += delay
                    snapshot.batch_completed += 1
                self.scenario_monitor.record_completion(self.current_time, delay)
                avg_util = np.mean([1.0 - n.cpu_free / max(1, n.cpu_total) for n in self.nodes])
                self.scenario_monitor.record_utilization(self.current_time, float(avg_util))
                is_violation = bool(task.finish_time > task.deadline)
                if is_violation:
                    snapshot.total_vio += 1
                slack = task.deadline - task.finish_time
                if slack >= 0:
                    snapshot.early_sum += slack
                    late_amount = 0.0
                else:
                    late_amount = -slack
                    snapshot.late_sum += late_amount

                # Per-class window diagnostics.
                prefix = "rt" if task.task_type == "RT" else ("ai" if task.task_type == "AI" else "batch")
                setattr(snapshot, f"{prefix}_late_sum", float(getattr(snapshot, f"{prefix}_late_sum", 0.0)) + float(late_amount))
                if is_violation:
                    setattr(snapshot, f"{prefix}_vio_count", int(getattr(snapshot, f"{prefix}_vio_count", 0)) + 1)
                # Only per-task transmission energy is directly attributable here; compute energy is window-integrated.
                setattr(snapshot, f"{prefix}_energy_sum", float(getattr(snapshot, f"{prefix}_energy_sum", 0.0)) + float(getattr(task, "energy_consumed", 0.0)))
                self._on_task_finished_cohort(task, delay)
            if self.current_time >= CFG.SESSION_DURATION and next_event_time == float('inf'):
                break
        state_after, stable, _ = self.scenario_monitor.get_state(self.current_time)
        backlog_count = self._system_backlog_count()
        metrics = snapshot.to_metrics(self.cumulative_energy, backlog=backlog_count, cumulative_energy_real=self.cumulative_energy_real)
        metrics["unfinished_end"] = backlog_count
        cohort_rows_this_window = self._finalize_ready_cohorts(self.current_time, force=False, reason="all_completed")
        metrics["feedback_mode"] = str(getattr(CFG, "FEEDBACK_MODE", "window"))
        metrics["cohort_id"] = int(self.current_cohort_id) if self.current_cohort_id is not None else None
        cur_cohort = self.cohorts.get(self.current_cohort_id) if self.current_cohort_id is not None else None
        metrics["cohort_arrivals"] = int(cur_cohort.total_tasks) if cur_cohort is not None else 0
        metrics["cohort_feedback_count"] = int(len(cohort_rows_this_window))
        metrics["cohort_feedback_cost_mean"] = float(np.mean([r["cohort_cost"] for r in cohort_rows_this_window])) if cohort_rows_this_window else np.nan
        metrics["cohort_active_count"] = int(sum(1 for c in self.cohorts.values() if (not c.finalized and c.total_tasks > 0)))
        metrics["cohort_finalized_total"] = int(self.cohort_finalized_total)
        metrics["cohort_pending_tasks"] = int(sum(len(c.unfinished_tasks) for c in self.cohorts.values() if not c.finalized))
        metrics = _cbo_metric_reference_patch(self, metrics)
        self.scenario_monitor.update_window_feedback(metrics)
        log_state = eval_state if eval_state is not None else state_after
        self.window_history.append((log_state, self.current_theta, metrics["cost"]))

        # 窗口结束后的 context 只用于诊断日志，不用于本轮 BO tell。
        context_metrics_after = {**self.scenario_monitor.last_metrics, **self.scenario_monitor.window_feedback}
        context_vec_after = self.scenario_monitor.get_context_vector(self.current_time, metrics=context_metrics_after)
        context_vec_for_bo = list(eval_context) if eval_context is not None else context_vec_after

        if (not self._use_cohort_feedback()) and self.scheduler_type == "Boltzmann" and self.agent is not None and not getattr(self, "disable_internal_agent_tell", False):
            control_sample = list(getattr(self, "current_control_vector", self.current_theta))
            state_arg = eval_state if getattr(self.agent, "use_state_partition", False) else None
            context_arg = context_vec_for_bo if getattr(self.agent, "use_context", False) else None
            self.agent.tell(control_sample, metrics["cost"], state=state_arg, context=context_arg)
            self.scheduler.update_beta(metrics["cost"])

        if window_scheduler_debugs:
            last_sched_debug = dict(window_scheduler_debugs[-1])
            alpha_vals = []
            for d in window_scheduler_debugs:
                try:
                    a = d.get("scheduler_alpha", None)
                    if a is not None and np.isfinite(float(a)):
                        alpha_vals.append(float(a))
                except Exception:
                    pass
            last_sched_debug["scheduler_alpha_mean"] = float(np.mean(alpha_vals)) if alpha_vals else None
        else:
            last_sched_debug = {}
        self._last_window_scheduler_debug = last_sched_debug
        self._record_window_log(metrics, log_state, stable, self.current_theta, context_vec=context_vec_for_bo)
        self.perf_log.setdefault("feedback_state_after", []).append(str(state_after) if state_after is not None else "None")
        self.perf_log.setdefault("context_vector_after", []).append(list(context_vec_after) if context_vec_after is not None else [])
        self.next_bo_time = target_end + CFG.BO_INTERVAL
        self.bo_step += 1
        return (state_after, self.current_theta, metrics["reward"], stable, metrics, time.time() - t0)

    def _schedule_next(self):
        """从工作负载生成器里取下一个任务，并压入事件堆。"""
        task, arr = self.workload.get_next_task(self.current_time)
        if task and arr <= CFG.SESSION_DURATION + 1.0:
            heapq.heappush(self.events, Event(arr, EventType.TASK_ARRIVAL, task))

    def _try_start(self, node):
        """尝试把节点 ready_queue 中可执行的任务启动起来。"""
        for t in list(node.ready_queue):
            if node.allocate(t):
                node.ready_queue.remove(t)
                speed = node.effective_speed(t)
                t.finish_time = self.current_time + t.cpu_cycles / (speed + 1e-9)
                heapq.heappush(self.events, Event(t.finish_time, EventType.TASK_FINISH, t))

# ==========================================
# 6. 绘图与分析 (Plotting)
# ==========================================
def save_detailed_data(baseline_log, fed_log, rr_log=None):
    logs = [("Baseline", baseline_log), ("Federated", fed_log)]
    if rr_log: logs.append(("RoundRobin", rr_log))
    for label, log in logs:
        alphas = log.get("alpha", [])
        feature_cols = {}
        for idx, name in enumerate(CFG.FEATURE_NAMES):
            feature_cols[name] = [a[idx] if isinstance(a, (list, tuple)) and len(a) > idx else None for a in alphas]
        df_dict = {
            "Time": log["time"],
            "Reward": log.get("reward", []),
            "Total_Energy": log["total_energy"],
            "Cumulative_Energy": log["cumulative_energy"],
            "Avg_Latency": log["avg_delay"],
            "Avg_Latency_RT": log.get("avg_delay_rt", []),
            "Avg_Latency_Batch": log.get("avg_delay_batch", []),
            "Avg_Latency_AI": log.get("avg_delay_ai", []),
            "Violation_Rate": log.get("vio_rate", []),
            "SLA_Success_Rate": log["sla_success_rate"],
            "Avg_Energy_Per_Task": log.get("avg_energy", []),
            "Avg_Earliness": log.get("avg_earliness", []),
            "Avg_Lateness": log.get("avg_lateness", []),
        }
        df_dict.update(feature_cols)
        df = pd.DataFrame(df_dict)
        csv_path = os.path.join(SAVE_DIR, f"detailed_data_{label}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed data for {label} to {csv_path}")

def smooth(data, window=5):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window) / window, mode='valid')

def ema_smooth(scalars, weight=0.85):
    if not scalars: return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_comparison(baseline_log, fed_log, rr_log=None):
    min_len = min(len(baseline_log['reward']), len(fed_log['reward']))
    if rr_log: min_len = min(min_len, len(rr_log['reward']))
    iters = np.arange(min_len)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    ax1 = axes[0]
    ax1.grid(True, linestyle='-', alpha=0.7)
    if rr_log:
        lat_rr = ema_smooth(rr_log['avg_delay'][:min_len], weight=0.85)
        ax1.plot(iters, lat_rr, label="Round Robin", color='orange', linestyle='--')
    lat_base = ema_smooth(baseline_log['avg_delay'][:min_len], weight=0.85)
    lat_fed = ema_smooth(fed_log['avg_delay'][:min_len], weight=0.85)
    ax1.plot(iters, lat_base, label="BO-Local (Baseline)", color='blue', linestyle='-.')
    ax1.plot(iters, lat_fed, label="Proposed Algorithm (BO-Fed)", color='green', linewidth=2.5, linestyle='-')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Average Latency (s)")
    ax1.set_title("Latency comparison")
    ax1.legend(loc='upper right')
    ax2 = axes[1]
    ax2.grid(True, linestyle='-', alpha=0.7)
    if rr_log:
        ce_rr = np.array(rr_log['cumulative_energy'][:min_len])
        ce_base = np.array(baseline_log['cumulative_energy'][:min_len])
        ce_fed = np.array(fed_log['cumulative_energy'][:min_len])
        savings_base = ce_rr - ce_base
        savings_fed = ce_rr - ce_fed
        ax2.plot(iters, savings_base, label="BO-Local Savings", color='blue', linestyle='-.')
        ax2.plot(iters, savings_fed, label="Proposed Algorithm Savings", color='purple', linewidth=2.0)
        ax2.set_ylabel("Cumulative Energy Savings (J)")
    else:
        ax2.plot(iters, baseline_log['cumulative_energy'][:min_len], label="BO-Local", color='blue', linestyle='-.')
        ax2.plot(iters, fed_log['cumulative_energy'][:min_len], label="Proposed Algorithm", color='purple', linewidth=2.0)
        ax2.set_ylabel("Cumulative Energy (J)")
    ax2.set_xlabel("Iterations")
    ax2.set_title("Cumulative Energy Savings Comparison")
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "paper_style_convergence.png"), dpi=300)
def plot_best_so_far(baseline_log, fed_log, rr_log=None):
    def bsf(seq):
        if not seq: return []
        out = []
        m = -float("inf")
        for x in seq:
            m = max(m, x)
            out.append(m)
        return out
    min_len = min(len(baseline_log.get("reward", [])), len(fed_log.get("reward", [])))
    if rr_log: min_len = min(min_len, len(rr_log.get("reward", [])))
    iters = np.arange(min_len)
    r_base = bsf(baseline_log["reward"][:min_len])
    r_fed = bsf(fed_log["reward"][:min_len])
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.plot(iters, r_base, label="BO-Local Best-So-Far", color='blue', linestyle='-.')
    ax.plot(iters, r_fed, label="BO-Fed Best-So-Far", color='green', linewidth=2.0)
    if rr_log:
        r_rr = bsf(rr_log["reward"][:min_len])
        ax.plot(iters, r_rr, label="Round-Robin Best-So-Far", color='orange', linestyle='--')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Reward (higher is better)")
    ax.set_title("Best-So-Far Reward")
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "trend_bsf.png"), dpi=300)
def plot_convergence_metrics(baseline_log, fed_log, rr_log=None):
    min_len = min(len(baseline_log['time']), len(fed_log['time']))
    if rr_log: min_len = min(min_len, len(rr_log['time']))
    iters = np.arange(min_len)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    s_base = ema_smooth(baseline_log['sla_success_rate'][:min_len], weight=0.85)
    s_fed = ema_smooth(fed_log['sla_success_rate'][:min_len], weight=0.85)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].plot(iters[:len(s_base)], s_base, label="BO-Local", color='blue', linestyle='-.')
    axes[0].plot(iters[:len(s_fed)], s_fed, label="BO-Fed", color='green')
    if rr_log:
        s_rr = ema_smooth(rr_log['sla_success_rate'][:min_len], weight=0.85)
        axes[0].plot(iters[:len(s_rr)], s_rr, label="Round-Robin", color='orange', linestyle='--')
    axes[0].set_title("SLA Success Rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower right')
    e_base = ema_smooth(baseline_log.get('avg_energy', [])[:min_len], weight=0.85)
    e_fed = ema_smooth(fed_log.get('avg_energy', [])[:min_len], weight=0.85)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    if e_base:
        axes[1].plot(iters[:len(e_base)], e_base, label="BO-Local Avg Energy", color='blue', linestyle='-.')
    if e_fed:
        axes[1].plot(iters[:len(e_fed)], e_fed, label="BO-Fed Avg Energy", color='purple')
    if rr_log and rr_log.get('avg_energy'):
        e_rr = ema_smooth(rr_log['avg_energy'][:min_len], weight=0.85)
        axes[1].plot(iters[:len(e_rr)], e_rr, label="Round-Robin Avg Energy", color='orange', linestyle='--')
    axes[1].set_title("Average Energy per Task (J)")
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "convergence_metrics.png"), dpi=300)

def plot_acq_process(acq_history):
    if not acq_history: return
    steps_to_show = acq_history[-4:]
    fig, axes = plt.subplots(len(steps_to_show), 1, figsize=(10, 4 * len(steps_to_show)))
    if len(steps_to_show) == 1: axes = [axes]
    for i, step_data in enumerate(steps_to_show):
        ax = axes[i]
        cands = np.array(step_data["candidates"])
        acq_vals = np.array(step_data["acq_values"])
        best = step_data["best_selected"]
        if len(cands) > 0 and len(cands) == len(acq_vals):
            sc = ax.scatter(cands[:, 0], cands[:, 1], c=acq_vals, cmap='viridis', s=50)
            plt.colorbar(sc, ax=ax)
        if best is not None and len(best) >= 2:
            ax.scatter(best[0], best[1], color='red', s=200, marker='*')
        ax.set_title(f"BO Step {step_data['step']}")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "acq_process.png"))

# ==========================================
# 7. 主流程 (Main)
# ==========================================
# 你可以把主流程理解为两层：
# - 内层：run_continuous() 在一个时间窗口内做事件驱动仿真。
# - 外层：run_baseline_batch()/run_federated_batch() 每轮先 ask theta，再跑一个窗口，再 tell 反馈。
def aggregate_logs(logs):
    if not logs:
        return {}
    min_len = min(len(l.get("reward", [])) for l in logs)
    result = {}
    all_keys = set()
    for l in logs:
        all_keys.update(l.keys())
    for key in all_keys:
        if key == "state":
            continue
        val0 = next((l.get(key) for l in logs if isinstance(l.get(key), list)), None)
        if not isinstance(val0, list):
            continue
        if min_len == 0:
            result[key] = []
            continue
        # build matrix padded to min_len so shorter lists won't raise IndexError
        matrix = []
        for l in logs:
            lst = l.get(key, []) or []
            row = []
            for j in range(min_len):
                if j < len(lst):
                    row.append(lst[j])
                else:
                    row.append(None)
            matrix.append(row)

        # find a non-None sample to determine element shape
        first_non_none = None
        for row in matrix:
            for v in row:
                if v is not None:
                    first_non_none = v
                    break
            if first_non_none is not None:
                break
        if first_non_none is None:
            result[key] = [None] * min_len
            continue
        if key in {
            "deploy_policy", "deploy_source", "used_theta_source",
            "tr_update_mode", "tr_update_signal", "cbo_tr_update_reason",
            "selected_reason", "selected_candidate_source", "selected_source",
            "actual_tr_anchor_mode", "actual_tr_anchor_source", "actual_tr_anchor_reason",
            "anchor_fallback_reason", "runtime_anchor_override",
            "state_beta_boost_reason", "service_guard_mode", "service_guard_reason",
            "macro_context_key", "energy_metric_source", "cbo_reference_mode", "cbo_reference_status", "cbo_objective_mode", "bo_training_cost_source",
            "scheduler_tradeoff_mode", "scheduler_score_norm_mode",
        }:
            result[key] = [next((row[j] for row in matrix if row[j] is not None), None) for j in range(min_len)]
            continue

        try:
            if isinstance(first_non_none, (list, tuple, np.ndarray)):
                inner_len = len(first_non_none)
                arr = np.full((len(matrix), min_len, inner_len), np.nan, dtype=float)
                for i in range(len(matrix)):
                    for j in range(min_len):
                        v = matrix[i][j]
                        if v is None:
                            continue
                        arr[i, j, :] = np.array(v, dtype=float)
                mean_vals = np.nanmean(arr, axis=0)
                result[key] = mean_vals.tolist()
            else:
                arr = np.full((len(matrix), min_len), np.nan, dtype=float)
                for i in range(len(matrix)):
                    for j in range(min_len):
                        v = matrix[i][j]
                        if v is None:
                            arr[i, j] = np.nan
                        else:
                            try:
                                arr[i, j] = float(v)
                            except Exception:
                                arr[i, j] = np.nan
                mean_vals = np.nanmean(arr, axis=0)
                result[key] = mean_vals.tolist()
        except Exception:
            result[key] = [None] * min_len
    return result

def summarize_metrics(logs, label):
    valid_logs = [
        l for l in logs
        if l.get("reward") and l.get("avg_delay") and l.get("sla_success_rate") and l.get("cumulative_energy")
    ]
    if not valid_logs:
        print(f"[{label}] Metrics Summary: no valid logs")
        return

    rewards = [float(np.mean(l["reward"])) for l in valid_logs]
    delays = [float(np.mean(l["avg_delay"])) for l in valid_logs]
    sla_success = [float(np.mean(l["sla_success_rate"])) for l in valid_logs]
    total_energy = [float(l["cumulative_energy"][-1]) for l in valid_logs]
    print(f"[{label}] Metrics Summary: Avg Reward: {np.mean(rewards):.3f}, Avg Delay: {np.mean(delays):.3f}s, SLA Success: {np.mean(sla_success):.3f}, Total Energy: {np.mean(total_energy):.2f} J")

def _eval_theta_once(theta, batch_size, seed):
    fac = ConnectedFactory(fid=0, name="Eval", seed=seed, node_config=CFG.NODES_CFG)
    fac.reset(use_batch=False)
    horizon = max(CFG.BO_INTERVAL, float(batch_size) / max(1e-9, CFG.BATCH_POISSON_LAMBDA))
    fac.run_continuous(theta, window_end=horizon)
    log = fac.perf_log
    idx = -1
    res = {}
    res.update(theta_to_named_dict(theta))
    res.update({
        "Reward": log["reward"][idx],
        "Total_Energy": log["total_energy"][idx],
        "Avg_Latency": log["avg_delay"][idx],
        "Avg_Latency_RT": log.get("avg_delay_rt", [None])[idx],
        "Avg_Latency_Batch": log.get("avg_delay_batch", [None])[idx],
        "Avg_Latency_AI": log.get("avg_delay_ai", [None])[idx],
        "SLA_Success_Rate": log["sla_success_rate"][idx],
        "Violation_Rate": log.get("vio_rate", [None])[idx],
        "Avg_Energy_Per_Task": log.get("avg_energy", [None])[idx],
    })
    return res

def run_param_analysis(samples=40, local_delta=0.08):
    bounds = get_control_bounds(CFG.DIM_THETA)
    low = torch.tensor(bounds[0], dtype=torch.double)
    high = torch.tensor(bounds[1], dtype=torch.double)
    base = default_control_vector(fill=1.5)
    sample_gen = torch.Generator().manual_seed(resolve_base_seed(CFG.BASE_SEED, stream=700))
    thetas = []
    thetas.append(base)
    dim = len(low)
    for _ in range(samples):
        r = torch.rand(dim, dtype=torch.double, generator=sample_gen)
        t = (low + r * (high - low)).tolist()
        thetas.append(t)
    for d in range(dim):
        td = base.copy()
        td[d] = float(max(low[d], base[d] * (1.0 - local_delta)))
        thetas.append(td)
        td2 = base.copy()
        td2[d] = float(min(high[d], base[d] * (1.0 + local_delta)))
        thetas.append(td2)
    rows = []
    for t in thetas:
        r = _eval_theta_once(t, CFG.TASKS_PER_BATCH, CFG.BASE_SEED)
        rows.append(r)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "param_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    cols_theta = list(CFG.FEATURE_NAMES)
    cols_metric = ["Reward", "Total_Energy", "Avg_Latency", "SLA_Success_Rate"]
    corr = {}
    for m in cols_metric:
        corr[m] = []
        for c in cols_theta:
            s = df[[c, m]].corr(method="spearman").iloc[0, 1]
            corr[m].append(s)
    corr_df = pd.DataFrame(corr, index=cols_theta)
    out_corr = os.path.join(SAVE_DIR, "param_sensitivity_corr.csv")
    corr_df.to_csv(out_corr)

def run_extreme_param_test():
    prev_fixed = CFG.USE_FIXED_RNG
    CFG.USE_FIXED_RNG = True
    extremes = [
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        [3.0, 0.8, 0.8, 0.8, 2.5, 2.0],
        [0.8, 3.0, 0.8, 2.5, 0.8, 2.0],
        [0.8, 0.8, 3.0, 2.0, 2.0, 0.8],
        [2.5, 1.5, 1.2, 1.0, 2.5, 2.0],
    ]
    rows = []
    for t in extremes:
        r = _eval_theta_once(t, CFG.TASKS_PER_BATCH, CFG.BASE_SEED)
        rows.append(r)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "extreme_param_test.csv")
    df.to_csv(out_csv, index=False)
    CFG.USE_FIXED_RNG = prev_fixed

def run_param_scan(dim_name="RT_E", points=8):
    names = list(CFG.FEATURE_NAMES)
    if dim_name not in names: dim_idx = 0
    else: dim_idx = names.index(dim_name)
    bounds = get_control_bounds(CFG.DIM_THETA)
    low = torch.tensor(bounds[0], dtype=torch.double)
    high = torch.tensor(bounds[1], dtype=torch.double)
    base = default_control_vector(fill=1.5)
    xs = np.linspace(float(low[dim_idx]), float(high[dim_idx]), points)
    rows = []
    for v in xs:
        th = base.copy()
        th[dim_idx] = float(v)
        r = _eval_theta_once(th, CFG.TASKS_PER_BATCH, CFG.BASE_SEED)
        rows.append(r)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, f"scan_{dim_name}.csv")
    df.to_csv(out_csv, index=False)


# ==========================================
# 5.5 参数敏感度与节点偏好诊断
# ==========================================
SENSITIVITY_SCENARIO_PRESETS = {
    "default": None,  # 使用当前 CFG.TASK_TYPE_PROBS
    "balanced": {"RT": 0.34, "Batch": 0.33, "AI": 0.33},
    "rt_high": {"RT": 0.70, "Batch": 0.20, "AI": 0.10},
    "batch_high": {"RT": 0.10, "Batch": 0.70, "AI": 0.20},
    "ai_high": {"RT": 0.20, "Batch": 0.10, "AI": 0.70},
}


def _normalize_task_probs(probs):
    vals = {t: float(probs.get(t, 0.0)) for t in TASK_TYPE_ORDER}
    total = sum(vals.values())
    if total <= 0:
        return dict(CFG.TASK_TYPE_PROBS)
    return {t: vals[t] / total for t in TASK_TYPE_ORDER}


def _parse_sensitivity_scenarios(spec):
    """解析敏感度场景。

    支持：
    - default,rt_high,batch_high,ai_high
    - 自定义 name:RT,Batch,AI，例如 mix1:0.2,0.5,0.3
    """
    if not spec:
        spec = "default,rt_high,batch_high,ai_high"
    scenarios = []
    for raw in str(spec).split(','):
        raw = raw.strip()
        if not raw:
            continue
        if ':' in raw:
            name, vals = raw.split(':', 1)
            parts = [float(x) for x in vals.replace('/', ',').split(',') if x.strip()]
            if len(parts) != 3:
                raise ValueError(f"自定义场景 {raw} 需要三个比例：RT,Batch,AI")
            probs = _normalize_task_probs({"RT": parts[0], "Batch": parts[1], "AI": parts[2]})
            scenarios.append((name.strip(), probs))
        else:
            key = raw.lower()
            if key not in SENSITIVITY_SCENARIO_PRESETS:
                raise ValueError(f"未知敏感度场景 {raw}. 可选: {list(SENSITIVITY_SCENARIO_PRESETS)} 或 name:rt,batch,ai")
            preset = SENSITIVITY_SCENARIO_PRESETS[key]
            probs = dict(CFG.TASK_TYPE_PROBS) if preset is None else _normalize_task_probs(preset)
            scenarios.append((key, probs))
    return scenarios


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def _mean_or_nan(values):
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _std_or_nan(values):
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.std(vals)) if vals else np.nan


def _sum_alloc_by_type(log):
    """把多个窗口的分任务分配计数相加。"""
    result = {t: [0 for _ in CFG.NODES_CFG] for t in TASK_TYPE_ORDER}
    for item in log.get("alloc_by_type", []):
        if not isinstance(item, dict):
            continue
        for t in TASK_TYPE_ORDER:
            vals = item.get(t, [])
            for i, v in enumerate(vals[:len(CFG.NODES_CFG)]):
                try:
                    result[t][i] += int(v)
                except Exception:
                    pass
    return result


def _allocation_top_nodes(alloc_counts, k=3):
    order = sorted(range(len(alloc_counts)), key=lambda i: (-alloc_counts[i], i))
    return order[:k]


def _eval_theta_windows(theta, seed, windows=3, scenario_name="default", task_probs=None):
    """固定一个 theta 连续运行若干 BO 窗口，用于敏感度诊断。

    注意：这里不调用 agent.ask，因此不是 BO 搜索；只是评价某个 theta 在同一任务分布下的平均表现。
    """
    old_probs = dict(CFG.TASK_TYPE_PROBS)
    if task_probs is not None:
        CFG.TASK_TYPE_PROBS = _normalize_task_probs(task_probs)
    try:
        fac = ConnectedFactory(fid=0, name=f"Sensitivity_{scenario_name}", seed=seed, node_config=CFG.NODES_CFG)
        fac.reset(use_batch=False)
        for _ in range(max(1, int(windows))):
            fac.current_control_label = "SensitivityFixedTheta"
            fac.run_continuous(theta)
        log = fac.perf_log
        rewards = log.get("reward", [])
        costs = [-_safe_float(x) for x in rewards]
        alloc_by_type = _sum_alloc_by_type(log)
        row = {
            "Seed": int(seed),
            "Windows": int(windows),
            "Mean_Cost": _mean_or_nan(costs),
            "Std_Cost_Window": _std_or_nan(costs),
            "Mean_Reward": _mean_or_nan(rewards),
            "Mean_Avg_Delay": _mean_or_nan(log.get("avg_delay", [])),
            "Mean_Avg_Delay_RT": _mean_or_nan(log.get("avg_delay_rt", [])),
            "Mean_Avg_Delay_Batch": _mean_or_nan(log.get("avg_delay_batch", [])),
            "Mean_Avg_Delay_AI": _mean_or_nan(log.get("avg_delay_ai", [])),
            "Mean_Avg_Energy": _mean_or_nan(log.get("avg_energy", [])),
            "Mean_Total_Energy": _mean_or_nan(log.get("total_energy", [])),
            "Final_Cumulative_Energy": _safe_float(log.get("cumulative_energy", [np.nan])[-1]) if log.get("cumulative_energy") else np.nan,
            "Mean_SLA_Success": _mean_or_nan(log.get("sla_success_rate", [])),
            "Mean_Violation_Rate": _mean_or_nan(log.get("vio_rate", [])),
            "Mean_Backlog": _mean_or_nan(log.get("backlog", [])),
            "Mean_Arrivals": _mean_or_nan(log.get("arrivals_total", [])),
            "Mean_Completed": _mean_or_nan(log.get("completed_total", [])),
            "Alloc_By_Type_JSON": json.dumps(alloc_by_type, ensure_ascii=False),
            "Alloc_RT_Top3": json.dumps(_allocation_top_nodes(alloc_by_type.get("RT", []), 3)),
            "Alloc_Batch_Top3": json.dumps(_allocation_top_nodes(alloc_by_type.get("Batch", []), 3)),
            "Alloc_AI_Top3": json.dumps(_allocation_top_nodes(alloc_by_type.get("AI", []), 3)),
        }
        return row
    finally:
        CFG.TASK_TYPE_PROBS = old_probs


def _candidate_scores_for_task(theta, task_type, origin_node_id=0, seed=0, norm_mode="rolling"):
    """静态计算某任务类型在代表性 origin 下的候选节点排序。

    这个函数不抽样、不执行任务，只复用调度器的 score/feasibility/opportunity 逻辑，
    用来回答：改变某个参数是否改变了该任务的节点偏好排序。
    """
    props = CFG.TASK_PROPS[task_type]
    task = Task(
        id=f"pref-{task_type}-{origin_node_id}",
        create_time=0.0,
        data_size=props["data"],
        cpu_req=props["cpu"],
        duration_base=props["dur"],
        task_type=task_type,
        deadline_factor=props["deadline_factor"],
        origin_node_id=int(origin_node_id),
    )
    nodes = [Node(cfg) for cfg in CFG.NODES_CFG]
    scheduler = ConstrainedBoltzmannScheduler(np_rng=np.random.default_rng(resolve_base_seed(seed, stream=810)), norm_mode=norm_mode)
    scheduler.current_time = 0.0
    latency_weights, energy_weights, _ = split_task_weights(theta)
    controls = extract_scheduler_controls(theta)
    scheduler.beta = float(controls.get("beta", scheduler.beta))
    latency_w = float(latency_weights.get(task_type, 1.0))
    energy_w = float(energy_weights.get(task_type, 1.0))
    base_risk_w = float(CFG.TASK_RISK_WEIGHTS.get(task_type, CFG.DEADLINE_WEIGHT))
    risk_w = base_risk_w * float(controls.get("risk_scale", getattr(CFG, "RISK_SCALE_DEFAULT", 1.0)))
    queue_w = float(controls.get("queue_w", getattr(CFG, "QUEUE_WEIGHT_DEFAULT", 1.0))) if getattr(CFG, "USE_QUEUE_PRESSURE_SCORE", True) else 0.0

    raw_infos = [scheduler._node_score(task, idx, node, latency_w, energy_w, risk_w) for idx, node in enumerate(nodes)]
    candidates, feasibility_debug = scheduler._apply_feasibility_filter(task, nodes, raw_infos, controls)
    if not candidates:
        candidates = [min(raw_infos, key=lambda c: c.get("latency_total", 1e18))]
        feasibility_debug["fallback_reason"] = "empty_after_all_filters_static_diag"
    norm_e, norm_l, norm_r, norm_q, norm_debug = scheduler._compute_norms_with_queue(task, candidates)
    alpha, _alpha_source, tradeoff_mode = scheduler._resolve_scheduler_alpha(task_type, latency_w, energy_w)
    for i, c in enumerate(candidates):
        c["norm_e"] = float(norm_e[i])
        c["norm_l"] = float(norm_l[i])
        c["norm_risk"] = float(norm_r[i])
        c["norm_queue"] = float(norm_q[i])
        score, latency_energy_component, energy_component = scheduler._score_candidate_components(
            c, latency_w, energy_w, risk_w, queue_w=queue_w, alpha=alpha, tradeoff_mode=tradeoff_mode
        )
        c["latency_energy_component"] = latency_energy_component
        c["base_latency_energy_score"] = latency_energy_component
        c["service_component"] = latency_energy_component  # Deprecated alias.
        c["energy_component"] = energy_component
        c["score"] = score
    opportunity_candidates, opportunity_debug = scheduler._apply_opportunity_window(candidates, controls)
    opp_nodes = {int(c["node_idx"]) for c in opportunity_candidates}
    sorted_candidates = sorted(candidates, key=lambda c: (float(c.get("score", np.inf)), int(c.get("node_idx", 999))))
    candidate_rank = {int(c["node_idx"]): i + 1 for i, c in enumerate(sorted_candidates)}
    sorted_opp = sorted(opportunity_candidates, key=lambda c: (float(c.get("score", np.inf)), int(c.get("node_idx", 999))))
    opp_rank = {int(c["node_idx"]): i + 1 for i, c in enumerate(sorted_opp)}

    rows = []
    for c in sorted_candidates:
        node_idx = int(c["node_idx"])
        node_cfg = CFG.NODES_CFG[node_idx]
        rows.append({
            "Task_Type": task_type,
            "Origin_Node": int(origin_node_id),
            "Node": node_idx,
            "Node_Role": str(node_cfg.get("role", "")),
            "Node_Workshop": _get_node_site_from_cfg(node_cfg, default_site=node_idx // 2),
            "Is_Cloud": bool(_node_is_cloud(node_cfg)),
            "Rank_Candidate": int(candidate_rank[node_idx]),
            "Rank_Opportunity": int(opp_rank[node_idx]) if node_idx in opp_rank else np.nan,
            "In_Opportunity": int(node_idx in opp_nodes),
            "Score": float(c.get("score", np.nan)),
            "Latency": float(c.get("latency_total", np.nan)),
            "Energy": float(c.get("energy_raw", np.nan)),
            "Risk": float(c.get("deadline_risk", np.nan)),
            "Queue": float(c.get("queue_pressure", np.nan)),
            "Slack": float(c.get("slack", np.nan)),
            "Predicted_Finish": float(c.get("predicted_finish", np.nan)),
            "Feasible_Count": int(len(candidates)),
            "Opportunity_Count": int(len(opportunity_candidates)),
            "Fallback_Reason": feasibility_debug.get("fallback_reason"),
            "Cloud_Pressure": feasibility_debug.get("cloud_pressure"),
        })
    return rows


def _diagnose_node_preferences(theta, seed=0, origins=None, norm_mode="rolling"):
    if origins is None:
        origins = [cfg["id"] for cfg in CFG.NODES_CFG if (not _node_is_cloud(cfg)) and cfg.get("id") in [0, 2, 4, 6, 8]]
        if not origins:
            origins = [0]
    rows = []
    for task_type in TASK_TYPE_ORDER:
        for origin in origins:
            rows.extend(_candidate_scores_for_task(theta, task_type, origin_node_id=origin, seed=seed, norm_mode=norm_mode))
    return rows


def _rank_map(df, task_type, origin, rank_col="Rank_Candidate"):
    sub = df[(df["Task_Type"] == task_type) & (df["Origin_Node"] == origin)]
    out = {}
    for _, r in sub.iterrows():
        val = r.get(rank_col, np.nan)
        if pd.notna(val):
            out[int(r["Node"])] = int(val)
    return out


def _topk_from_rank_map(rank_map, k=3):
    return [node for node, _ in sorted(rank_map.items(), key=lambda kv: (kv[1], kv[0]))[:k]]


def _kendall_distance_rank_maps(a, b):
    nodes = sorted(set(a.keys()) | set(b.keys()))
    n = len(nodes)
    if n <= 1:
        return 0.0
    missing_rank = n + 1
    inv = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            ni, nj = nodes[i], nodes[j]
            da = (a.get(ni, missing_rank) - a.get(nj, missing_rank))
            db = (b.get(ni, missing_rank) - b.get(nj, missing_rank))
            if da == 0 or db == 0:
                continue
            total += 1
            if da * db < 0:
                inv += 1
    return float(inv / total) if total > 0 else 0.0


def _preference_compare_rows(base_pref_df, pref_df, scenario_name, dimension, value, theta_id):
    rows = []
    for task_type in TASK_TYPE_ORDER:
        origins = sorted(set(base_pref_df[base_pref_df["Task_Type"] == task_type]["Origin_Node"].tolist()) |
                         set(pref_df[pref_df["Task_Type"] == task_type]["Origin_Node"].tolist()))
        for origin in origins:
            base_rank = _rank_map(base_pref_df, task_type, origin)
            cur_rank = _rank_map(pref_df, task_type, origin)
            if not base_rank or not cur_rank:
                continue
            base_top1 = _topk_from_rank_map(base_rank, 1)[0]
            cur_top1 = _topk_from_rank_map(cur_rank, 1)[0]
            base_top3 = set(_topk_from_rank_map(base_rank, 3))
            cur_top3 = set(_topk_from_rank_map(cur_rank, 3))
            union = base_top3 | cur_top3
            jaccard_dist = 1.0 - (len(base_top3 & cur_top3) / max(1, len(union)))
            rows.append({
                "Scenario": scenario_name,
                "Dimension": dimension,
                "Value": value,
                "Theta_ID": theta_id,
                "Task_Type": task_type,
                "Origin_Node": int(origin),
                "Top1_Base": int(base_top1),
                "Top1_Current": int(cur_top1),
                "Top1_Changed": int(base_top1 != cur_top1),
                "Top3_Base": json.dumps(sorted(base_top3)),
                "Top3_Current": json.dumps(sorted(cur_top3)),
                "Top3_Jaccard_Distance": float(jaccard_dist),
                "Kendall_Distance": float(_kendall_distance_rank_maps(base_rank, cur_rank)),
            })
    return rows


def _spearman_corr_safe(x, y):
    try:
        ser = pd.Series(x, dtype="float64")
        val = ser.corr(pd.Series(y, dtype="float64"), method="spearman")
        return float(val) if pd.notna(val) else np.nan
    except Exception:
        return np.nan


def run_full_sensitivity_analysis(points=5, seeds=2, windows=3, scenario_spec="default,rt_high,batch_high,ai_high", output_dir=None, greedy=False, pref_norm_mode="rolling"):
    """保留 11 维控制量，做参数敏感度 + 节点偏好诊断。

    输出文件：
    - sensitivity_eval_raw.csv：每个 theta、seed 的窗口评价原始结果；
    - sensitivity_eval_summary.csv：跨 seed 聚合后的性能结果；
    - sensitivity_preference_raw.csv：每个 theta 下 RT/Batch/AI 的候选节点排序；
    - sensitivity_preference_compare.csv：相对 baseline 的排序变化；
    - sensitivity_dimension_summary.csv：每个参数维度的综合敏感度结论。
    """
    root = output_dir or os.path.abspath("sensitivity_full_11d")
    os.makedirs(root, exist_ok=True)
    old_random = CFG.USE_BOLTZMANN_RANDOM
    if greedy:
        CFG.USE_BOLTZMANN_RANDOM = False
    try:
        bounds = get_control_bounds(CFG.DIM_THETA)
        low = np.array(bounds[0], dtype=float)
        high = np.array(bounds[1], dtype=float)
        base_theta = default_control_vector(fill=1.5)
        scenarios = _parse_sensitivity_scenarios(scenario_spec)
        eval_rows = []
        pref_rows = []
        pref_cmp_rows = []

        for scenario_name, probs in scenarios:
            print(f"\n=== Sensitivity scenario: {scenario_name}, probs={probs} ===")
            # baseline preference for each scenario; static preference does not depend on task probs,
            # but keeping scenario in output makes downstream analysis easier.
            base_pref_rows = _diagnose_node_preferences(base_theta, seed=CFG.BASE_SEED, norm_mode=pref_norm_mode)
            base_pref_df = pd.DataFrame(base_pref_rows)
            for r in base_pref_rows:
                rr = dict(r)
                rr.update({"Scenario": scenario_name, "Dimension": "BASE", "Value": np.nan, "Theta_ID": "BASE"})
                rr.update(theta_to_named_dict(base_theta))
                pref_rows.append(rr)

            theta_jobs = []
            theta_jobs.append(("BASE", "BASE", np.nan, list(base_theta)))
            for dim_idx, dim_name in enumerate(CFG.FEATURE_NAMES):
                xs = np.linspace(low[dim_idx], high[dim_idx], max(2, int(points)))
                for v in xs:
                    theta = list(base_theta)
                    theta[dim_idx] = float(v)
                    theta_id = f"{dim_name}={float(v):.6g}"
                    theta_jobs.append((dim_name, theta_id, float(v), theta))

            for job_idx, (dim_name, theta_id, value, theta) in enumerate(theta_jobs, start=1):
                print(f"  [{scenario_name}] {job_idx}/{len(theta_jobs)} {theta_id}")
                # 静态节点偏好诊断：每个 theta 只做一次。
                pref = _diagnose_node_preferences(theta, seed=CFG.BASE_SEED, norm_mode=pref_norm_mode)
                pref_df = pd.DataFrame(pref)
                for r in pref:
                    rr = dict(r)
                    rr.update({"Scenario": scenario_name, "Dimension": dim_name, "Value": value, "Theta_ID": theta_id})
                    rr.update(theta_to_named_dict(theta))
                    pref_rows.append(rr)
                pref_cmp_rows.extend(_preference_compare_rows(base_pref_df, pref_df, scenario_name, dim_name, value, theta_id))

                for sidx in range(max(1, int(seeds))):
                    seed = CFG.BASE_SEED + sidx
                    row = _eval_theta_windows(theta, seed=seed, windows=windows, scenario_name=scenario_name, task_probs=probs)
                    row.update({"Scenario": scenario_name, "Dimension": dim_name, "Value": value, "Theta_ID": theta_id})
                    row.update(theta_to_named_dict(theta))
                    eval_rows.append(row)

        eval_raw = pd.DataFrame(eval_rows)
        pref_raw = pd.DataFrame(pref_rows)
        pref_cmp = pd.DataFrame(pref_cmp_rows)

        eval_raw.to_csv(os.path.join(root, "sensitivity_eval_raw.csv"), index=False)
        pref_raw.to_csv(os.path.join(root, "sensitivity_preference_raw.csv"), index=False)
        pref_cmp.to_csv(os.path.join(root, "sensitivity_preference_compare.csv"), index=False)

        group_cols = ["Scenario", "Dimension", "Theta_ID", "Value"]
        metric_cols = [
            "Mean_Cost", "Mean_Reward", "Mean_Avg_Delay", "Mean_Avg_Delay_RT", "Mean_Avg_Delay_Batch", "Mean_Avg_Delay_AI",
            "Mean_Avg_Energy", "Mean_Total_Energy", "Final_Cumulative_Energy", "Mean_SLA_Success", "Mean_Violation_Rate", "Mean_Backlog",
            "Mean_Arrivals", "Mean_Completed"
        ]
        agg_spec = {m: ["mean", "std"] for m in metric_cols if m in eval_raw.columns}
        eval_summary = eval_raw.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
        eval_summary.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col for col in eval_summary.columns]
        eval_summary.to_csv(os.path.join(root, "sensitivity_eval_summary.csv"), index=False)

        dim_rows = []
        for scenario_name, _ in scenarios:
            base_cost_vals = eval_raw[(eval_raw["Scenario"] == scenario_name) & (eval_raw["Dimension"] == "BASE")]["Mean_Cost"].tolist()
            base_cost = _mean_or_nan(base_cost_vals)
            for dim_name in CFG.FEATURE_NAMES:
                sub = eval_summary[(eval_summary["Scenario"] == scenario_name) & (eval_summary["Dimension"] == dim_name)].copy()
                if sub.empty:
                    continue
                values = sub["Value_"].tolist() if "Value_" in sub.columns else sub["Value"].tolist()
                cost_mean_col = "Mean_Cost_mean"
                cost_std_col = "Mean_Cost_std"
                costs = sub[cost_mean_col].tolist() if cost_mean_col in sub.columns else []
                delays = sub["Mean_Avg_Delay_mean"].tolist() if "Mean_Avg_Delay_mean" in sub.columns else []
                energies = sub["Mean_Avg_Energy_mean"].tolist() if "Mean_Avg_Energy_mean" in sub.columns else []
                vios = sub["Mean_Violation_Rate_mean"].tolist() if "Mean_Violation_Rate_mean" in sub.columns else []
                cost_range = float(np.nanmax(costs) - np.nanmin(costs)) if costs else np.nan
                delay_range = float(np.nanmax(delays) - np.nanmin(delays)) if delays else np.nan
                energy_range = float(np.nanmax(energies) - np.nanmin(energies)) if energies else np.nan
                vio_range = float(np.nanmax(vios) - np.nanmin(vios)) if vios else np.nan
                avg_seed_std = _mean_or_nan(sub[cost_std_col].tolist()) if cost_std_col in sub.columns else np.nan
                rel_cost_range = cost_range / max(abs(base_cost), 1e-9) if np.isfinite(cost_range) and np.isfinite(base_cost) else np.nan
                noise_ratio = avg_seed_std / max(cost_range, 1e-9) if np.isfinite(avg_seed_std) and np.isfinite(cost_range) else np.nan
                cmp_sub = pref_cmp[(pref_cmp["Scenario"] == scenario_name) & (pref_cmp["Dimension"] == dim_name)]
                mean_kendall = _mean_or_nan(cmp_sub.get("Kendall_Distance", [])) if not cmp_sub.empty else np.nan
                top1_change = _mean_or_nan(cmp_sub.get("Top1_Changed", [])) if not cmp_sub.empty else np.nan
                top3_dist = _mean_or_nan(cmp_sub.get("Top3_Jaccard_Distance", [])) if not cmp_sub.empty else np.nan

                if (np.isfinite(rel_cost_range) and rel_cost_range < 0.02 and
                    np.isfinite(top1_change) and top1_change < 0.10 and
                    np.isfinite(mean_kendall) and mean_kendall < 0.05):
                    decision = "weak_fix_candidate"
                elif np.isfinite(noise_ratio) and noise_ratio > 1.0 and (not np.isfinite(rel_cost_range) or rel_cost_range < 0.05):
                    decision = "noise_dominated"
                elif np.isfinite(top1_change) and top1_change >= 0.25:
                    decision = "preference_sensitive_keep_or_ablate"
                elif np.isfinite(rel_cost_range) and rel_cost_range >= 0.05:
                    decision = "performance_sensitive_keep"
                else:
                    decision = "unclear_need_repeat"

                dim_rows.append({
                    "Scenario": scenario_name,
                    "Dimension": dim_name,
                    "Base_Cost": base_cost,
                    "Cost_Range": cost_range,
                    "Relative_Cost_Range": rel_cost_range,
                    "Avg_Seed_Cost_Std": avg_seed_std,
                    "Seed_Noise_To_Effect_Ratio": noise_ratio,
                    "Cost_Spearman_With_Value": _spearman_corr_safe(values, costs),
                    "Delay_Range": delay_range,
                    "Energy_Range": energy_range,
                    "Violation_Range": vio_range,
                    "Mean_Kendall_Distance": mean_kendall,
                    "Top1_Change_Rate": top1_change,
                    "Top3_Jaccard_Distance": top3_dist,
                    "Decision": decision,
                })
        dim_summary = pd.DataFrame(dim_rows)
        dim_summary.to_csv(os.path.join(root, "sensitivity_dimension_summary.csv"), index=False)
        print(f"\n=== Sensitivity finished. Outputs saved to: {root} ===")
        return dim_summary
    finally:
        CFG.USE_BOLTZMANN_RANDOM = old_random

def _safe_json(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _log_get_safe(log, key, i, default=None):
    vals = log.get(key, [])
    if isinstance(vals, (list, tuple, np.ndarray)) and i < len(vals):
        return vals[i]
    return default


def _log_get_non_nan(log, key, i, default=None):
    val = _log_get_safe(log, key, i, default)
    try:
        if isinstance(val, float) and np.isnan(val):
            return default
    except Exception:
        pass
    return val


def _log_get_ffill(log, key, i, default=None):
    vals = log.get(key, [])
    if isinstance(vals, (list, tuple, np.ndarray)) and vals:
        idx = min(i, len(vals) - 1)
        for j in range(idx, -1, -1):
            val = vals[j]
            try:
                if isinstance(val, (float, np.floating)) and np.isnan(val):
                    continue
            except Exception:
                pass
            return val
    return default


def _is_missing_value(v):
    if v is None:
        return True
    try:
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return True
    except Exception:
        pass
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _first_present(*values, default=None):
    for v in values:
        if not _is_missing_value(v):
            return v
    return default


def group_log_to_dataframe(log, group_key, group_label):
    n = len(log.get("time", []))
    alphas = log.get("alpha", [])
    controls = log.get("control_vector", [])
    allocs = log.get("alloc", [])
    bsf_reward = best_so_far(log.get("reward", []))
    rows = []
    for i in range(n):
        theta = alphas[i] if i < len(alphas) else [None] * CFG.DIM_THETA
        theta = list(theta) if isinstance(theta, (list, tuple, np.ndarray)) else [None] * CFG.DIM_THETA
        while len(theta) < CFG.DIM_THETA:
            theta.append(None)
        control = controls[i] if i < len(controls) else []
        alloc = allocs[i] if i < len(allocs) else []
        selected_candidate_source = _first_present(
            _log_get_ffill(log, "selected_candidate_source", i, None),
            _log_get_ffill(log, "selected_source", i, None),
            _log_get_ffill(log, "best_mu_candidate_source", i, None),
            _log_get_ffill(log, "best_acq_candidate_source", i, None),
        )
        if _is_missing_value(selected_candidate_source):
            deploy_src = str(_log_get_ffill(log, "deploy_source", i, "") or "")
            if "greedy_posterior_mean" in deploy_src:
                selected_candidate_source = "posterior_mean_candidate"
            elif "acquisition" in deploy_src:
                selected_candidate_source = "acquisition_candidate"
            elif deploy_src:
                selected_candidate_source = deploy_src
        macro_gate_mode_row = _log_get_non_nan(log, "cbo_macro_gate_mode", i, str(getattr(CFG, "CBO_MACRO_GATE_MODE", "off")))
        default_source_pool = "macro_pool" if str(macro_gate_mode_row).strip().lower() == "hierarchical" else None
        actual_anchor_reason_row = _log_get_ffill(log, "actual_tr_anchor_reason", i, None)
        runtime_anchor_override_row = _log_get_ffill(log, "runtime_anchor_override", i, None)
        anchor_override_used_row = _log_get_ffill(log, "anchor_override_used", i, None)
        anchor_override_reason_row = _log_get_ffill(log, "anchor_override_reason", i, None)
        runtime_anchor_override_reason_row = _log_get_ffill(log, "runtime_anchor_override_reason", i, None)
        try:
            anchor_override_used_bool = (not _is_missing_value(anchor_override_used_row)) and int(float(anchor_override_used_row)) != 0
        except Exception:
            anchor_override_used_bool = False
        if _is_missing_value(runtime_anchor_override_reason_row):
            reason_text = str(actual_anchor_reason_row or "")
            if (not _is_missing_value(runtime_anchor_override_row)) and (anchor_override_used_bool or "runtime_override" in reason_text):
                runtime_anchor_override_reason_row = f"runtime_anchor_override={runtime_anchor_override_row}"
        if _is_missing_value(anchor_override_reason_row) and not _is_missing_value(runtime_anchor_override_reason_row):
            anchor_override_reason_row = runtime_anchor_override_reason_row
        row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Iteration_轮次": i + 1,
            "Time_时间": log.get("time", [None] * n)[i],
            "Reward_奖励": log.get("reward", [None] * n)[i],
            "Eval_Cost_最终评估Cost": log.get("eval_cost", [None] * n)[i] if i < len(log.get("eval_cost", [])) else (-_safe_float(log.get("reward", [np.nan] * n)[i])),
            "BO_Training_Cost_BO训练Cost": log.get("bo_training_cost", [None] * n)[i] if i < len(log.get("bo_training_cost", [])) else None,
            "BO_Training_Feedback_训练反馈模式": log.get("bo_training_feedback_score", [None] * n)[i] if i < len(log.get("bo_training_feedback_score", [])) else None,
            "BO_Training_Feedback_Note_训练反馈说明": log.get("bo_training_feedback_note", [None] * n)[i] if i < len(log.get("bo_training_feedback_note", [])) else None,
            "Paired_Baseline_Key_配对基线": log.get("paired_baseline_key", [None] * n)[i] if i < len(log.get("paired_baseline_key", [])) else None,
            "Paired_Baseline_Cost_配对基线Cost": log.get("paired_baseline_cost", [None] * n)[i] if i < len(log.get("paired_baseline_cost", [])) else None,
            "Paired_Delta_Cost_相对基线DeltaCost": log.get("paired_delta_cost", [None] * n)[i] if i < len(log.get("paired_delta_cost", [])) else None,
            "Paired_Delta_Relative_Pct_相对基线百分比": log.get("paired_delta_relative_pct", [None] * n)[i] if i < len(log.get("paired_delta_relative_pct", [])) else None,
            "Paired_Note_配对说明": log.get("paired_note", [None] * n)[i] if i < len(log.get("paired_note", [])) else None,
            "Refactor_Version_重构版本": log.get("refactor_version", [None] * n)[i] if i < len(log.get("refactor_version", [])) else None,
            "Feedback_Confidence_反馈可信度": log.get("feedback_confidence", [None] * n)[i] if i < len(log.get("feedback_confidence", [])) else None,
            "Confidence_Completion_Ratio_可信度完成率": log.get("confidence_completion_ratio", [None] * n)[i] if i < len(log.get("confidence_completion_ratio", [])) else None,
            "Confidence_Unfinished_Ratio_可信度未完成比例": log.get("confidence_unfinished_ratio", [None] * n)[i] if i < len(log.get("confidence_unfinished_ratio", [])) else None,
            "Confidence_Task_Mix_L1_任务比例偏移": log.get("confidence_task_mix_l1", [None] * n)[i] if i < len(log.get("confidence_task_mix_l1", [])) else None,
            "BO_History_Mode_历史模式": _log_get_non_nan(log, "bo_history_mode", i, str(getattr(CFG, "BO_HISTORY_MODE", "recent"))),
            "BO_Recent_Window_最近历史窗口": log.get("bo_recent_window", [None] * n)[i] if i < len(log.get("bo_recent_window", [])) else None,
            "effective_history_mode": _log_get_non_nan(log, "effective_history_mode", i, str(getattr(CFG, "BO_HISTORY_MODE", "recent"))),
            "effective_recent_window": log.get("effective_recent_window", [None] * n)[i] if i < len(log.get("effective_recent_window", [])) else None,
            "history_override_source": _log_get_non_nan(log, "history_override_source", i, "unknown"),
            "bo_train_sample_count": log.get("bo_train_sample_count", [None] * n)[i] if i < len(log.get("bo_train_sample_count", [])) else None,
            "BO_Confidence_Min_最小可信度": log.get("bo_confidence_min", [None] * n)[i] if i < len(log.get("bo_confidence_min", [])) else None,
            "Best_So_Far_Reward_历史最优奖励": bsf_reward[i] if i < len(bsf_reward) else None,
            "deploy_policy": _log_get_safe(log, "deploy_policy", i, None),
            "deploy_source": _log_get_safe(log, "deploy_source", i, None),
            "explore_used": _log_get_safe(log, "explore_used", i, None),
            "incumbent_available": _log_get_safe(log, "incumbent_available", i, None),
            "incumbent_cost": _log_get_safe(log, "incumbent_cost", i, None),
            "current_candidate_cost": _log_get_safe(log, "current_candidate_cost", i, None),
            "current_train_cost": _log_get_safe(log, "current_train_cost", i, None),
            "best_so_far_cost": _log_get_safe(log, "best_so_far_cost", i, None),
            "best_so_far_iter": _log_get_safe(log, "best_so_far_iter", i, None),
            "used_theta_source": _log_get_safe(log, "used_theta_source", i, None),
            "posterior_mu": _log_get_safe(log, "posterior_mu", i, None),
            "posterior_sigma": _log_get_safe(log, "posterior_sigma", i, None),
            "candidate_count_safe": _log_get_safe(log, "candidate_count_safe", i, None),
            "history_select_mode": _log_get_non_nan(log, "history_select_mode", i, str(getattr(CFG, "CBO_HISTORY_SELECT_MODE", "recent"))),
            "selected_recent_count": _log_get_ffill(log, "selected_recent_count", i, None),
            "selected_macro_count": _log_get_ffill(log, "selected_macro_count", i, None),
            "selected_context_count": _log_get_ffill(log, "selected_context_count", i, None),
            "selected_elite_count": _log_get_ffill(log, "selected_elite_count", i, None),
            "selected_diverse_count": _log_get_ffill(log, "selected_diverse_count", i, None),
            "selected_total_count": _log_get_ffill(log, "selected_total_count", i, None),
            "cbo_macro_gate_mode": macro_gate_mode_row,
            "macro_total_arrivals_norm": _log_get_ffill(log, "macro_total_arrivals_norm", i, None),
            "macro_rt_ratio": _log_get_ffill(log, "macro_rt_ratio", i, None),
            "macro_batch_ratio": _log_get_ffill(log, "macro_batch_ratio", i, None),
            "macro_similarity_max": _log_get_ffill(log, "macro_similarity_max", i, None),
            "macro_similarity_mean": _log_get_ffill(log, "macro_similarity_mean", i, None),
            "macro_similarity_p50": _log_get_ffill(log, "macro_similarity_p50", i, None),
            "macro_similarity_p90": _log_get_ffill(log, "macro_similarity_p90", i, None),
            "selected_macro_mean_similarity": _log_get_ffill(log, "selected_macro_mean_similarity", i, None),
            "selected_macro_min_similarity": _log_get_ffill(log, "selected_macro_min_similarity", i, None),
            "selected_macro_max_similarity": _log_get_ffill(log, "selected_macro_max_similarity", i, None),
            "macro_pool_count": _log_get_ffill(log, "macro_pool_count", i, None),
            "macro_pool_mean_similarity": _log_get_ffill(log, "macro_pool_mean_similarity", i, None),
            "macro_pool_min_similarity": _log_get_ffill(log, "macro_pool_min_similarity", i, None),
            "macro_pool_max_similarity": _log_get_ffill(log, "macro_pool_max_similarity", i, None),
            "macro_pool_p50_similarity": _log_get_ffill(log, "macro_pool_p50_similarity", i, None),
            "macro_pool_p90_similarity": _log_get_ffill(log, "macro_pool_p90_similarity", i, None),
            "selected_from_macro_pool_count": _log_get_ffill(log, "selected_from_macro_pool_count", i, None),
            "selected_outside_macro_pool_count": _log_get_ffill(log, "selected_outside_macro_pool_count", i, None),
            "macro_gate_fallback_used": _log_get_ffill(log, "macro_gate_fallback_used", i, None),
            "macro_gate_fallback_reason": _log_get_ffill(log, "macro_gate_fallback_reason", i, None),
            "context_selection_source_pool": _first_present(_log_get_ffill(log, "context_selection_source_pool", i, None), default_source_pool),
            "elite_selection_source_pool": _first_present(_log_get_ffill(log, "elite_selection_source_pool", i, None), default_source_pool),
            "tr_anchor_source_pool": _first_present(_log_get_ffill(log, "tr_anchor_source_pool", i, None), default_source_pool),
            "macro_k": _log_get_ffill(log, "macro_k", i, None),
            "macro_lengthscale_total": _log_get_ffill(log, "macro_lengthscale_total", i, None),
            "macro_lengthscale_rt": _log_get_ffill(log, "macro_lengthscale_rt", i, None),
            "macro_lengthscale_batch": _log_get_ffill(log, "macro_lengthscale_batch", i, None),
            "context_similarity_max": _log_get_ffill(log, "context_similarity_max", i, None),
            "context_similarity_mean": _log_get_ffill(log, "context_similarity_mean", i, None),
            "elite_best_robust_score": _log_get_ffill(log, "elite_best_robust_score", i, None),
            "elite_best_eval_count": _log_get_ffill(log, "elite_best_eval_count", i, None),
            "elite_best_mean_cost": _log_get_ffill(log, "elite_best_mean_cost", i, None),
            "elite_best_std_cost": _log_get_ffill(log, "elite_best_std_cost", i, None),
            "robust_incumbent_available": _log_get_ffill(log, "robust_incumbent_available", i, None),
            "robust_incumbent_score": _log_get_ffill(log, "robust_incumbent_score", i, None),
            "robust_incumbent_eval_count": _log_get_ffill(log, "robust_incumbent_eval_count", i, None),
            "robust_incumbent_mean_cost": _log_get_ffill(log, "robust_incumbent_mean_cost", i, None),
            "robust_incumbent_std_cost": _log_get_ffill(log, "robust_incumbent_std_cost", i, None),
            "robust_incumbent_context_similarity": _log_get_ffill(log, "robust_incumbent_context_similarity", i, None),
            "robust_incumbent_theta": _safe_json(_log_get_ffill(log, "robust_incumbent_theta", i, None)),
            "robust_incumbent_used": _log_get_ffill(log, "robust_incumbent_used", i, None),
            "robust_incumbent_reason": _log_get_ffill(log, "robust_incumbent_reason", i, None),
            "cbo_tr_mode": _log_get_non_nan(log, "cbo_tr_mode", i, str(getattr(CFG, "CBO_TR_MODE", "off"))),
            "cbo_tr_anchor_mode": _log_get_non_nan(log, "cbo_tr_anchor_mode", i, str(getattr(CFG, "CBO_TR_ANCHOR_MODE", "posterior_mean"))),
            "cbo_tr_radius": _log_get_ffill(log, "cbo_tr_radius", i, None),
            "cbo_tr_anchor_theta": _safe_json(_log_get_ffill(log, "cbo_tr_anchor_theta", i, None)),
            "tr_update_mode": _log_get_ffill(log, "tr_update_mode", i, None),
            "tr_baseline_mean": _log_get_ffill(log, "tr_baseline_mean", i, None),
            "tr_current_mean": _log_get_ffill(log, "tr_current_mean", i, None),
            "tr_improve_pct": _log_get_ffill(log, "tr_improve_pct", i, None),
            "tr_worse_pct": _log_get_ffill(log, "tr_worse_pct", i, None),
            "tr_update_signal": _log_get_ffill(log, "tr_update_signal", i, None),
            "tr_update_patience_count": _log_get_ffill(log, "tr_update_patience_count", i, None),
            "cbo_tr_radius_before_update": _log_get_ffill(log, "cbo_tr_radius_before_update", i, None),
            "actual_tr_anchor_mode": _log_get_ffill(log, "actual_tr_anchor_mode", i, None),
            "actual_tr_anchor_source": _log_get_ffill(log, "actual_tr_anchor_source", i, None),
            "actual_tr_anchor_theta": _safe_json(_log_get_ffill(log, "actual_tr_anchor_theta", i, None)),
            "actual_tr_anchor_reason": actual_anchor_reason_row,
            "anchor_override_used": anchor_override_used_row,
            "anchor_override_reason": anchor_override_reason_row,
            "anchor_fallback_used": _log_get_ffill(log, "anchor_fallback_used", i, None),
            "anchor_fallback_reason": _log_get_ffill(log, "anchor_fallback_reason", i, None),
            "anchor_theta_distance_to_prev": _log_get_ffill(log, "anchor_theta_distance_to_prev", i, None),
            "anchor_theta_distance_to_robust_elite": _log_get_ffill(log, "anchor_theta_distance_to_robust_elite", i, None),
            "anchor_theta_distance_to_context_best": _log_get_ffill(log, "anchor_theta_distance_to_context_best", i, None),
            "anchor_theta_distance_to_recent_best": _log_get_ffill(log, "anchor_theta_distance_to_recent_best", i, None),
            "cbo_tr_candidate_count": _log_get_ffill(log, "cbo_tr_candidate_count", i, None),
            "cbo_global_candidate_count": _log_get_ffill(log, "cbo_global_candidate_count", i, None),
            "cbo_tr_update_reason": _log_get_ffill(log, "cbo_tr_update_reason", i, None),
            "cbo_tr_success_count": _log_get_ffill(log, "cbo_tr_success_count", i, None),
            "cbo_tr_failure_count": _log_get_ffill(log, "cbo_tr_failure_count", i, None),
            "cbo_select_mode": _log_get_non_nan(log, "cbo_select_mode", i, str(getattr(CFG, "CBO_SELECT_MODE", "greedy"))),
            "cbo_topk": _log_get_ffill(log, "cbo_topk", i, None),
            "cbo_select_temperature": _log_get_ffill(log, "cbo_select_temperature", i, None),
            "cbo_epsilon": _log_get_ffill(log, "cbo_epsilon", i, None),
            "cbo_acq_beta": _log_get_ffill(log, "cbo_acq_beta", i, None),
            "cbo_acq_beta_mode": _log_get_ffill(log, "cbo_acq_beta_mode", i, None),
            "beta_eff": _log_get_ffill(log, "beta_eff", i, None),
            "radius_norm": _log_get_ffill(log, "radius_norm", i, None),
            "radius_beta_component": _log_get_ffill(log, "radius_beta_component", i, None),
            "state_beta_boost_used": _log_get_ffill(log, "state_beta_boost_used", i, None),
            "state_beta_boost_reason": _log_get_ffill(log, "state_beta_boost_reason", i, None),
            "actual_score_formula": _log_get_ffill(log, "actual_score_formula", i, None),
            "selected_candidate_score": _first_present(_log_get_ffill(log, "selected_candidate_score", i, None), _log_get_ffill(log, "selected_score", i, None)),
            "selected_candidate_beta_eff": _log_get_ffill(log, "selected_candidate_beta_eff", i, None),
            "selected_candidate_rank_by_score": _first_present(_log_get_ffill(log, "selected_candidate_rank_by_score", i, None), _log_get_ffill(log, "selected_rank_by_score", i, None)),
            "selected_candidate_rank_by_sigma": _first_present(_log_get_ffill(log, "selected_candidate_rank_by_sigma", i, None), _log_get_ffill(log, "selected_rank_by_sigma", i, None)),
            "service_guard_mode": _log_get_ffill(log, "service_guard_mode", i, None),
            "service_guard_available": _log_get_ffill(log, "service_guard_available", i, None),
            "service_guard_penalty": _log_get_ffill(log, "service_guard_penalty", i, None),
            "service_guard_reason": _log_get_ffill(log, "service_guard_reason", i, None),
            "good_region_available": _log_get_ffill(log, "good_region_available", i, None),
            "good_region_best_iter": _log_get_ffill(log, "good_region_best_iter", i, None),
            "good_region_best_rolling50_cost": _log_get_ffill(log, "good_region_best_rolling50_cost", i, None),
            "good_region_anchor_theta": _safe_json(_log_get_ffill(log, "good_region_anchor_theta", i, None)),
            "good_region_anchor_source": _log_get_ffill(log, "good_region_anchor_source", i, None),
            "distance_to_good_region_anchor": _log_get_ffill(log, "distance_to_good_region_anchor", i, None),
            "current_vs_good_region_gap_pct": _log_get_ffill(log, "current_vs_good_region_gap_pct", i, None),
            "predicted_cost": _log_get_ffill(log, "predicted_cost", i, None),
            "actual_cost": _log_get_ffill(log, "actual_cost", i, None),
            "prediction_error": _log_get_ffill(log, "prediction_error", i, None),
            "surprise": _log_get_ffill(log, "surprise", i, None),
            "cost_gap_pct": _log_get_ffill(log, "cost_gap_pct", i, None),
            "residual_trigger": _log_get_ffill(log, "residual_trigger", i, None),
            "condition_trigger": _log_get_ffill(log, "condition_trigger", i, None),
            "radius_min_stuck_count": _log_get_ffill(log, "radius_min_stuck_count", i, None),
            "force_explore_countdown": _log_get_ffill(log, "force_explore_countdown", i, None),
            "runtime_anchor_override": runtime_anchor_override_row,
            "runtime_anchor_override_reason": runtime_anchor_override_reason_row,
            "cbo_tr_radius_after_update": _log_get_ffill(log, "cbo_tr_radius_after_update", i, None),
            "selected_reason": _log_get_ffill(log, "selected_reason", i, None),
            "selected_candidate_source": selected_candidate_source,
            "selected_candidate_mu": _first_present(_log_get_ffill(log, "selected_candidate_mu", i, None), _log_get_ffill(log, "selected_mu", i, None)),
            "selected_candidate_sigma": _first_present(_log_get_ffill(log, "selected_candidate_sigma", i, None), _log_get_ffill(log, "selected_sigma", i, None)),
            "selected_candidate_acq": _first_present(_log_get_ffill(log, "selected_candidate_acq", i, None), _log_get_ffill(log, "selected_acq", i, None)),
            "selected_candidate_rank_by_mu": _first_present(_log_get_ffill(log, "selected_candidate_rank_by_mu", i, None), _log_get_ffill(log, "selected_rank_by_mu", i, None)),
            "selected_candidate_rank_by_acq": _first_present(_log_get_ffill(log, "selected_candidate_rank_by_acq", i, None), _log_get_ffill(log, "selected_rank_by_acq", i, None)),
            "best_mu_candidate_source": _log_get_ffill(log, "best_mu_candidate_source", i, None),
            "best_acq_candidate_source": _log_get_ffill(log, "best_acq_candidate_source", i, None),
            "num_candidates": _log_get_ffill(log, "num_candidates", i, None),
            "num_tr_candidates": _log_get_ffill(log, "num_tr_candidates", i, None),
            "num_global_candidates": _log_get_ffill(log, "num_global_candidates", i, None),
            "Avg_Delay_平均时延": log.get("avg_delay", [None] * n)[i],
            "Avg_Delay_RT_实时平均时延": log.get("avg_delay_rt", [None] * n)[i],
            "Avg_Delay_Batch_批量平均时延": log.get("avg_delay_batch", [None] * n)[i],
            "Avg_Delay_AI_AI平均时延": log.get("avg_delay_ai", [None] * n)[i],
            "SLA_Success_Rate_SLA成功率": log.get("sla_success_rate", [None] * n)[i],
            "Violation_Rate_违约率": log.get("vio_rate", [None] * n)[i],
            "Avg_Energy_平均能耗": log.get("avg_energy", [None] * n)[i],
            "Total_Energy_优化目标能耗": log.get("total_energy", [None] * n)[i],
            "Total_Energy_Real_真实常开能耗": log.get("total_energy_real", [None] * n)[i] if i < len(log.get("total_energy_real", [])) else None,
            "Compute_Dynamic_Energy_动态计算能耗": log.get("compute_dynamic_energy", [None] * n)[i] if i < len(log.get("compute_dynamic_energy", [])) else None,
            "Compute_Idle_Energy_目标空闲能耗": log.get("compute_idle_energy", [None] * n)[i] if i < len(log.get("compute_idle_energy", [])) else None,
            "Transmission_Energy_传输能耗": log.get("transmission_energy", [None] * n)[i] if i < len(log.get("transmission_energy", [])) else None,
            "Cumulative_Energy_累计目标能耗": log.get("cumulative_energy", [None] * n)[i],
            "Cumulative_Energy_Real_累计真实能耗": log.get("cumulative_energy_real", [None] * n)[i] if i < len(log.get("cumulative_energy_real", [])) else None,
            "Backlog_积压任务数": log.get("backlog", [None] * n)[i],
            "macro_context_key": _log_get_safe(log, "macro_context_key", i, None),
            "window_arrivals_total": _log_get_safe(log, "window_arrivals_total", i, None),
            "window_completed_total": _log_get_safe(log, "window_completed_total", i, None),
            "window_unfinished_total": _log_get_safe(log, "window_unfinished_total", i, None),
            "unfinished_rate": _log_get_safe(log, "unfinished_rate", i, None),
            "backlog_growth": _log_get_safe(log, "backlog_growth", i, None),
            "backlog_growth_rate": _log_get_safe(log, "backlog_growth_rate", i, None),
            "energy_per_arrival": _log_get_safe(log, "energy_per_arrival", i, None),
            "energy_metric_source": _log_get_safe(log, "energy_metric_source", i, None),
            "class_imbalance_available": _log_get_safe(log, "class_imbalance_available", i, None),
            "min_class_success_rate": _log_get_safe(log, "min_class_success_rate", i, None),
            "class_imbalance_penalty": _log_get_safe(log, "class_imbalance_penalty", i, None),
            "cbo_reference_mode": _log_get_safe(log, "cbo_reference_mode", i, None),
            "cbo_reference_available": _log_get_safe(log, "cbo_reference_available", i, None),
            "cbo_reference_status": _log_get_safe(log, "cbo_reference_status", i, None),
            "cbo_reference_round_count": _log_get_safe(log, "cbo_reference_round_count", i, None),
            "cbo_reference_frozen": _log_get_safe(log, "cbo_reference_frozen", i, None),
            "delay_ref": _log_get_safe(log, "delay_ref", i, None),
            "energy_per_arrival_ref": _log_get_safe(log, "energy_per_arrival_ref", i, None),
            "unfinished_rate_ref": _log_get_safe(log, "unfinished_rate_ref", i, None),
            "success_rate_ref": _log_get_safe(log, "success_rate_ref", i, None),
            "eval_cost_ref": _log_get_safe(log, "eval_cost_ref", i, None),
            "delay_norm": _log_get_safe(log, "delay_norm", i, None),
            "energy_norm": _log_get_safe(log, "energy_norm", i, None),
            "unfinished_norm": _log_get_safe(log, "unfinished_norm", i, None),
            "eval_cost_norm": _log_get_safe(log, "eval_cost_norm", i, None),
            "success_shortfall": _log_get_safe(log, "success_shortfall", i, None),
            "success_shortfall_norm": _log_get_safe(log, "success_shortfall_norm", i, None),
            "service_norm": _log_get_safe(log, "service_norm", i, None),
            "normalized_tradeoff_score": _log_get_safe(log, "normalized_tradeoff_score", i, None),
            "cbo_objective_mode": _log_get_safe(log, "cbo_objective_mode", i, None),
            "tradeoff_alpha": _log_get_safe(log, "tradeoff_alpha", i, None),
            "bo_training_cost_source": _log_get_safe(log, "bo_training_cost_source", i, None),
            "scheduler_tradeoff_mode": _log_get_safe(log, "scheduler_tradeoff_mode", i, None),
            "scheduler_score_norm_mode": _log_get_safe(log, "scheduler_score_norm_mode", i, None),
            "scheduler_alpha_last": _log_get_safe(log, "scheduler_alpha_last", i, None),
            "scheduler_alpha_mean": _log_get_safe(log, "scheduler_alpha_mean", i, None),
            "selected_latency_component_last": _log_get_safe(log, "selected_latency_component_last", i, None),
            "selected_energy_component_last": _log_get_safe(log, "selected_energy_component_last", i, None),
            "selected_risk_penalty_last": _log_get_safe(log, "selected_risk_penalty_last", i, None),
            "selected_queue_penalty_last": _log_get_safe(log, "selected_queue_penalty_last", i, None),
            "selected_latency_energy_component_last": _log_get_safe(log, "selected_latency_energy_component_last", i, None),
            "selected_service_component_last": _log_get_safe(log, "selected_service_component_last", i, None),  # Deprecated alias.
            "selected_norm_e_last": _log_get_safe(log, "selected_norm_e_last", i, None),
            "selected_norm_l_last": _log_get_safe(log, "selected_norm_l_last", i, None),
            "selected_norm_risk_last": _log_get_safe(log, "selected_norm_risk_last", i, None),
            "selected_norm_queue_last": _log_get_safe(log, "selected_norm_queue_last", i, None),
            "selected_score_last": _log_get_safe(log, "selected_score_last", i, None),
            "Zero_Completion_Penalty_零完成惩罚": log.get("zero_completion_penalty", [None] * n)[i],
            "Beta_Boltzmann系数": log.get("beta", [None] * n)[i],
            "Control_Label_控制标签": log.get("control_label", [None] * n)[i] if i < len(log.get("control_label", [])) else None,
            "Control_Vector_控制向量": _safe_json(control),
            "Alloc_By_Type_分任务节点分配": _safe_json(log.get("alloc_by_type", [None] * n)[i] if i < len(log.get("alloc_by_type", [])) else None),
            "Context_Label_情景标签": log.get("context_label", [None] * n)[i] if i < len(log.get("context_label", [])) else None,
            "Context_Vector_情景向量": _safe_json(log.get("context_vector", [None] * n)[i] if i < len(log.get("context_vector", [])) else None),
            "Arrival_Rate_到达率": log.get("arrival_rate", [None] * n)[i],
            "Avg_Util_平均利用率": log.get("avg_util", [None] * n)[i],
            "RT_Ratio_实时占比": log.get("rt_arrival_ratio", [None] * n)[i],
            "Batch_Ratio_批量占比": log.get("batch_arrival_ratio", [None] * n)[i],
            "AI_Ratio_AI占比": log.get("ai_arrival_ratio", [None] * n)[i],
            "Arrivals_Total_本轮到达任务数": log.get("arrivals_total", [None] * n)[i],
            "Arrivals_RT_本轮实时任务数": log.get("arrivals_rt", [None] * n)[i],
            "Arrivals_Batch_本轮批量任务数": log.get("arrivals_batch", [None] * n)[i],
            "Arrivals_AI_本轮AI任务数": log.get("arrivals_ai", [None] * n)[i],
            "Completed_Total_本轮完成任务数": log.get("completed_total", [None] * n)[i],
            "Completed_RT_本轮完成实时任务数": log.get("completed_rt", [None] * n)[i],
            "Completed_Batch_本轮完成批量任务数": log.get("completed_batch", [None] * n)[i],
            "Completed_AI_本轮完成AI任务数": log.get("completed_ai", [None] * n)[i],

            # Per-class window diagnostics.
            "RT_Window_Avg_Lateness_实时窗口平均超期": log.get("avg_lateness_rt", [None] * n)[i] if i < len(log.get("avg_lateness_rt", [])) else None,
            "Batch_Window_Avg_Lateness_批任务窗口平均超期": log.get("avg_lateness_batch", [None] * n)[i] if i < len(log.get("avg_lateness_batch", [])) else None,
            "AI_Window_Avg_Lateness_AI窗口平均超期": log.get("avg_lateness_ai", [None] * n)[i] if i < len(log.get("avg_lateness_ai", [])) else None,
            "RT_Window_Vio_Rate_实时窗口违约率": log.get("vio_rate_rt", [None] * n)[i] if i < len(log.get("vio_rate_rt", [])) else None,
            "Batch_Window_Vio_Rate_批任务窗口违约率": log.get("vio_rate_batch", [None] * n)[i] if i < len(log.get("vio_rate_batch", [])) else None,
            "AI_Window_Vio_Rate_AI窗口违约率": log.get("vio_rate_ai", [None] * n)[i] if i < len(log.get("vio_rate_ai", [])) else None,
            "RT_Window_Completion_Ratio_实时窗口完成比例": log.get("completion_ratio_rt", [None] * n)[i] if i < len(log.get("completion_ratio_rt", [])) else None,
            "Batch_Window_Completion_Ratio_批任务窗口完成比例": log.get("completion_ratio_batch", [None] * n)[i] if i < len(log.get("completion_ratio_batch", [])) else None,
            "AI_Window_Completion_Ratio_AI窗口完成比例": log.get("completion_ratio_ai", [None] * n)[i] if i < len(log.get("completion_ratio_ai", [])) else None,
            "RT_Window_Avg_Energy_实时窗口平均能耗": log.get("avg_energy_rt", [None] * n)[i] if i < len(log.get("avg_energy_rt", [])) else None,
            "Batch_Window_Avg_Energy_批任务窗口平均能耗": log.get("avg_energy_batch", [None] * n)[i] if i < len(log.get("avg_energy_batch", [])) else None,
            "AI_Window_Avg_Energy_AI窗口平均能耗": log.get("avg_energy_ai", [None] * n)[i] if i < len(log.get("avg_energy_ai", [])) else None,
            "RT_Window_Class_Cost_实时窗口分类Cost": log.get("window_rt_class_cost", [None] * n)[i] if i < len(log.get("window_rt_class_cost", [])) else None,
            "Batch_Window_Class_Cost_批任务窗口分类Cost": log.get("window_batch_class_cost", [None] * n)[i] if i < len(log.get("window_batch_class_cost", [])) else None,
            "AI_Window_Class_Cost_AI窗口分类Cost": log.get("window_ai_class_cost", [None] * n)[i] if i < len(log.get("window_ai_class_cost", [])) else None,
            "Unfinished_End_轮末未完成任务数": log.get("unfinished_end", [None] * n)[i],
            "Feedback_Mode_反馈模式": log.get("feedback_mode", [None] * n)[i] if i < len(log.get("feedback_mode", [])) else None,
            "Cohort_ID_任务批次ID": log.get("cohort_id", [None] * n)[i] if i < len(log.get("cohort_id", [])) else None,
            "Cohort_Arrivals_批次到达任务数": log.get("cohort_arrivals", [None] * n)[i] if i < len(log.get("cohort_arrivals", [])) else None,
            "Cohort_Feedback_Count_本轮反馈批次数": log.get("cohort_feedback_count", [None] * n)[i] if i < len(log.get("cohort_feedback_count", [])) else None,
            "Cohort_Feedback_Cost_Mean_本轮批次反馈平均Cost": log.get("cohort_feedback_cost_mean", [None] * n)[i] if i < len(log.get("cohort_feedback_cost_mean", [])) else None,
            "Cohort_Active_Count_活跃批次数": log.get("cohort_active_count", [None] * n)[i] if i < len(log.get("cohort_active_count", [])) else None,
            "Cohort_Finalized_Total_累计已反馈批次数": log.get("cohort_finalized_total", [None] * n)[i] if i < len(log.get("cohort_finalized_total", [])) else None,
            "Cohort_Pending_Tasks_批次未完成任务数": log.get("cohort_pending_tasks", [None] * n)[i] if i < len(log.get("cohort_pending_tasks", [])) else None,
            "Training_Sample_Count_建模样本数": log.get("training_sample_count", [None] * n)[i],
            "Recent_Sample_Count_最近样本数": log.get("recent_sample_count", [None] * n)[i],
            "Archive_Sample_Count_归档样本数": log.get("archive_sample_count", [None] * n)[i],
            "Neighbor_K_相似邻居数": log.get("neighbor_k", [None] * n)[i],
            "Candidate_Count_候选点数量": log.get("candidate_count", [None] * n)[i],
            "Trust_Radius_TR半径": log.get("trust_radius", [None] * n)[i],
            "Pivot_局部搜索中心": _safe_json(log.get("pivot_theta", [None] * n)[i] if i < len(log.get("pivot_theta", [])) else None),
            "TopK_History_参考历史点": _safe_json(log.get("topk_history", [None] * n)[i] if i < len(log.get("topk_history", [])) else None),
            "TopK_Similarity_相似度列表": _safe_json(log.get("topk_similarity", [None] * n)[i] if i < len(log.get("topk_similarity", [])) else None),
            "Best_Selected_本轮最终选点": _safe_json(log.get("best_selected_theta", [None] * n)[i] if i < len(log.get("best_selected_theta", [])) else None),
        }
        bilingual_feature_names = [
            "W_RT_Latency_RT时延权重",
            "W_Batch_Latency_Batch时延权重",
            "W_AI_Latency_AI时延权重",
            "W_RT_Energy_RT能耗权重",
            "W_Batch_Energy_Batch能耗权重",
            "W_AI_Energy_AI能耗权重",
        ]
        for idx, name in enumerate(bilingual_feature_names):
            row[name] = theta[idx] if idx < len(theta) else None
        if isinstance(alloc, (list, tuple)):
            row["Alloc_Counts_节点分配统计"] = _safe_json(alloc)
            for node_idx, value in enumerate(alloc):
                row[f"Alloc_Node_{node_idx}_节点{node_idx}分配数"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_context_debug_dataframe(log, group_key, group_label):
    n = len(log.get("time", []))
    rows = []
    for i in range(n):
        row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Iteration_轮次": i + 1,
            "Time_时间": log.get("time", [None] * n)[i],
            "Context_Label_情景标签": log.get("context_label", [None] * n)[i] if i < len(log.get("context_label", [])) else None,
            "Context_Vector_情景向量": _safe_json(log.get("context_vector", [None] * n)[i] if i < len(log.get("context_vector", [])) else None),
            "Arrival_Rate_到达率": log.get("arrival_rate", [None] * n)[i],
            "Avg_Util_平均利用率": log.get("avg_util", [None] * n)[i],
            "Backlog_积压任务数": log.get("backlog", [None] * n)[i],
            "Violation_Rate_违约率": log.get("vio_rate", [None] * n)[i],
            "RT_Ratio_实时占比": log.get("rt_arrival_ratio", [None] * n)[i],
            "Batch_Ratio_批量占比": log.get("batch_arrival_ratio", [None] * n)[i],
            "AI_Ratio_AI占比": log.get("ai_arrival_ratio", [None] * n)[i],
            "Training_Sample_Count_建模样本数": log.get("training_sample_count", [None] * n)[i],
            "Recent_Sample_Count_最近样本数": log.get("recent_sample_count", [None] * n)[i],
            "Archive_Sample_Count_归档样本数": log.get("archive_sample_count", [None] * n)[i],
            "Neighbor_K_相似邻居数": log.get("neighbor_k", [None] * n)[i],
            "Candidate_Count_候选点数量": log.get("candidate_count", [None] * n)[i],
            "Trust_Radius_TR半径": log.get("trust_radius", [None] * n)[i],
            "Pivot_局部搜索中心": _safe_json(log.get("pivot_theta", [None] * n)[i] if i < len(log.get("pivot_theta", [])) else None),
            "TopK_History_参考历史点": _safe_json(log.get("topk_history", [None] * n)[i] if i < len(log.get("topk_history", [])) else None),
            "TopK_Similarity_相似度列表": _safe_json(log.get("topk_similarity", [None] * n)[i] if i < len(log.get("topk_similarity", [])) else None),
            "Best_Selected_本轮最终选点": _safe_json(log.get("best_selected_theta", [None] * n)[i] if i < len(log.get("best_selected_theta", [])) else None),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_alloc_debug_dataframe(log, group_key, group_label):
    n = len(log.get("time", []))
    allocs = log.get("alloc", [])
    rows = []
    for i in range(n):
        alloc = allocs[i] if i < len(allocs) else []
        row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Iteration_轮次": i + 1,
            "Time_时间": log.get("time", [None] * n)[i],
            "Arrivals_Total_本轮到达任务数": log.get("arrivals_total", [None] * n)[i],
            "Completed_Total_本轮完成任务数": log.get("completed_total", [None] * n)[i],
            "Unfinished_End_轮末未完成任务数": log.get("unfinished_end", [None] * n)[i],
            "Alloc_Counts_节点分配统计": _safe_json(alloc),
        }
        if isinstance(alloc, (list, tuple)):
            for node_idx, value in enumerate(alloc):
                row[f"Alloc_Node_{node_idx}_节点{node_idx}分配数"] = value
        rows.append(row)
    return pd.DataFrame(rows)


# ===============================================================
# 诊断增强：每轮平均指标、cohort 学习曲线、按任务类型节点分配
# ===============================================================
PLOT_COLOR_CYCLE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
    "#56B4E9", "#E69F00", "#000000", "#8A2BE2", "#A52A2A"
]


def _log_get(log, key, i, default=None):
    vals = log.get(key, []) if isinstance(log, dict) else []
    if isinstance(vals, list) and i < len(vals):
        return vals[i]
    return default


def _json_load_maybe(x):
    if x is None:
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _normalize_alloc_by_type(value, node_count):
    """把 log 里的 alloc_by_type 统一整理成 RT/Batch/AI -> 节点计数列表。"""
    empty = {t: [0.0 for _ in range(node_count)] for t in TASK_TYPE_ORDER}
    value = _json_load_maybe(value)
    if not isinstance(value, dict):
        return empty, True
    out = {}
    missing = False
    for task_type in TASK_TYPE_ORDER:
        vals = value.get(task_type, value.get(task_type.lower(), None))
        if vals is None:
            out[task_type] = [0.0 for _ in range(node_count)]
            missing = True
            continue
        if isinstance(vals, np.ndarray):
            vals = vals.tolist()
        if not isinstance(vals, (list, tuple)):
            out[task_type] = [0.0 for _ in range(node_count)]
            missing = True
            continue
        clean = [_safe_float(v, 0.0) for v in list(vals)[:node_count]]
        if len(clean) < node_count:
            clean += [0.0 for _ in range(node_count - len(clean))]
        out[task_type] = clean
    return out, missing


def _node_meta(node_idx):
    cfg = CFG.NODES_CFG[int(node_idx)]
    return {
        "Node_ID_节点": int(node_idx),
        "Node_Role_节点角色": str(cfg.get("role", "")),
        "Node_Workshop_车间": int(cfg.get("workshop", cfg.get("site", -1))),
        "Node_Is_Cloud_是否云": bool(_node_is_cloud(cfg)),
        "Node_CPU_节点CPU": int(cfg.get("cpu", 0)),
        "Node_Speed_节点速度": float(cfg.get("speed", 0.0)),
    }


def build_alloc_by_type_debug_dataframe(log, group_key, group_label, run_index=None):
    """长表：每轮、每类任务、每个节点一行，用于判断 BO 是否改变了任务类型级分配。"""
    n = len(log.get("time", []))
    node_count = len(CFG.NODES_CFG)
    rows = []
    for i in range(n):
        alloc_raw = _log_get(log, "alloc_by_type", i, default=None)
        alloc_by_type, missing_alloc_by_type = _normalize_alloc_by_type(alloc_raw, node_count)
        control_vec = _json_load_maybe(_log_get(log, "control_vector", i, default=[]))
        if not isinstance(control_vec, (list, tuple)):
            control_vec = []
        base_row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Run_Index_重复编号": run_index,
            "Iteration_轮次": i + 1,
            "Time_时间": _log_get(log, "time", i, None),
            "Feedback_Mode_反馈模式": _log_get(log, "feedback_mode", i, None),
            "Reward_评分": _log_get(log, "reward", i, None),
            "Cost_代价": -_safe_float(_log_get(log, "reward", i, np.nan)),
            "Avg_Delay_平均时延": _log_get(log, "avg_delay", i, None),
            "Avg_Energy_平均能耗": _log_get(log, "avg_energy", i, None),
            "Violation_Rate_违约率": _log_get(log, "vio_rate", i, None),
            "SLA_Success_Rate_SLA成功率": _log_get(log, "sla_success_rate", i, None),
            "Backlog_积压": _log_get(log, "backlog", i, None),
            "Arrivals_Total_总到达": _log_get(log, "arrivals_total", i, 0),
            "Completed_Total_总完成": _log_get(log, "completed_total", i, 0),
            "RT_Arrivals_实时到达": _log_get(log, "arrivals_rt", i, None),
            "Batch_Arrivals_批任务到达": _log_get(log, "arrivals_batch", i, None),
            "AI_Arrivals_AI到达": _log_get(log, "arrivals_ai", i, None),
            "RT_Ratio_实时占比": _log_get(log, "rt_arrival_ratio", i, None),
            "Batch_Ratio_批任务占比": _log_get(log, "batch_arrival_ratio", i, None),
            "AI_Ratio_AI占比": _log_get(log, "ai_arrival_ratio", i, None),
            "Missing_Alloc_By_Type_缺失类型分配": bool(missing_alloc_by_type),
            "Control_Vector_控制向量": _safe_json(list(control_vec)),
        }
        for idx, name in enumerate(REDUCED4_FEATURE_NAMES if 'REDUCED4_FEATURE_NAMES' in globals() else ["Theta0", "Theta1", "Theta2", "Theta3"]):
            base_row[f"Control_{name}"] = _safe_float(control_vec[idx], np.nan) if idx < len(control_vec) else np.nan
        for task_type in TASK_TYPE_ORDER:
            counts = alloc_by_type.get(task_type, [0.0 for _ in range(node_count)])
            task_total = float(np.nansum(counts))
            cloud_total = 0.0
            edge_total = 0.0
            for node_idx, count in enumerate(counts):
                if _node_is_cloud(CFG.NODES_CFG[int(node_idx)]):
                    cloud_total += float(count)
                else:
                    edge_total += float(count)
            for node_idx, count in enumerate(counts):
                row = dict(base_row)
                row.update(_node_meta(node_idx))
                row.update({
                    "Task_Type_任务类型": task_type,
                    "Alloc_Count_分配数": float(count),
                    "Task_Type_Total_该类型总分配": task_total,
                    "Alloc_Ratio_In_Task_该类型内占比": float(count) / task_total if task_total > 0 else 0.0,
                    "Task_Type_Cloud_Total_该类型云分配": cloud_total,
                    "Task_Type_Edge_Total_该类型边缘分配": edge_total,
                    "Task_Type_Cloud_Ratio_该类型云占比": cloud_total / task_total if task_total > 0 else 0.0,
                    "Task_Type_Edge_Ratio_该类型边缘占比": edge_total / task_total if task_total > 0 else 0.0,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def build_alloc_by_type_summary_dataframe(alloc_type_df):
    if alloc_type_df is None or alloc_type_df.empty:
        return pd.DataFrame()
    group_cols = [
        "Group_Key_方法键", "Group_Label_方法名称", "Task_Type_任务类型",
        "Node_ID_节点", "Node_Role_节点角色", "Node_Workshop_车间", "Node_Is_Cloud_是否云"
    ]
    summary = alloc_type_df.groupby(group_cols, dropna=False)["Alloc_Count_分配数"].sum().reset_index()
    totals = summary.groupby(["Group_Key_方法键", "Task_Type_任务类型"], dropna=False)["Alloc_Count_分配数"].sum().reset_index()
    totals = totals.rename(columns={"Alloc_Count_分配数": "Task_Type_Total_该类型总分配"})
    summary = summary.merge(totals, on=["Group_Key_方法键", "Task_Type_任务类型"], how="left")
    summary["Alloc_Ratio_In_Task_该类型内占比"] = summary.apply(
        lambda r: float(r["Alloc_Count_分配数"]) / float(r["Task_Type_Total_该类型总分配"])
        if float(r["Task_Type_Total_该类型总分配"]) > 0 else 0.0,
        axis=1,
    )
    cloud_rows = []
    for (g, label, t), sub in summary.groupby(["Group_Key_方法键", "Group_Label_方法名称", "Task_Type_任务类型"], dropna=False):
        total = float(sub["Alloc_Count_分配数"].sum())
        cloud = float(sub.loc[sub["Node_Is_Cloud_是否云"] == True, "Alloc_Count_分配数"].sum())
        edge = total - cloud
        cloud_rows.append({
            "Group_Key_方法键": g,
            "Group_Label_方法名称": label,
            "Task_Type_任务类型": t,
            "Task_Type_Total_该类型总分配": total,
            "Cloud_Total_云分配": cloud,
            "Edge_Total_边缘分配": edge,
            "Cloud_Ratio_云占比": cloud / total if total > 0 else 0.0,
            "Edge_Ratio_边缘占比": edge / total if total > 0 else 0.0,
        })
    cloud_df = pd.DataFrame(cloud_rows)
    summary = summary.merge(
        cloud_df,
        on=["Group_Key_方法键", "Group_Label_方法名称", "Task_Type_任务类型", "Task_Type_Total_该类型总分配"],
        how="left",
    )
    return summary


def plot_alloc_by_type_summary(summary_df, save_dir, group_key, group_label):
    if summary_df is None or summary_df.empty:
        return
    os.makedirs(save_dir, exist_ok=True)
    task_types = [t for t in TASK_TYPE_ORDER if t in set(summary_df["Task_Type_任务类型"])]
    node_ids = sorted(summary_df["Node_ID_节点"].dropna().astype(int).unique().tolist())
    pivot = summary_df.pivot_table(index="Task_Type_任务类型", columns="Node_ID_节点", values="Alloc_Count_分配数", aggfunc="sum", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(task_types), dtype=float)
    for j, node_id in enumerate(node_ids):
        vals = np.array([float(pivot.loc[t, node_id]) if t in pivot.index and node_id in pivot.columns else 0.0 for t in task_types])
        if np.sum(vals) <= 0:
            continue
        role = str(CFG.NODES_CFG[int(node_id)].get("role", ""))
        ax.bar(task_types, vals, bottom=bottom, label=f"N{node_id}-{role}", color=PLOT_COLOR_CYCLE[j % len(PLOT_COLOR_CYCLE)])
        bottom += vals
    ax.set_title(f"{group_label} - Allocation by Task Type")
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Allocated Tasks")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{group_key}_alloc_by_type_stacked_任务类型节点堆叠图.png"), dpi=300)
    plt.close(fig)

    cloud_df = summary_df.drop_duplicates(["Group_Key_方法键", "Task_Type_任务类型"])[["Task_Type_任务类型", "Cloud_Ratio_云占比", "Edge_Ratio_边缘占比"]].copy()
    cloud_df = cloud_df.set_index("Task_Type_任务类型").reindex(task_types).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(task_types))
    width = 0.35
    ax.bar(x - width / 2, cloud_df["Edge_Ratio_边缘占比"].fillna(0.0).values, width, label="Edge", color="#0072B2")
    ax.bar(x + width / 2, cloud_df["Cloud_Ratio_云占比"].fillna(0.0).values, width, label="Cloud", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(task_types)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Ratio")
    ax.set_title(f"{group_label} - Edge vs Cloud Ratio by Task Type")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{group_key}_alloc_by_type_cloud_ratio_任务类型云边占比.png"), dpi=300)
    plt.close(fig)


def plot_alloc_by_type_method_compare(all_summary_df, save_dir):
    if all_summary_df is None or all_summary_df.empty:
        return
    os.makedirs(save_dir, exist_ok=True)
    method_labels = all_summary_df[["Group_Key_方法键", "Group_Label_方法名称"]].drop_duplicates().sort_values("Group_Key_方法键")
    ratio_rows = []
    for _, method in method_labels.iterrows():
        g = method["Group_Key_方法键"]
        label = method["Group_Label_方法名称"]
        sub_g = all_summary_df[all_summary_df["Group_Key_方法键"] == g]
        for task_type in TASK_TYPE_ORDER:
            sub_t = sub_g[sub_g["Task_Type_任务类型"] == task_type]
            if sub_t.empty:
                cloud_ratio, total = np.nan, 0.0
            else:
                one = sub_t.iloc[0]
                cloud_ratio = _safe_float(one.get("Cloud_Ratio_云占比", np.nan))
                total = _safe_float(one.get("Task_Type_Total_该类型总分配", 0.0), 0.0)
            ratio_rows.append({
                "Group_Key_方法键": g,
                "Group_Label_方法名称": label,
                "Task_Type_任务类型": task_type,
                "Cloud_Ratio_云占比": cloud_ratio,
                "Task_Type_Total_该类型总分配": total,
            })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(os.path.join(save_dir, "alloc_by_type_method_cloud_ratio_compare_方法云占比对比.csv"), index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(TASK_TYPE_ORDER))
    methods = method_labels["Group_Key_方法键"].tolist()
    width = 0.8 / max(1, len(methods))
    for j, g in enumerate(methods):
        sub = ratio_df[ratio_df["Group_Key_方法键"] == g]
        vals = []
        for t in TASK_TYPE_ORDER:
            row = sub[sub["Task_Type_任务类型"] == t]
            vals.append(float(row["Cloud_Ratio_云占比"].iloc[0]) if not row.empty else np.nan)
        label = method_labels.loc[method_labels["Group_Key_方法键"] == g, "Group_Label_方法名称"].iloc[0]
        color = get_method_style(g, {"label": label}, fallback_idx=j).get("color", PLOT_COLOR_CYCLE[j % len(PLOT_COLOR_CYCLE)])
        ax.bar(x - 0.4 + width / 2 + j * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_TYPE_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Cloud Allocation Ratio")
    ax.set_title("Cloud Allocation Ratio by Method and Task Type")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "alloc_by_type_method_cloud_ratio_compare_方法云占比对比.png"), dpi=300)
    plt.close(fig)


def plot_round_mean_energy_delay_score(group_logs, save_dir=SCENARIO_SAVE_DIR):
    """每轮重复平均后的能耗、时延、评分曲线。"""
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    phase_ranges = get_bo_phase_ranges()
    metric_specs = [
        ("avg_energy", "Avg Energy per Completed Task", "Avg Energy"),
        ("avg_delay", "Avg Delay", "Delay (s)"),
        ("reward", "Score / Reward", "Reward (higher is better)"),
    ]
    for idx2, (group_key, info) in enumerate(group_logs.items()):
        log = aggregate_logs(info["logs"])
        style = get_method_style(group_key, info, fallback_idx=idx2)
        for ax, (key, title, ylabel) in zip(axes, metric_specs):
            vals = log.get(key, [])
            if not vals:
                continue
            vals = ema_smooth(vals, weight=0.80)
            ax.plot(np.arange(1, len(vals) + 1), vals, label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.2)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.45)
    for ax in axes:
        for phase in phase_ranges[:-1]:
            ax.axvline(phase["iter_end"], color="#555555", linestyle=":", alpha=0.8)
        ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("BO Iteration")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scenario_round_mean_energy_delay_score_每轮平均能耗时延评分.png"), dpi=300)
    plt.close(fig)


def plot_cohort_learning_curves(group_logs, save_dir=SCENARIO_SAVE_DIR):
    """画 cohort_cost 和 best-so-far cohort_reward，诊断 BO 学习反馈。"""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for idx2, (group_key, info) in enumerate(group_logs.items()):
        frames = []
        for raw_log in info.get("logs", []):
            cdf = build_cohort_feedback_dataframe(raw_log, group_key, info["label"])
            if not cdf.empty:
                frames.append(cdf)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if "Window_Index_窗口序号" not in df.columns:
            continue
        df["Window_Index_窗口序号"] = pd.to_numeric(df["Window_Index_窗口序号"], errors="coerce")
        df["Cohort_Cost_批次Cost"] = pd.to_numeric(df["Cohort_Cost_批次Cost"], errors="coerce")
        df = df.dropna(subset=["Window_Index_窗口序号", "Cohort_Cost_批次Cost"])
        if df.empty:
            continue
        mean_df = df.groupby("Window_Index_窗口序号", as_index=False)["Cohort_Cost_批次Cost"].mean().sort_values("Window_Index_窗口序号")
        x = mean_df["Window_Index_窗口序号"].astype(int).values
        cost = mean_df["Cohort_Cost_批次Cost"].values.astype(float)
        reward = -cost
        bsf_reward = best_so_far(reward.tolist())
        style = get_method_style(group_key, info, fallback_idx=idx2)
        axes[0].plot(x, ema_smooth(cost.tolist(), weight=0.75), label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.2)
        axes[1].plot(x, bsf_reward, label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.2)
    axes[0].set_title("Cohort Feedback Cost by BO Iteration")
    axes[0].set_ylabel("Cohort Cost (lower is better)")
    axes[1].set_title("Best-so-far Cohort Reward")
    axes[1].set_ylabel("Best-so-far Cohort Reward")
    axes[1].set_xlabel("BO Iteration / Feedback Window")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scenario_cohort_learning_curves_批次反馈学习曲线.png"), dpi=300)
    plt.close(fig)


def plot_theta_trajectory(group_logs, save_dir=SCENARIO_SAVE_DIR):
    """画低维控制变量轨迹，自动兼容 reduced4 / reduced6。"""
    # 先扫描所有 control_vector，判断最大维度。
    max_dim = 0
    for info in group_logs.values():
        log0 = aggregate_logs(info.get("logs", [])) if info.get("logs") else info.get("mean", {})
        for c in log0.get("control_vector", []):
            cc = _json_load_maybe(c)
            if isinstance(cc, (list, tuple, np.ndarray)):
                max_dim = max(max_dim, len(cc))
                break
    if max_dim <= 0:
        return
    names = _control_feature_names_for_vector([np.nan] * max_dim) if '_control_feature_names_for_vector' in globals() else [f"Theta{i}" for i in range(max_dim)]
    fig, axes = plt.subplots(max_dim, 1, figsize=(13, max(3.0 * max_dim, 8)), sharex=True)
    if max_dim == 1:
        axes = [axes]
    for idx2, (group_key, info) in enumerate(group_logs.items()):
        log = aggregate_logs(info["logs"])
        controls = log.get("control_vector", [])
        if not controls:
            continue
        arr = []
        for c in controls:
            cc = _json_load_maybe(c)
            if isinstance(cc, np.ndarray):
                cc = cc.tolist()
            if isinstance(cc, (list, tuple)):
                row = []
                for v in list(cc)[:max_dim]:
                    row.append(_safe_float(v))
                while len(row) < max_dim:
                    row.append(np.nan)
                arr.append(row)
        if not arr:
            continue
        arr = np.array(arr, dtype=float)
        style = get_method_style(group_key, info, fallback_idx=idx2)
        x = np.arange(1, arr.shape[0] + 1)
        for d in range(min(max_dim, arr.shape[1])):
            axes[d].plot(x, arr[:, d], label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
    for d, ax in enumerate(axes):
        label_name = names[d] if d < len(names) else f"Theta{d}"
        ax.set_title(f"Control Trajectory - {label_name}")
        ax.set_ylabel(label_name)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("BO Iteration")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "scenario_theta_trajectory_控制参数轨迹.png"), dpi=300)
    plt.close(fig)

def _as_clean_numeric_list(values):
    """把 None / 字符串 / 非数值项转成可安全 np 统计的 float list。"""
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]
    out = []
    for v in values:
        try:
            if v is None:
                out.append(np.nan)
            else:
                fv = float(v)
                out.append(fv if np.isfinite(fv) else np.nan)
        except Exception:
            out.append(np.nan)
    return out


def _safe_nanmean(values):
    vals = _as_clean_numeric_list(values)
    if not vals:
        return np.nan
    arr = np.array(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _safe_nanmax(values):
    vals = _as_clean_numeric_list(values)
    if not vals:
        return np.nan
    arr = np.array(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmax(arr))


def _safe_last(values):
    vals = _as_clean_numeric_list(values)
    if not vals:
        return np.nan
    for v in reversed(vals):
        if not np.isnan(v):
            return float(v)
    return np.nan


def _recovery_iteration(backlog_seq, phase, threshold=1.0):
    start = max(0, phase["iter_start"] - 1)
    end = min(len(backlog_seq), phase["iter_end"])
    for idx in range(start, end):
        try:
            value = backlog_seq[idx]
            if value is None:
                continue
            if float(value) <= float(threshold):
                return idx + 1
        except Exception:
            continue
    return np.nan


def build_key_metric_summary_dataframe(group_logs):
    """核心指标统计：整体 + 阶段 + 峰值 + 恢复。

    修复点：aggregate_logs 对某些非数值字段会产生 None；这里统一转为 np.nan，
    避免 Colab / NumPy 2.x 下 np.nanmean([None, None]) 报错。
    """
    rows = []
    phases = get_bo_phase_ranges()
    for group_key, info in group_logs.items():
        log = aggregate_logs(info["logs"])
        reward = _as_clean_numeric_list(log.get("reward", []))
        delay = _as_clean_numeric_list(log.get("avg_delay", []))
        energy = _as_clean_numeric_list(log.get("avg_energy", []))
        total_energy = _as_clean_numeric_list(log.get("total_energy", []))
        vio = _as_clean_numeric_list(log.get("vio_rate", []))
        sla = _as_clean_numeric_list(log.get("sla_success_rate", []))
        backlog = _as_clean_numeric_list(log.get("backlog", []))
        completion = _as_clean_numeric_list(log.get("completion_rate", []))
        cumulative_energy = _as_clean_numeric_list(log.get("cumulative_energy", []))
        cumulative_energy_real = _as_clean_numeric_list(log.get("cumulative_energy_real", []))

        row = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": info["label"],
            "Overall_Mean_Reward_整体平均评分": _safe_nanmean(reward),
            "Overall_Mean_Cost_整体平均代价": _safe_nanmean([-x if not np.isnan(x) else np.nan for x in reward]),
            "Overall_Mean_Avg_Delay_整体平均时延": _safe_nanmean(delay),
            "Overall_Mean_Avg_Energy_整体平均能耗": _safe_nanmean(energy),
            "Overall_Mean_Violation_整体平均违约率": _safe_nanmean(vio),
            "Overall_Mean_SLA_整体平均SLA成功率": _safe_nanmean(sla),
            "Overall_Mean_Backlog_整体平均积压": _safe_nanmean(backlog),
            "Overall_Peak_Backlog_整体积压峰值": _safe_nanmax(backlog),
            "Final_Reward_最终评分": _safe_last(reward),
            "Final_Avg_Delay_最终平均时延": _safe_last(delay),
            "Final_Avg_Energy_最终平均能耗": _safe_last(energy),
            "Final_Backlog_最终积压": _safe_last(backlog),
            "Final_Cumulative_Objective_Energy_最终累计目标能耗": _safe_last(cumulative_energy),
            "Final_Cumulative_Real_Energy_最终累计真实能耗": _safe_last(cumulative_energy_real),
        }
        for phase in phases:
            pidx = phase["phase_idx"]
            def seg(vals):
                return _as_clean_numeric_list(_slice_metric_by_phase(vals, phase))
            for name, vals, cn in [
                ("Reward", reward, "评分"), ("Avg_Delay", delay, "平均时延"), ("Avg_Energy", energy, "平均能耗"),
                ("Total_Energy", total_energy, "窗口总能耗"), ("Violation", vio, "违约率"), ("SLA", sla, "SLA成功率"),
                ("Backlog", backlog, "积压"), ("Completion_Rate", completion, "完成率"),
            ]:
                ss = seg(vals)
                row[f"Phase{pidx}_{name}_Mean_{cn}均值"] = _safe_nanmean(ss)
                row[f"Phase{pidx}_{name}_Final_{cn}末值"] = _safe_last(ss)
                row[f"Phase{pidx}_{name}_Peak_{cn}峰值"] = _safe_nanmax(ss)
            if pidx >= 2:
                row[f"Phase{pidx}_Backlog_Recovery_Iter_积压恢复轮次"] = _recovery_iteration(backlog, phase, threshold=1.0)
        rows.append(row)
    return pd.DataFrame(rows)

def save_extra_diagnostics(group_logs):
    """统一保存增强诊断输出。"""
    all_alloc_type_summary_frames = []
    for group_key, info in group_logs.items():
        alloc_type_frames = []
        for run_idx, raw_log in enumerate(info.get("logs", []), start=1):
            atdf = build_alloc_by_type_debug_dataframe(raw_log, group_key=group_key, group_label=info["label"], run_index=run_idx)
            if not atdf.empty:
                alloc_type_frames.append(atdf)
        if alloc_type_frames:
            alloc_type_df = pd.concat(alloc_type_frames, ignore_index=True)
            alloc_type_df.to_csv(os.path.join(SCENARIO_SAVE_DIR, f"{group_key}_alloc_by_type_debug_任务类型节点分配调试.csv"), index=False)
            alloc_type_summary_df = build_alloc_by_type_summary_dataframe(alloc_type_df)
            alloc_type_summary_df.to_csv(os.path.join(SCENARIO_SAVE_DIR, f"{group_key}_alloc_by_type_summary_任务类型节点分配汇总.csv"), index=False)
            all_alloc_type_summary_frames.append(alloc_type_summary_df)
            plot_alloc_by_type_summary(alloc_type_summary_df, save_dir=SCENARIO_SAVE_DIR, group_key=group_key, group_label=info["label"])
    if all_alloc_type_summary_frames:
        all_summary = pd.concat(all_alloc_type_summary_frames, ignore_index=True)
        all_summary.to_csv(os.path.join(SCENARIO_SAVE_DIR, "alloc_by_type_all_methods_summary_全部方法任务类型分配汇总.csv"), index=False)
        plot_alloc_by_type_method_compare(all_summary, SCENARIO_SAVE_DIR)
    key_summary = build_key_metric_summary_dataframe(group_logs)
    key_summary.to_csv(os.path.join(SCENARIO_SAVE_DIR, "key_metric_summary_核心指标统计.csv"), index=False)
    plot_round_mean_energy_delay_score(group_logs, save_dir=SCENARIO_SAVE_DIR)
    plot_cohort_learning_curves(group_logs, save_dir=SCENARIO_SAVE_DIR)
    plot_theta_trajectory(group_logs, save_dir=SCENARIO_SAVE_DIR)

def plot_group_alloc_heatmaps(group_logs, save_dir=SCENARIO_SAVE_DIR, prefix="scenario"):
    group_items = list(group_logs.items())
    fig, axes = plt.subplots(len(group_items), 1, figsize=(12, 2.6 * max(1, len(group_items))))
    if len(group_items) == 1:
        axes = [axes]
    for ax, (group_key, info) in zip(axes, group_items):
        log = aggregate_logs(info["logs"])
        alloc = np.array(log.get("alloc", []), dtype=float) if log.get("alloc") else np.zeros((1, len(CFG.NODES_CFG)))
        if alloc.ndim == 1:
            alloc = alloc.reshape(1, -1)
        im = ax.imshow(alloc.T, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_title(f"Node Allocation Heatmap - {info['label']}")
        ax.set_xlabel("BO Iteration")
        ax.set_ylabel("Node Index")
        ax.set_yticks(np.arange(len(CFG.NODES_CFG)))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_alloc_heatmaps.png"), dpi=300)
    plt.close(fig)

def plot_group_task_delay_bars(group_logs, save_dir=SCENARIO_SAVE_DIR, prefix="scenario"):
    labels = []
    rt_vals = []
    batch_vals = []
    ai_vals = []
    for _, info in group_logs.items():
        log = aggregate_logs(info["logs"])
        labels.append(info["label"])
        rt_vals.append(float(np.nanmean(log.get("avg_delay_rt", [np.nan]))))
        batch_vals.append(float(np.nanmean(log.get("avg_delay_batch", [np.nan]))))
        ai_vals.append(float(np.nanmean(log.get("avg_delay_ai", [np.nan]))))
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    ax.bar(x - width, rt_vals, width=width, label="RT")
    ax.bar(x, batch_vals, width=width, label="Batch")
    ax.bar(x + width, ai_vals, width=width, label="AI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Average Delay (s)")
    ax.set_title("Average Delay by Task Type Across Control Groups")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_task_delay_bars.png"), dpi=300)
    plt.close(fig)

def build_scenario_method_groups():
    """定义情景对比实验中的方法组。

    默认四条线：
    - fixednorm_fixed_balanced：固定归一化 + 固定平衡权重
    - fixednorm_vanilla_bo：固定归一化 + 普通 BO
    - fixednorm_context_bo：固定归一化 + 情景 BO
    - fixednorm_context_tr_bo：固定归一化 + 情景 BO + TR

    可选扩展：
    - rollingnorm_fixed_balanced：滚动归一化 + 固定平衡权重
    - rollingnorm_vanilla_bo：滚动归一化 + 普通 BO
    - rollingnorm_context_bo：滚动归一化 + 情景 BO
    - rollingnorm_context_tr_bo：滚动归一化 + 情景 BO + TR
    """
    anchor_balanced, anchor_rt, anchor_energy = default_scenario_anchor_points()

    def cbo_lite_group(label_suffix, context_mode):
        return {
            "label": f"Reduced6 CBO-Lite {label_suffix}",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "deploy_policy": "greedy",
            "method_family": "cbo_lite_context_ablation",
            "context_mode": str(context_mode),
            "history_mode": "recent_confidence",
            "recent_window": 80,
            "confidence_min": 0.35,
            "confidence_min_samples": 12,
            "agent_kwargs": reduced6_lite_context_agent_kwargs(use_trust_region=False, anchor_mode="none", context_mode=context_mode),
        }

    def vanilla_agent_kwargs():
        return {
            "dim": CFG.DIM_THETA,
            "bounds": get_control_bounds(CFG.DIM_THETA),
            "feature_names": list(CFG.FEATURE_NAMES),
            "use_context": False,
            "use_state_partition": False,
            "use_trust_region": False,
            "anchor_points": [anchor_balanced, anchor_rt, anchor_energy],
        }

    def context_agent_kwargs(use_trust_region=False):
        return {
            "dim": CFG.DIM_THETA,
            "bounds": get_control_bounds(CFG.DIM_THETA),
            "feature_names": list(CFG.FEATURE_NAMES),
            "use_context": True,
            "use_state_partition": True,
            "use_trust_region": bool(use_trust_region),
            "context_dim": len(CFG.CONTEXT_FEATURE_NAMES),
            "context_bounds": CFG.CONTEXT_BOUNDS,
            "anchor_points": [anchor_balanced, anchor_rt, anchor_energy],
        }

    return {
        "fixednorm_fixed_balanced": {
            "label": "FixedNorm Fixed Balanced",
            "norm_mode": "fixed",
            "agent": None,
            "fixed_theta": anchor_balanced,
        },
        "fixednorm_vanilla_bo": {
            "label": "FixedNorm Vanilla BO",
            "norm_mode": "fixed",
            "agent_kwargs": vanilla_agent_kwargs(),
        },
        "reduced4_fixed_mid": {
            "label": "Reduced4 Fixed Mid",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced4",
            "fixed_theta": [2.55, 2.55, 2.55, 0.50],
        },
        "reduced4_fixed_recommended": {
            "label": "Reduced4 Tuned Fixed (Sensitivity)",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced4",
            "fixed_theta": [5.00, 5.00, 5.00, 0.05],
        },
        "reduced4_vanilla_bo": {
            "label": "Reduced4 Vanilla BO (Basic Anchors)",
            "norm_mode": "fixed",
            "control_mode": "reduced4",
            "agent_kwargs": reduced4_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="basic"),
        },
        "reduced4_context_bo": {
            "label": "Reduced4 Contextual BO (Basic Anchors)",
            "norm_mode": "fixed",
            "control_mode": "reduced4",
            "agent_kwargs": reduced4_agent_kwargs(use_context=True, use_trust_region=False, anchor_mode="basic"),
        },
        "reduced4_context_tr_bo": {
            "label": "Reduced4 Contextual BO + TR (Basic Anchors)",
            "norm_mode": "fixed",
            "control_mode": "reduced4",
            "agent_kwargs": reduced4_agent_kwargs(use_context=True, use_trust_region=True, anchor_mode="basic"),
        },
        "reduced4_vanilla_bo_anchor": {
            "label": "Reduced4 Vanilla BO + Anchor",
            "norm_mode": "fixed",
            "control_mode": "reduced4",
            "agent_kwargs": reduced4_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="recommended"),
        },
        "reduced4_context_bo_anchor": {
            "label": "Reduced4 Contextual BO + Anchor",
            "norm_mode": "fixed",
            "control_mode": "reduced4",
            "agent_kwargs": reduced4_agent_kwargs(use_context=True, use_trust_region=False, anchor_mode="recommended"),
        },
        "reduced4_context_tr_bo_anchor": {
            "label": "Reduced4 Contextual BO + TR + Anchor",
            "norm_mode": "fixed",
            "control_mode": "reduced4",
            "agent_kwargs": reduced4_agent_kwargs(use_context=True, use_trust_region=True, anchor_mode="recommended"),
        },
        "reduced6_fixed_mid": {
            "label": "Reduced6 Fixed Mid",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "fixed_theta": [2.55, 2.55, 2.55, 1.0, 1.0, 0.50],
        },
        "reduced6_fixed_tuned": {
            "label": "Reduced6 Tuned Fixed",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "fixed_theta": [5.00, 5.00, 5.00, 1.0, 1.0, 0.05],
        },
        "reduced6_fixed_queue_high": {
            "label": "Reduced6 Fixed Queue High",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "fixed_theta": [3.00, 3.00, 3.00, 5.0, 1.0, 0.30],
        },
        "reduced6_fixed_risk_high": {
            "label": "Reduced6 Fixed Risk High",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "fixed_theta": [3.00, 3.00, 3.00, 1.0, 5.0, 0.30],
        },
        "reduced6_fixed_edge_safe": {
            "label": "Reduced6 Fixed Edge Safe",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "fixed_theta": [2.00, 2.00, 2.00, 3.0, 2.0, 0.90],
        },
        "direct_round_robin": {
            "label": "RoundRobin Direct",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "method_family": "direct_baseline",
            "scheduler_type": "round_robin_direct",
            "fixed_theta": [3.00, 3.00, 3.00, 1.0, 1.0, 0.30],
        },
        "direct_greedy_cost": {
            "label": "Greedy Direct Cost",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "method_family": "direct_baseline",
            "scheduler_type": "greedy_direct_cost",
            "fixed_theta": [3.00, 3.00, 3.00, 1.0, 1.0, 0.30],
        },
        "direct_least_load": {
            "label": "LeastLoad Direct",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "method_family": "direct_baseline",
            "scheduler_type": "least_load_direct",
            "fixed_theta": [3.00, 3.00, 3.00, 1.0, 1.0, 0.30],
        },
        "direct_queue_aware_greedy": {
            "label": "QueueAware Greedy Direct",
            "norm_mode": "fixed",
            "agent": None,
            "control_mode": "reduced6",
            "method_family": "direct_baseline",
            "scheduler_type": "queue_aware_greedy_direct",
            "fixed_theta": [3.00, 3.00, 3.00, 1.0, 1.0, 0.30],
        },
        "reduced6_bo_ei": {
            "label": "Reduced6 BO-EI Cold Start",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "deploy_policy": "ei",
            "method_family": "main_bo",
            "agent_kwargs": reduced6_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="none"),
        },
        "reduced6_bo_greedy": {
            "label": "Reduced6 BO-Greedy Cold Start",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "deploy_policy": "greedy",
            "method_family": "main_bo",
            "agent_kwargs": reduced6_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="none"),
        },
        "reduced6_bo_greedy_recent_conf": {
            "label": "Reduced6 BO-Greedy Recent+Confidence",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "deploy_policy": "greedy",
            "method_family": "recent_confidence_bo",
            "history_mode": "recent_confidence",
            "recent_window": 80,
            "confidence_min": 0.35,
            "confidence_min_samples": 12,
            "agent_kwargs": reduced6_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="none"),
        },
        "reduced6_cbo_lite_recent_conf": {
            "label": "Reduced6 CBO-Lite Recent+Confidence",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "deploy_policy": "greedy",
            "method_family": "cbo_lite_recent_confidence",
            "context_mode": "lite",
            "history_mode": "recent_confidence",
            "recent_window": 80,
            "confidence_min": 0.35,
            "confidence_min_samples": 12,
            "agent_kwargs": reduced6_lite_context_agent_kwargs(use_trust_region=False, anchor_mode="none", context_mode="lite"),
        },
        "reduced6_cbo_lite_full": cbo_lite_group("Full Context", "full"),
        "reduced6_cbo_lite_load_only": cbo_lite_group("Load Context", "load_only"),
        "reduced6_cbo_lite_util_only": cbo_lite_group("Util Context", "util_only"),
        "reduced6_cbo_lite_pressure_only": cbo_lite_group("Pressure Context", "pressure_only"),
        "reduced6_cbo_lite_no_cloud": cbo_lite_group("No Cloud Context", "no_cloud"),
        "reduced6_cbo_lite_no_arrival": cbo_lite_group("No Arrival Context", "no_arrival"),
        # v6.1 新增：任务结构/上一窗口任务数 context，用于验证 RT/Batch/AI 场景区分度。
        "reduced6_cbo_lite_taskmix": cbo_lite_group("TaskMix Context", "taskmix"),
        "reduced6_cbo_lite_recent_mix": cbo_lite_group("RecentMix Context", "recent_mix"),
        "reduced6_cbo_lite_prev_counts": cbo_lite_group("PrevCounts Context", "prev_counts"),
        "reduced6_cbo_lite_pressure_taskmix": cbo_lite_group("Pressure+TaskMix Context", "pressure_taskmix"),
        "reduced6_cbo_lite_pressure_recent_mix": cbo_lite_group("Pressure+RecentMix Context", "pressure_recent_mix"),
        "reduced6_cbo_lite_pressure_counts": cbo_lite_group("Pressure+Counts Context", "pressure_counts"),
        "reduced6_cbo_lite_pressure_taskmix_counts": cbo_lite_group("Pressure+TaskMix+Counts Context", "pressure_taskmix_counts"),
        "reduced6_cbo_lite_full_taskmix": cbo_lite_group("Full+TaskMix Context", "full_taskmix"),
        "reduced6_cbo_lite_full_taskmix_counts": cbo_lite_group("Full+TaskMix+Counts Context", "full_taskmix_counts"),
        "reduced6_cbo_greedy_legacy": {
            "label": "Reduced6 CBO-Greedy Legacy",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "deploy_policy": "greedy",
            "method_family": "legacy_context",
            "agent_kwargs": reduced6_agent_kwargs(use_context=True, use_trust_region=False, anchor_mode="none"),
        },
        "reduced6_cbo_tr_greedy_legacy": {
            "label": "Reduced6 CBO-TR-Greedy Legacy",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "deploy_policy": "greedy",
            "method_family": "legacy_context",
            "agent_kwargs": reduced6_agent_kwargs(use_context=True, use_trust_region=True, anchor_mode="none"),
        },
        # Legacy anchor methods: retained for backwards compatibility, not part of v2 default main path.
        "reduced6_vanilla_bo_anchor": {
            "label": "Reduced6 Vanilla BO + Anchor",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "agent_kwargs": reduced6_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="recommended"),
        },
        "reduced6_context_bo_anchor": {
            "label": "Reduced6 Contextual BO + Anchor",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "agent_kwargs": reduced6_agent_kwargs(use_context=True, use_trust_region=False, anchor_mode="recommended"),
        },
        "reduced6_context_tr_bo_anchor": {
            "label": "Reduced6 Contextual BO + TR + Anchor",
            "norm_mode": "fixed",
            "control_mode": "reduced6",
            "agent_kwargs": reduced6_agent_kwargs(use_context=True, use_trust_region=True, anchor_mode="recommended"),
        },
        "rollingnorm_fixed_balanced": {
            "label": "RollingNorm Fixed Balanced",
            "norm_mode": "rolling",
            "agent": None,
            "fixed_theta": anchor_balanced,
        },
        "rollingnorm_vanilla_bo": {
            "label": "RollingNorm Vanilla BO",
            "norm_mode": "rolling",
            "agent_kwargs": vanilla_agent_kwargs(),
        },
        "fixednorm_context_bo": {
            "label": "FixedNorm Contextual BO",
            "norm_mode": "fixed",
            "agent_kwargs": context_agent_kwargs(use_trust_region=False),
        },
        "fixednorm_context_tr_bo": {
            "label": "FixedNorm Contextual BO + TR",
            "norm_mode": "fixed",
            "agent_kwargs": context_agent_kwargs(use_trust_region=True),
        },
        "rollingnorm_context_bo": {
            "label": "RollingNorm Contextual BO",
            "norm_mode": "rolling",
            "agent_kwargs": context_agent_kwargs(use_trust_region=False),
        },
        "rollingnorm_context_tr_bo": {
            "label": "RollingNorm Contextual BO + TR",
            "norm_mode": "rolling",
            "agent_kwargs": context_agent_kwargs(use_trust_region=True),
        },
    }


# 备份版默认方法集与 current/run_500_bestcbo_direct36.sh 保持一致：
# fixed_mid,fixed_tuned,bo-greedy,cbo-pressure-taskmix-counts,
# greedy-direct-cost,least-load-direct,queue-aware-greedy-direct
# 注意：默认不包含 RoundRobin / 轮询基线。
DEFAULT_SCENARIO_KEYS = list(getattr(CFG, "DEFAULT_SELECTED_KEYS_NO_RR", [
    "reduced6_fixed_mid",
    "reduced6_fixed_tuned",
    "reduced6_bo_greedy",
    "reduced6_cbo_lite_pressure_taskmix_counts",
    "direct_greedy_cost",
    "direct_least_load",
    "direct_queue_aware_greedy",
]))


# ===============================================================
# 低维 4D BO 主实验配置
# ---------------------------------------------------------------
# 保留任务类型差异，同时把 11 维扩展控制压缩到 4 维：
# [RT时延偏好, Batch时延偏好, AI时延偏好, 云门控]
# 其余维度采用敏感度分析后的稳定固定值。
# ===============================================================
REDUCED4_FEATURE_NAMES = ["W_RT_Latency", "W_Batch_Latency", "W_AI_Latency", "Cloud_Gate"]
# 4D 主实验中，Cloud_Gate 不再使用完整 11D 的 [0.05, 0.95] 宽范围。
# 前期敏感度分析显示过高的 Cloud_Gate 往往不利于 Batch/AI 场景；
# 因此 4D BO 将云门控收缩到 [0.05, 0.60]，保留 RT 场景的中等云门控空间，
# 同时避免 BO 浪费大量轮次搜索明显较弱的高门控区域。
REDUCED4_CLOUD_GATE_BOUNDS = (0.05, 0.60)
REDUCED4_BOUNDS = [
    [float(CFG.CONTROL_WEIGHT_BOUNDS[0]), float(CFG.CONTROL_WEIGHT_BOUNDS[0]), float(CFG.CONTROL_WEIGHT_BOUNDS[0]), float(REDUCED4_CLOUD_GATE_BOUNDS[0])],
    [float(CFG.CONTROL_WEIGHT_BOUNDS[1]), float(CFG.CONTROL_WEIGHT_BOUNDS[1]), float(CFG.CONTROL_WEIGHT_BOUNDS[1]), float(REDUCED4_CLOUD_GATE_BOUNDS[1])],
]

REDUCED4_FIXED_VALUES = {
    "W_RT_Energy": 1.5,
    "W_Batch_Energy": 1.5,
    "W_AI_Energy": 1.5,
    "W_Queue": 0.0,
    "W_Risk_Scale": 1.0,
    "Beta_Control": 8.0,
    "Opportunity_Rho": 0.0,
}


def reduced4_basic_anchor_points():
    """原始 4D BO 引导点，用于对比旧版本搜索行为。"""
    return [
        [2.55, 2.55, 2.55, 0.50],
        [5.00, 2.55, 2.55, 0.50],
        [2.55, 5.00, 5.00, 0.05],
    ]


def reduced4_recommended_anchor_points():
    """改进后的 4D BO 引导点。

    重点把敏感度分析得到的强区域 [5,5,5,0.05] 纳入 BO 初始样本，
    避免 Context/TR BO 在有限轮次内长期停留在中性区域附近。
    """
    return [
        [2.55, 2.55, 2.55, 0.50],   # 中性固定点
        [5.00, 5.00, 5.00, 0.05],   # sensitivity-tuned strong point
        [5.00, 5.00, 5.00, 0.10],   # 强时延 + 低云门控邻域
        [5.00, 5.00, 3.00, 0.10],   # AI 权重稍弱的邻域
        [3.00, 5.00, 5.00, 0.10],   # RT 权重稍弱的邻域
        [5.00, 3.00, 5.00, 0.10],   # Batch 权重稍弱的邻域
    ]


def reduced4_anchor_points(anchor_mode="recommended"):
    """根据 anchor_mode 返回 4D BO 引导点。"""
    mode = str(anchor_mode or "recommended").lower()
    if mode in {"basic", "old", "vanilla"}:
        return reduced4_basic_anchor_points()
    return reduced4_recommended_anchor_points()


def reduced4_to_full_theta(theta4):
    """把 4D 控制向量映射回调度器使用的完整 11D theta。"""
    t = list(theta4)
    if len(t) < 4:
        base = reduced4_anchor_points()[0]
        t = t + base[len(t):]
    rt_lat = float(t[0])
    batch_lat = float(t[1])
    ai_lat = float(t[2])
    cloud_gate = float(np.clip(float(t[3]), float(REDUCED4_CLOUD_GATE_BOUNDS[0]), float(REDUCED4_CLOUD_GATE_BOUNDS[1])))
    full = default_control_vector(fill=1.5)
    names = list(CFG.FEATURE_NAMES)

    def set_name(name, value):
        if name in names:
            full[names.index(name)] = float(value)

    set_name("W_RT_Latency", rt_lat)
    set_name("W_Batch_Latency", batch_lat)
    set_name("W_AI_Latency", ai_lat)
    for k, v in REDUCED4_FIXED_VALUES.items():
        set_name(k, v)
    set_name("Cloud_Gate", cloud_gate)
    return full


def map_group_theta_to_full(theta, group_cfg):
    if group_cfg.get("control_mode") == "reduced4":
        return reduced4_to_full_theta(theta)
    if group_cfg.get("control_mode") == "reduced6":
        return reduced6_to_full_theta(theta)
    return list(theta)


def reduced4_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="none"):
    """创建 reduced4 BO agent 参数。默认 cold-start；旧 anchor 只作为 legacy 消融。"""
    mode = str(anchor_mode or "none").strip().lower()
    anchors = [] if mode in {"none", "cold", "cold_start", "no_anchor", "off"} else reduced4_anchor_points(anchor_mode=mode)
    return {
        "dim": 4,
        "bounds": REDUCED4_BOUNDS,
        "feature_names": list(REDUCED4_FEATURE_NAMES),
        "use_context": bool(use_context),
        "use_state_partition": bool(use_context),
        "use_trust_region": bool(use_trust_region),
        "context_dim": len(CFG.CONTEXT_FEATURE_NAMES) if use_context else 0,
        "context_bounds": CFG.CONTEXT_BOUNDS if use_context else None,
        "anchor_points": anchors,
    }


# ===============================================================
# 低维 6D Trade-off BO 实验配置
# ---------------------------------------------------------------
# reduced4 只调三类时延权重 + 云门控。压力场景下真正容易变化的
# 往往是队列权重 W_Queue 和风险权重 W_Risk_Scale，因此补充 reduced6：
# [RT时延, Batch时延, AI时延, 队列压力, 风险缩放, 云门控]
# ===============================================================
REDUCED6_FEATURE_NAMES = [
    "W_RT_Latency", "W_Batch_Latency", "W_AI_Latency",
    "W_Queue", "W_Risk_Scale", "Cloud_Gate"
]
REDUCED6_CLOUD_GATE_BOUNDS = (0.05, 0.95)
REDUCED6_BOUNDS = [
    [float(CFG.CONTROL_WEIGHT_BOUNDS[0]), float(CFG.CONTROL_WEIGHT_BOUNDS[0]), float(CFG.CONTROL_WEIGHT_BOUNDS[0]),
     float(CFG.CONTROL_QUEUE_BOUNDS[0]), float(CFG.CONTROL_RISK_SCALE_BOUNDS[0]), float(REDUCED6_CLOUD_GATE_BOUNDS[0])],
    [float(CFG.CONTROL_WEIGHT_BOUNDS[1]), float(CFG.CONTROL_WEIGHT_BOUNDS[1]), float(CFG.CONTROL_WEIGHT_BOUNDS[1]),
     float(CFG.CONTROL_QUEUE_BOUNDS[1]), float(CFG.CONTROL_RISK_SCALE_BOUNDS[1]), float(REDUCED6_CLOUD_GATE_BOUNDS[1])],
]

REDUCED6_FIXED_VALUES = {
    "W_RT_Energy": 1.5,
    "W_Batch_Energy": 1.5,
    "W_AI_Energy": 1.5,
    "Beta_Control": 8.0,
    "Opportunity_Rho": 0.0,
}


def reduced6_to_full_theta(theta6):
    """把 6D trade-off 控制向量映射回完整 11D theta。"""
    base = [3.0, 3.0, 3.0, 1.0, 1.0, 0.30]
    t = list(theta6)
    if len(t) < 6:
        t = t + base[len(t):]
    rt_lat = float(t[0])
    batch_lat = float(t[1])
    ai_lat = float(t[2])
    queue_w = float(np.clip(float(t[3]), *CFG.CONTROL_QUEUE_BOUNDS))
    risk_scale = float(np.clip(float(t[4]), *CFG.CONTROL_RISK_SCALE_BOUNDS))
    cloud_gate = float(np.clip(float(t[5]), float(REDUCED6_CLOUD_GATE_BOUNDS[0]), float(REDUCED6_CLOUD_GATE_BOUNDS[1])))
    full = default_control_vector(fill=1.5)
    names = list(CFG.FEATURE_NAMES)

    def set_name(name, value):
        if name in names:
            full[names.index(name)] = float(value)

    set_name("W_RT_Latency", rt_lat)
    set_name("W_Batch_Latency", batch_lat)
    set_name("W_AI_Latency", ai_lat)
    set_name("W_Queue", queue_w)
    set_name("W_Risk_Scale", risk_scale)
    set_name("Cloud_Gate", cloud_gate)
    for k, v in REDUCED6_FIXED_VALUES.items():
        set_name(k, v)
    return full


def reduced6_basic_anchor_points():
    return [
        [2.55, 2.55, 2.55, 1.0, 1.0, 0.50],
        [5.00, 5.00, 5.00, 1.0, 1.0, 0.05],
        [3.00, 3.00, 3.00, 5.0, 1.0, 0.30],
    ]


def reduced6_recommended_anchor_points():
    return [
        [2.55, 2.55, 2.55, 1.0, 1.0, 0.50],  # 中性点
        [5.00, 5.00, 5.00, 1.0, 1.0, 0.05],  # 原 tuned fixed 邻域
        [3.00, 3.00, 3.00, 5.0, 1.0, 0.30],  # 高队列压力权重
        [3.00, 3.00, 3.00, 1.0, 5.0, 0.30],  # 高 deadline 风险权重
        [2.00, 2.00, 2.00, 3.0, 2.0, 0.90],  # 保守边缘优先
        [4.00, 4.00, 4.00, 3.0, 2.0, 0.20],  # 高时延 + 中队列
    ]


def reduced6_anchor_points(anchor_mode="recommended"):
    mode = str(anchor_mode or "recommended").lower()
    if mode in {"basic", "old", "vanilla"}:
        return reduced6_basic_anchor_points()
    return reduced6_recommended_anchor_points()


def reduced6_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="none"):
    """创建 reduced6 BO agent 参数。

    v2 主线默认 cold-start：anchor_mode="none" 时不提供 anchor_points，
    BO 第 1 轮开始就由自身采样/模型选择 theta。旧 anchor 行为仍可用，
    但只作为 legacy 消融。
    """
    mode = str(anchor_mode or "none").strip().lower()
    anchors = [] if mode in {"none", "cold", "cold_start", "no_anchor", "off"} else reduced6_anchor_points(anchor_mode=mode)
    return {
        "dim": 6,
        "bounds": REDUCED6_BOUNDS,
        "feature_names": list(REDUCED6_FEATURE_NAMES),
        "use_context": bool(use_context),
        "use_state_partition": bool(use_context),
        "use_trust_region": bool(use_trust_region),
        "context_dim": len(CFG.CONTEXT_FEATURE_NAMES) if use_context else 0,
        "context_bounds": CFG.CONTEXT_BOUNDS if use_context else None,
        "anchor_points": anchors,
    }


def _control_feature_names_for_vector(vec):
    try:
        n = len(vec)
    except Exception:
        n = 0
    if n == 6:
        return list(REDUCED6_FEATURE_NAMES)
    if n == 4:
        return list(REDUCED4_FEATURE_NAMES)
    return [f"Theta{i}" for i in range(n)]

def create_scenario_agent(group_cfg, seed):
    """按实验分组配置创建 agent。"""
    if group_cfg.get("agent_kwargs") is None:
        return None
    kwargs = dict(group_cfg["agent_kwargs"])
    kwargs["py_rng"] = random.Random(resolve_base_seed(seed, stream=7100 + CFG.DIM_THETA))
    kwargs["torch_gen"] = torch.Generator().manual_seed(resolve_base_seed(seed, stream=7200 + CFG.DIM_THETA))
    return FederatedBOAgent(**kwargs)

def run_scenario_group(seed, group_key, group_cfg):
    """运行一组情景方法对比实验，支持完整 11D 和 reduced4 两类控制向量。"""
    fac = ConnectedFactory(fid=0, name=group_cfg["label"], seed=seed, node_config=CFG.NODES_CFG, scheduler_type=group_cfg.get("scheduler_type", "Boltzmann"), norm_mode=group_cfg.get("norm_mode", "rolling"))
    fac.reset(use_batch=False)
    fac.agent = create_scenario_agent(group_cfg, seed)
    configure_refactor_agent(fac.agent, group_cfg)
    fac.perf_log["group_key"] = group_key
    fac.perf_log["group_label"] = group_cfg["label"]
    is_reduced = group_cfg.get("control_mode") == "reduced4"
    fac.disable_internal_agent_tell = bool(is_reduced and fac.agent is not None)

    for i in range(CFG.BO_ITERATIONS):
        state, _, _ = fac.scenario_monitor.get_state(fac.current_time)
        ctx = fac.scenario_monitor.get_context_vector(fac.current_time)
        if fac.agent is None:
            theta_control = list(group_cfg["fixed_theta"])
            ask_state = state
            ask_ctx = ctx
        else:
            ask_state = state if getattr(fac.agent, "use_state_partition", False) else None
            ask_ctx = ctx if getattr(fac.agent, "use_context", False) else None
            theta_control = fac.agent.ask(state=ask_state, context=ask_ctx)

        theta_full = map_group_theta_to_full(theta_control, group_cfg)
        fac.current_control_vector = list(theta_full)
        fac.current_control_label = group_cfg.get("label", group_key)
        _, _, _, _, metrics, _ = fac.run_continuous(
            theta_full,
            eval_state=ask_state if fac.agent is not None else state,
            eval_context=ask_ctx if fac.agent is not None else ctx,
            feedback_control=theta_control,
        )
        if (not fac._use_cohort_feedback()) and is_reduced and fac.agent is not None:
            state_arg = ask_state if getattr(fac.agent, "use_state_partition", False) else None
            context_arg = ask_ctx if getattr(fac.agent, "use_context", False) else None
            fac.agent.tell(theta_control, metrics["cost"], state=state_arg, context=context_arg)
            fac.scheduler.update_beta(metrics["cost"])

        if (i + 1) % 10 == 0:
            print(f"  [{group_cfg['label']}] Iteration {i + 1}/{CFG.BO_ITERATIONS}")
    if fac._use_cohort_feedback() and bool(getattr(CFG, "COHORT_FORCE_FINALIZE_AT_RUN_END", True)):
        fac._finalize_ready_cohorts(fac.current_time, force=True, reason="run_end")
    fac.perf_log["cohort_feedback_debug_rows"] = list(getattr(fac, "cohort_feedback_rows", []))
    return fac.perf_log

def best_so_far(seq):
    """计算 best-so-far 序列，用于观察 BO 是否更快找到历史最好值。"""
    out = []
    best = -float("inf")
    for x in seq:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            out.append(best if out else np.nan)
            continue
        best = max(best, float(x))
        out.append(best)
    return out


def get_bo_phase_ranges():
    """根据 LAMBDA_SCHEDULE 和 BO_INTERVAL，把时间段映射到 BO 迭代阶段。"""
    phases = []
    prev_end_iter = 0
    total_iters = int(CFG.BO_ITERATIONS)
    for idx, (start_t, end_t, lam) in enumerate(CFG.LAMBDA_SCHEDULE, start=1):
        end_iter = int(min(total_iters, max(prev_end_iter + 1, round(end_t / max(CFG.BO_INTERVAL, 1e-9)))))
        start_iter = prev_end_iter + 1
        phases.append({
            "phase_idx": idx,
            "start_time": float(start_t),
            "end_time": float(end_t),
            "lambda": float(lam),
            "iter_start": int(start_iter),
            "iter_end": int(end_iter),
        })
        prev_end_iter = end_iter
        if prev_end_iter >= total_iters:
            break
    return phases


def _slice_metric_by_phase(metric_list, phase):
    start = max(0, phase["iter_start"] - 1)
    end = min(len(metric_list), phase["iter_end"])
    return metric_list[start:end]


def _first_reach_iteration(seq, target):
    for idx, value in enumerate(seq, start=1):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if float(value) >= float(target):
            return idx
    return np.nan


def save_scenario_phase_summary(group_logs):
    """保存阶段统计，便于比较场景切换前后谁恢复更快。"""
    rows = []
    phases = get_bo_phase_ranges()
    for group_key, info in group_logs.items():
        log = aggregate_logs(info["logs"])
        reward = _as_clean_numeric_list(log.get("reward", []))
        delay = _as_clean_numeric_list(log.get("avg_delay", []))
        sla = _as_clean_numeric_list(log.get("sla_success_rate", []))
        bsf = best_so_far(reward)
        row = {
            "Group_Key": group_key,
            "Group_Label": info["label"],
            "Overall_Avg_Reward": _safe_nanmean(reward),
            "Overall_Avg_Delay": _safe_nanmean(delay),
            "Final_BSF_Reward": float(bsf[-1]) if bsf else np.nan,
        }
        if len(bsf) >= 2 and not np.isnan(bsf[0]) and not np.isnan(bsf[-1]):
            target = float(bsf[0] + 0.9 * (bsf[-1] - bsf[0]))
            row["BSF_90pct_Target"] = target
            row["BSF_90pct_Hit_Iter"] = _first_reach_iteration(bsf, target)
        else:
            row["BSF_90pct_Target"] = np.nan
            row["BSF_90pct_Hit_Iter"] = np.nan
        for phase in phases:
            pidx = phase["phase_idx"]
            r_seg = _slice_metric_by_phase(reward, phase)
            d_seg = _slice_metric_by_phase(delay, phase)
            s_seg = _slice_metric_by_phase(sla, phase)
            row[f"Phase{pidx}_Iter_Start"] = phase["iter_start"]
            row[f"Phase{pidx}_Iter_End"] = phase["iter_end"]
            row[f"Phase{pidx}_Lambda"] = phase["lambda"]
            row[f"Phase{pidx}_Avg_Reward"] = _safe_nanmean(r_seg)
            row[f"Phase{pidx}_Avg_Delay"] = _safe_nanmean(d_seg)
            row[f"Phase{pidx}_Avg_SLA"] = _safe_nanmean(s_seg)
            if pidx >= 2:
                early_end = min(phase["iter_end"], phase["iter_start"] + 9)
                early_phase = dict(phase)
                early_phase["iter_end"] = early_end
                r_early = _slice_metric_by_phase(reward, early_phase)
                d_early = _slice_metric_by_phase(delay, early_phase)
                row[f"Phase{pidx}_Early10_Avg_Reward"] = _safe_nanmean(r_early)
                row[f"Phase{pidx}_Early10_Avg_Delay"] = _safe_nanmean(d_early)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(SCENARIO_SAVE_DIR, "scenario_phase_summary.csv"), index=False)


def build_cohort_feedback_dataframe(log, group_key, group_label):
    rows = []
    for run_idx, row in enumerate(log.get("cohort_feedback_debug_rows", []) or []):
        if not isinstance(row, dict):
            continue
        out = {
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": group_label,
            "Run_Index_重复序号": run_idx,
            "Cohort_ID_任务批次ID": row.get("cohort_id"),
            "Window_Index_窗口序号": row.get("window_index"),
            "Start_Time_批次开始时间": row.get("start_time"),
            "Finalize_Time_反馈时间": row.get("finalize_time"),
            "Age_Windows_跨窗口数": row.get("age_windows"),
            "Finalize_Reason_反馈原因": row.get("reason"),
            "Total_Tasks_批次总任务数": row.get("total_tasks"),
            "Completed_Tasks_完成任务数": row.get("completed_tasks"),
            "Unfinished_Tasks_未完成任务数": row.get("unfinished_tasks"),
            "Completion_Ratio_完成比例": row.get("completion_ratio"),
            "Confidence_置信度": row.get("confidence"),
            "Cohort_Cost_批次Cost": row.get("cohort_cost"),
            "Cohort_Reward_批次Reward": row.get("cohort_reward"),
            "Avg_Energy_Est_估计平均能耗": row.get("avg_energy_est"),
            "Censored_Avg_Delay_截尾平均时延": row.get("censored_avg_delay"),
            "Completed_Avg_Delay_已完成平均时延": row.get("completed_avg_delay"),
            "Effective_Violation_Rate_有效违约率": row.get("effective_violation_rate"),
            "Avg_Lateness_Effective_有效平均超期": row.get("avg_lateness_effective"),
            "Unfinished_Ratio_未完成比例": row.get("unfinished_ratio"),
            "Pending_Area_Per_Task_单位任务积压面积": row.get("pending_area_per_task"),
            "RT_Arrivals_实时到达": row.get("rt_arrivals"),
            "Batch_Arrivals_批任务到达": row.get("batch_arrivals"),
            "AI_Arrivals_AI到达": row.get("ai_arrivals"),
            "RT_Completed_实时完成": row.get("rt_completed"),
            "Batch_Completed_批任务完成": row.get("batch_completed"),
            "AI_Completed_AI完成": row.get("ai_completed"),

            # Per-class cohort diagnostics: completed-only, censored/effective, energy and unfinished.
            "RT_Unfinished_实时未完成": row.get("rt_unfinished"),
            "Batch_Unfinished_批任务未完成": row.get("batch_unfinished"),
            "AI_Unfinished_AI未完成": row.get("ai_unfinished"),
            "RT_Censored_Avg_Delay_实时截尾平均时延": row.get("rt_censored_avg_delay"),
            "Batch_Censored_Avg_Delay_批任务截尾平均时延": row.get("batch_censored_avg_delay"),
            "AI_Censored_Avg_Delay_AI截尾平均时延": row.get("ai_censored_avg_delay"),
            "RT_Effective_Avg_Lateness_实时有效平均超期": row.get("rt_effective_avg_lateness"),
            "Batch_Effective_Avg_Lateness_批任务有效平均超期": row.get("batch_effective_avg_lateness"),
            "AI_Effective_Avg_Lateness_AI有效平均超期": row.get("ai_effective_avg_lateness"),
            "RT_Effective_Vio_Rate_实时有效违约率": row.get("rt_effective_vio_rate"),
            "Batch_Effective_Vio_Rate_批任务有效违约率": row.get("batch_effective_vio_rate"),
            "AI_Effective_Vio_Rate_AI有效违约率": row.get("ai_effective_vio_rate"),
            "RT_Unfinished_Ratio_实时未完成比例": row.get("rt_unfinished_ratio"),
            "Batch_Unfinished_Ratio_批任务未完成比例": row.get("batch_unfinished_ratio"),
            "AI_Unfinished_Ratio_AI未完成比例": row.get("ai_unfinished_ratio"),
            "RT_Avg_Energy_Est_实时估计平均能耗": row.get("rt_avg_energy_est"),
            "Batch_Avg_Energy_Est_批任务估计平均能耗": row.get("batch_avg_energy_est"),
            "AI_Avg_Energy_Est_AI估计平均能耗": row.get("ai_avg_energy_est"),
            "RT_Avg_Est_Latency_实时估计平均时延": row.get("rt_avg_estimated_latency"),
            "Batch_Avg_Est_Latency_批任务估计平均时延": row.get("batch_avg_estimated_latency"),
            "AI_Avg_Est_Latency_AI估计平均时延": row.get("ai_avg_estimated_latency"),

            "Theta_Control_反馈控制向量": _safe_json(row.get("theta_control")),
            "Theta_Full_完整调度向量": _safe_json(row.get("theta_full")),
            "Context_绑定情景": _safe_json(row.get("context")),
            "State_绑定状态": row.get("state"),
            # Dual feedback diagnostics: window provisional -> delayed refined replacement.
            "Sample_ID_样本ID": row.get("sample_id"),
            "Dual_Refined_Cost_精反馈Cost": row.get("dual_refined_cost"),
            "Dual_Refined_Source_精反馈来源": row.get("dual_refined_source"),
            "Dual_Class_Aggregation_分类合成方式": row.get("dual_class_aggregation"),
            "Dual_Replace_Success_是否替换窗口样本": row.get("dual_replace_success"),
            "Dual_Window_Provisional_Cost_窗口临时Cost": row.get("dual_window_provisional_cost"),
            "Dual_Refined_Delta_vs_Window_精反馈减窗口Cost": row.get("dual_refined_delta_vs_window"),
            "Dual_Refined_Ratio_vs_Window_精反馈除窗口Cost": row.get("dual_refined_ratio_vs_window"),
            "Dual_Energy_Term_能耗项": row.get("dual_energy_term"),
            "Dual_Pending_Term_积压面积项": row.get("dual_pending_term"),
            "Dual_Class_Weighted_Term_分类合成项": row.get("dual_class_weighted_term"),
            "Dual_Ref_Probs_参考比例": _safe_json(row.get("dual_ref_probs")),
            "Dual_Actual_Probs_实际比例": _safe_json(row.get("dual_actual_probs")),
            "Dual_Equal_Probs_等权比例": _safe_json(row.get("dual_equal_probs")),
            "RT_Class_Cost_实时分类Cost": row.get("dual_rt_class_cost"),
            "RT_Avg_Delay_实时已完成平均时延": row.get("dual_rt_avg_delay_completed"),
            "RT_Avg_Lateness_实时已完成平均超期": row.get("dual_rt_avg_lateness_completed"),
            "RT_Vio_Rate_实时已完成违约率": row.get("dual_rt_vio_rate_completed"),
            "RT_Avg_Delay_Used_实时精反馈使用时延": row.get("dual_rt_avg_delay_used"),
            "RT_Avg_Lateness_Used_实时精反馈使用超期": row.get("dual_rt_avg_lateness_used"),
            "RT_Vio_Rate_Used_实时精反馈使用违约率": row.get("dual_rt_vio_rate_used"),
            "RT_Metric_Mode_实时精反馈指标模式": row.get("dual_rt_metric_mode"),
            "RT_Completion_Ratio_实时完成比例": row.get("dual_rt_completion_ratio"),
            "RT_Weight_Used_实时合成权重": row.get("dual_rt_weight_used"),
            "Batch_Class_Cost_批任务分类Cost": row.get("dual_batch_class_cost"),
            "Batch_Avg_Delay_批任务已完成平均时延": row.get("dual_batch_avg_delay_completed"),
            "Batch_Avg_Lateness_批任务已完成平均超期": row.get("dual_batch_avg_lateness_completed"),
            "Batch_Vio_Rate_批任务已完成违约率": row.get("dual_batch_vio_rate_completed"),
            "Batch_Avg_Delay_Used_批任务精反馈使用时延": row.get("dual_batch_avg_delay_used"),
            "Batch_Avg_Lateness_Used_批任务精反馈使用超期": row.get("dual_batch_avg_lateness_used"),
            "Batch_Vio_Rate_Used_批任务精反馈使用违约率": row.get("dual_batch_vio_rate_used"),
            "Batch_Metric_Mode_批任务精反馈指标模式": row.get("dual_batch_metric_mode"),
            "Batch_Completion_Ratio_批任务完成比例": row.get("dual_batch_completion_ratio"),
            "Batch_Weight_Used_批任务合成权重": row.get("dual_batch_weight_used"),
            "AI_Class_Cost_AI分类Cost": row.get("dual_ai_class_cost"),
            "AI_Avg_Delay_AI已完成平均时延": row.get("dual_ai_avg_delay_completed"),
            "AI_Avg_Lateness_AI已完成平均超期": row.get("dual_ai_avg_lateness_completed"),
            "AI_Vio_Rate_AI已完成违约率": row.get("dual_ai_vio_rate_completed"),
            "AI_Avg_Delay_Used_AI精反馈使用时延": row.get("dual_ai_avg_delay_used"),
            "AI_Avg_Lateness_Used_AI精反馈使用超期": row.get("dual_ai_avg_lateness_used"),
            "AI_Vio_Rate_Used_AI精反馈使用违约率": row.get("dual_ai_vio_rate_used"),
            "AI_Metric_Mode_AI精反馈指标模式": row.get("dual_ai_metric_mode"),
            "AI_Completion_Ratio_AI完成比例": row.get("dual_ai_completion_ratio"),
            "AI_Weight_Used_AI合成权重": row.get("dual_ai_weight_used"),
        }
        rows.append(out)
    return pd.DataFrame(rows)

def save_scenario_experiment_csvs(group_logs):
    summary_rows = []
    for group_key, info in group_logs.items():
        mean_log = aggregate_logs(info["logs"])
        round_df = group_log_to_dataframe(mean_log, group_key, info["label"])
        round_df.to_csv(os.path.join(SCENARIO_SAVE_DIR, f"{group_key}_round_summary_轮次汇总.csv"), index=False)

        context_source_log = info["logs"][0] if len(info.get("logs", [])) == 1 else mean_log
        context_df = build_context_debug_dataframe(context_source_log, group_key, info["label"])
        context_df.to_csv(os.path.join(SCENARIO_SAVE_DIR, f"{group_key}_context_debug_情景调试.csv"), index=False)

        alloc_df = build_alloc_debug_dataframe(mean_log, group_key, info["label"])
        alloc_df.to_csv(os.path.join(SCENARIO_SAVE_DIR, f"{group_key}_alloc_debug_节点分配调试.csv"), index=False)

        cohort_frames = []
        for raw_log in info.get("logs", []):
            cdf = build_cohort_feedback_dataframe(raw_log, group_key, info["label"])
            if not cdf.empty:
                cohort_frames.append(cdf)
        if cohort_frames:
            pd.concat(cohort_frames, ignore_index=True).to_csv(
                os.path.join(SCENARIO_SAVE_DIR, f"{group_key}_cohort_feedback_debug_批次反馈调试.csv"),
                index=False
            )

        # 兼容旧版列名与新版能耗拆分列名。
        cumulative_energy_col = None
        for candidate_col in [
            "Cumulative_Energy_累计目标能耗",
            "Cumulative_Energy_累计能耗",
        ]:
            if candidate_col in round_df.columns:
                cumulative_energy_col = candidate_col
                break
        cumulative_energy_real_col = "Cumulative_Energy_Real_累计真实能耗" if "Cumulative_Energy_Real_累计真实能耗" in round_df.columns else None

        summary_rows.append({
            "Group_Key_方法键": group_key,
            "Group_Label_方法名称": info["label"],
            "Mean_Reward_平均奖励": float(np.nanmean(round_df["Reward_奖励"])) if not round_df.empty else np.nan,
            "Mean_Avg_Delay_平均时延": float(np.nanmean(round_df["Avg_Delay_平均时延"])) if not round_df.empty else np.nan,
            "Mean_Violation_Rate_平均违约率": float(np.nanmean(round_df["Violation_Rate_违约率"])) if not round_df.empty else np.nan,
            "Mean_SLA_Success_Rate_平均SLA成功率": float(np.nanmean(round_df["SLA_Success_Rate_SLA成功率"])) if not round_df.empty else np.nan,
            "Final_Cumulative_Energy_最终累计目标能耗": float(round_df[cumulative_energy_col].iloc[-1]) if (not round_df.empty and cumulative_energy_col is not None) else np.nan,
            "Final_Cumulative_Energy_Real_最终累计真实能耗": float(round_df[cumulative_energy_real_col].iloc[-1]) if (not round_df.empty and cumulative_energy_real_col is not None) else np.nan,
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(SCENARIO_SAVE_DIR, "scenario_experiment_summary_实验汇总.csv"), index=False)

METHOD_STYLE_MAP = {
    "fixednorm_fixed_balanced": {"color": "#7F7F7F", "linestyle": "--", "label": "Fixed Balanced"},
    "fixednorm_vanilla_bo": {"color": "#0072B2", "linestyle": "-", "label": "Vanilla BO"},
    "fixednorm_context_bo": {"color": "#009E73", "linestyle": "-", "label": "Context BO"},
    "fixednorm_context_tr_bo": {"color": "#D55E00", "linestyle": "-", "label": "Context BO + TR"},
    "reduced4_fixed_mid": {"color": "#6E6E6E", "linestyle": "--", "label": "Fixed Mid"},
    "reduced4_fixed_recommended": {"color": "#000000", "linestyle": "-", "label": "Tuned Fixed"},
    "reduced4_vanilla_bo": {"color": "#0072B2", "linestyle": "-", "label": "Vanilla BO"},
    "reduced4_context_bo": {"color": "#009E73", "linestyle": "-", "label": "Context BO"},
    "reduced4_context_tr_bo": {"color": "#D55E00", "linestyle": "-", "label": "Context BO + TR"},
    "reduced4_vanilla_bo_anchor": {"color": "#56B4E9", "linestyle": "-", "label": "Vanilla BO + Anchor"},
    "reduced4_context_bo_anchor": {"color": "#CC79A7", "linestyle": "-", "label": "Context BO + Anchor"},
    "reduced4_context_tr_bo_anchor": {"color": "#E69F00", "linestyle": "-", "label": "Context BO + TR + Anchor"},
    "direct_round_robin": {"color": "#999999", "linestyle": "--", "label": "RoundRobin Direct"},
    "direct_greedy_cost": {"color": "#CC6677", "linestyle": "--", "label": "Greedy Direct Cost"},
    "direct_least_load": {"color": "#117733", "linestyle": "--", "label": "LeastLoad Direct"},
    "direct_queue_aware_greedy": {"color": "#882255", "linestyle": "--", "label": "QueueAware Greedy Direct"},
}


def get_method_style(group_key, info, fallback_idx=0):
    style = dict(METHOD_STYLE_MAP.get(group_key, {}))
    if "color" not in style:
        style["color"] = f"C{fallback_idx}"
    if "linestyle" not in style:
        style["linestyle"] = "-"
    if "label" not in style:
        style["label"] = info.get("label", group_key)
    return style


def plot_scenario_convergence(group_logs):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    phase_ranges = get_bo_phase_ranges()
    for idx2, (group_key, info) in enumerate(group_logs.items()):
        log = aggregate_logs(info["logs"])
        style = get_method_style(group_key, info, fallback_idx=idx2)
        reward = ema_smooth(log.get("reward", []), weight=0.85)
        delay = ema_smooth(log.get("avg_delay", []), weight=0.85)
        if reward:
            axes[0].plot(np.arange(1, len(reward) + 1), reward, label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
        if delay:
            axes[1].plot(np.arange(1, len(delay) + 1), delay, label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
    for ax in axes:
        for phase in phase_ranges[:-1]:
            ax.axvline(phase["iter_end"], color="gray", linestyle=":", alpha=0.7)
    axes[0].set_title("Scenario-aware BO Reward Convergence")
    axes[0].set_xlabel("BO Iteration")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="best")
    axes[1].set_title("Scenario-aware BO Average Delay")
    axes[1].set_xlabel("BO Iteration")
    axes[1].set_ylabel("Average Delay (s)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIO_SAVE_DIR, "scenario_convergence.png"), dpi=300)
    plt.close(fig)


def plot_scenario_best_so_far(group_logs):
    """绘制 best-so-far reward，用来更直观看 BO 的收敛速度。"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    phase_ranges = get_bo_phase_ranges()
    for idx2, (group_key, info) in enumerate(group_logs.items()):
        log = aggregate_logs(info["logs"])
        style = get_method_style(group_key, info, fallback_idx=idx2)
        bsf = best_so_far(log.get("reward", []))
        if bsf:
            ax.plot(np.arange(1, len(bsf) + 1), bsf, label=style["label"], color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
    for phase in phase_ranges[:-1]:
        ax.axvline(phase["iter_end"], color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Scenario-aware BO Best-So-Far Reward")
    ax.set_xlabel("BO Iteration")
    ax.set_ylabel("Best-So-Far Reward")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIO_SAVE_DIR, "scenario_best_so_far.png"), dpi=300)
    plt.close(fig)


def _write_refactor_config_snapshot(output_dir, selected_keys=None, groups=None):
    """保存本次运行的关键配置，方便复现实验。"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "refactor_version": REFACTOR_VERSION,
            "bo_iterations": int(CFG.BO_ITERATIONS),
            "bo_interval": float(CFG.BO_INTERVAL),
            "session_duration": float(CFG.SESSION_DURATION),
            "feedback_mode": str(getattr(CFG, "FEEDBACK_MODE", "window")),
            "feedback_score": str(getattr(CFG, "BO_TRAINING_FEEDBACK_SCORE", "window_original")),
            "bo_training_feedback_score": str(getattr(CFG, "BO_TRAINING_FEEDBACK_SCORE", "window_original")),
            "cbo_reference_mode": str(getattr(CFG, "CBO_REFERENCE_MODE", "off")),
            "cbo_reference_calibration_rounds": int(getattr(CFG, "CBO_REFERENCE_CALIBRATION_ROUNDS", 30)),
            "cbo_reference_min_rounds": int(getattr(CFG, "CBO_REFERENCE_MIN_ROUNDS", 5)),
            "cbo_reference_stat": str(getattr(CFG, "CBO_REFERENCE_STAT", "median")),
            "cbo_reference_trim_pct": float(getattr(CFG, "CBO_REFERENCE_TRIM_PCT", 0.1)),
            "cbo_objective_mode": str(getattr(CFG, "CBO_OBJECTIVE_MODE", "eval_cost")),
            "cbo_tradeoff_alpha": float(getattr(CFG, "CBO_TRADEOFF_ALPHA", 0.8)),
            "cbo_target_success_rate": float(getattr(CFG, "CBO_TARGET_SUCCESS_RATE", 0.995)),
            "cbo_unfinished_penalty_weight": float(getattr(CFG, "CBO_UNFINISHED_PENALTY_WEIGHT", 5.0)),
            "cbo_success_shortfall_weight": float(getattr(CFG, "CBO_SUCCESS_SHORTFALL_WEIGHT", 2.0)),
            "cbo_backlog_growth_penalty_weight": float(getattr(CFG, "CBO_BACKLOG_GROWTH_PENALTY_WEIGHT", 2.0)),
            "cbo_class_imbalance_weight": float(getattr(CFG, "CBO_CLASS_IMBALANCE_WEIGHT", 0.0)),
            "scheduler_tradeoff_mode": str(getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy")),
            "scheduler_tradeoff_alpha": float(getattr(CFG, "SCHEDULER_TRADEOFF_ALPHA", 0.85)),
            "scheduler_alpha_min": float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60)),
            "scheduler_alpha_max": float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97)),
            "scheduler_service_latency_weight": float(getattr(CFG, "SCHEDULER_SERVICE_LATENCY_WEIGHT", 1.0)),
            "scheduler_service_risk_weight": float(getattr(CFG, "SCHEDULER_SERVICE_RISK_WEIGHT", 1.0)),
            "scheduler_service_queue_weight": float(getattr(CFG, "SCHEDULER_SERVICE_QUEUE_WEIGHT", 1.0)),
            "scheduler_energy_weight": float(getattr(CFG, "SCHEDULER_ENERGY_WEIGHT", 1.0)),
            "scheduler_score_norm_mode": str(getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")),
            "scheduler_norm_clip_max": float(getattr(CFG, "SCHEDULER_NORM_CLIP_MAX", 3.0)),
            "scheduler_norm_eps": float(getattr(CFG, "SCHEDULER_NORM_EPS", 1e-6)),
            "scheduler_norm_ema_alpha": float(getattr(CFG, "SCHEDULER_NORM_EMA_ALPHA", 0.995)),
            "deploy_policy_arg": _deploy_policy_arg(),
            "effective_deploy_policy": os.environ.get("SAFEBO_POLICY", None),
            "safe_bo_policy": os.environ.get("SAFEBO_POLICY", None),
            "SAFEBO_POLICY": os.environ.get("SAFEBO_POLICY", None),
            "lambda_schedule": list(getattr(CFG, "LAMBDA_SCHEDULE", [])),
            "task_type_probs": dict(getattr(CFG, "TASK_TYPE_PROBS", {})),
            "use_task_type_adaptation": bool(getattr(CFG, "USE_TASK_TYPE_ADAPTATION", False)),
            "cloud_delay_mult": float(getattr(CFG, "CLOUD_DELAY_MULT", 1.0)),
            "cloud_energy_mult": float(getattr(CFG, "CLOUD_ENERGY_MULT", 1.0)),
            "cloud_speed_mult": float(getattr(CFG, "CLOUD_SPEED_MULT", 1.0)),
            "default_scenario_keys": list(DEFAULT_SCENARIO_KEYS),
            "selected_keys": list(selected_keys) if selected_keys is not None else None,
            "method_deploy_policy_map": method_deploy_policy_map(groups or {}),
            "method_history_policy_map": method_history_policy_map(groups or {}),
            "bo_history_mode": str(getattr(CFG, "BO_HISTORY_MODE", "all")),
            "bo_recent_window": int(getattr(CFG, "BO_RECENT_WINDOW", 80)),
            "bo_confidence_min": float(getattr(CFG, "BO_CONFIDENCE_MIN", 0.35)),
            "bo_confidence_min_samples": int(getattr(CFG, "BO_CONFIDENCE_MIN_SAMPLES", 12)),
            "cbo_history_select_mode": str(getattr(CFG, "CBO_HISTORY_SELECT_MODE", "recent")),
            "cbo_context_k": int(getattr(CFG, "CBO_CONTEXT_K", 50)),
            "cbo_elite_k": int(getattr(CFG, "CBO_ELITE_K", 20)),
            "cbo_diverse_k": int(getattr(CFG, "CBO_DIVERSE_K", 20)),
            "cbo_robust_score_mode": str(getattr(CFG, "CBO_ROBUST_SCORE_MODE", "none")),
            "cbo_robust_std_weight": float(getattr(CFG, "CBO_ROBUST_STD_WEIGHT", 0.5)),
            "cbo_theta_merge_eps": float(getattr(CFG, "CBO_THETA_MERGE_EPS", 0.05)),
            "cbo_context_sim_threshold": float(getattr(CFG, "CBO_CONTEXT_SIM_THRESHOLD", 0.0)),
            "cbo_tr_mode": str(getattr(CFG, "CBO_TR_MODE", "off")),
            "cbo_tr_radius_init": float(getattr(CFG, "CBO_TR_RADIUS_INIT", getattr(CFG, "TRUST_RADIUS_INIT", 0.10))),
            "cbo_tr_radius_min": float(getattr(CFG, "CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))),
            "cbo_tr_radius_max": float(getattr(CFG, "CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35))),
            "cbo_tr_grow": float(getattr(CFG, "CBO_TR_GROW", getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15))),
            "cbo_tr_shrink": float(getattr(CFG, "CBO_TR_SHRINK", getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92))),
            "cbo_tr_update_mode": str(getattr(CFG, "CBO_TR_UPDATE_MODE", "best_so_far")),
            "cbo_tr_compare_window": int(getattr(CFG, "CBO_TR_COMPARE_WINDOW", 30)),
            "cbo_tr_baseline_window": int(getattr(CFG, "CBO_TR_BASELINE_WINDOW", 60)),
            "cbo_tr_improve_pct": float(getattr(CFG, "CBO_TR_IMPROVE_PCT", 0.015)),
            "cbo_tr_worsen_pct": float(getattr(CFG, "CBO_TR_WORSEN_PCT", 0.03)),
            "cbo_tr_deadband_pct": float(getattr(CFG, "CBO_TR_DEADBAND_PCT", 0.01)),
            "cbo_tr_update_patience": int(getattr(CFG, "CBO_TR_UPDATE_PATIENCE", 2)),
            "cbo_tr_anchor_mode": str(getattr(CFG, "CBO_TR_ANCHOR_MODE", "posterior_mean")),
            "cbo_robust_incumbent_mode": str(getattr(CFG, "CBO_ROBUST_INCUMBENT_MODE", "off")),
            "cbo_macro_gate_mode": str(getattr(CFG, "CBO_MACRO_GATE_MODE", "off")),
            "cbo_macro_k": int(getattr(CFG, "CBO_MACRO_K", 100)),
            "cbo_macro_total_scale": str(getattr(CFG, "CBO_MACRO_TOTAL_SCALE", "auto")),
            "cbo_macro_lengthscale_total": float(getattr(CFG, "CBO_MACRO_LENGTHSCALE_TOTAL", 1.0)),
            "cbo_macro_lengthscale_rt": float(getattr(CFG, "CBO_MACRO_LENGTHSCALE_RT", 0.15)),
            "cbo_macro_lengthscale_batch": float(getattr(CFG, "CBO_MACRO_LENGTHSCALE_BATCH", 0.15)),
            "cbo_macro_alpha": float(getattr(CFG, "CBO_MACRO_ALPHA", 1.0)),
            "cbo_dump_candidates": bool(getattr(CFG, "CBO_DUMP_CANDIDATES", False)),
            "cbo_dump_candidates_every": int(getattr(CFG, "CBO_DUMP_CANDIDATES_EVERY", 20)),
            "cbo_dump_candidates_topn": int(getattr(CFG, "CBO_DUMP_CANDIDATES_TOPN", 30)),
            "cbo_select_mode": str(getattr(CFG, "CBO_SELECT_MODE", "greedy")),
            "cbo_topk": int(getattr(CFG, "CBO_TOPK", 5)),
            "cbo_select_temperature": float(getattr(CFG, "CBO_SELECT_TEMPERATURE", 0.20)),
            "cbo_epsilon": float(getattr(CFG, "CBO_EPSILON", 0.10)),
            "cbo_acq_beta": float(getattr(CFG, "CBO_ACQ_BETA", 3.0)),
            "cbo_acq_beta_mode": str(getattr(CFG, "CBO_ACQ_BETA_MODE", "fixed")),
            "cbo_beta_min": float(getattr(CFG, "CBO_BETA_MIN", 0.1)),
            "cbo_beta_max": float(getattr(CFG, "CBO_BETA_MAX", 2.0)),
            "cbo_radius_beta_power": float(getattr(CFG, "CBO_RADIUS_BETA_POWER", 1.0)),
            "cbo_radius_stable_rebound_pct": float(getattr(CFG, "CBO_RADIUS_STABLE_REBOUND_PCT", 0.02)),
            "cbo_radius_unstable_rebound_pct": float(getattr(CFG, "CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04)),
            "cbo_radius_surprise_boost_threshold": float(getattr(CFG, "CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0)),
            "cbo_radius_beta_boost": float(getattr(CFG, "CBO_RADIUS_BETA_BOOST", 1.5)),
            "cbo_radius_beta_cap": float(getattr(CFG, "CBO_RADIUS_BETA_CAP", 3.0)),
            "cbo_service_guard_mode": str(getattr(CFG, "CBO_SERVICE_GUARD_MODE", "off")),
            "cbo_service_guard_delay_pct": float(getattr(CFG, "CBO_SERVICE_GUARD_DELAY_PCT", 0.03)),
            "cbo_service_guard_backlog_pct": float(getattr(CFG, "CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03)),
            "cbo_surprise_window": int(getattr(CFG, "CBO_SURPRISE_WINDOW", 10)),
            "cbo_surprise_z_threshold": float(getattr(CFG, "CBO_SURPRISE_Z_THRESHOLD", 2.0)),
            "cbo_surprise_cost_gap_pct": float(getattr(CFG, "CBO_SURPRISE_COST_GAP_PCT", 0.03)),
            "cbo_sigma_floor": float(getattr(CFG, "CBO_SIGMA_FLOOR", 1e-6)),
            "cbo_radius_reset": float(getattr(CFG, "CBO_RADIUS_RESET", 0.12)),
            "cbo_radius_min_stuck_rounds": int(getattr(CFG, "CBO_RADIUS_MIN_STUCK_ROUNDS", 10)),
            "cbo_rebound_window": int(getattr(CFG, "CBO_REBOUND_WINDOW", 20)),
            "cbo_rebound_threshold_pct": float(getattr(CFG, "CBO_REBOUND_THRESHOLD_PCT", 0.03)),
            "cbo_selection_cooldown": int(getattr(CFG, "CBO_SELECTION_COOLDOWN", 5)),
            "cbo_condition_anchor_switch": str(getattr(CFG, "CBO_CONDITION_ANCHOR_SWITCH", "context_best")),
            "lite_context_feature_names": list(LITE_CONTEXT_FEATURE_NAMES),
            "lite_context_mode_specs": {k: {"label": v.get("label"), "feature_names": [LITE_CONTEXT_FEATURE_NAMES[i] for i in v.get("indices", [])]} for k, v in LITE_CONTEXT_MODE_SPECS.items()},
            "notes": "v3 adds recent/confidence BO and CBO-lite. BO remains cold-start; low-confidence window feedback can be filtered for GP training.",
        }
        with open(os.path.join(output_dir, "refactor_run_config.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write refactor_run_config.json: {e}")

def run_scenario_method_experiments(repeat_runs=1, selected_keys=None, output_dir=None):
    """运行 Fixed / Vanilla BO / Context BO / Context+TR 对比。

    selected_keys: 可选，指定只跑哪些方法，例如 ["fixed_balanced", "vanilla_bo", "context_tr_bo"]。
    output_dir:    可选，指定本次实验所有图和 CSV 的保存目录。
    """
    global SCENARIO_SAVE_DIR
    old_save_dir = SCENARIO_SAVE_DIR
    if output_dir is not None:
        SCENARIO_SAVE_DIR = os.path.abspath(output_dir)
        os.makedirs(SCENARIO_SAVE_DIR, exist_ok=True)

    print("=== BO Refactor v2 Main Experiment ===")
    groups = build_scenario_method_groups()
    if selected_keys is None:
        selected_keys = list(DEFAULT_SCENARIO_KEYS)
    if selected_keys is not None:
        selected_keys = normalize_selected_method_keys([str(k).strip() for k in selected_keys if str(k).strip()])
        groups = {k: v for k, v in groups.items() if k in selected_keys}
    if not groups:
        raise ValueError(f"No valid method groups selected: {selected_keys}")
    groups = apply_deploy_policy_override(groups)
    groups = apply_history_policy_override(groups)
    groups = apply_cbo_stability_policy_override(groups)
    _write_refactor_config_snapshot(SCENARIO_SAVE_DIR, selected_keys=selected_keys, groups=groups)
    group_logs = {k: {"label": v["label"], "logs": []} for k, v in groups.items()}
    # v6.2 runtime logging: saved incrementally after each method finishes.
    runtime_rows = []
    runtime_csv_path = os.path.join(SCENARIO_SAVE_DIR, "method_runtime_summary.csv")

    for run_idx in range(max(1, repeat_runs)):
        seed = CFG.BASE_SEED + run_idx
        print(f"[Repeat {run_idx + 1}/{max(1, repeat_runs)}] seed={seed}")
        for group_key, group_cfg in groups.items():
            method_t0 = time.perf_counter()
            log = run_scenario_group(seed, group_key, group_cfg)
            method_elapsed = time.perf_counter() - method_t0
            group_logs[group_key]["logs"].append(log)

            agent_kwargs = group_cfg.get("agent_kwargs", {}) or {}
            try:
                control_dim = int(len(group_cfg.get("fixed_theta", []))) if group_cfg.get("fixed_theta") is not None else int(agent_kwargs.get("dim", 0) or 0)
            except Exception:
                control_dim = int(agent_kwargs.get("dim", 0) or 0)
            try:
                context_dim = int(agent_kwargs.get("context_dim", 0) or 0)
            except Exception:
                context_dim = 0
            total_model_dim = int(control_dim + context_dim)
            iter_sec = log.get("runtime_iter_elapsed_sec", []) if isinstance(log, dict) else []
            runtime_rows.append({
                "repeat_idx": int(run_idx + 1),
                "seed": int(seed),
                "method_key": str(group_key),
                "method_label": str(group_cfg.get("label", group_key)),
                "method_family": str(group_cfg.get("method_family", "fixed" if group_cfg.get("agent") is None and group_cfg.get("fixed_theta") is not None else "unknown")),
                "control_mode": str(group_cfg.get("control_mode", "full")),
                "context_mode": str(group_cfg.get("context_mode", "none")),
                "deploy_policy": str(group_cfg.get("deploy_policy", "")),
                "deploy_policy_source": str(group_cfg.get("deploy_policy_source", "")),
                "control_dim": int(control_dim),
                "context_dim": int(context_dim),
                "total_model_dim": int(total_model_dim),
                "bo_iterations": int(CFG.BO_ITERATIONS),
                "elapsed_sec": float(method_elapsed),
                "elapsed_min": float(method_elapsed / 60.0),
                "sec_per_iter": float(method_elapsed / max(1, int(CFG.BO_ITERATIONS))),
                "iter_elapsed_mean_sec": float(np.nanmean(iter_sec)) if len(iter_sec) else np.nan,
                "iter_elapsed_p50_sec": float(np.nanmedian(iter_sec)) if len(iter_sec) else np.nan,
                "iter_elapsed_p90_sec": float(np.nanpercentile(iter_sec, 90)) if len(iter_sec) else np.nan,
                "feedback_score": str(getattr(CFG, "BO_TRAINING_FEEDBACK_SCORE", "window_original")),
            })
            try:
                pd.DataFrame(runtime_rows).to_csv(runtime_csv_path, index=False, encoding="utf-8-sig")
            except Exception as e:
                print(f"[WARN] failed to save runtime summary: {e}", flush=True)
            print(
                f"[TIME] repeat={run_idx + 1} method={group_key} "
                f"control_dim={control_dim} context_dim={context_dim} model_input_dim={total_model_dim} "
                f"elapsed={method_elapsed:.1f}s sec/iter={method_elapsed / max(1, int(CFG.BO_ITERATIONS)):.3f}",
                flush=True,
            )
    # v6.2 runtime logging: long table with one row per method / iteration.
    try:
        iter_rows = []
        for group_key, info in group_logs.items():
            for log_idx, log in enumerate(info.get("logs", []), start=1):
                secs = log.get("runtime_iter_elapsed_sec", []) if isinstance(log, dict) else []
                for iter_idx, sec in enumerate(secs, start=1):
                    iter_rows.append({
                        "method_key": group_key,
                        "method_label": info.get("label", group_key),
                        "repeat_log_idx": int(log_idx),
                        "iteration": int(iter_idx),
                        "iter_elapsed_sec": float(sec),
                    })
        if iter_rows:
            pd.DataFrame(iter_rows).to_csv(os.path.join(SCENARIO_SAVE_DIR, "method_runtime_by_iter.csv"), index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] failed to save per-iteration runtime table: {e}", flush=True)

    for group_key, info in group_logs.items():
        summarize_metrics(info["logs"], info["label"])
    save_scenario_experiment_csvs(group_logs)
    save_scenario_phase_summary(group_logs)
    plot_scenario_convergence(group_logs)
    plot_scenario_best_so_far(group_logs)
    plot_group_alloc_heatmaps(group_logs, save_dir=SCENARIO_SAVE_DIR, prefix="scenario")
    plot_group_task_delay_bars(group_logs, save_dir=SCENARIO_SAVE_DIR, prefix="scenario")
    save_extra_diagnostics(group_logs)
    if output_dir is not None:
        SCENARIO_SAVE_DIR = old_save_dir
    return group_logs


def generate_task_ratio_grid(step=10, min_ratio=10):
    """生成三类任务比例网格。

    默认 step=10, min_ratio=10 时，RT/Batch/AI 均至少 10%，且总和为 100%，共 36 个场景。
    返回值单位为百分比整数，例如 (10, 40, 50)。
    """
    step = int(step)
    min_ratio = int(min_ratio)
    if step <= 0:
        raise ValueError("step must be positive")
    if min_ratio < 0:
        raise ValueError("min_ratio must be non-negative")

    ratios = []
    for rt in range(min_ratio, 100 - 2 * min_ratio + 1, step):
        for batch in range(min_ratio, 100 - rt - min_ratio + 1, step):
            ai = 100 - rt - batch
            if ai < min_ratio:
                continue
            # ai 是由 100 - rt - batch 自动补齐的。
            # 只要求三类比例均不低于 min_ratio；不再要求 ai 必须整除 step。
            # 这样 --ratio-step 30 --ratio-min 10 会生成 (10,10,80), (10,40,50) 等粗网格场景。
            ratios.append((rt, batch, ai))
    return ratios


def ratio_folder_name(rt, batch, ai):
    return f"RT{int(rt):02d}_Batch{int(batch):02d}_AI{int(ai):02d}"


def _summarize_ratio_group_logs(group_logs, rt, batch, ai, scenario_dir):
    """把单个比例场景下各方法的结果汇总成若干行。"""
    rows = []
    for group_key, info in group_logs.items():
        mean_log = aggregate_logs(info["logs"])
        reward = list(mean_log.get("reward", []))
        delay = list(mean_log.get("avg_delay", []))
        energy = list(mean_log.get("avg_energy", []))
        vio = list(mean_log.get("vio_rate", []))
        backlog = list(mean_log.get("backlog", []))
        bsf = best_so_far(reward)
        rows.append({
            "RT_Ratio": rt / 100.0,
            "Batch_Ratio": batch / 100.0,
            "AI_Ratio": ai / 100.0,
            "RT_Percent": int(rt),
            "Batch_Percent": int(batch),
            "AI_Percent": int(ai),
            "Scenario_Key": ratio_folder_name(rt, batch, ai),
            "Scenario_Dir": scenario_dir,
            "Group_Key": group_key,
            "Group_Label": info["label"],
            "Mean_Reward": float(np.nanmean(reward)) if reward else np.nan,
            "Final_Reward": float(reward[-1]) if reward else np.nan,
            "Best_So_Far_Final": float(bsf[-1]) if bsf else np.nan,
            "Mean_Avg_Delay": float(np.nanmean(delay)) if delay else np.nan,
            "Final_Avg_Delay": float(delay[-1]) if delay else np.nan,
            "Mean_Avg_Energy": float(np.nanmean(energy)) if energy else np.nan,
            "Mean_Violation_Rate": float(np.nanmean(vio)) if vio else np.nan,
            "Mean_Backlog": float(np.nanmean(backlog)) if backlog else np.nan,
        })
    return rows


def run_ratio_grid_experiments(repeat_runs=1, step=10, min_ratio=10, output_root=None, selected_keys=None):
    """按任务比例网格运行 36 场景实验，并分文件夹保存图和 CSV。

    默认 step=10, min_ratio=10：
    - RT、Batch、AI 三类任务比例均至少 10%；
    - 三者相加为 100%；
    - 一共 36 个比例场景。

    每个场景会单独保存到：
        output_root/RTxx_Batchxx_AIxx/
    其中包含 scenario_convergence.png、scenario_best_so_far.png、各方法轮次 CSV、情景调试 CSV、节点分配 CSV 等。
    根目录会额外保存 ratio_grid_summary_all.csv。
    """
    global SCENARIO_SAVE_DIR
    ratios = generate_task_ratio_grid(step=step, min_ratio=min_ratio)
    if len(ratios) == 0:
        raise ValueError(f"No ratio scenarios generated. Check step/min_ratio. Got step={step}, min_ratio={min_ratio}. Valid condition: step>0 and 3*min_ratio<=100.")

    old_probs = dict(CFG.TASK_TYPE_PROBS)
    old_save_dir = SCENARIO_SAVE_DIR
    root = output_root or os.path.abspath(os.path.join(old_save_dir, f"ratio_grid_step{int(step)}_min{int(min_ratio)}_adapt{int(bool(getattr(CFG, 'USE_TASK_TYPE_ADAPTATION', False)))}"))
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)

    ratio_rows = [
        {
            "Scenario_Index": idx,
            "Scenario_Key": ratio_folder_name(rt, batch, ai),
            "RT_Percent": int(rt),
            "Batch_Percent": int(batch),
            "AI_Percent": int(ai),
            "RT_Ratio": rt / 100.0,
            "Batch_Ratio": batch / 100.0,
            "AI_Ratio": ai / 100.0,
        }
        for idx, (rt, batch, ai) in enumerate(ratios, start=1)
    ]
    pd.DataFrame(ratio_rows).to_csv(os.path.join(root, "ratio_grid_scenarios.csv"), index=False)

    all_summary_rows = []
    print(f"=== Ratio Grid Experiments: {len(ratios)} scenarios ===")
    print(f"Output root: {root}")

    try:
        for idx, (rt, batch, ai) in enumerate(ratios, start=1):
            scenario_key = ratio_folder_name(rt, batch, ai)
            scenario_dir = os.path.join(root, scenario_key)
            os.makedirs(scenario_dir, exist_ok=True)

            CFG.TASK_TYPE_PROBS = {"RT": rt / 100.0, "Batch": batch / 100.0, "AI": ai / 100.0}

            config_payload = {
                "scenario_index": idx,
                "scenario_count": len(ratios),
                "scenario_key": scenario_key,
                "task_type_probs": dict(CFG.TASK_TYPE_PROBS),
                "lambda_schedule": list(CFG.LAMBDA_SCHEDULE),
                "bo_iterations": int(CFG.BO_ITERATIONS),
                "bo_interval": float(CFG.BO_INTERVAL),
                "repeat_runs": int(max(1, repeat_runs)),
                "selected_keys": selected_keys,
                "use_task_type_adaptation": bool(getattr(CFG, "USE_TASK_TYPE_ADAPTATION", False)),
                "normalization": "支持 fixed 和 rolling 两种归一化；默认四线为 fixednorm基线/BO + rollingnorm基线/BO",
            }
            with open(os.path.join(scenario_dir, "scenario_config.json"), "w", encoding="utf-8") as f:
                json.dump(config_payload, f, ensure_ascii=False, indent=2)

            print(f"\n[{idx}/{len(ratios)}] Scenario {scenario_key}: {CFG.TASK_TYPE_PROBS}")
            group_logs = run_scenario_method_experiments(
                repeat_runs=max(1, repeat_runs),
                selected_keys=selected_keys,
                output_dir=scenario_dir,
            )
            all_summary_rows.extend(_summarize_ratio_group_logs(group_logs, rt, batch, ai, scenario_dir))
            pd.DataFrame(all_summary_rows).to_csv(os.path.join(root, "ratio_grid_summary_all.csv"), index=False)
    finally:
        CFG.TASK_TYPE_PROBS = old_probs
        SCENARIO_SAVE_DIR = old_save_dir

    summary_df = pd.DataFrame(all_summary_rows)
    summary_csv = os.path.join(root, "ratio_grid_summary_all.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n=== Ratio grid finished. Summary saved to: {summary_csv} ===")
    return summary_df

def run_baseline_batch(seed):
    """运行本地版 Contextual BO + TR。"""
    fac = ConnectedFactory(fid=0, name="Fac_A_Local_ContextTR", seed=seed, node_config=CFG.NODES_CFG)
    fac.reset(use_batch=False)
    for i in range(CFG.BO_ITERATIONS):
        state, _, _ = fac.scenario_monitor.get_state(fac.current_time)
        context_vec = fac.scenario_monitor.get_context_vector(fac.current_time)
        theta = fac.agent.ask(state=state, context=context_vec)
        fac.current_control_label = "Contextual BO + TR (Local)"
        fac.run_continuous(theta, eval_state=state, eval_context=context_vec)
        if (i + 1) % 10 == 0: print(f"  [BO-Local-ContextTR] Iteration {i + 1}/{CFG.BO_ITERATIONS}")
    return fac.perf_log

def run_rr_batch(seed):
    """运行 Round-Robin 作为简单基线。"""
    """Run Round Robin baseline for comparison"""
    fac = ConnectedFactory(fid=0, name="Fac_A_RR", seed=seed, node_config=CFG.NODES_CFG, scheduler_type="RoundRobin")
    fac.reset(use_batch=False)
    for i in range(CFG.BO_ITERATIONS):
        theta = default_control_vector(fill=1.0)
        fac.run_continuous(theta)
        if (i + 1) % 10 == 0: print(f"  [Round-Robin] Iteration {i + 1}/{CFG.BO_ITERATIONS}")
    return fac.perf_log

def run_federated_batch(seed):
    """运行联邦版 Contextual BO + TR。

    过程：
    1) worker 工厂各自在线推进；
    2) worker 对候选 theta 预测(mu, sigma)；
    3) 云端聚合 score_fed；
    4) 主工厂 A 采用聚合后最优候选并真实执行。
    """
    aggregator = FederatedAggregator()
    fed_rng = random.Random(resolve_base_seed(seed, stream=900))

    workers = []
    for fid, name in [(1, "Fac_B"), (2, "Fac_C")]:
        fac = ConnectedFactory(fid=fid, name=name, seed=seed + fid * 10, node_config=CFG.NODES_CFG)
        fac.reset(use_batch=False)
        workers.append(fac)

    fac_a = ConnectedFactory(fid=0, name="Fac_A_Fed_ContextTR", seed=seed, node_config=CFG.NODES_CFG)
    fac_a.reset(use_batch=False)

    for i in range(CFG.BO_ITERATIONS):
        candidate_pool: List[List[float]] = []

        # Workers advance online each BO round so their contexts/states stay fresh.
        for w in workers:
            s_w, _, _ = w.scenario_monitor.get_state(w.current_time)
            ctx_w = w.scenario_monitor.get_context_vector(w.current_time)
            theta_w = w.agent.ask(state=s_w, context=ctx_w)
            w.current_control_label = "Worker Contextual BO + TR"
            w.run_continuous(theta_w, eval_state=s_w, eval_context=ctx_w)
            candidate_pool.append(list(theta_w))
            if w.agent.prev_best is not None:
                candidate_pool.append(list(w.agent.prev_best))

        s_a, _, _ = fac_a.scenario_monitor.get_state(fac_a.current_time)
        ctx_a = fac_a.scenario_monitor.get_context_vector(fac_a.current_time)
        local_theta = fac_a.agent.ask(state=s_a, context=ctx_a)
        candidate_pool.append(list(local_theta))

        low = fac_a.agent.bounds[0].tolist()
        high = fac_a.agent.bounds[1].tolist()

        def sample_in_bounds():
            return [low[d] + (high[d] - low[d]) * fed_rng.random() for d in range(fac_a.agent.dim)]

        for w in workers:
            if w.agent.prev_best is not None:
                base = w.agent.prev_best
                for _ in range(2):
                    jitter = [
                        min(max(base[d] + (high[d] - low[d]) * 0.05 * (fed_rng.random() - 0.5), low[d]), high[d])
                        for d in range(fac_a.agent.dim)
                    ]
                    candidate_pool.append(jitter)

        target_pool = min(18, max(10, len(candidate_pool) + 4))
        while len(candidate_pool) < target_pool:
            candidate_pool.append(sample_in_bounds())

        seen = set()
        unique = []
        for t in candidate_pool:
            key = tuple(round(float(x), 6) for x in t)
            if key in seen:
                continue
            seen.add(key)
            unique.append(list(t))
        candidate_pool = unique[:24]

        prediction_packets = []
        for w in workers:
            s_w, _, _ = w.scenario_monitor.get_state(w.current_time)
            ctx_w = w.scenario_monitor.get_context_vector(w.current_time)
            try:
                preds = w.agent.predict_candidates(candidate_pool, state=s_w, context=ctx_w)
                if preds:
                    prediction_packets.append({"factory_id": w.id, "state": s_w, "predictions": preds})
            except Exception as e:
                print(f"Worker {w.id} predict failed: {e}")

        aggs = aggregator.aggregate_predictions(prediction_packets)

        fac_a.perf_log.setdefault("fed_candidate_pool", []).append(candidate_pool)
        if aggs:
            mus = [a["mu_fed"] for a in aggs]
            sigs = [a["sigma_fed"] for a in aggs]
            scores = [a["score_fed"] for a in aggs]
            fac_a.perf_log.setdefault("fed_mu", []).append(mus)
            fac_a.perf_log.setdefault("fed_sigma", []).append(sigs)
            fac_a.perf_log.setdefault("fed_score", []).append(scores)
            best = sorted(aggs, key=lambda x: x["score_fed"], reverse=True)[0]
            selected = list(best["theta"])
        else:
            selected = list(local_theta)
            fac_a.perf_log.setdefault("fed_mu", []).append([])
            fac_a.perf_log.setdefault("fed_sigma", []).append([])
            fac_a.perf_log.setdefault("fed_score", []).append([])

        fac_a.perf_log.setdefault("fed_selected_theta", []).append(selected)
        fac_a.current_control_label = "Contextual BO + TR (Federated)"
        fac_a.run_continuous(selected, eval_state=s_a, eval_context=ctx_a)
        if (i + 1) % 10 == 0:
            print(f"  [BO-Fed-ContextTR] Iteration {i + 1}/{CFG.BO_ITERATIONS}")

    return fac_a.perf_log, fac_a.agent.acq_history


def parse_task_probs_arg(spec):
    """解析 --task-probs，例如 0.2,0.4,0.4 或 20,40,40。"""
    if spec is None or str(spec).strip() == "":
        return None
    parts = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--task-probs must be RT,Batch,AI, e.g. 0.2,0.4,0.4 or 20,40,40")
    if max(parts) > 1.0:
        parts = [x / 100.0 for x in parts]
    return _normalize_task_probs({"RT": parts[0], "Batch": parts[1], "AI": parts[2]})


def parse_task_prob_schedule_arg(spec):
    """解析 --task-prob-schedule。

    格式："0:4000:20,40,40;4000:9000:40,10,50;9000:12000:20,40,40"
    """
    if spec is None or str(spec).strip() == "":
        return None
    schedule = []
    for block in str(spec).split(";"):
        block = block.strip()
        if not block:
            continue
        parts = block.split(":")
        if len(parts) != 3:
            raise ValueError("Each task-prob-schedule block must be start:end:RT,Batch,AI")
        start, end = float(parts[0]), float(parts[1])
        probs = parse_task_probs_arg(parts[2])
        schedule.append((start, end, probs))
    return schedule


def parse_lambda_schedule_arg(spec):
    """解析 --lambda-schedule，例如 0:4000:1.0,4000:9000:2.2,9000:12000:1.2。"""
    if spec is None or str(spec).strip() == "":
        return None
    schedule = []
    for block in str(spec).split(","):
        block = block.strip()
        if not block:
            continue
        parts = block.split(":")
        if len(parts) != 3:
            raise ValueError("Each lambda schedule block must be start:end:lambda")
        schedule.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not schedule:
        return None
    return schedule


def parse_lambda_values_arg(spec):
    if spec is None or str(spec).strip() == "":
        return [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def _lambda_folder_name(lam):
    return f"lambda_{float(lam):.2f}".replace(".", "p")


def run_pressure_scan_experiments(repeat_runs=1, lambda_values=None, output_root=None, selected_keys=None, task_probs=None):
    """压力标定：扫描多个 lambda，只运行指定方法并汇总退化点。"""
    global SCENARIO_SAVE_DIR
    if lambda_values is None:
        lambda_values = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    if selected_keys is None:
        selected_keys = ["reduced4_fixed_mid", "reduced4_fixed_recommended"]
    old_schedule = list(CFG.LAMBDA_SCHEDULE)
    old_thresholds = CFG.ARRIVAL_THRESHOLDS
    old_probs = dict(CFG.TASK_TYPE_PROBS)
    old_prob_schedule = getattr(CFG, "TASK_TYPE_PROB_SCHEDULE", None)
    old_save_dir = SCENARIO_SAVE_DIR
    root = output_root or os.path.abspath("pressure_scan_outputs")
    os.makedirs(root, exist_ok=True)
    all_rows = []
    try:
        if task_probs is not None:
            CFG.TASK_TYPE_PROBS = _normalize_task_probs(task_probs)
            CFG.TASK_TYPE_PROB_SCHEDULE = None
        for idx, lam in enumerate(lambda_values, start=1):
            CFG.LAMBDA_SCHEDULE = [(0.0, float(CFG.SESSION_DURATION), float(lam))]
            CFG.ARRIVAL_THRESHOLDS = infer_arrival_thresholds(CFG.LAMBDA_SCHEDULE)
            scenario_dir = os.path.join(root, _lambda_folder_name(lam))
            os.makedirs(scenario_dir, exist_ok=True)
            with open(os.path.join(scenario_dir, "pressure_scan_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "lambda": float(lam),
                    "lambda_schedule": list(CFG.LAMBDA_SCHEDULE),
                    "task_type_probs": dict(CFG.TASK_TYPE_PROBS),
                    "bo_iterations": int(CFG.BO_ITERATIONS),
                    "bo_interval": float(CFG.BO_INTERVAL),
                    "repeat_runs": int(max(1, repeat_runs)),
                    "selected_keys": selected_keys,
                    "cloud_delay_mult": float(getattr(CFG, "CLOUD_DELAY_MULT", 1.0)),
                    "cloud_energy_mult": float(getattr(CFG, "CLOUD_ENERGY_MULT", 1.0)),
                    "cloud_speed_mult": float(getattr(CFG, "CLOUD_SPEED_MULT", 1.0)),
                }, f, ensure_ascii=False, indent=2)
            print(f"\n[Pressure {idx}/{len(lambda_values)}] lambda={lam}, output={scenario_dir}")
            group_logs = run_scenario_method_experiments(
                repeat_runs=max(1, repeat_runs),
                selected_keys=selected_keys,
                output_dir=scenario_dir,
            )
            key_path = os.path.join(scenario_dir, "key_metric_summary_核心指标统计.csv")
            if os.path.exists(key_path):
                df = pd.read_csv(key_path)
                df.insert(0, "Lambda", float(lam))
                all_rows.extend(df.to_dict("records"))
            pd.DataFrame(all_rows).to_csv(os.path.join(root, "pressure_scan_summary_all.csv"), index=False)
    finally:
        CFG.LAMBDA_SCHEDULE = old_schedule
        CFG.ARRIVAL_THRESHOLDS = old_thresholds
        CFG.TASK_TYPE_PROBS = old_probs
        CFG.TASK_TYPE_PROB_SCHEDULE = old_prob_schedule
        SCENARIO_SAVE_DIR = old_save_dir
    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(root, "pressure_scan_summary_all.csv"), index=False)
    print(f"\n=== Pressure scan finished. Summary saved to: {os.path.join(root, 'pressure_scan_summary_all.csv')} ===")
    return summary

def main():
    """主入口：跑本地 Context+TR、Round-Robin、联邦 Context+TR 三组。"""
    print("=== Scenario-aware Continuous-Window Federated BO Simulation ===")
    baseline_logs, federated_logs, rr_logs = [], [], []
    last_fed_acq_history = []
    for i in range(CFG.REPEAT_RUNS):
        seed = CFG.BASE_SEED + i
        baseline_logs.append(run_baseline_batch(seed))
        rr_logs.append(run_rr_batch(seed))
        fed_log, fed_acq_hist = run_federated_batch(seed)
        federated_logs.append(fed_log)
        last_fed_acq_history = fed_acq_hist
    
    summarize_metrics(baseline_logs, "BO-Local-ContextTR")
    summarize_metrics(rr_logs, "Round-Robin")
    summarize_metrics(federated_logs, "BO-Federated-ContextTR")
    
    base_mean = aggregate_logs(baseline_logs)
    rr_mean = aggregate_logs(rr_logs)
    fed_mean = aggregate_logs(federated_logs)
    
    save_detailed_data(base_mean, fed_mean, rr_mean)
    plot_comparison(base_mean, fed_mean, rr_mean)
    plot_convergence_metrics(base_mean, fed_mean, rr_mean)
    plot_best_so_far(base_mean, fed_mean, rr_mean)
    plot_acq_process(last_fed_acq_history)




# ===============================================================
# SAFE BO + CBO GREEDY + DUAL FEEDBACK PATCH V3
# ===============================================================
import os

_DUAL_ORIG_USE_COHORT_FEEDBACK = ConnectedFactory._use_cohort_feedback
_DUAL_ORIG_ON_TASK_FINISHED_COHORT = ConnectedFactory._on_task_finished_cohort


def _safebo_env(name, default, cast=str):
    try:
        return cast(os.environ.get(name, default))
    except Exception:
        return default


def _safebo_policy_name(group_cfg=None):
    p = None
    if group_cfg is not None:
        p = group_cfg.get("deploy_policy", None)
    if p is None:
        p = os.environ.get("SAFEBO_POLICY", "ei")
    return str(p or "ei").strip().lower()


def _is_cbo_method_key(group_key, group_cfg=None):
    key = str(group_key or "").lower()
    family = str((group_cfg or {}).get("method_family", "")).lower() if isinstance(group_cfg, dict) else ""
    label = str((group_cfg or {}).get("label", "")).lower() if isinstance(group_cfg, dict) else ""
    return ("cbo" in key) or ("cbo" in family) or ("cbo" in label)


def _deploy_policy_arg():
    val = getattr(CFG, "DEPLOY_POLICY_ARG", None)
    if val is None:
        val = os.environ.get("SAFEBO_POLICY_ARG", None)
    return str(val).strip().lower() if val is not None and str(val).strip() else None


def apply_deploy_policy_override(groups):
    """Apply CLI deploy-policy only to CBO-like methods; fixed/direct baselines stay unchanged."""
    policy = _deploy_policy_arg()
    if not policy:
        return groups
    for group_key, group_cfg in groups.items():
        if _is_cbo_method_key(group_key, group_cfg):
            group_cfg["deploy_policy"] = policy
            group_cfg["deploy_policy_source"] = "cli_override_cbo_only"
        else:
            group_cfg.setdefault("deploy_policy_source", "method_default_or_not_applicable")
    return groups


def method_deploy_policy_map(groups):
    out = {}
    for group_key, group_cfg in (groups or {}).items():
        out[str(group_key)] = {
            "deploy_policy": group_cfg.get("deploy_policy"),
            "deploy_policy_source": group_cfg.get("deploy_policy_source", "method_default"),
            "method_family": group_cfg.get("method_family"),
            "is_cbo_method": bool(_is_cbo_method_key(group_key, group_cfg)),
        }
    return out


def _argv_has_option(option_name):
    """Return True when an argparse option was explicitly supplied.

    Supports both "--x value" and "--x=value" forms. This is needed because
    argparse defaults are non-None for backwards compatibility, while method
    configs may have their own defaults that should only be overridden by an
    explicit CLI request.
    """
    opt = str(option_name)
    return any(str(arg) == opt or str(arg).startswith(opt + "=") for arg in sys.argv[1:])


def _history_mode_arg():
    if not _argv_has_option("--bo-history-mode"):
        return None
    return _cfg_history_mode()


def _recent_window_arg():
    if not _argv_has_option("--bo-recent-window"):
        return None
    return _cfg_recent_window()


def apply_history_policy_override(groups):
    """Apply CLI history settings to CBO-like methods only.

    Method configs such as cbo_lite_group() can define their own history_mode /
    recent_window defaults. A CLI --bo-history-mode or --bo-recent-window should
    override those CBO method-level defaults, while fixed/direct baselines are
    unaffected. If no CLI flag was explicitly supplied, preserve method defaults.
    """
    mode = _history_mode_arg()
    window = _recent_window_arg()
    if mode is None and window is None:
        for group_key, group_cfg in (groups or {}).items():
            group_cfg.setdefault("history_override_source", "method_default")
        return groups
    for group_key, group_cfg in (groups or {}).items():
        if _is_cbo_method_key(group_key, group_cfg):
            if mode is not None:
                group_cfg["history_mode"] = str(mode)
            if window is not None:
                group_cfg["recent_window"] = int(window)
            group_cfg["history_override_source"] = "cli_override_cbo_only"
        else:
            group_cfg.setdefault("history_override_source", "method_default_or_not_applicable")
    return groups


def _cbo_cli_option(name, attr, default=None):
    if not _argv_has_option(name):
        return None
    return getattr(CFG, attr, default)


def apply_cbo_stability_policy_override(groups):
    values = {
        "cbo_history_select_mode": _cbo_cli_option("--cbo-history-select-mode", "CBO_HISTORY_SELECT_MODE", "recent"),
        "cbo_context_k": _cbo_cli_option("--cbo-context-k", "CBO_CONTEXT_K", 50),
        "cbo_elite_k": _cbo_cli_option("--cbo-elite-k", "CBO_ELITE_K", 20),
        "cbo_diverse_k": _cbo_cli_option("--cbo-diverse-k", "CBO_DIVERSE_K", 20),
        "cbo_robust_score_mode": _cbo_cli_option("--cbo-robust-score-mode", "CBO_ROBUST_SCORE_MODE", "none"),
        "cbo_robust_std_weight": _cbo_cli_option("--cbo-robust-std-weight", "CBO_ROBUST_STD_WEIGHT", 0.5),
        "cbo_theta_merge_eps": _cbo_cli_option("--cbo-theta-merge-eps", "CBO_THETA_MERGE_EPS", 0.05),
        "cbo_context_sim_threshold": _cbo_cli_option("--cbo-context-sim-threshold", "CBO_CONTEXT_SIM_THRESHOLD", 0.0),
        "cbo_tr_mode": _cbo_cli_option("--cbo-tr-mode", "CBO_TR_MODE", "off"),
        "cbo_tr_radius_init": _cbo_cli_option("--cbo-tr-radius-init", "CBO_TR_RADIUS_INIT", getattr(CFG, "TRUST_RADIUS_INIT", 0.10)),
        "cbo_tr_radius_min": _cbo_cli_option("--cbo-tr-radius-min", "CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04)),
        "cbo_tr_radius_max": _cbo_cli_option("--cbo-tr-radius-max", "CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35)),
        "cbo_tr_grow": _cbo_cli_option("--cbo-tr-grow", "CBO_TR_GROW", getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15)),
        "cbo_tr_shrink": _cbo_cli_option("--cbo-tr-shrink", "CBO_TR_SHRINK", getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92)),
        "cbo_tr_update_mode": _cbo_cli_option("--cbo-tr-update-mode", "CBO_TR_UPDATE_MODE", "best_so_far"),
        "cbo_tr_compare_window": _cbo_cli_option("--cbo-tr-compare-window", "CBO_TR_COMPARE_WINDOW", 30),
        "cbo_tr_baseline_window": _cbo_cli_option("--cbo-tr-baseline-window", "CBO_TR_BASELINE_WINDOW", 60),
        "cbo_tr_improve_pct": _cbo_cli_option("--cbo-tr-improve-pct", "CBO_TR_IMPROVE_PCT", 0.015),
        "cbo_tr_worsen_pct": _cbo_cli_option("--cbo-tr-worsen-pct", "CBO_TR_WORSEN_PCT", 0.03),
        "cbo_tr_deadband_pct": _cbo_cli_option("--cbo-tr-deadband-pct", "CBO_TR_DEADBAND_PCT", 0.01),
        "cbo_tr_update_patience": _cbo_cli_option("--cbo-tr-update-patience", "CBO_TR_UPDATE_PATIENCE", 2),
        "cbo_tr_anchor_mode": _cbo_cli_option("--cbo-tr-anchor-mode", "CBO_TR_ANCHOR_MODE", "posterior_mean"),
        "cbo_robust_incumbent_mode": _cbo_cli_option("--cbo-robust-incumbent-mode", "CBO_ROBUST_INCUMBENT_MODE", "off"),
        "cbo_macro_gate_mode": _cbo_cli_option("--cbo-macro-gate-mode", "CBO_MACRO_GATE_MODE", "off"),
        "cbo_macro_k": _cbo_cli_option("--cbo-macro-k", "CBO_MACRO_K", 100),
        "cbo_macro_total_scale": _cbo_cli_option("--cbo-macro-total-scale", "CBO_MACRO_TOTAL_SCALE", "auto"),
        "cbo_macro_lengthscale_total": _cbo_cli_option("--cbo-macro-lengthscale-total", "CBO_MACRO_LENGTHSCALE_TOTAL", 1.0),
        "cbo_macro_lengthscale_rt": _cbo_cli_option("--cbo-macro-lengthscale-rt", "CBO_MACRO_LENGTHSCALE_RT", 0.15),
        "cbo_macro_lengthscale_batch": _cbo_cli_option("--cbo-macro-lengthscale-batch", "CBO_MACRO_LENGTHSCALE_BATCH", 0.15),
        "cbo_macro_alpha": _cbo_cli_option("--cbo-macro-alpha", "CBO_MACRO_ALPHA", 1.0),
        "cbo_dump_candidates": _cbo_cli_option("--cbo-dump-candidates", "CBO_DUMP_CANDIDATES", False),
        "cbo_dump_candidates_every": _cbo_cli_option("--cbo-dump-candidates-every", "CBO_DUMP_CANDIDATES_EVERY", 20),
        "cbo_dump_candidates_topn": _cbo_cli_option("--cbo-dump-candidates-topn", "CBO_DUMP_CANDIDATES_TOPN", 30),
        "cbo_select_mode": _cbo_cli_option("--cbo-select-mode", "CBO_SELECT_MODE", "greedy"),
        "cbo_topk": _cbo_cli_option("--cbo-topk", "CBO_TOPK", 5),
        "cbo_select_temperature": _cbo_cli_option("--cbo-select-temperature", "CBO_SELECT_TEMPERATURE", 0.20),
        "cbo_epsilon": _cbo_cli_option("--cbo-epsilon", "CBO_EPSILON", 0.10),
        "cbo_acq_beta": _cbo_cli_option("--cbo-acq-beta", "CBO_ACQ_BETA", 3.0),
        "cbo_acq_beta_mode": _cbo_cli_option("--cbo-acq-beta-mode", "CBO_ACQ_BETA_MODE", "fixed"),
        "cbo_beta_min": _cbo_cli_option("--cbo-beta-min", "CBO_BETA_MIN", 0.1),
        "cbo_beta_max": _cbo_cli_option("--cbo-beta-max", "CBO_BETA_MAX", 2.0),
        "cbo_radius_beta_power": _cbo_cli_option("--cbo-radius-beta-power", "CBO_RADIUS_BETA_POWER", 1.0),
        "cbo_radius_stable_rebound_pct": _cbo_cli_option("--cbo-radius-stable-rebound-pct", "CBO_RADIUS_STABLE_REBOUND_PCT", 0.02),
        "cbo_radius_unstable_rebound_pct": _cbo_cli_option("--cbo-radius-unstable-rebound-pct", "CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04),
        "cbo_radius_surprise_boost_threshold": _cbo_cli_option("--cbo-radius-surprise-boost-threshold", "CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0),
        "cbo_radius_beta_boost": _cbo_cli_option("--cbo-radius-beta-boost", "CBO_RADIUS_BETA_BOOST", 1.5),
        "cbo_radius_beta_cap": _cbo_cli_option("--cbo-radius-beta-cap", "CBO_RADIUS_BETA_CAP", 3.0),
        "cbo_service_guard_mode": _cbo_cli_option("--cbo-service-guard-mode", "CBO_SERVICE_GUARD_MODE", "off"),
        "cbo_service_guard_delay_pct": _cbo_cli_option("--cbo-service-guard-delay-pct", "CBO_SERVICE_GUARD_DELAY_PCT", 0.03),
        "cbo_service_guard_backlog_pct": _cbo_cli_option("--cbo-service-guard-backlog-pct", "CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03),
        "cbo_surprise_window": _cbo_cli_option("--cbo-surprise-window", "CBO_SURPRISE_WINDOW", 10),
        "cbo_surprise_z_threshold": _cbo_cli_option("--cbo-surprise-z-threshold", "CBO_SURPRISE_Z_THRESHOLD", 2.0),
        "cbo_surprise_cost_gap_pct": _cbo_cli_option("--cbo-surprise-cost-gap-pct", "CBO_SURPRISE_COST_GAP_PCT", 0.03),
        "cbo_sigma_floor": _cbo_cli_option("--cbo-sigma-floor", "CBO_SIGMA_FLOOR", 1e-6),
        "cbo_radius_reset": _cbo_cli_option("--cbo-radius-reset", "CBO_RADIUS_RESET", 0.12),
        "cbo_radius_min_stuck_rounds": _cbo_cli_option("--cbo-radius-min-stuck-rounds", "CBO_RADIUS_MIN_STUCK_ROUNDS", 10),
        "cbo_rebound_window": _cbo_cli_option("--cbo-rebound-window", "CBO_REBOUND_WINDOW", 20),
        "cbo_rebound_threshold_pct": _cbo_cli_option("--cbo-rebound-threshold-pct", "CBO_REBOUND_THRESHOLD_PCT", 0.03),
        "cbo_selection_cooldown": _cbo_cli_option("--cbo-selection-cooldown", "CBO_SELECTION_COOLDOWN", 5),
        "cbo_condition_anchor_switch": _cbo_cli_option("--cbo-condition-anchor-switch", "CBO_CONDITION_ANCHOR_SWITCH", "context_best"),
    }
    any_explicit = any(v is not None for v in values.values())
    for group_key, group_cfg in (groups or {}).items():
        if _is_cbo_method_key(group_key, group_cfg):
            if any_explicit:
                for k, v in values.items():
                    if v is not None:
                        group_cfg[k] = v
                group_cfg["cbo_stability_override_source"] = "cli_override_cbo_only"
            else:
                group_cfg.setdefault("cbo_history_select_mode", "recent")
                group_cfg.setdefault("cbo_robust_score_mode", "none")
                group_cfg.setdefault("cbo_tr_mode", "off")
                group_cfg.setdefault("cbo_robust_incumbent_mode", "off")
                group_cfg.setdefault("cbo_macro_gate_mode", "off")
                group_cfg.setdefault("cbo_dump_candidates", False)
                group_cfg.setdefault("cbo_stability_override_source", "method_default")
        else:
            group_cfg.setdefault("cbo_stability_override_source", "not_applicable")
    return groups


def method_history_policy_map(groups):
    out = {}
    for group_key, group_cfg in (groups or {}).items():
        out[str(group_key)] = {
            "history_mode": group_cfg.get("history_mode"),
            "recent_window": group_cfg.get("recent_window"),
            "history_override_source": group_cfg.get("history_override_source", "method_default"),
            "method_family": group_cfg.get("method_family"),
            "is_cbo_method": bool(_is_cbo_method_key(group_key, group_cfg)),
            "history_select_mode": group_cfg.get("cbo_history_select_mode", _cfg_cbo_history_select_mode("recent")),
            "effective_history_mode": group_cfg.get("history_mode", _cfg_history_mode("all")),
            "effective_recent_window": group_cfg.get("recent_window", _cfg_recent_window()),
            "context_k": group_cfg.get("cbo_context_k", _cfg_cbo_int("CBO_CONTEXT_K", 50)),
            "elite_k": group_cfg.get("cbo_elite_k", _cfg_cbo_int("CBO_ELITE_K", 20)),
            "diverse_k": group_cfg.get("cbo_diverse_k", _cfg_cbo_int("CBO_DIVERSE_K", 20)),
            "robust_score_mode": group_cfg.get("cbo_robust_score_mode", _cfg_cbo_str("CBO_ROBUST_SCORE_MODE", "none")),
            "scheduler_tradeoff_mode": str(getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy")),
            "scheduler_score_norm_mode": str(getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")),
            "scheduler_tradeoff_alpha": float(getattr(CFG, "SCHEDULER_TRADEOFF_ALPHA", 0.85)),
            "scheduler_alpha_min": float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60)),
            "scheduler_alpha_max": float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97)),
            "tr_mode": group_cfg.get("cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")),
            "tr_anchor_mode": group_cfg.get("cbo_tr_anchor_mode", _cfg_cbo_str("CBO_TR_ANCHOR_MODE", "posterior_mean")),
            "tr_update_mode": group_cfg.get("cbo_tr_update_mode", _cfg_cbo_str("CBO_TR_UPDATE_MODE", "best_so_far")),
            "tr_compare_window": group_cfg.get("cbo_tr_compare_window", _cfg_cbo_int("CBO_TR_COMPARE_WINDOW", 30)),
            "tr_baseline_window": group_cfg.get("cbo_tr_baseline_window", _cfg_cbo_int("CBO_TR_BASELINE_WINDOW", 60)),
            "tr_improve_pct": group_cfg.get("cbo_tr_improve_pct", _cfg_cbo_float("CBO_TR_IMPROVE_PCT", 0.015)),
            "tr_worsen_pct": group_cfg.get("cbo_tr_worsen_pct", _cfg_cbo_float("CBO_TR_WORSEN_PCT", 0.03)),
            "tr_deadband_pct": group_cfg.get("cbo_tr_deadband_pct", _cfg_cbo_float("CBO_TR_DEADBAND_PCT", 0.01)),
            "tr_update_patience": group_cfg.get("cbo_tr_update_patience", _cfg_cbo_int("CBO_TR_UPDATE_PATIENCE", 2)),
            "robust_incumbent_mode": group_cfg.get("cbo_robust_incumbent_mode", _cfg_cbo_str("CBO_ROBUST_INCUMBENT_MODE", "off")),
            "macro_gate_mode": group_cfg.get("cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")),
            "macro_k": group_cfg.get("cbo_macro_k", _cfg_cbo_int("CBO_MACRO_K", 100)),
            "macro_total_scale": group_cfg.get("cbo_macro_total_scale", getattr(CFG, "CBO_MACRO_TOTAL_SCALE", "auto")),
            "macro_lengthscale_total": group_cfg.get("cbo_macro_lengthscale_total", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_TOTAL", 1.0)),
            "macro_lengthscale_rt": group_cfg.get("cbo_macro_lengthscale_rt", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_RT", 0.15)),
            "macro_lengthscale_batch": group_cfg.get("cbo_macro_lengthscale_batch", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_BATCH", 0.15)),
            "macro_alpha": group_cfg.get("cbo_macro_alpha", _cfg_cbo_float("CBO_MACRO_ALPHA", 1.0)),
            "dump_candidates": group_cfg.get("cbo_dump_candidates", bool(getattr(CFG, "CBO_DUMP_CANDIDATES", False))),
            "dump_candidates_every": group_cfg.get("cbo_dump_candidates_every", _cfg_cbo_int("CBO_DUMP_CANDIDATES_EVERY", 20)),
            "dump_candidates_topn": group_cfg.get("cbo_dump_candidates_topn", _cfg_cbo_int("CBO_DUMP_CANDIDATES_TOPN", 30)),
            "select_mode": group_cfg.get("cbo_select_mode", _cfg_cbo_str("CBO_SELECT_MODE", "greedy")),
            "topk": group_cfg.get("cbo_topk", _cfg_cbo_int("CBO_TOPK", 5)),
            "select_temperature": group_cfg.get("cbo_select_temperature", _cfg_cbo_float("CBO_SELECT_TEMPERATURE", 0.20)),
            "epsilon": group_cfg.get("cbo_epsilon", _cfg_cbo_float("CBO_EPSILON", 0.10)),
            "acq_beta": group_cfg.get("cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0)),
            "acq_beta_mode": group_cfg.get("cbo_acq_beta_mode", _cfg_cbo_str("CBO_ACQ_BETA_MODE", "fixed")),
            "beta_min": group_cfg.get("cbo_beta_min", _cfg_cbo_float("CBO_BETA_MIN", 0.1)),
            "beta_max": group_cfg.get("cbo_beta_max", _cfg_cbo_float("CBO_BETA_MAX", 2.0)),
            "radius_beta_power": group_cfg.get("cbo_radius_beta_power", _cfg_cbo_float("CBO_RADIUS_BETA_POWER", 1.0)),
            "radius_stable_rebound_pct": group_cfg.get("cbo_radius_stable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_STABLE_REBOUND_PCT", 0.02)),
            "radius_unstable_rebound_pct": group_cfg.get("cbo_radius_unstable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04)),
            "radius_surprise_boost_threshold": group_cfg.get("cbo_radius_surprise_boost_threshold", _cfg_cbo_float("CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0)),
            "radius_beta_boost": group_cfg.get("cbo_radius_beta_boost", _cfg_cbo_float("CBO_RADIUS_BETA_BOOST", 1.5)),
            "radius_beta_cap": group_cfg.get("cbo_radius_beta_cap", _cfg_cbo_float("CBO_RADIUS_BETA_CAP", 3.0)),
            "service_guard_mode": group_cfg.get("cbo_service_guard_mode", _cfg_cbo_str("CBO_SERVICE_GUARD_MODE", "off")),
            "service_guard_delay_pct": group_cfg.get("cbo_service_guard_delay_pct", _cfg_cbo_float("CBO_SERVICE_GUARD_DELAY_PCT", 0.03)),
            "service_guard_backlog_pct": group_cfg.get("cbo_service_guard_backlog_pct", _cfg_cbo_float("CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03)),
            "surprise_window": group_cfg.get("cbo_surprise_window", _cfg_cbo_int("CBO_SURPRISE_WINDOW", 10)),
            "surprise_z_threshold": group_cfg.get("cbo_surprise_z_threshold", _cfg_cbo_float("CBO_SURPRISE_Z_THRESHOLD", 2.0)),
            "surprise_cost_gap_pct": group_cfg.get("cbo_surprise_cost_gap_pct", _cfg_cbo_float("CBO_SURPRISE_COST_GAP_PCT", 0.03)),
            "radius_reset": group_cfg.get("cbo_radius_reset", _cfg_cbo_float("CBO_RADIUS_RESET", 0.12)),
            "radius_min_stuck_rounds": group_cfg.get("cbo_radius_min_stuck_rounds", _cfg_cbo_int("CBO_RADIUS_MIN_STUCK_ROUNDS", 10)),
            "rebound_window": group_cfg.get("cbo_rebound_window", _cfg_cbo_int("CBO_REBOUND_WINDOW", 20)),
            "rebound_threshold_pct": group_cfg.get("cbo_rebound_threshold_pct", _cfg_cbo_float("CBO_REBOUND_THRESHOLD_PCT", 0.03)),
            "selection_cooldown": group_cfg.get("cbo_selection_cooldown", _cfg_cbo_int("CBO_SELECTION_COOLDOWN", 5)),
            "condition_anchor_switch": group_cfg.get("cbo_condition_anchor_switch", _cfg_cbo_str("CBO_CONDITION_ANCHOR_SWITCH", "context_best")),
        }
    return out


def _dual_feedback_mode():
    return str(getattr(CFG, "FEEDBACK_MODE", "window")).strip().lower()


def _dual_is_enabled():
    return _dual_feedback_mode() in {"dual", "dual_feedback", "window_cohort", "window_refine"}


def _dual_patch_use_cohort_feedback(self):
    mode = _dual_feedback_mode()
    if mode in {"dual", "dual_feedback", "window_cohort", "window_refine"}:
        return True
    return _DUAL_ORIG_USE_COHORT_FEEDBACK(self)


ConnectedFactory._use_cohort_feedback = _dual_patch_use_cohort_feedback


def _dual_agent_replace_sample(agent, sample_id, refined_cost, refined_meta=None):
    """Replace the provisional window feedback of one BO sample by delayed refined feedback.

    The agent stores y=-cost in local_recent. In dual mode we first append a provisional
    window sample, then later replace that sample's y with the refined cohort/class score.
    The previous window cost is saved for diagnosis/export.
    """
    if agent is None or sample_id is None:
        return False
    refined_y = -float(refined_cost)
    refined_meta = dict(refined_meta or {})
    for rec in reversed(list(getattr(agent, "local_recent", []))):
        if isinstance(rec, dict) and rec.get("sample_id") == sample_id:
            prev_cost = rec.get("feedback_cost", None)
            try:
                prev_cost_float = float(prev_cost) if prev_cost is not None else None
            except Exception:
                prev_cost_float = None
            rec["window_provisional_cost"] = prev_cost_float
            rec["y"] = refined_y
            rec["feedback_cost"] = float(refined_cost)
            rec["feedback_source"] = "dual_refined"
            rec["refined_meta"] = refined_meta
            if prev_cost_float is not None:
                rec["refined_delta_vs_window"] = float(refined_cost) - float(prev_cost_float)
                rec["refined_ratio_vs_window"] = float(refined_cost) / max(1e-12, abs(float(prev_cost_float)))
            else:
                rec["refined_delta_vs_window"] = None
                rec["refined_ratio_vs_window"] = None
            try:
                agent._dual_last_replace_meta = {
                    "sample_id": sample_id,
                    "window_provisional_cost": prev_cost_float,
                    "refined_cost": float(refined_cost),
                    "refined_delta_vs_window": rec.get("refined_delta_vs_window"),
                    "refined_ratio_vs_window": rec.get("refined_ratio_vs_window"),
                }
            except Exception:
                pass
            # Keep incumbent consistent after replacement.
            try:
                best = None
                for rr in getattr(agent, "local_recent", []):
                    if isinstance(rr, dict):
                        if best is None or float(rr.get("y", -1e99)) > float(best.get("y", -1e99)):
                            best = rr
                if best is not None:
                    agent.prev_best_value = float(best.get("y"))
                    agent.prev_best = list(best.get("theta"))
            except Exception:
                pass
            return True
    return False


def _dual_update_class_attrs(cohort, task, delay):
    """Add per-task-type delay/lateness/violation sums to cohort without changing dataclass."""
    ttype = str(getattr(task, "task_type", "Batch"))
    prefix = "rt" if ttype == "RT" else ("ai" if ttype == "AI" else "batch")
    setattr(cohort, f"{prefix}_delay_sum", float(getattr(cohort, f"{prefix}_delay_sum", 0.0)) + float(delay))
    late = max(0.0, float(getattr(task, "finish_time", 0.0)) - float(getattr(task, "deadline", 0.0)))
    setattr(cohort, f"{prefix}_lateness_sum", float(getattr(cohort, f"{prefix}_lateness_sum", 0.0)) + late)
    if float(getattr(task, "finish_time", 0.0)) > float(getattr(task, "deadline", 0.0)):
        setattr(cohort, f"{prefix}_vio_count", int(getattr(cohort, f"{prefix}_vio_count", 0)) + 1)


def _dual_on_task_finished_cohort(self, task, delay):
    cid = getattr(task, "cohort_id", None)
    if cid is not None and cid in getattr(self, "cohorts", {}):
        cohort = self.cohorts.get(cid)
        if cohort is not None and not getattr(cohort, "finalized", False):
            _dual_update_class_attrs(cohort, task, delay)
    return _DUAL_ORIG_ON_TASK_FINISHED_COHORT(self, task, delay)


ConnectedFactory._on_task_finished_cohort = _dual_on_task_finished_cohort


def _dual_ref_probs_from_cfg():
    probs = getattr(CFG, "DUAL_FEEDBACK_REF_PROBS", None)
    if probs is None:
        probs = getattr(CFG, "TASK_TYPE_PROBS", {"RT": 1/3, "Batch": 1/3, "AI": 1/3})
    probs = _normalize_task_probs(probs)
    return {"RT": float(probs.get("RT", 0.0)), "Batch": float(probs.get("Batch", 0.0)), "AI": float(probs.get("AI", 0.0))}


def _dual_class_refined_cost(cohort, row, aggregation=None):
    """Class-normalized refined score.

    The BO/CBO agent still needs one scalar feedback value, but this scalar can be
    composed from separated RT/Batch/AI diagnostics. This function therefore always
    exports per-class details and lets aggregation choose how to combine them.

    aggregation options:
    - class / class_weighted: use configured/nominal task probabilities as weights.
    - class_equal: give RT/Batch/AI equal weights.
    - class_actual: use this cohort's realized arrival ratios; useful for diagnosis, not
      the default, because it reintroduces task-mix randomness into the target.
    - class_worst: use the worst class cost; conservative fairness/safety feedback.
    """
    source = str(aggregation or os.environ.get("DUAL_REFINED_SOURCE", getattr(CFG, "DUAL_REFINED_SOURCE", "class"))).strip().lower()
    unfinished_weight = float(getattr(CFG, "DUAL_CLASS_UNFINISHED_WEIGHT", getattr(CFG, "COHORT_UNFINISHED_PENALTY", 1000.0)))
    include_energy = bool(int(os.environ.get("DUAL_INCLUDE_ENERGY", "1")))
    metric_mode = str(os.environ.get("DUAL_CLASS_METRIC", getattr(CFG, "DUAL_CLASS_METRIC", "completed"))).strip().lower()
    if metric_mode not in {"completed", "effective", "censored"}:
        metric_mode = "completed"

    ref_probs = _dual_ref_probs_from_cfg()
    total_arrivals = max(1, int(getattr(cohort, "rt_arrivals", 0)) + int(getattr(cohort, "batch_arrivals", 0)) + int(getattr(cohort, "ai_arrivals", 0)))
    actual_probs = {
        "RT": int(getattr(cohort, "rt_arrivals", 0)) / total_arrivals,
        "Batch": int(getattr(cohort, "batch_arrivals", 0)) / total_arrivals,
        "AI": int(getattr(cohort, "ai_arrivals", 0)) / total_arrivals,
    }
    equal_probs = {"RT": 1.0 / 3.0, "Batch": 1.0 / 3.0, "AI": 1.0 / 3.0}

    if source in {"class_equal", "equal", "separate_equal"}:
        weights = equal_probs
        aggregation_name = "class_equal"
    elif source in {"class_actual", "actual", "arrival_weighted"}:
        weights = actual_probs
        aggregation_name = "class_actual"
    elif source in {"class_worst", "worst", "max"}:
        weights = None
        aggregation_name = "class_worst"
    else:
        weights = ref_probs
        aggregation_name = "class_weighted"

    class_costs = {}
    class_total = 0.0
    detail = {
        "class_aggregation": aggregation_name,
        "ref_probs": ref_probs,
        "actual_probs": actual_probs,
        "equal_probs": equal_probs,
    }

    for name, prefix in [("RT", "rt"), ("Batch", "batch"), ("AI", "ai")]:
        arrivals = int(getattr(cohort, f"{prefix}_arrivals", 0))
        completed = int(getattr(cohort, f"{prefix}_completed", 0))
        delay_sum = float(getattr(cohort, f"{prefix}_delay_sum", 0.0))
        late_sum = float(getattr(cohort, f"{prefix}_lateness_sum", 0.0))
        vio_count = int(getattr(cohort, f"{prefix}_vio_count", 0))

        if arrivals <= 0:
            avg_delay_completed = 0.0
            avg_late_completed = 0.0
            vio_rate_completed = 0.0
            completion_ratio = None
            avg_delay = 0.0
            avg_late = 0.0
            vio_rate = 0.0
            class_cost = None
        else:
            avg_delay_completed = delay_sum / max(1, completed) if completed > 0 else 0.0
            avg_late_completed = late_sum / max(1, completed) if completed > 0 else 0.0
            vio_rate_completed = vio_count / max(1, completed) if completed > 0 else 0.0
            completion_ratio = completed / max(1, arrivals)

            if metric_mode in {"effective", "censored"}:
                avg_delay = float(row.get(f"{prefix}_censored_avg_delay", avg_delay_completed))
                avg_late = float(row.get(f"{prefix}_effective_avg_lateness", avg_late_completed))
                vio_rate = float(row.get(f"{prefix}_effective_vio_rate", vio_rate_completed))
            else:
                avg_delay = avg_delay_completed
                avg_late = avg_late_completed
                vio_rate = vio_rate_completed

            class_cost = (
                float(CFG.ALPHA_LATENCY) * avg_delay
                + float(CFG.SLA_PENALTY_WEIGHT) * vio_rate
                + float(CFG.LATE_PENALTY_WEIGHT) * avg_late
                + unfinished_weight * (1.0 - completion_ratio)
            )
            class_costs[name] = float(class_cost)
            if weights is not None:
                class_total += float(weights.get(name, 0.0)) * float(class_cost)

        detail[f"{prefix}_class_cost"] = None if class_cost is None else float(class_cost)
        detail[f"{prefix}_avg_delay_completed"] = float(avg_delay_completed)
        detail[f"{prefix}_avg_lateness_completed"] = float(avg_late_completed)
        detail[f"{prefix}_vio_rate_completed"] = float(vio_rate_completed)
        detail[f"{prefix}_avg_delay_used"] = float(avg_delay)
        detail[f"{prefix}_avg_lateness_used"] = float(avg_late)
        detail[f"{prefix}_vio_rate_used"] = float(vio_rate)
        detail[f"{prefix}_metric_mode"] = metric_mode
        detail[f"{prefix}_completion_ratio"] = None if completion_ratio is None else float(completion_ratio)
        detail[f"{prefix}_weight_used"] = None if weights is None else float(weights.get(name, 0.0))

    if aggregation_name == "class_worst":
        class_total = max(class_costs.values()) if class_costs else 0.0

    energy_term = float(row.get("avg_energy_est", 0.0)) if include_energy else 0.0
    pending_term = float(getattr(CFG, "COHORT_PENDING_AREA_WEIGHT", 5.0)) * float(row.get("pending_area_per_task", 0.0))
    refined = energy_term + float(class_total) + pending_term
    detail["metric_mode"] = metric_mode
    detail["energy_term"] = float(energy_term)
    detail["pending_term"] = float(pending_term)
    detail["class_weighted_term"] = float(class_total)
    return float(refined), detail


def _dual_choose_refined_cost(cohort, row):
    source = str(os.environ.get("DUAL_REFINED_SOURCE", getattr(CFG, "DUAL_REFINED_SOURCE", "class"))).strip().lower()
    if source in {"cohort", "cohort_cost", "raw_cohort"}:
        return float(row["cohort_cost"]), {"refined_source": "cohort_cost", "class_aggregation": "none"}
    cost, detail = _dual_class_refined_cost(cohort, row, aggregation=source)
    detail["refined_source"] = "class_cost"
    return float(cost), detail


def _dual_finalize_ready_cohorts(self, now, force=False, reason="all_completed"):
    """Dual-aware cohort finalization.

    Normal cohort_complete mode keeps old behavior: cohort directly tells BO.
    Dual mode: finalized cohort replaces its matching provisional window sample when possible.
    """
    if not self._use_cohort_feedback():
        return []
    finalized_rows = []
    dual = _dual_is_enabled()
    pending = getattr(self, "dual_pending_refinements", None)
    if pending is None:
        self.dual_pending_refinements = {}
        pending = self.dual_pending_refinements
    for cohort in list(self.cohorts.values()):
        if cohort.finalized or cohort.total_tasks <= 0:
            continue
        ready = (cohort.completed_tasks >= cohort.total_tasks)
        if not (ready or force):
            continue
        final_reason = reason if ready else str(reason or "forced")
        row = self._cohort_metrics(cohort, now, reason=final_reason)
        refined_cost, refined_meta = _dual_choose_refined_cost(cohort, row) if dual else (float(row["cohort_cost"]), {"refined_source": "cohort_cost"})
        row["dual_refined_cost"] = float(refined_cost)
        row["dual_refined_source"] = str(refined_meta.get("refined_source", "cohort_cost"))
        for kk, vv in refined_meta.items():
            if kk != "ref_probs":
                row[f"dual_{kk}"] = vv
        row["dual_ref_probs"] = refined_meta.get("ref_probs")
        cohort.finalized = True
        cohort.finalize_time = float(now)
        cohort.finalize_reason = final_reason
        cohort.feedback_cost = float(refined_cost if dual else row["cohort_cost"])
        cohort.confidence = float(row["confidence"])
        self.cohort_finalized_total += 1
        self.cohort_feedback_rows.append(row)
        finalized_rows.append(row)

        if self.scheduler_type == "Boltzmann" and self.agent is not None:
            if dual:
                sid = getattr(cohort, "sample_id", None)
                if sid is None:
                    sid = f"w{int(cohort.window_index)}_c{int(cohort.cohort_id)}"
                    cohort.sample_id = sid
                ok = _dual_agent_replace_sample(self.agent, sid, refined_cost, refined_meta={**row, "sample_id": sid})
                replace_meta = getattr(self.agent, "_dual_last_replace_meta", {}) if self.agent is not None else {}
                row["sample_id"] = sid
                row["dual_replace_success"] = bool(ok)
                row["dual_window_provisional_cost"] = replace_meta.get("window_provisional_cost") if ok else None
                row["dual_refined_delta_vs_window"] = replace_meta.get("refined_delta_vs_window") if ok else None
                row["dual_refined_ratio_vs_window"] = replace_meta.get("refined_ratio_vs_window") if ok else None
                if not ok:
                    pending[sid] = {"cost": float(refined_cost), "meta": {**row, "sample_id": sid}}
                self.scheduler.update_beta(refined_cost)
            else:
                state_arg = cohort.state if getattr(self.agent, "use_state_partition", False) else None
                context_arg = cohort.context if getattr(self.agent, "use_context", False) else None
                self.agent.tell(cohort.theta_control, row["cohort_cost"], state=state_arg, context=context_arg)
                self.scheduler.update_beta(row["cohort_cost"])
    return finalized_rows


ConnectedFactory._finalize_ready_cohorts = _dual_finalize_ready_cohorts


def _dual_apply_pending_refinements(fac, agent):
    pending = getattr(fac, "dual_pending_refinements", {}) or {}
    if not pending:
        return 0
    applied = 0
    for sid in list(pending.keys()):
        item = pending.get(sid) or {}
        if _dual_agent_replace_sample(agent, sid, item.get("cost"), refined_meta=item.get("meta", {})):
            applied += 1
            pending.pop(sid, None)
    return applied


def _safebo_dedup_clip(candidates, low, high, dim):
    out = []
    seen = set()
    for c in candidates:
        if c is None:
            continue
        cc = list(c)
        if len(cc) < dim:
            cc = cc + [cc[-1] if cc else 1.0] * (dim - len(cc))
        cc = [float(min(max(cc[d], low[d]), high[d])) for d in range(dim)]
        key = tuple(round(v, 6) for v in cc)
        if key in seen:
            continue
        seen.add(key)
        out.append(cc)
    return out


def _safebo_dedup_clip_with_sources(candidates, sources, low, high, dim, limit=None):
    out = []
    out_sources = []
    seen = set()
    for idx, c in enumerate(candidates):
        if c is None:
            continue
        cc = list(c)
        if len(cc) < dim:
            cc = cc + [cc[-1] if cc else 1.0] * (dim - len(cc))
        cc = [float(min(max(cc[d], low[d]), high[d])) for d in range(dim)]
        key = tuple(round(v, 6) for v in cc)
        if key in seen:
            continue
        seen.add(key)
        out.append(cc)
        out_sources.append(str(sources[idx] if idx < len(sources) else "unknown"))
        if limit is not None and len(out) >= int(limit):
            break
    return out, out_sources


def _safebo_candidate_pool(agent, state=None, context=None, n_candidates=None):
    n_candidates = int(n_candidates if n_candidates is not None else _safebo_env("SAFEBO_CANDIDATES", 160, int))
    radius = float(_safebo_env("SAFEBO_TR_RADIUS", 0.12, float))
    dim = int(agent.dim)
    low = agent.bounds[0].tolist()
    high = agent.bounds[1].tolist()
    candidates = []
    sources = []

    for p in getattr(agent, "anchor_points", []) or []:
        try:
            candidates.append(agent._normalize_theta(p))
        except Exception:
            candidates.append(list(p))
        sources.append("anchor_point")

    if getattr(agent, "prev_best", None) is not None:
        candidates.append(list(agent.prev_best))
        sources.append("prev_best")
        center = list(agent.prev_best)
        for _ in range(max(8, n_candidates // 4)):
            cand = []
            for d in range(dim):
                span = (high[d] - low[d]) * radius
                cand.append(center[d] + span * (2.0 * agent.py_rng.random() - 1.0))
            candidates.append(cand)
            sources.append("prev_best_tr")

    try:
        if getattr(agent, "use_context", False):
            _, _, records = agent._training_data(state=state)
            pivot = agent._select_pivot_theta(context, records)
            if pivot is not None:
                candidates.append(list(pivot))
                sources.append("context_pivot")
                for _ in range(max(8, n_candidates // 4)):
                    cand = []
                    for d in range(dim):
                        span = (high[d] - low[d]) * radius
                        cand.append(pivot[d] + span * (2.0 * agent.py_rng.random() - 1.0))
                    candidates.append(cand)
                    sources.append("context_pivot_tr")
            tr_mode = str(getattr(agent, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) or "off").lower()
            if tr_mode != "off":
                anchor, anchor_debug = _cbo_resolve_actual_tr_anchor(agent, context, records)
                anchor_mode = str(anchor_debug.get("effective_tr_anchor_mode", getattr(agent, "cbo_tr_anchor_mode", "posterior_mean")) or "posterior_mean")
                anchor_source = str(anchor_debug.get("actual_tr_anchor_source", "no_anchor"))
                if anchor is not None:
                    candidates.append(list(anchor))
                    sources.append("actual_tr_anchor")
                    tr_added = max(8, n_candidates // 3)
                    for _ in range(tr_added):
                        cand = []
                        for d in range(dim):
                            span = (high[d] - low[d]) * float(getattr(agent, "trust_radius", radius))
                            cand.append(anchor[d] + span * (2.0 * agent.py_rng.random() - 1.0))
                        candidates.append(cand)
                        sources.append("trust_region")
                    debug = dict(getattr(agent, "last_debug_info", {}) or {})
                    debug.update({
                        "cbo_tr_mode": tr_mode,
                        "cbo_tr_anchor_mode": anchor_mode,
                        "cbo_tr_radius": float(getattr(agent, "trust_radius", radius)),
                        "cbo_tr_anchor_theta": list(anchor),
                        "cbo_tr_anchor_source": str(anchor_source),
                        "cbo_tr_candidate_count": int(tr_added),
                        "cbo_global_candidate_count": int(n_candidates),
                        "cbo_tr_update_reason": getattr(agent, "cbo_tr_update_reason", "safe_candidate_pool"),
                        "cbo_tr_success_count": int(getattr(agent, "cbo_tr_success_count", 0)),
                        "cbo_tr_failure_count": int(getattr(agent, "cbo_tr_failure_count", 0)),
                    })
                    debug.update(anchor_debug)
                    agent.last_debug_info = debug
                else:
                    debug = dict(getattr(agent, "last_debug_info", {}) or {})
                    debug.update({
                        "cbo_tr_mode": tr_mode,
                        "cbo_tr_anchor_mode": anchor_mode,
                        "cbo_tr_radius": float(getattr(agent, "trust_radius", radius)),
                        "cbo_tr_anchor_theta": None,
                        "cbo_tr_anchor_source": str(anchor_source),
                        "cbo_tr_candidate_count": 0,
                        "cbo_global_candidate_count": int(n_candidates),
                    })
                    debug.update(anchor_debug)
                    agent.last_debug_info = debug
    except Exception:
        pass

    while len(candidates) < n_candidates:
        candidates.append(agent._sample_in_bounds(low, high))
        sources.append("global_random")

    candidates, sources = _safebo_dedup_clip_with_sources(candidates, sources, low, high, dim, limit=max(1, n_candidates))
    try:
        agent._last_candidate_sources = list(sources)
    except Exception:
        pass
    return candidates




def _cbo_numpy_argmax_safe(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return 0
    return int(np.nanargmax(arr))


def _cbo_force_exploration_active(agent):
    try:
        return int(getattr(agent, "cbo_force_explore_countdown", 0)) > 0
    except Exception:
        return False


def _cbo_select_index_from_scores(agent, mu, sigma, score, default_reason="greedy_posterior_mean"):
    """Select a candidate index from posterior scores.

    Old behavior is preserved when cbo_select_mode=greedy and no residual/condition trigger is active.
    When a trigger is active, topK/epsilon/randomized modes can avoid always choosing rank 1.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    score = np.asarray(score, dtype=float)
    n = int(len(score))
    if n <= 0:
        return 0, "empty_candidate_fallback"
    beta_mode = str(getattr(agent, "cbo_acq_beta_mode", _cfg_cbo_str("CBO_ACQ_BETA_MODE", "fixed")) or "fixed").strip().lower()
    # Backward compatibility: fixed beta keeps the old SAFEBO=greedy_mean
    # posterior-mean choice. Adaptive beta modes use the score directly.
    if str(default_reason) == "greedy_posterior_mean" and beta_mode == "fixed":
        greedy_idx = _cbo_numpy_argmax_safe(mu)
    else:
        greedy_idx = _cbo_numpy_argmax_safe(score)
    select_mode = str(getattr(agent, "cbo_select_mode", _cfg_cbo_str("CBO_SELECT_MODE", "greedy")) or "greedy").strip().lower()
    triggered = _cbo_force_exploration_active(agent) or select_mode in {"topk_stochastic", "epsilon_greedy", "randomized_ucb"}
    if not triggered or select_mode == "greedy":
        try:
            agent.cbo_last_actual_beta_used = float(getattr(agent, "cbo_last_beta_eff", getattr(agent, "cbo_acq_beta", 0.0)))
        except Exception:
            pass
        return greedy_idx, default_reason

    k = max(1, min(int(getattr(agent, "cbo_topk", _cfg_cbo_int("CBO_TOPK", 5))), n))
    finite_score = np.where(np.isfinite(score), score, -1e300)
    top_idx = np.argsort(-finite_score)[:k]
    if len(top_idx) == 0:
        return greedy_idx, default_reason

    # epsilon-greedy: usually greedy, occasionally top-K random.
    if select_mode == "epsilon_greedy":
        eps = float(np.clip(getattr(agent, "cbo_epsilon", _cfg_cbo_float("CBO_EPSILON", 0.10)), 0.0, 1.0))
        if agent.py_rng.random() >= eps and not _cbo_force_exploration_active(agent):
            return greedy_idx, default_reason
        return int(top_idx[int(agent.py_rng.random() * len(top_idx)) % len(top_idx)]), "epsilon_greedy_topk"

    # randomized UCB: sample a temporary beta and rank mu + beta*sigma.
    if select_mode == "randomized_ucb":
        base_beta = max(0.0, float(getattr(agent, "cbo_last_beta_eff", getattr(agent, "cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0)))))
        beta_sample = float(base_beta * (0.5 + agent.py_rng.random())) if base_beta > 0 else 0.0
        try:
            agent.cbo_last_actual_beta_used = float(beta_sample)
        except Exception:
            pass
        randomized_score = mu + beta_sample * sigma
        return _cbo_numpy_argmax_safe(randomized_score), f"randomized_ucb_beta={beta_sample:.3g}"

    # top-K stochastic: softmax over top-K acquisition scores.
    temp = max(1e-9, float(getattr(agent, "cbo_select_temperature", _cfg_cbo_float("CBO_SELECT_TEMPERATURE", 0.20))))
    vals = finite_score[top_idx]
    vals = vals - np.nanmax(vals)
    probs = np.exp(np.clip(vals / temp, -60.0, 60.0))
    if not np.isfinite(probs).all() or float(np.sum(probs)) <= 0:
        return int(top_idx[int(agent.py_rng.random() * len(top_idx)) % len(top_idx)]), "topk_stochastic_uniform"
    probs = probs / np.sum(probs)
    # Use Python RNG for reproducibility with agent stream.
    r = agent.py_rng.random()
    cum = 0.0
    for idx, p in zip(top_idx, probs):
        cum += float(p)
        if r <= cum:
            return int(idx), "topk_stochastic"
    return int(top_idx[-1]), "topk_stochastic_tail"


def _cbo_recent_costs(agent, window=None):
    records = []
    try:
        records = [agent._unpack_sample(s) for s in getattr(agent, "local_recent", [])]
    except Exception:
        records = []
    if window is not None and int(window) > 0:
        records = records[-int(window):]
    vals = []
    for r in records:
        try:
            vals.append(-float(r.get("y")))
        except Exception:
            pass
    return [v for v in vals if np.isfinite(v)]


def _cbo_update_residual_condition_state(agent, actual_cost):
    """Update residual/condition diagnostics and possibly reset TR radius.

    This function is called after a true window feedback is observed. It compares the
    selected candidate's predicted cost against actual Eval/Train cost and can trigger
    a temporary exploration mode for the next few selections.
    """
    debug = dict(getattr(agent, "last_debug_info", {}) or {})
    tr_mode = str(getattr(agent, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) or "off").lower()
    select_mode = str(getattr(agent, "cbo_select_mode", _cfg_cbo_str("CBO_SELECT_MODE", "greedy")) or "greedy").lower()
    sigma_floor = max(1e-12, float(getattr(agent, "cbo_sigma_floor", _cfg_cbo_float("CBO_SIGMA_FLOOR", 1e-6))))
    mu = debug.get("selected_candidate_mu", debug.get("posterior_mu"))
    sigma = debug.get("selected_candidate_sigma", debug.get("posterior_sigma"))
    predicted_cost = np.nan
    surprise = np.nan
    raw_error = np.nan
    try:
        if mu is not None and np.isfinite(float(mu)):
            predicted_cost = -float(mu)
            raw_error = float(actual_cost) - predicted_cost
            sig = sigma_floor if sigma is None or not np.isfinite(float(sigma)) else max(float(sigma), sigma_floor)
            surprise = raw_error / sig
    except Exception:
        pass

    hist = list(getattr(agent, "cbo_surprise_history", []))
    hist.append({
        "actual_cost": float(actual_cost),
        "predicted_cost": float(predicted_cost) if np.isfinite(predicted_cost) else np.nan,
        "prediction_error": float(raw_error) if np.isfinite(raw_error) else np.nan,
        "surprise": float(surprise) if np.isfinite(surprise) else np.nan,
        "radius": float(getattr(agent, "trust_radius", np.nan)),
    })
    max_hist = max(20, int(getattr(agent, "cbo_surprise_window", _cfg_cbo_int("CBO_SURPRISE_WINDOW", 10))) * 5)
    agent.cbo_surprise_history = hist[-max_hist:]

    # radius-min stuck counter
    r = float(getattr(agent, "trust_radius", np.nan))
    r_min = float(getattr(agent, "cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))))
    if np.isfinite(r) and r <= r_min * 1.001:
        agent.cbo_radius_min_stuck_count = int(getattr(agent, "cbo_radius_min_stuck_count", 0)) + 1
    else:
        agent.cbo_radius_min_stuck_count = 0

    # Recent rebound relative to recent best.
    rebound_window = max(2, int(getattr(agent, "cbo_rebound_window", _cfg_cbo_int("CBO_REBOUND_WINDOW", 20))))
    recent_costs = _cbo_recent_costs(agent, window=rebound_window)
    recent_best = min(recent_costs) if recent_costs else np.nan
    cost_gap_pct = 0.0
    if np.isfinite(recent_best) and abs(recent_best) > 1e-9:
        cost_gap_pct = (float(actual_cost) - float(recent_best)) / abs(float(recent_best))

    z_thr = float(getattr(agent, "cbo_surprise_z_threshold", _cfg_cbo_float("CBO_SURPRISE_Z_THRESHOLD", 2.0)))
    gap_thr = float(getattr(agent, "cbo_surprise_cost_gap_pct", _cfg_cbo_float("CBO_SURPRISE_COST_GAP_PCT", 0.03)))
    rebound_thr = float(getattr(agent, "cbo_rebound_threshold_pct", _cfg_cbo_float("CBO_REBOUND_THRESHOLD_PCT", 0.03)))
    stuck_thr = int(getattr(agent, "cbo_radius_min_stuck_rounds", _cfg_cbo_int("CBO_RADIUS_MIN_STUCK_ROUNDS", 10)))
    residual_trigger = bool(np.isfinite(surprise) and surprise >= z_thr and cost_gap_pct >= gap_thr)
    condition_trigger = bool(cost_gap_pct >= rebound_thr or int(getattr(agent, "cbo_radius_min_stuck_count", 0)) >= stuck_thr)
    trigger = (tr_mode == "residual_adaptive" and residual_trigger) or (tr_mode == "condition_adaptive" and condition_trigger)

    if trigger:
        reset = float(getattr(agent, "cbo_radius_reset", _cfg_cbo_float("CBO_RADIUS_RESET", 0.12)))
        r_max = float(getattr(agent, "cbo_tr_radius_max", _cfg_cbo_float("CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35))))
        r_min = float(getattr(agent, "cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))))
        agent.trust_radius = float(np.clip(max(reset, r), r_min, r_max))
        agent.cbo_force_explore_countdown = max(int(getattr(agent, "cbo_force_explore_countdown", 0)), int(getattr(agent, "cbo_selection_cooldown", _cfg_cbo_int("CBO_SELECTION_COOLDOWN", 5))))
        anchor_switch = str(getattr(agent, "cbo_condition_anchor_switch", _cfg_cbo_str("CBO_CONDITION_ANCHOR_SWITCH", "context_best")) or "off").lower()
        if anchor_switch != "off":
            agent.cbo_runtime_anchor_override = anchor_switch
        agent.cbo_tr_update_reason = ("residual_surprise_reset" if residual_trigger else "condition_rebound_or_radius_stuck_reset")
    else:
        # Decrease countdown after each observed feedback.
        if int(getattr(agent, "cbo_force_explore_countdown", 0)) > 0:
            agent.cbo_force_explore_countdown = max(0, int(getattr(agent, "cbo_force_explore_countdown", 0)) - 1)
        if int(getattr(agent, "cbo_force_explore_countdown", 0)) <= 0:
            agent.cbo_runtime_anchor_override = None

    debug.update({
        "cbo_select_mode": select_mode,
        "cbo_topk": int(getattr(agent, "cbo_topk", _cfg_cbo_int("CBO_TOPK", 5))),
        "cbo_select_temperature": float(getattr(agent, "cbo_select_temperature", _cfg_cbo_float("CBO_SELECT_TEMPERATURE", 0.20))),
        "cbo_epsilon": float(getattr(agent, "cbo_epsilon", _cfg_cbo_float("CBO_EPSILON", 0.10))),
        "cbo_acq_beta": float(getattr(agent, "cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0))),
        "predicted_cost": float(predicted_cost) if np.isfinite(predicted_cost) else np.nan,
        "actual_cost": float(actual_cost),
        "prediction_error": float(raw_error) if np.isfinite(raw_error) else np.nan,
        "surprise": float(surprise) if np.isfinite(surprise) else np.nan,
        "cost_gap_pct": float(cost_gap_pct),
        "residual_trigger": int(residual_trigger),
        "condition_trigger": int(condition_trigger),
        "radius_min_stuck_count": int(getattr(agent, "cbo_radius_min_stuck_count", 0)),
        "force_explore_countdown": int(getattr(agent, "cbo_force_explore_countdown", 0)),
        "runtime_anchor_override": getattr(agent, "cbo_runtime_anchor_override", None),
        "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", np.nan)),
    })
    agent.last_debug_info = debug
    return debug

def _safebo_posterior_mean_theta(agent, state=None, context=None):
    if getattr(agent, "anchor_points", None) and agent.step_count < len(agent.anchor_points):
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "anchor_or_original_ask", "mu": None, "sigma": None, "candidate_count": 1}

    try:
        _, _, records = agent._training_data(state=state)
    except Exception:
        records = []

    if len(records) < 2:
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "cold_start_original_ask", "mu": None, "sigma": None, "candidate_count": 1}

    model_pack = agent.fit_local_gp(state=state)
    if model_pack is None:
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "fit_failed_original_ask", "mu": None, "sigma": None, "candidate_count": 1}

    candidates = _safebo_candidate_pool(agent, state=state, context=context)
    if not candidates:
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "empty_pool_original_ask", "mu": None, "sigma": None, "candidate_count": 1}
    sources = list(getattr(agent, "_last_candidate_sources", []) or [])
    if len(sources) < len(candidates):
        sources += ["posterior_mean_candidate"] * (len(candidates) - len(sources))
    sources = sources[:len(candidates)]

    try:
        gp = model_pack["gp"]
        y_mean = float(model_pack["y_mean"].detach().view(-1)[0].item())
        y_std = float(model_pack["y_std"].detach().view(-1)[0].item())
        bounds_full = model_pack["bounds"]
        xs = torch.tensor([agent._compose_features(t, context=context) for t in candidates], dtype=torch.double)
        xs_norm = torch.clamp(normalize(xs, bounds_full), 0.0, 1.0)
        with torch.no_grad():
            posterior = gp.posterior(xs_norm)
            mu_std = posterior.mean.detach().view(-1)
            var_std = posterior.variance.detach().view(-1)
            mu = mu_std * y_std + y_mean
            sigma = torch.sqrt(torch.clamp(var_std * (y_std ** 2), min=0.0))
            beta_info = _cbo_beta_eff_info(agent)
            score = mu + float(beta_info.get("beta_eff", getattr(agent, "cbo_acq_beta", getattr(agent, "beta_init", 3.0)))) * sigma
            score_np, service_penalty, guard_info = _cbo_service_guard_apply(agent, score.detach().cpu().numpy())
            beta_info.update(guard_info)
            try:
                agent.cbo_last_beta_info = dict(beta_info)
            except Exception:
                pass
            best_idx, select_reason = _cbo_select_index_from_scores(
                agent,
                mu.detach().cpu().numpy(),
                sigma.detach().cpu().numpy(),
                score_np,
                default_reason="greedy_posterior_mean",
            )
        theta = list(candidates[best_idx])
        agent.step_count += 1
        agent.last_theta = list(theta)
        try:
            agent.acq_history.append({
                "step": int(agent.step_count),
                "candidates": [list(c) for c in candidates],
                "acq_values": [float(v) for v in list(score_np)],
                "best_selected": list(theta),
                "model_state_dict": gp.state_dict(),
                "selection_policy": str(select_reason),
            })
        except Exception:
            pass
        if 0 <= best_idx < len(sources) and str(select_reason) != "greedy_posterior_mean":
            sources[best_idx] = str(select_reason)
        recent_records = [agent._unpack_sample(s) for s in getattr(agent, "local_recent", [])]
        recent_best = list(max(recent_records, key=lambda r: float(r.get("y", -1e300))).get("theta", [])) if recent_records else None
        robust_theta = None
        try:
            robust_theta, _ = agent._compute_robust_incumbent(context=context)
        except Exception:
            robust_theta = None
        cand_rows, cand_summary = _cbo_candidate_rows(
            agent, candidates, sources,
            mu.detach().cpu().numpy(), sigma.detach().cpu().numpy(), score_np,
            best_idx, selected_reason=str(select_reason),
            deploy_policy="greedy_mean", deploy_source=str(select_reason),
            anchor=(getattr(agent, "cbo_last_actual_anchor_debug", {}) or {}).get("actual_tr_anchor_theta", getattr(agent, "prev_best", None)),
            robust_theta=robust_theta, recent_best=recent_best,
            beta_eff=beta_info.get("beta_eff"), service_penalty=service_penalty,
        )
        debug = dict(getattr(agent, "last_debug_info", {}) or {})
        debug.update(cand_summary)
        debug.update(beta_info)
        debug.update({
            "selected_candidate_source": str(sources[best_idx] if 0 <= best_idx < len(sources) else "posterior_mean_candidate"),
            "selected_candidate_mu": float(mu[best_idx].item()),
            "selected_candidate_sigma": float(sigma[best_idx].item()),
            "selected_candidate_acq": float(score_np[best_idx]),
            "selected_candidate_score": float(score_np[best_idx]),
            "selected_candidate_beta_eff": float(beta_info.get("beta_eff", np.nan)),
            "actual_beta_used": float(getattr(agent, "cbo_last_actual_beta_used", beta_info.get("beta_eff", np.nan))),
            "cbo_select_mode": str(getattr(agent, "cbo_select_mode", "greedy")),
            "selected_reason": str(select_reason),
            "candidate_diagnostic_rows": cand_rows,
        })
        agent.last_debug_info = debug
        return theta, {
            "selection": str(select_reason),
            "mu": float(mu[best_idx].item()),
            "sigma": float(sigma[best_idx].item()),
            "candidate_count": int(len(candidates)),
            **cand_summary,
            "selected_candidate_source": str(sources[best_idx] if 0 <= best_idx < len(sources) else "posterior_mean_candidate"),
            "selected_candidate_mu": float(mu[best_idx].item()),
            "selected_candidate_sigma": float(sigma[best_idx].item()),
            "selected_candidate_acq": float(score_np[best_idx]),
            "selected_candidate_score": float(score_np[best_idx]),
            "selected_candidate_beta_eff": float(beta_info.get("beta_eff", np.nan)),
            "actual_beta_used": float(getattr(agent, "cbo_last_actual_beta_used", beta_info.get("beta_eff", np.nan))),
            "cbo_select_mode": str(getattr(agent, "cbo_select_mode", "greedy")),
            "selected_reason": str(select_reason),
            "candidate_diagnostic_rows": cand_rows,
            **beta_info,
        }
    except Exception as e:
        theta = agent.ask(state=state, context=context)
        return theta, {"selection": "posterior_failed_original_ask_" + type(e).__name__, "mu": None, "sigma": None, "candidate_count": 1}


def _safebo_select_theta(agent, state=None, context=None, group_cfg=None):
    policy = _safebo_policy_name(group_cfg)
    warmup = int(_safebo_env("SAFEBO_WARMUP", 10, int))
    explore_prob = float(_safebo_env("SAFEBO_EXPLORE_PROB", 0.20, float))
    incumbent_available = bool(getattr(agent, "prev_best", None) is not None)
    incumbent_cost = -float(agent.prev_best_value) if getattr(agent, "prev_best_value", None) is not None else None
    incumbent_iter = getattr(agent, "prev_best_iter", None)

    def pack_info(deploy_policy, deploy_source, explore_used, **extra):
        info = {
            "deploy_policy": deploy_policy,
            "deploy_source": deploy_source,
            "used_theta_source": deploy_source,
            "explore_used": int(explore_used),
            "incumbent_available": bool(incumbent_available),
            "incumbent_cost": incumbent_cost,
            "current_candidate_cost": None,
            "current_train_cost": None,
            "best_so_far_cost": incumbent_cost,
            "best_so_far_iter": incumbent_iter,
            "posterior_mu": None,
            "posterior_sigma": None,
            "candidate_count_safe": None,
        }
        info.update(extra)
        return info

    if policy in {"ei", "default", "original", "acq", "explore"}:
        theta = agent.ask(state=state, context=context)
        return theta, pack_info("ei", "acquisition_candidate", 1)

    if policy in {"greedy", "greedy_mean", "posterior_mean", "mean"}:
        theta, info = _safebo_posterior_mean_theta(agent, state=state, context=context)
        extra_info = dict(info or {})
        extra_info.update({"posterior_mu": info.get("mu"), "posterior_sigma": info.get("sigma"), "candidate_count_safe": info.get("candidate_count")})
        return theta, pack_info("greedy_mean", info.get("selection", "greedy_posterior_mean"), 0, **extra_info)

    if policy in {"incumbent", "incumbent_safe", "safe", "safe_bo"}:
        try:
            _, _, records = agent._training_data(state=state)
            n_records = len(records)
        except Exception:
            n_records = 0
        if getattr(agent, "prev_best", None) is None or n_records < warmup:
            theta = agent.ask(state=state, context=context)
            return theta, pack_info("incumbent_safe", "warmup_acquisition_candidate", 1)
        if agent.py_rng.random() < explore_prob:
            theta = agent.ask(state=state, context=context)
            return theta, pack_info("incumbent_safe", "exploration_candidate", 1)
        theta = list(agent.prev_best)
        return theta, pack_info("incumbent_safe", "incumbent_prev_best", 0)

    theta = agent.ask(state=state, context=context)
    return theta, pack_info("unknown_fallback_ei_" + policy, "fallback_acquisition_candidate", 1)


_ORIG_SAFEBO_SELECT_THETA_STABILITY = _safebo_select_theta


def _safebo_select_theta(agent, state=None, context=None, group_cfg=None):
    if agent is not None:
        try:
            agent._active_context = context
        except Exception:
            pass
    theta, info = _ORIG_SAFEBO_SELECT_THETA_STABILITY(agent, state=state, context=context, group_cfg=group_cfg)
    robust_mode = str(getattr(agent, "cbo_robust_incumbent_mode", _cfg_cbo_str("CBO_ROBUST_INCUMBENT_MODE", "off")) if agent is not None else "off").strip().lower()
    robust_theta, robust_info = (None, {"robust_incumbent_available": False, "robust_incumbent_reason": "agent_none"})
    if agent is not None:
        try:
            robust_theta, robust_info = agent._compute_robust_incumbent(context=context)
        except Exception as exc:
            robust_info = {"robust_incumbent_available": False, "robust_incumbent_reason": "robust_error:" + type(exc).__name__}
    robust_info = dict(robust_info or {})
    robust_info.setdefault("robust_incumbent_used", False)
    if robust_mode == "deploy" and robust_theta is not None and robust_info.get("robust_incumbent_available"):
        eval_count = int(robust_info.get("robust_incumbent_eval_count") or 0)
        sim = float(robust_info.get("robust_incumbent_context_similarity") or 0.0)
        threshold = float(getattr(agent, "cbo_context_sim_threshold", _cfg_cbo_float("CBO_CONTEXT_SIM_THRESHOLD", 0.0)))
        posterior_mu = info.get("posterior_mu")
        predicted_current_cost = -float(posterior_mu) if posterior_mu is not None else np.nan
        robust_score = float(robust_info.get("robust_incumbent_score", np.nan))
        bo_not_clearly_better = (not np.isfinite(predicted_current_cost)) or (np.isfinite(robust_score) and predicted_current_cost >= robust_score * 0.98)
        if (eval_count >= 2 or not robust_info.get("robust_incumbent_available") is False) and sim >= threshold and bo_not_clearly_better:
            theta = list(robust_theta)
            info["deploy_source"] = "robust_incumbent"
            info["used_theta_source"] = "robust_incumbent"
            info["explore_used"] = 0
            robust_info["robust_incumbent_used"] = True
            robust_info["robust_incumbent_reason"] = "deployed_robust_score"
        else:
            robust_info["robust_incumbent_reason"] = f"not_deployed eval_count={eval_count} sim={sim:.3f} bo_not_clearly_better={bo_not_clearly_better}"
    elif robust_mode == "recommend_only" and robust_info.get("robust_incumbent_available"):
        robust_info["robust_incumbent_reason"] = "recommend_only"
    info.update(robust_info)
    if agent is not None:
        debug = dict(getattr(agent, "last_debug_info", {}) or {})
        debug.update(dict(getattr(agent, "last_history_debug", {}) or {}))
        for k in [
            "history_select_mode", "effective_history_mode", "effective_recent_window",
            "selected_recent_count", "selected_macro_count", "selected_context_count", "selected_elite_count",
            "selected_diverse_count", "selected_total_count", "context_similarity_max",
            "context_similarity_mean", "elite_best_robust_score", "elite_best_eval_count",
            "elite_best_mean_cost", "elite_best_std_cost", "cbo_tr_mode", "cbo_tr_anchor_mode",
            "cbo_tr_radius", "cbo_tr_anchor_theta", "cbo_tr_candidate_count",
            "cbo_global_candidate_count", "cbo_tr_update_reason", "cbo_tr_success_count",
            "cbo_tr_failure_count", "tr_update_mode", "tr_baseline_mean", "tr_current_mean",
            "tr_improve_pct", "tr_worse_pct", "tr_update_signal", "tr_update_patience_count",
            "cbo_tr_radius_before_update", "cbo_tr_radius_after_update",
            "cbo_macro_gate_mode", "macro_total_arrivals_norm", "macro_rt_ratio",
            "macro_batch_ratio", "macro_similarity_max", "macro_similarity_mean",
            "macro_similarity_p50", "macro_similarity_p90", "selected_macro_mean_similarity",
            "selected_macro_min_similarity", "selected_macro_max_similarity", "macro_k",
            "macro_lengthscale_total", "macro_lengthscale_rt", "macro_lengthscale_batch",
            "macro_pool_count", "macro_pool_mean_similarity", "macro_pool_min_similarity",
            "macro_pool_max_similarity", "macro_pool_p50_similarity", "macro_pool_p90_similarity",
            "selected_from_macro_pool_count", "selected_outside_macro_pool_count",
            "macro_gate_fallback_used", "macro_gate_fallback_reason",
            "context_selection_source_pool", "elite_selection_source_pool", "tr_anchor_source_pool",
            "selected_candidate_source", "selected_candidate_mu", "selected_candidate_sigma",
            "selected_candidate_acq", "selected_candidate_score", "selected_candidate_beta_eff",
            "selected_candidate_rank_by_score", "selected_candidate_rank_by_mu",
            "selected_candidate_rank_by_sigma", "selected_candidate_rank_by_acq",
            "best_mu_candidate_source", "best_acq_candidate_source", "num_candidates",
            "num_tr_candidates", "num_global_candidates", "candidate_diagnostic_rows",
            "cbo_select_mode", "cbo_topk", "cbo_select_temperature", "cbo_epsilon", "cbo_acq_beta",
            "cbo_acq_beta_mode", "beta_eff", "radius_norm", "radius_beta_component",
            "state_beta_boost_used", "state_beta_boost_reason", "actual_score_formula",
            "actual_beta_used", "service_guard_mode", "service_guard_available",
            "service_guard_penalty", "service_guard_reason",
            "actual_tr_anchor_mode", "actual_tr_anchor_source", "actual_tr_anchor_theta",
            "actual_tr_anchor_reason", "anchor_override_used", "anchor_override_reason",
            "anchor_fallback_used", "anchor_fallback_reason", "anchor_theta_distance_to_prev",
            "anchor_theta_distance_to_robust_elite", "anchor_theta_distance_to_context_best",
            "anchor_theta_distance_to_recent_best", "runtime_anchor_override_reason",
            "predicted_cost", "actual_cost", "prediction_error", "surprise", "cost_gap_pct",
            "residual_trigger", "condition_trigger", "radius_min_stuck_count",
            "force_explore_countdown", "runtime_anchor_override", "cbo_tr_radius_after_update", "selected_reason",
        ]:
            if k in debug:
                info[k] = debug.get(k)
        info.setdefault("history_select_mode", getattr(agent, "cbo_history_select_mode", _cfg_cbo_history_select_mode("recent")))
        info.setdefault("effective_history_mode", getattr(agent, "history_mode", _cfg_history_mode("all")))
        info.setdefault("effective_recent_window", getattr(agent, "recent_window", _cfg_recent_window()))
        info.setdefault("cbo_tr_mode", getattr(agent, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")))
        info.setdefault("cbo_tr_anchor_mode", getattr(agent, "cbo_tr_anchor_mode", _cfg_cbo_str("CBO_TR_ANCHOR_MODE", "posterior_mean")))
        info.setdefault("cbo_tr_radius", float(getattr(agent, "trust_radius", np.nan)))
        if str(getattr(agent, "cbo_macro_gate_mode", "")).strip().lower() == "hierarchical":
            for _pool_key in ["context_selection_source_pool", "elite_selection_source_pool", "tr_anchor_source_pool"]:
                if _is_missing_value(info.get(_pool_key)):
                    info[_pool_key] = "macro_pool"
    return theta, info


# ===============================================================
# REFACTOR V1: BO training feedback selector
# ---------------------------------------------------------------
# Eval_Cost:  最终系统评价，默认仍然是 WindowSnapshot.to_metrics() 里的 metrics["cost"]。
# Train_Cost: 真正 tell 给 BO 的单标量反馈。默认等于 Eval_Cost；
#             后续如果比较反馈设计，只改这里，不再散落到 run_scenario_group 里。
# ===============================================================


# ===============================================================
# Reference-normalized metric patch: scenario baseline + tradeoff score
# Defaults preserve old behavior unless explicitly enabled from CLI.
# ===============================================================
def _cbo_metric_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _cbo_metric_clip_ratio(v, lo=None, hi=None):
    x = _cbo_metric_float(v, np.nan)
    if not np.isfinite(x):
        return np.nan
    lo = float(getattr(CFG, "CBO_NORMALIZED_RATIO_CLIP_MIN", 0.2) if lo is None else lo)
    hi = float(getattr(CFG, "CBO_NORMALIZED_RATIO_CLIP_MAX", 5.0) if hi is None else hi)
    return float(np.clip(x, lo, hi))


def _cbo_metric_reference_stat(vals, stat="median", trim_pct=0.1):
    arr = np.array([_cbo_metric_float(v) for v in vals], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    stat = str(stat or "median").lower()
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "trimmed_mean":
        arr.sort()
        k = int(np.floor(float(trim_pct) * arr.size))
        if k > 0 and arr.size > 2 * k:
            arr = arr[k:-k]
        return float(np.mean(arr)) if arr.size else np.nan
    return float(np.median(arr))


def _cbo_macro_context_key():
    try:
        lambdas = []
        for item in getattr(CFG, "LAMBDA_SCHEDULE", []):
            if len(item) >= 3:
                lambdas.append(float(item[2]))
        lam = float(np.median(lambdas)) if lambdas else np.nan
    except Exception:
        lam = np.nan
    try:
        probs = get_task_type_probs_at_time(0.0)
        rt = int(round(100 * float(probs.get("RT", 0.0))))
        batch = int(round(100 * float(probs.get("Batch", 0.0))))
        ai = int(round(100 * float(probs.get("AI", 0.0))))
    except Exception:
        rt, batch, ai = 0, 0, 0
    if np.isfinite(lam):
        return f"lambda_{lam:.3g}_mix_{rt}_{batch}_{ai}"
    return f"taskmix_{rt}_{batch}_{ai}"


def _cbo_load_reference_from_file():
    path = str(getattr(CFG, "CBO_REFERENCE_FILE", "") or "").strip()
    if not path:
        return None, "empty_reference_file"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = _cbo_macro_context_key()
        if isinstance(data, dict) and key in data and isinstance(data[key], dict):
            return data[key], "loaded_macro_key"
        if isinstance(data, dict):
            return data, "loaded_single_reference"
        return None, "invalid_reference_json"
    except Exception as e:
        return None, f"load_failed:{type(e).__name__}"


def _cbo_write_reference_if_needed(ref):
    out_path = str(getattr(CFG, "CBO_REFERENCE_OUTPUT_FILE", "") or "").strip()
    if not out_path or not isinstance(ref, dict):
        return
    try:
        if os.path.dirname(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        key = ref.get("macro_context_key", _cbo_macro_context_key())
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({key: ref}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _cbo_build_reference(records):
    if not records:
        return None
    stat = str(getattr(CFG, "CBO_REFERENCE_STAT", "median"))
    trim = float(getattr(CFG, "CBO_REFERENCE_TRIM_PCT", 0.1))
    ref = {
        "delay_ref": _cbo_metric_reference_stat([r.get("avg_delay") for r in records], stat, trim),
        "energy_per_arrival_ref": _cbo_metric_reference_stat([r.get("energy_per_arrival") for r in records], stat, trim),
        "unfinished_rate_ref": _cbo_metric_reference_stat([r.get("unfinished_rate") for r in records], stat, trim),
        "success_rate_ref": _cbo_metric_reference_stat([r.get("sla_success_rate") for r in records], stat, trim),
        "eval_cost_ref": _cbo_metric_reference_stat([r.get("cost") for r in records], stat, trim),
        "arrivals_ref": _cbo_metric_reference_stat([r.get("arrivals_total") for r in records], stat, trim),
        "created_at_iter": int(len(records)),
        "calibration_rounds": int(len(records)),
        "reference_stat": stat,
        "macro_context_key": _cbo_macro_context_key(),
    }
    for k in ["delay_ref", "energy_per_arrival_ref", "eval_cost_ref"]:
        if not np.isfinite(_cbo_metric_float(ref.get(k))) or abs(_cbo_metric_float(ref.get(k))) < 1e-12:
            ref[k] = np.nan
    return ref


def _cbo_metric_reference_patch(factory, metrics):
    # Compute extra metric diagnostics and optional normalized tradeoff objective.
    # Mutates and returns metrics. Falls back to Eval_Cost if reference unavailable.
    if metrics is None:
        metrics = {}
    ref_mode = str(getattr(CFG, "CBO_REFERENCE_MODE", "off")).lower()
    objective_mode = str(getattr(CFG, "CBO_OBJECTIVE_MODE", "eval_cost")).lower()
    eps = 1e-9

    arrivals = int(_cbo_metric_float(metrics.get("arrivals_total", metrics.get("arrivals", 0)), 0))
    completed = int(_cbo_metric_float(metrics.get("completed_total", metrics.get("task_count", metrics.get("completed", 0))), 0))
    unfinished = int(_cbo_metric_float(metrics.get("unfinished_end", metrics.get("backlog", 0)), 0))
    total_energy = _cbo_metric_float(metrics.get("total_energy", np.nan))
    avg_energy = _cbo_metric_float(metrics.get("avg_energy", np.nan))

    if np.isfinite(total_energy):
        energy_per_arrival = float(total_energy) / max(arrivals, 1)
        energy_metric_source = "Total_Energy_per_arrival"
    else:
        energy_per_arrival = avg_energy
        energy_metric_source = "Avg_Energy"

    unfinished_rate = float(unfinished) / max(arrivals, 1)
    if factory is not None:
        prev_backlog = getattr(factory, "_cbo_prev_backlog_end", None)
        factory._cbo_prev_backlog_end = int(unfinished)
    else:
        prev_backlog = None
    backlog_growth = 0.0 if prev_backlog is None else max(0.0, float(unfinished) - float(prev_backlog))
    backlog_growth_rate = float(backlog_growth) / max(arrivals, 1)

    comp_rt = _cbo_metric_float(metrics.get("completion_ratio_rt", np.nan))
    comp_batch = _cbo_metric_float(metrics.get("completion_ratio_batch", np.nan))
    comp_ai = _cbo_metric_float(metrics.get("completion_ratio_ai", np.nan))
    class_vals = [x for x in [comp_rt, comp_batch, comp_ai] if np.isfinite(x)]
    class_imbalance_available = len(class_vals) == 3
    min_class_success_rate = float(np.min(class_vals)) if class_vals else np.nan
    class_imbalance_penalty = float(np.max(class_vals) - np.min(class_vals)) if len(class_vals) == 3 else np.nan

    metrics.update({
        "macro_context_key": _cbo_macro_context_key(),
        "window_arrivals_total": int(arrivals),
        "window_completed_total": int(completed),
        "window_unfinished_total": int(unfinished),
        "unfinished_rate": float(unfinished_rate),
        "backlog_growth": float(backlog_growth),
        "backlog_growth_rate": float(backlog_growth_rate),
        "energy_per_arrival": float(energy_per_arrival) if np.isfinite(energy_per_arrival) else np.nan,
        "energy_metric_source": str(energy_metric_source),
        "class_imbalance_available": bool(class_imbalance_available),
        "min_class_success_rate": min_class_success_rate,
        "class_imbalance_penalty": class_imbalance_penalty,
    })

    if factory is None:
        ref_records = []
        ref_frozen = False
        ref = None
        ref_status = "no_factory"
    else:
        if not hasattr(factory, "_cbo_reference_records"):
            factory._cbo_reference_records = []
            factory._cbo_reference_frozen = False
            factory._cbo_reference = None
            factory._cbo_reference_status = "new"
        ref_records = factory._cbo_reference_records
        ref_frozen = bool(getattr(factory, "_cbo_reference_frozen", False))
        ref = getattr(factory, "_cbo_reference", None)
        ref_status = getattr(factory, "_cbo_reference_status", "new")

    if ref_mode == "off":
        ref_status = "off"
    elif ref_mode == "load":
        if ref is None:
            ref, ref_status = _cbo_load_reference_from_file()
            if factory is not None:
                factory._cbo_reference = ref
                factory._cbo_reference_frozen = ref is not None
        ref_frozen = ref is not None
    elif ref_mode in {"calibrate", "auto_macro"}:
        if not ref_frozen and factory is not None:
            ref_records.append(dict(metrics))
            n_rec = len(ref_records)
            min_rounds = int(getattr(CFG, "CBO_REFERENCE_MIN_ROUNDS", 5))
            calib_rounds = int(getattr(CFG, "CBO_REFERENCE_CALIBRATION_ROUNDS", 30))
            if n_rec < min_rounds:
                ref_status = "calibrating_min_rounds"
            elif n_rec < calib_rounds:
                ref_status = "calibrating"
                ref = _cbo_build_reference(ref_records)
                factory._cbo_reference = ref
            else:
                ref = _cbo_build_reference(ref_records[-calib_rounds:])
                factory._cbo_reference = ref
                factory._cbo_reference_frozen = bool(getattr(CFG, "CBO_REFERENCE_FREEZE_AFTER_CALIBRATION", True))
                ref_frozen = bool(factory._cbo_reference_frozen)
                ref_status = "frozen" if ref_frozen else "calibrated_unfrozen"
                _cbo_write_reference_if_needed(ref)
        else:
            ref_status = "frozen" if ref_frozen else ref_status
    else:
        ref_status = f"unknown_reference_mode:{ref_mode}"

    ref_available = isinstance(ref, dict)
    delay_ref = _cbo_metric_float(ref.get("delay_ref") if ref_available else np.nan)
    energy_ref = _cbo_metric_float(ref.get("energy_per_arrival_ref") if ref_available else np.nan)
    unfinished_ref = _cbo_metric_float(ref.get("unfinished_rate_ref") if ref_available else np.nan)
    success_ref = _cbo_metric_float(ref.get("success_rate_ref") if ref_available else np.nan)
    eval_ref = _cbo_metric_float(ref.get("eval_cost_ref") if ref_available else np.nan)

    avg_delay = _cbo_metric_float(metrics.get("avg_delay", np.nan))
    eval_cost = _cbo_metric_float(metrics.get("cost", np.nan))
    sla_success = _cbo_metric_float(metrics.get("sla_success_rate", 1.0), 1.0)

    delay_norm = _cbo_metric_clip_ratio(avg_delay / delay_ref) if np.isfinite(delay_ref) and abs(delay_ref) > eps else np.nan
    energy_norm = _cbo_metric_clip_ratio(energy_per_arrival / energy_ref) if np.isfinite(energy_ref) and abs(energy_ref) > eps else np.nan
    eval_cost_norm = _cbo_metric_clip_ratio(eval_cost / eval_ref) if np.isfinite(eval_ref) and abs(eval_ref) > eps else np.nan
    unfinished_norm = _cbo_metric_clip_ratio(unfinished_rate / unfinished_ref) if np.isfinite(unfinished_ref) and unfinished_ref > eps else np.nan

    target_success = float(getattr(CFG, "CBO_TARGET_SUCCESS_RATE", 0.995))
    success_shortfall = max(0.0, target_success - float(sla_success)) if np.isfinite(sla_success) else np.nan
    success_shortfall_norm = success_shortfall / max(1.0 - target_success, eps) if np.isfinite(success_shortfall) else np.nan

    service_norm = np.nan
    if np.isfinite(delay_norm):
        service_norm = float(delay_norm)
        service_norm += float(getattr(CFG, "CBO_UNFINISHED_PENALTY_WEIGHT", 5.0)) * float(unfinished_rate)
        if np.isfinite(success_shortfall_norm):
            service_norm += float(getattr(CFG, "CBO_SUCCESS_SHORTFALL_WEIGHT", 2.0)) * float(success_shortfall_norm)
        service_norm += float(getattr(CFG, "CBO_BACKLOG_GROWTH_PENALTY_WEIGHT", 2.0)) * float(backlog_growth_rate)
        if bool(class_imbalance_available) and np.isfinite(class_imbalance_penalty):
            service_norm += float(getattr(CFG, "CBO_CLASS_IMBALANCE_WEIGHT", 0.0)) * float(class_imbalance_penalty)

    alpha = float(np.clip(float(getattr(CFG, "CBO_TRADEOFF_ALPHA", 0.8)),
                          float(getattr(CFG, "CBO_ALPHA_MIN", 0.6)),
                          float(getattr(CFG, "CBO_ALPHA_MAX", 0.95))))
    normalized_tradeoff_score = np.nan
    if np.isfinite(service_norm) and np.isfinite(energy_norm):
        normalized_tradeoff_score = float(alpha * service_norm + (1.0 - alpha) * energy_norm)

    metrics.update({
        "cbo_reference_mode": str(ref_mode),
        "cbo_reference_available": bool(ref_available),
        "cbo_reference_status": str(ref_status),
        "cbo_reference_round_count": int(len(ref_records)),
        "cbo_reference_frozen": bool(ref_frozen),
        "delay_ref": delay_ref,
        "energy_per_arrival_ref": energy_ref,
        "unfinished_rate_ref": unfinished_ref,
        "success_rate_ref": success_ref,
        "eval_cost_ref": eval_ref,
        "delay_norm": delay_norm,
        "energy_norm": energy_norm,
        "unfinished_norm": unfinished_norm,
        "eval_cost_norm": eval_cost_norm,
        "success_shortfall": success_shortfall,
        "success_shortfall_norm": success_shortfall_norm,
        "service_norm": service_norm,
        "normalized_tradeoff_score": normalized_tradeoff_score,
        "cbo_objective_mode": str(objective_mode),
        "tradeoff_alpha": float(alpha),
        "bo_training_cost_source": "normalized_tradeoff_score" if objective_mode == "normalized_tradeoff" and np.isfinite(normalized_tradeoff_score) else "Eval_Cost_or_feedback_score",
    })
    return metrics


def _cbo_log_reference_fields(perf_log, metrics):
    keys = [
        "macro_context_key", "window_arrivals_total", "window_completed_total", "window_unfinished_total",
        "unfinished_rate", "backlog_growth", "backlog_growth_rate", "energy_per_arrival", "energy_metric_source",
        "class_imbalance_available", "min_class_success_rate", "class_imbalance_penalty",
        "cbo_reference_mode", "cbo_reference_available", "cbo_reference_status", "cbo_reference_round_count", "cbo_reference_frozen",
        "delay_ref", "energy_per_arrival_ref", "unfinished_rate_ref", "success_rate_ref", "eval_cost_ref",
        "delay_norm", "energy_norm", "unfinished_norm", "eval_cost_norm",
        "success_shortfall", "success_shortfall_norm", "service_norm", "normalized_tradeoff_score",
        "cbo_objective_mode", "tradeoff_alpha", "bo_training_cost_source",
    ]
    for k in keys:
        perf_log.setdefault(k, []).append(metrics.get(k, None))

REFACTOR_VERSION = "bo_refactor_v5_paired_delta_crn_tr_residual_topk"


def _feedback_score_mode():
    """读取 BO 训练反馈模式。默认 window_original，等价于 v3 的 metrics['cost']。"""
    mode = getattr(CFG, "BO_TRAINING_FEEDBACK_SCORE", None)
    if mode is None:
        mode = os.environ.get("BO_TRAINING_FEEDBACK_SCORE", "window_original")
    mode = str(mode or "window_original").strip().lower()
    aliases = {
        "window": "window_original",
        "original": "window_original",
        "eval": "window_original",
        "cost": "window_original",
        "simple": "task_effective",
        "simple_effective": "task_effective",
        "effective": "task_effective",
        "effective_simple": "task_effective",
        "task": "task_effective",
        "task_effective_simple": "task_effective",
        "simple_backlog": "task_effective_backlog",
        "effective_backlog": "task_effective_backlog",
        "simple_backlog_violation": "task_effective_backlog_violation",
        "effective_backlog_violation": "task_effective_backlog_violation",
        "task_effective_backlog_violation": "task_effective_backlog_violation",
        "paired": "paired_fixed_mid_delta",
        "paired_delta": "paired_fixed_mid_delta",
        "delta": "paired_fixed_mid_delta",
        "crn": "paired_fixed_mid_delta",
        "crn_delta": "paired_fixed_mid_delta",
        "paired_fixed": "paired_fixed_mid_delta",
        "paired_mid": "paired_fixed_mid_delta",
        "paired_fixed_mid": "paired_fixed_mid_delta",
        "paired_fixed_mid_delta": "paired_fixed_mid_delta",
    }
    return aliases.get(mode, mode)


def _refactor_effective_task_cost(metrics, include_backlog=False, include_violation=False):
    """简化 BO 训练反馈：只保留连续、少重复的三类信号。

    这个函数刻意不使用 class cost、pending area、completion ratio、zero-completion
    penalty 等复杂项，避免 BO 训练目标变成“大杂烩”。

    当前可从窗口聚合指标中稳定获得的信号是：
    - avg_energy: 平均能耗
    - avg_delay: 完成任务平均延迟
    - avg_lateness: 完成任务平均超期时长

    如果 include_backlog=True，仅加入轻量 backlog 项，作为积压保护。
    """
    avg_energy = _safe_float(metrics.get("avg_energy", 0.0), 0.0)
    avg_delay = _safe_float(metrics.get("avg_delay", 0.0), 0.0)
    avg_late = _safe_float(metrics.get("avg_lateness", 0.0), 0.0)
    cost = (
        avg_energy
        + float(getattr(CFG, "ALPHA_LATENCY", 100.0)) * avg_delay
        + float(getattr(CFG, "LATE_PENALTY_WEIGHT", 300.0)) * avg_late
    )
    if include_backlog:
        cost += float(getattr(CFG, "REF_SIMPLE_BACKLOG_WEIGHT", 0.25)) * float(getattr(CFG, "BACKLOG_WEIGHT", 200.0)) * _safe_float(metrics.get("backlog", 0.0), 0.0)
    if include_violation:
        vio = _safe_float(
            metrics.get(
                "effective_violation_rate",
                metrics.get("violation_rate", metrics.get("sla_violation_rate", 0.0)),
            ),
            0.0,
        )
        cost += float(getattr(CFG, "SLA_PENALTY_WEIGHT", 1500.0)) * vio
    return float(cost)




def _paired_delta_enabled():
    """Whether scenario runs should create a shadow baseline window for CRN-style paired feedback."""
    return _feedback_score_mode() in {"paired_fixed_mid_delta", "paired_delta", "crn_delta"}


def _paired_baseline_key_for_group(group_cfg=None):
    """Pick a baseline policy with the same control family as the current group."""
    requested = str(getattr(CFG, "PAIRED_BASELINE_KEY", "") or os.environ.get("PAIRED_BASELINE_KEY", "")).strip()
    if requested:
        return USER_METHOD_ALIASES.get(requested, USER_METHOD_ALIASES.get(requested.lower(), requested))
    mode = str((group_cfg or {}).get("control_mode", "reduced6"))
    if mode == "reduced4":
        return "reduced4_fixed_mid"
    return "reduced6_fixed_mid"


def _run_paired_shadow_baseline(factory_snapshot, group_cfg, ask_state=None, ask_ctx=None, window_end=None):
    """Run a baseline policy on a deep-copied factory state for one identical window.

    Simulation-only CRN diagnostic/training option. It does not claim to be
    available in a real online system. The shadow factory starts from the same
    queues, events, node states, and workload RNG state as the deployed theta.
    """
    baseline_key = _paired_baseline_key_for_group(group_cfg)
    groups = build_scenario_method_groups()
    if baseline_key not in groups:
        raise KeyError(f"Unknown paired baseline key: {baseline_key}")
    baseline_cfg = dict(groups[baseline_key])
    if "fixed_theta" not in baseline_cfg:
        raise ValueError(f"Paired baseline must be a fixed policy, got: {baseline_key}")

    shadow = factory_snapshot
    shadow.agent = None
    shadow.disable_internal_agent_tell = True
    theta_control = list(baseline_cfg["fixed_theta"])
    theta_full = map_group_theta_to_full(theta_control, baseline_cfg)
    shadow.current_control_vector = list(theta_full)
    shadow.current_control_label = "paired_baseline:" + baseline_key
    if window_end is None:
        window_end = float(shadow.current_time) + float(getattr(CFG, "BO_INTERVAL", 40.0))
    _, _, _, _, baseline_metrics, _ = shadow.run_continuous(
        theta_full,
        eval_state=ask_state,
        eval_context=ask_ctx,
        feedback_control=theta_control,
        window_end=window_end,
    )
    return baseline_key, theta_control, baseline_metrics


def _attach_paired_delta_metrics(metrics, baseline_key, baseline_metrics):
    eval_cost = float(_safe_float(metrics.get("cost", np.nan), np.nan))
    baseline_cost = float(_safe_float(baseline_metrics.get("cost", np.nan), np.nan))
    delta = eval_cost - baseline_cost
    rel = 100.0 * delta / max(1e-12, abs(baseline_cost)) if np.isfinite(delta) and np.isfinite(baseline_cost) else np.nan
    fields = {
        "paired_baseline_key": str(baseline_key),
        "paired_baseline_cost": baseline_cost,
        "paired_delta_cost": float(delta),
        "paired_delta_relative_pct": float(rel),
        "paired_eval_cost": eval_cost,
        "paired_baseline_reward": float(_safe_float(baseline_metrics.get("reward", -baseline_cost), -baseline_cost)),
        "paired_baseline_avg_delay": float(_safe_float(baseline_metrics.get("avg_delay", np.nan), np.nan)),
        "paired_baseline_avg_energy": float(_safe_float(baseline_metrics.get("avg_energy", np.nan), np.nan)),
        "paired_baseline_sla_success_rate": float(_safe_float(baseline_metrics.get("sla_success_rate", np.nan), np.nan)),
        "paired_baseline_backlog": float(_safe_float(baseline_metrics.get("backlog", np.nan), np.nan)),
        "paired_note": "simulation_only_shadow_baseline_same_window_state_and_rng",
    }
    metrics.update(fields)
    return fields


def log_paired_delta_feedback(fac, metrics):
    keys = [
        "paired_baseline_key", "paired_baseline_cost", "paired_delta_cost", "paired_delta_relative_pct",
        "paired_eval_cost", "paired_baseline_reward", "paired_baseline_avg_delay",
        "paired_baseline_avg_energy", "paired_baseline_sla_success_rate", "paired_baseline_backlog",
        "paired_note",
    ]
    for k in keys:
        fac.perf_log.setdefault(k, []).append(metrics.get(k, None))

def select_bo_training_feedback_cost(metrics, fac=None, group_key=None):
    """统一选择 BO tell 使用的训练 cost。

    返回 (train_cost, feedback_mode, note)。默认返回原始 window cost，保持 v3 行为。
    """
    mode = _feedback_score_mode()
    eval_cost = float(_safe_float(metrics.get("cost", np.nan), np.nan))
    objective_mode = str(getattr(CFG, "CBO_OBJECTIVE_MODE", "eval_cost")).strip().lower()
    if objective_mode == "normalized_tradeoff":
        nt = float(_safe_float(metrics.get("normalized_tradeoff_score", np.nan), np.nan))
        if np.isfinite(nt):
            return nt, "normalized_tradeoff", "normalized service-energy tradeoff score"
        return eval_cost, "normalized_tradeoff_missing", "normalized_tradeoff_missing_fallback_to_eval_cost"
    if mode in {"paired_fixed_mid_delta", "paired_delta", "crn_delta"}:
        if "paired_delta_cost" in metrics and np.isfinite(_safe_float(metrics.get("paired_delta_cost"), np.nan)):
            return float(metrics["paired_delta_cost"]), "paired_fixed_mid_delta", "eval_cost_minus_shadow_fixed_mid_cost"
        return eval_cost, "paired_fixed_mid_delta_missing", "paired_delta_missing_fallback_to_eval_cost"
    if mode in {"window_original", "window_cost", "legacy_window"}:
        return eval_cost, "window_original", "same_as_eval_cost"
    if mode in {"task_effective"}:
        return _refactor_effective_task_cost(metrics, include_backlog=False), "task_effective", "energy_delay_lateness_only"
    if mode in {"task_effective_backlog"}:
        return _refactor_effective_task_cost(metrics, include_backlog=True), "task_effective_backlog", "energy_delay_lateness_light_backlog"
    if mode in {"task_effective_backlog_violation"}:
        return _refactor_effective_task_cost(metrics, include_backlog=True, include_violation=True), "task_effective_backlog_violation", "energy_delay_lateness_light_backlog_violation"
    # dual/cohort 的真正反馈路径仍由 legacy 逻辑控制；这里不抢它们的延迟替换机制。
    if mode in {"dual", "cohort", "cohort_complete", "legacy_dual", "legacy_cohort"}:
        return eval_cost, mode, "legacy_feedback_path_kept"
    return eval_cost, "window_original", f"unknown_feedback_score_fallback:{mode}"


def log_bo_training_feedback(fac, metrics, train_cost, mode, note):
    """把 BO 训练反馈显式写入 perf_log，避免和最终评价 cost 混淆。"""
    eval_cost = float(_safe_float(metrics.get("cost", np.nan), np.nan))
    for key, value in {
        "eval_cost": eval_cost,
        "bo_training_cost": float(train_cost),
        "bo_training_feedback_score": str(mode),
        "bo_training_feedback_note": str(note),
        "refactor_version": REFACTOR_VERSION,
    }.items():
        fac.perf_log.setdefault(key, []).append(value)


# ===============================================================
# REFACTOR V6: recent / confidence BO, CBO-lite and context ablation helpers
# ---------------------------------------------------------------
# 目标：不改变 BO 冷启动本质，而是降低 noisy window feedback 的误导。
# - recent:        只用最近窗口样本训练 GP，避免很老状态污染。
# - confidence:    对低可信窗口样本过滤/弱化，避免少完成、强偏移窗口误导 GP。
# - CBO-lite:      只使用少量窗口开始状态特征，不再把大量结果指标塞进 context。
# - CBO context ablation: full/load/util/pressure/no_cloud/no_arrival，用于判断哪些状态信息真正有用。
# ===============================================================

# v6.1: CBO-lite/context 信息扩展
# ---------------------------------------------------------------
# 原 v6 CBO-lite 只看 6 维窗口开始压力状态：arrival/backlog/util/queue/cloud-gap。
# 现在新增任务结构与上一窗口任务数，解决 RT-heavy / Batch-heavy / AI-heavy 场景中
# “压力相似但任务含义不同”的问题。
# 注意：这些特征都来自当前窗口开始前可观测信息或外部已知场景配置，不使用本窗口结束结果。
LITE_CONTEXT_FEATURE_NAMES = [
    # 原 6 维压力状态
    "arrival_rate_recent",          # 0: 最近到达率/当前估计到达强度
    "start_backlog",                # 1: 当前窗口开始积压任务数
    "start_avg_util",               # 2: 当前窗口开始平均节点利用率
    "start_max_util",               # 3: 当前窗口开始最大节点利用率
    "start_queue_total",            # 4: 当前窗口开始 ready queue 总数
    "edge_cloud_pressure_gap",       # 5: 边缘压力 - 云压力

    # 当前场景/当前时间段已知任务比例：固定 36 场景时特别有用
    "cfg_rt_prob",                  # 6
    "cfg_batch_prob",               # 7
    "cfg_ai_prob",                  # 8

    # 上一窗口真实到达任务数，按参考量归一化，反映短期任务结构偏移
    "prev_rt_arrivals_norm",        # 9
    "prev_batch_arrivals_norm",     # 10
    "prev_ai_arrivals_norm",        # 11

    # 上一窗口真实到达占比，避免只用 cfg_probs 导致 CBO 看不到随机比例偏移
    "prev_rt_arrival_ratio",        # 12
    "prev_batch_arrival_ratio",     # 13
    "prev_ai_arrival_ratio",        # 14
]
LITE_CONTEXT_BOUNDS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.0, 500.0, 1.0, 1.0, 500.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
]

# v6.1: CBO-lite context 消融配置。
# 原模式全部保留；新增 taskmix / recent_mix / pressure_taskmix / pressure_taskmix_counts 等模式。
# 推荐后续先跑：cbo-pressure, cbo-taskmix, cbo-pressure-taskmix, cbo-pressure-taskmix-counts。
LITE_CONTEXT_MODE_SPECS = {
    # 原始 full：仅 6 维压力状态，保持兼容
    "lite": {"label": "full", "indices": [0, 1, 2, 3, 4, 5]},
    "full": {"label": "full", "indices": [0, 1, 2, 3, 4, 5]},
    "load_only": {"label": "load_only", "indices": [0, 1, 4]},
    "load": {"label": "load_only", "indices": [0, 1, 4]},
    "util_only": {"label": "util_only", "indices": [2, 3]},
    "util": {"label": "util_only", "indices": [2, 3]},
    "pressure_only": {"label": "pressure_only", "indices": [1, 2, 3, 4]},
    "pressure": {"label": "pressure_only", "indices": [1, 2, 3, 4]},
    "no_cloud": {"label": "no_cloud", "indices": [0, 1, 2, 3, 4]},
    "no_arrival": {"label": "no_arrival", "indices": [1, 2, 3, 4, 5]},

    # 新增：任务结构 context
    "taskmix": {"label": "taskmix", "indices": [6, 7, 8]},
    "task_mix": {"label": "taskmix", "indices": [6, 7, 8]},
    "recent_mix": {"label": "recent_mix", "indices": [12, 13, 14]},
    "prev_mix": {"label": "recent_mix", "indices": [12, 13, 14]},
    "prev_counts": {"label": "prev_counts", "indices": [9, 10, 11]},
    "counts": {"label": "prev_counts", "indices": [9, 10, 11]},

    # 新增：压力 + 任务结构。适合验证“RT/Batch/AI比例是否能改善情景区分”。
    "pressure_taskmix": {"label": "pressure_taskmix", "indices": [1, 2, 3, 4, 6, 7, 8]},
    "pressure_task_mix": {"label": "pressure_taskmix", "indices": [1, 2, 3, 4, 6, 7, 8]},
    "taskmix_pressure": {"label": "pressure_taskmix", "indices": [1, 2, 3, 4, 6, 7, 8]},
    "pressure_recent_mix": {"label": "pressure_recent_mix", "indices": [1, 2, 3, 4, 12, 13, 14]},
    "pressure_counts": {"label": "pressure_counts", "indices": [1, 2, 3, 4, 9, 10, 11]},
    "pressure_taskmix_counts": {"label": "pressure_taskmix_counts", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]},
    "pressure_task_mix_counts": {"label": "pressure_taskmix_counts", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]},
    "ptc": {"label": "pressure_taskmix_counts", "indices": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]},

    # 新增：全量扩展 context。维度高，只建议小规模对照。
    "full_taskmix": {"label": "full_taskmix", "indices": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
    "full_taskmix_counts": {"label": "full_taskmix_counts", "indices": list(range(15))},
}


def _lite_context_indices(context_mode="lite"):
    mode = str(context_mode or "lite").strip().lower()
    spec = LITE_CONTEXT_MODE_SPECS.get(mode, LITE_CONTEXT_MODE_SPECS["lite"])
    return list(spec["indices"])


def lite_context_feature_names(context_mode="lite"):
    return [LITE_CONTEXT_FEATURE_NAMES[i] for i in _lite_context_indices(context_mode)]


def lite_context_bounds(context_mode="lite"):
    idx = _lite_context_indices(context_mode)
    lo = [LITE_CONTEXT_BOUNDS[0][i] for i in idx]
    hi = [LITE_CONTEXT_BOUNDS[1][i] for i in idx]
    return [lo, hi]


def slice_lite_context_vector(full_vec, context_mode="lite"):
    idx = _lite_context_indices(context_mode)
    return [float(full_vec[i]) for i in idx]


def _cfg_history_mode(default="all"):
    return str(getattr(CFG, "BO_HISTORY_MODE", os.environ.get("BO_HISTORY_MODE", default)) or default).strip().lower()


def _cfg_recent_window(default=80):
    try:
        return int(getattr(CFG, "BO_RECENT_WINDOW", os.environ.get("BO_RECENT_WINDOW", default)))
    except Exception:
        return int(default)


def _cfg_confidence_min(default=0.35):
    try:
        return float(getattr(CFG, "BO_CONFIDENCE_MIN", os.environ.get("BO_CONFIDENCE_MIN", default)))
    except Exception:
        return float(default)


def _cfg_confidence_min_samples(default=12):
    try:
        return int(getattr(CFG, "BO_CONFIDENCE_MIN_SAMPLES", os.environ.get("BO_CONFIDENCE_MIN_SAMPLES", default)))
    except Exception:
        return int(default)


def _cfg_cbo_history_select_mode(default="recent"):
    return str(getattr(CFG, "CBO_HISTORY_SELECT_MODE", os.environ.get("CBO_HISTORY_SELECT_MODE", default)) or default).strip().lower()


def _cfg_cbo_int(name, default):
    try:
        return int(getattr(CFG, name, os.environ.get(name, default)))
    except Exception:
        return int(default)


def _cfg_cbo_float(name, default):
    try:
        return float(getattr(CFG, name, os.environ.get(name, default)))
    except Exception:
        return float(default)


def _cfg_cbo_str(name, default):
    return str(getattr(CFG, name, os.environ.get(name, default)) or default).strip().lower()


def _node_count_backlog(nodes):
    return int(sum(len(n.ready_queue) + len(n.running_tasks) for n in nodes))


def _node_count_ready_queue(nodes):
    return int(sum(len(n.ready_queue) for n in nodes))


def _pressure_for_nodes(nodes):
    if not nodes:
        return 0.0
    vals = []
    for n in nodes:
        util = float(n.utilization())
        q = len(n.ready_queue) + len(n.running_tasks)
        denom = max(1.0, float(getattr(n, "cpu_total", 1)) / 8.0)
        vals.append(0.65 * util + 0.35 * min(1.0, q / denom))
    return float(np.mean(vals)) if vals else 0.0


def _last_perf_value(fac, key, default=0.0):
    """读取上一窗口日志值。当前窗口开始时可观测，不使用本窗口未来结果。"""
    try:
        vals = getattr(fac, "perf_log", {}).get(key, [])
        if vals:
            v = vals[-1]
            if v is not None and np.isfinite(float(v)):
                return float(v)
    except Exception:
        pass
    return float(default)


def build_lite_context_vector(fac, base_context=None):
    """构造 CBO-lite 的窗口开始状态。

    只使用当前决策前能看到的状态，不使用本窗口结束后的结果指标。
    """
    try:
        m = fac.scenario_monitor.compute_metrics(fac.current_time)
    except Exception:
        m = {}
    arrival_rate = float(m.get("arrival_rate", 0.0))
    nodes = list(getattr(fac, "nodes", []))
    utils = [float(n.utilization()) for n in nodes] if nodes else [0.0]
    backlog = float(_node_count_backlog(nodes))
    queue_total = float(_node_count_ready_queue(nodes))
    edge_nodes = [n for n in nodes if not getattr(n, "is_cloud", False)]
    cloud_nodes = [n for n in nodes if getattr(n, "is_cloud", False)]
    edge_pressure = _pressure_for_nodes(edge_nodes)
    cloud_pressure = _pressure_for_nodes(cloud_nodes)
    gap = float(np.clip(edge_pressure - cloud_pressure, -1.0, 1.0))

    # 当前配置/当前时间段的任务结构。固定 36 场景时它就是该场景的任务比例；
    # 若启用 TASK_TYPE_PROB_SCHEDULE，它会随 current_time 分段变化。
    try:
        cfg_probs = get_task_type_probs_at_time(getattr(fac, "current_time", None))
    except Exception:
        cfg_probs = _normalize_task_probs(getattr(CFG, "TASK_TYPE_PROBS", {"RT": 1/3, "Batch": 1/3, "AI": 1/3}))
    cfg_rt = float(cfg_probs.get("RT", 0.0))
    cfg_batch = float(cfg_probs.get("Batch", 0.0))
    cfg_ai = float(cfg_probs.get("AI", 0.0))

    # 上一窗口实际到达数量与占比。这里用上一轮 perf_log，避免看当前窗口结束结果。
    prev_rt = _last_perf_value(fac, "arrivals_rt", 0.0)
    prev_batch = _last_perf_value(fac, "arrivals_batch", 0.0)
    prev_ai = _last_perf_value(fac, "arrivals_ai", 0.0)
    prev_total = max(0.0, prev_rt + prev_batch + prev_ai)
    # 参考值约等于 λ*BO_INTERVAL，用于把上一窗口任务数压到 0~1。
    # 如果当前 m 里没有 arrival_rate，则回退到 CFG 当前 λ 或 1.0。
    count_ref = max(1.0, float(arrival_rate) * float(getattr(CFG, "BO_INTERVAL", 1.0)))
    if count_ref <= 1.0:
        try:
            lam, _ = fac.workload._get_lambda(getattr(fac, "current_time", 0.0))
            count_ref = max(1.0, float(lam) * float(getattr(CFG, "BO_INTERVAL", 1.0)))
        except Exception:
            count_ref = max(1.0, float(getattr(CFG, "BO_INTERVAL", 1.0)))
    prev_rt_norm = float(np.clip(prev_rt / count_ref, 0.0, 1.0))
    prev_batch_norm = float(np.clip(prev_batch / count_ref, 0.0, 1.0))
    prev_ai_norm = float(np.clip(prev_ai / count_ref, 0.0, 1.0))
    if prev_total > 0:
        prev_rt_ratio = prev_rt / prev_total
        prev_batch_ratio = prev_batch / prev_total
        prev_ai_ratio = prev_ai / prev_total
    else:
        # 第 1 轮没有上一窗口，用配置比例兜底，避免冷启动 context 全 0。
        prev_rt_ratio = cfg_rt
        prev_batch_ratio = cfg_batch
        prev_ai_ratio = cfg_ai

    return [
        float(np.clip(arrival_rate, 0.0, 5.0)),
        float(np.clip(backlog, 0.0, 500.0)),
        float(np.clip(np.mean(utils), 0.0, 1.0)),
        float(np.clip(np.max(utils), 0.0, 1.0)),
        float(np.clip(queue_total, 0.0, 500.0)),
        gap,
        float(np.clip(cfg_rt, 0.0, 1.0)),
        float(np.clip(cfg_batch, 0.0, 1.0)),
        float(np.clip(cfg_ai, 0.0, 1.0)),
        prev_rt_norm,
        prev_batch_norm,
        prev_ai_norm,
        float(np.clip(prev_rt_ratio, 0.0, 1.0)),
        float(np.clip(prev_batch_ratio, 0.0, 1.0)),
        float(np.clip(prev_ai_ratio, 0.0, 1.0)),
    ]


def build_context_for_group(fac, group_cfg, base_context=None):
    mode = str(group_cfg.get("context_mode", "legacy") or "legacy").strip().lower()
    if mode in set(LITE_CONTEXT_MODE_SPECS.keys()) | {"state_lite", "cbo_lite"}:
        full_vec = build_lite_context_vector(fac, base_context=base_context)
        if mode in {"state_lite", "cbo_lite"}:
            mode = "lite"
        return slice_lite_context_vector(full_vec, context_mode=mode)
    return base_context


def compute_feedback_confidence(metrics, group_cfg=None):
    """估计当前窗口 feedback 可信度。

    它不是最终评价指标，只是告诉 BO：这一轮 observation 是否容易被随机到达、
    低完成率、强任务比例偏移污染。
    """
    arrivals = max(0.0, _safe_float(metrics.get("arrivals_total", metrics.get("arrivals", 0.0)), 0.0))
    completed = max(0.0, _safe_float(metrics.get("completed_total", metrics.get("task_count", 0.0)), 0.0))
    unfinished = max(0.0, _safe_float(metrics.get("unfinished_end", metrics.get("backlog", 0.0)), 0.0))
    comp_ratio = completed / max(1.0, arrivals)

    # 样本数量越多，单窗口均值越稳。阈值不宜太高，避免高负载窗口被过度丢弃。
    count_conf = min(1.0, math.log1p(arrivals) / math.log1p(float(getattr(CFG, "BO_CONFIDENCE_TASK_REF", 60.0))))
    complete_conf = min(1.0, comp_ratio / max(1e-9, float(getattr(CFG, "BO_CONFIDENCE_COMPLETION_REF", 0.65))))
    unfinished_ratio = unfinished / max(1.0, arrivals + unfinished)
    unfinished_conf = max(0.20, 1.0 - 0.60 * min(1.0, unfinished_ratio))

    ref = _normalize_task_probs(getattr(CFG, "TASK_TYPE_PROBS", {"RT": 1/3, "Batch": 1/3, "AI": 1/3}))
    actual = {
        "RT": _safe_float(metrics.get("rt_arrival_ratio", 0.0), 0.0),
        "Batch": _safe_float(metrics.get("batch_arrival_ratio", 0.0), 0.0),
        "AI": _safe_float(metrics.get("ai_arrival_ratio", 0.0), 0.0),
    }
    # L1 偏移最大约为 2；这里保留温和惩罚，避免单窗口比例抖动完全支配。
    mix_l1 = sum(abs(float(actual[k]) - float(ref.get(k, 0.0))) for k in ["RT", "Batch", "AI"])
    mix_conf = max(0.25, 1.0 - 0.50 * min(1.0, mix_l1))

    zero_penalty = _safe_float(metrics.get("zero_completion_penalty", 0.0), 0.0)
    zero_conf = 0.25 if zero_penalty > 0 else 1.0

    conf = (
        0.30 * count_conf
        + 0.30 * complete_conf
        + 0.20 * unfinished_conf
        + 0.15 * mix_conf
        + 0.05 * zero_conf
    )
    return float(np.clip(conf, 0.05, 1.0)), {
        "confidence_count": float(count_conf),
        "confidence_completion": float(complete_conf),
        "confidence_unfinished": float(unfinished_conf),
        "confidence_mix": float(mix_conf),
        "confidence_zero": float(zero_conf),
        "confidence_completion_ratio": float(comp_ratio),
        "confidence_unfinished_ratio": float(unfinished_ratio),
        "confidence_task_mix_l1": float(mix_l1),
    }


def log_feedback_confidence(fac, confidence, parts, group_cfg=None):
    data = {"feedback_confidence": float(confidence)}
    data.update(parts or {})
    data["bo_history_mode"] = str(group_cfg.get("history_mode", _cfg_history_mode())) if group_cfg else _cfg_history_mode()
    data["bo_recent_window"] = int(group_cfg.get("recent_window", _cfg_recent_window())) if group_cfg else _cfg_recent_window()
    data["effective_history_mode"] = data["bo_history_mode"]
    data["effective_recent_window"] = data["bo_recent_window"]
    data["history_override_source"] = str(group_cfg.get("history_override_source", "method_default")) if group_cfg else "global_default"
    data["bo_train_sample_count"] = int(getattr(getattr(fac, "agent", None), "last_debug_info", {}).get("training_sample_count", -1)) if getattr(fac, "agent", None) is not None else -1
    data["bo_confidence_min"] = float(group_cfg.get("confidence_min", _cfg_confidence_min())) if group_cfg else _cfg_confidence_min()
    for k, v in data.items():
        fac.perf_log.setdefault(k, []).append(v)


def configure_refactor_agent(agent, group_cfg):
    if agent is None:
        return
    if group_cfg is None:
        group_cfg = {}
    agent.history_mode = str(group_cfg.get("history_mode", _cfg_history_mode("all"))).strip().lower()
    agent.recent_window = int(group_cfg.get("recent_window", _cfg_recent_window()))
    agent.confidence_min = float(group_cfg.get("confidence_min", _cfg_confidence_min()))
    agent.confidence_min_samples = int(group_cfg.get("confidence_min_samples", _cfg_confidence_min_samples()))
    agent.context_mode = str(group_cfg.get("context_mode", "legacy"))
    agent.history_override_source = str(group_cfg.get("history_override_source", "method_default"))
    agent.is_cbo_stability_enabled = bool(_is_cbo_method_key(str(group_cfg.get("group_key", "")), group_cfg))
    agent.cbo_history_select_mode = str(group_cfg.get("cbo_history_select_mode", _cfg_cbo_history_select_mode("recent")) if agent.is_cbo_stability_enabled else "recent").strip().lower()
    agent.cbo_context_k = int(group_cfg.get("cbo_context_k", _cfg_cbo_int("CBO_CONTEXT_K", 50)))
    agent.cbo_elite_k = int(group_cfg.get("cbo_elite_k", _cfg_cbo_int("CBO_ELITE_K", 20)))
    agent.cbo_diverse_k = int(group_cfg.get("cbo_diverse_k", _cfg_cbo_int("CBO_DIVERSE_K", 20)))
    agent.cbo_robust_score_mode = str(group_cfg.get("cbo_robust_score_mode", _cfg_cbo_str("CBO_ROBUST_SCORE_MODE", "none")) if agent.is_cbo_stability_enabled else "none").strip().lower()
    agent.cbo_robust_std_weight = float(group_cfg.get("cbo_robust_std_weight", _cfg_cbo_float("CBO_ROBUST_STD_WEIGHT", 0.5)))
    agent.cbo_theta_merge_eps = float(group_cfg.get("cbo_theta_merge_eps", _cfg_cbo_float("CBO_THETA_MERGE_EPS", 0.05)))
    agent.cbo_context_sim_threshold = float(group_cfg.get("cbo_context_sim_threshold", _cfg_cbo_float("CBO_CONTEXT_SIM_THRESHOLD", 0.0)))
    agent.cbo_tr_mode = str(group_cfg.get("cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) if agent.is_cbo_stability_enabled else "off").strip().lower()
    agent.cbo_tr_anchor_mode = str(group_cfg.get("cbo_tr_anchor_mode", _cfg_cbo_str("CBO_TR_ANCHOR_MODE", "posterior_mean"))).strip().lower()
    agent.cbo_tr_radius_min = float(group_cfg.get("cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))))
    agent.cbo_tr_radius_max = float(group_cfg.get("cbo_tr_radius_max", _cfg_cbo_float("CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35))))
    agent.cbo_tr_grow = float(group_cfg.get("cbo_tr_grow", _cfg_cbo_float("CBO_TR_GROW", getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15))))
    agent.cbo_tr_shrink = float(group_cfg.get("cbo_tr_shrink", _cfg_cbo_float("CBO_TR_SHRINK", getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92))))
    agent.cbo_tr_update_mode = str(group_cfg.get("cbo_tr_update_mode", _cfg_cbo_str("CBO_TR_UPDATE_MODE", "best_so_far"))).strip().lower()
    agent.cbo_tr_compare_window = int(group_cfg.get("cbo_tr_compare_window", _cfg_cbo_int("CBO_TR_COMPARE_WINDOW", 30)))
    agent.cbo_tr_baseline_window = int(group_cfg.get("cbo_tr_baseline_window", _cfg_cbo_int("CBO_TR_BASELINE_WINDOW", 60)))
    agent.cbo_tr_improve_pct = float(group_cfg.get("cbo_tr_improve_pct", _cfg_cbo_float("CBO_TR_IMPROVE_PCT", 0.015)))
    agent.cbo_tr_worsen_pct = float(group_cfg.get("cbo_tr_worsen_pct", _cfg_cbo_float("CBO_TR_WORSEN_PCT", 0.03)))
    agent.cbo_tr_deadband_pct = float(group_cfg.get("cbo_tr_deadband_pct", _cfg_cbo_float("CBO_TR_DEADBAND_PCT", 0.01)))
    agent.cbo_tr_update_patience = int(group_cfg.get("cbo_tr_update_patience", _cfg_cbo_int("CBO_TR_UPDATE_PATIENCE", 2)))
    agent.cbo_tr_update_patience_count = int(getattr(agent, "cbo_tr_update_patience_count", 0))
    agent.cbo_tr_update_last_signal = str(getattr(agent, "cbo_tr_update_last_signal", ""))
    agent.cbo_tr_cost_history = list(getattr(agent, "cbo_tr_cost_history", []))
    agent.cbo_robust_incumbent_mode = str(group_cfg.get("cbo_robust_incumbent_mode", _cfg_cbo_str("CBO_ROBUST_INCUMBENT_MODE", "off")) if agent.is_cbo_stability_enabled else "off").strip().lower()
    agent.cbo_macro_gate_mode = str(group_cfg.get("cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")) if agent.is_cbo_stability_enabled else "off").strip().lower()
    agent.cbo_macro_k = int(group_cfg.get("cbo_macro_k", _cfg_cbo_int("CBO_MACRO_K", 100)))
    agent.cbo_macro_total_scale = str(group_cfg.get("cbo_macro_total_scale", getattr(CFG, "CBO_MACRO_TOTAL_SCALE", "auto")) or "auto")
    agent.cbo_macro_lengthscale_total = float(group_cfg.get("cbo_macro_lengthscale_total", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_TOTAL", 1.0)))
    agent.cbo_macro_lengthscale_rt = float(group_cfg.get("cbo_macro_lengthscale_rt", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_RT", 0.15)))
    agent.cbo_macro_lengthscale_batch = float(group_cfg.get("cbo_macro_lengthscale_batch", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_BATCH", 0.15)))
    agent.cbo_macro_alpha = float(group_cfg.get("cbo_macro_alpha", _cfg_cbo_float("CBO_MACRO_ALPHA", 1.0)))
    agent.cbo_dump_candidates = bool(group_cfg.get("cbo_dump_candidates", bool(getattr(CFG, "CBO_DUMP_CANDIDATES", False))))
    agent.cbo_dump_candidates_every = int(group_cfg.get("cbo_dump_candidates_every", _cfg_cbo_int("CBO_DUMP_CANDIDATES_EVERY", 20)))
    agent.cbo_dump_candidates_topn = int(group_cfg.get("cbo_dump_candidates_topn", _cfg_cbo_int("CBO_DUMP_CANDIDATES_TOPN", 30)))
    agent.cbo_select_mode = str(group_cfg.get("cbo_select_mode", _cfg_cbo_str("CBO_SELECT_MODE", "greedy"))).strip().lower()
    agent.cbo_topk = int(group_cfg.get("cbo_topk", _cfg_cbo_int("CBO_TOPK", 5)))
    agent.cbo_select_temperature = float(group_cfg.get("cbo_select_temperature", _cfg_cbo_float("CBO_SELECT_TEMPERATURE", 0.20)))
    agent.cbo_epsilon = float(group_cfg.get("cbo_epsilon", _cfg_cbo_float("CBO_EPSILON", 0.10)))
    agent.cbo_acq_beta = float(group_cfg.get("cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0)))
    agent.cbo_acq_beta_mode = str(group_cfg.get("cbo_acq_beta_mode", _cfg_cbo_str("CBO_ACQ_BETA_MODE", "fixed"))).strip().lower()
    agent.cbo_beta_min = float(group_cfg.get("cbo_beta_min", _cfg_cbo_float("CBO_BETA_MIN", 0.1)))
    agent.cbo_beta_max = float(group_cfg.get("cbo_beta_max", _cfg_cbo_float("CBO_BETA_MAX", 2.0)))
    agent.cbo_radius_beta_power = float(group_cfg.get("cbo_radius_beta_power", _cfg_cbo_float("CBO_RADIUS_BETA_POWER", 1.0)))
    agent.cbo_radius_stable_rebound_pct = float(group_cfg.get("cbo_radius_stable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_STABLE_REBOUND_PCT", 0.02)))
    agent.cbo_radius_unstable_rebound_pct = float(group_cfg.get("cbo_radius_unstable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04)))
    agent.cbo_radius_surprise_boost_threshold = float(group_cfg.get("cbo_radius_surprise_boost_threshold", _cfg_cbo_float("CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0)))
    agent.cbo_radius_beta_boost = float(group_cfg.get("cbo_radius_beta_boost", _cfg_cbo_float("CBO_RADIUS_BETA_BOOST", 1.5)))
    agent.cbo_radius_beta_cap = float(group_cfg.get("cbo_radius_beta_cap", _cfg_cbo_float("CBO_RADIUS_BETA_CAP", 3.0)))
    agent.cbo_service_guard_mode = str(group_cfg.get("cbo_service_guard_mode", _cfg_cbo_str("CBO_SERVICE_GUARD_MODE", "off"))).strip().lower()
    agent.cbo_service_guard_delay_pct = float(group_cfg.get("cbo_service_guard_delay_pct", _cfg_cbo_float("CBO_SERVICE_GUARD_DELAY_PCT", 0.03)))
    agent.cbo_service_guard_backlog_pct = float(group_cfg.get("cbo_service_guard_backlog_pct", _cfg_cbo_float("CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03)))
    agent.cbo_surprise_window = int(group_cfg.get("cbo_surprise_window", _cfg_cbo_int("CBO_SURPRISE_WINDOW", 10)))
    agent.cbo_surprise_z_threshold = float(group_cfg.get("cbo_surprise_z_threshold", _cfg_cbo_float("CBO_SURPRISE_Z_THRESHOLD", 2.0)))
    agent.cbo_surprise_cost_gap_pct = float(group_cfg.get("cbo_surprise_cost_gap_pct", _cfg_cbo_float("CBO_SURPRISE_COST_GAP_PCT", 0.03)))
    agent.cbo_sigma_floor = float(group_cfg.get("cbo_sigma_floor", _cfg_cbo_float("CBO_SIGMA_FLOOR", 1e-6)))
    agent.cbo_radius_reset = float(group_cfg.get("cbo_radius_reset", _cfg_cbo_float("CBO_RADIUS_RESET", 0.12)))
    agent.cbo_radius_min_stuck_rounds = int(group_cfg.get("cbo_radius_min_stuck_rounds", _cfg_cbo_int("CBO_RADIUS_MIN_STUCK_ROUNDS", 10)))
    agent.cbo_rebound_window = int(group_cfg.get("cbo_rebound_window", _cfg_cbo_int("CBO_REBOUND_WINDOW", 20)))
    agent.cbo_rebound_threshold_pct = float(group_cfg.get("cbo_rebound_threshold_pct", _cfg_cbo_float("CBO_REBOUND_THRESHOLD_PCT", 0.03)))
    agent.cbo_selection_cooldown = int(group_cfg.get("cbo_selection_cooldown", _cfg_cbo_int("CBO_SELECTION_COOLDOWN", 5)))
    agent.cbo_condition_anchor_switch = str(group_cfg.get("cbo_condition_anchor_switch", _cfg_cbo_str("CBO_CONDITION_ANCHOR_SWITCH", "context_best"))).strip().lower()
    agent.cbo_force_explore_countdown = int(getattr(agent, "cbo_force_explore_countdown", 0))
    agent.cbo_radius_min_stuck_count = int(getattr(agent, "cbo_radius_min_stuck_count", 0))
    agent.cbo_surprise_history = list(getattr(agent, "cbo_surprise_history", []))
    if agent.cbo_tr_mode != "off":
        agent.use_trust_region = True
    agent.trust_radius = float(group_cfg.get("cbo_tr_radius_init", getattr(CFG, "TRUST_RADIUS_INIT", getattr(agent, "trust_radius", 0.10))))
    agent.cbo_tr_success_count = int(getattr(agent, "cbo_tr_success_count", 0))
    agent.cbo_tr_failure_count = int(getattr(agent, "cbo_tr_failure_count", 0))
    agent.cbo_tr_update_reason = str(getattr(agent, "cbo_tr_update_reason", "init"))


def agent_tell_with_feedback_meta(agent, theta, cost, state=None, context=None, metrics=None, bo_iter=None, group_key=None, group_cfg=None, confidence=None, parts=None):
    agent.tell(theta, cost, state=state, context=context)
    try:
        rec = None
        for r in reversed(list(getattr(agent, "local_recent", []))):
            if isinstance(r, dict):
                rec = r
                break
        if rec is not None:
            rec["feedback_confidence"] = float(confidence if confidence is not None else 1.0)
            rec["bo_iter"] = int(bo_iter) if bo_iter is not None else None
            rec["group_key"] = str(group_key) if group_key is not None else None
            rec["history_mode"] = str(getattr(agent, "history_mode", _cfg_history_mode()))
            rec["feedback_metrics_meta"] = dict(parts or {})
            if isinstance(metrics, dict):
                rec["metrics"] = dict(metrics)
                for macro_key in [
                    "arrivals_total", "arrivals_rt", "arrivals_batch", "arrivals_ai",
                    "rt_arrival_ratio", "batch_arrival_ratio", "ai_arrival_ratio",
                    "completed_total", "task_count", "unfinished_end", "backlog",
                ]:
                    if macro_key in metrics:
                        rec[macro_key] = metrics.get(macro_key)
            try:
                theta_norm = agent._normalize_theta(theta)
                prev_best = getattr(agent, "prev_best", None)
                prev_best_value = getattr(agent, "prev_best_value", None)
                if (
                    prev_best is not None
                    and prev_best_value is not None
                    and np.allclose(np.array(prev_best, dtype=float), np.array(theta_norm, dtype=float), atol=1e-9, rtol=1e-9)
                    and abs(float(prev_best_value) + float(cost)) <= 1e-9 * max(1.0, abs(float(cost)))
                ):
                    agent.prev_best_iter = int(bo_iter) + 1 if bo_iter is not None else None
            except Exception:
                pass
    except Exception:
        pass


_ORIG_AGENT_COLLECT_SAMPLES = FederatedBOAgent._collect_samples


def _cbo_record_cost(rec):
    try:
        return -float(rec.get("y", np.nan))
    except Exception:
        return np.nan


def _cbo_theta_norm(agent, theta):
    theta = np.asarray(agent._normalize_theta(theta), dtype=float)
    low = np.asarray(agent.bounds[0].tolist(), dtype=float)
    high = np.asarray(agent.bounds[1].tolist(), dtype=float)
    denom = np.maximum(1e-12, high - low)
    return np.clip((theta - low) / denom, 0.0, 1.0)


def _cbo_context_similarity(agent, context, rec):
    if context is None or rec.get("context") is None:
        return 1.0 if context is None else 0.0
    try:
        return float(agent._context_similarity(context, rec.get("context")))
    except Exception:
        return 0.0


def _cbo_all_records(agent):
    records = []
    for bucket in getattr(agent, "local_archive", {}).values():
        records.extend([agent._unpack_sample(s) for s in bucket])
    records.extend([agent._unpack_sample(s) for s in getattr(agent, "local_recent", [])])
    return records


def _cbo_rec_value(rec, keys, default=np.nan):
    if rec is None:
        return default
    metrics = rec.get("metrics") if isinstance(rec, dict) else None
    for key in keys:
        try:
            if isinstance(rec, dict) and key in rec and rec.get(key) is not None:
                return float(rec.get(key))
            if isinstance(metrics, dict) and key in metrics and metrics.get(key) is not None:
                return float(metrics.get(key))
        except Exception:
            continue
    return default


def _cbo_macro_total_scale(agent, records):
    mode = str(getattr(agent, "cbo_macro_total_scale", getattr(CFG, "CBO_MACRO_TOTAL_SCALE", "auto")) or "auto").strip().lower()
    if mode != "auto":
        try:
            return max(1e-12, float(mode))
        except Exception:
            return 1.0
    totals = []
    for rec in records or []:
        macro = _cbo_macro_context_from_record(agent, rec, records=None, scale=1.0, allow_auto=False)
        total_raw = macro.get("total_arrivals_raw")
        if np.isfinite(total_raw) and total_raw > 0:
            totals.append(float(total_raw))
    if not totals:
        return 1.0
    vals = np.asarray(totals, dtype=float)
    return max(1.0, float(np.nanpercentile(vals, 90)))


def _cbo_macro_context_from_record(agent, rec, records=None, scale=None, allow_auto=True):
    if scale is None:
        scale = _cbo_macro_total_scale(agent, records or []) if allow_auto else 1.0
    rt = _cbo_rec_value(rec, ["arrivals_rt", "Prev_RT_Arrivals", "window_rt_arrivals", "rt_arrivals"], np.nan)
    batch = _cbo_rec_value(rec, ["arrivals_batch", "Prev_Batch_Arrivals", "window_batch_arrivals", "batch_arrivals"], np.nan)
    ai = _cbo_rec_value(rec, ["arrivals_ai", "Prev_AI_Arrivals", "window_ai_arrivals", "ai_arrivals"], np.nan)
    total = _cbo_rec_value(rec, ["arrivals_total", "task_count", "generated_total", "completed_total"], np.nan)
    if not np.isfinite(total):
        if np.isfinite(rt) or np.isfinite(batch) or np.isfinite(ai):
            total = float(np.nan_to_num(rt, nan=0.0) + np.nan_to_num(batch, nan=0.0) + np.nan_to_num(ai, nan=0.0))
    rt_ratio = _cbo_rec_value(rec, ["rt_arrival_ratio", "prev_rt_arrival_ratio", "RT_Ratio"], np.nan)
    batch_ratio = _cbo_rec_value(rec, ["batch_arrival_ratio", "prev_batch_arrival_ratio", "Batch_Ratio"], np.nan)
    if np.isfinite(total) and total > 0:
        if not np.isfinite(rt_ratio) and np.isfinite(rt):
            rt_ratio = float(rt) / max(1e-12, float(total))
        if not np.isfinite(batch_ratio) and np.isfinite(batch):
            batch_ratio = float(batch) / max(1e-12, float(total))
    elif isinstance(rec, dict) and rec.get("context") is not None and getattr(agent, "use_context", False):
        try:
            names = lite_context_feature_names(getattr(agent, "context_mode", "pressure_taskmix_counts"))
            ctx = list(agent._normalize_context(rec.get("context")))
            name_to_val = {str(n): float(ctx[i]) for i, n in enumerate(names[:len(ctx)])}
            rt_norm = name_to_val.get("prev_rt_arrivals_norm", np.nan)
            batch_norm = name_to_val.get("prev_batch_arrivals_norm", np.nan)
            ai_norm = name_to_val.get("prev_ai_arrivals_norm", np.nan)
            norm_total = float(np.nan_to_num(rt_norm, nan=0.0) + np.nan_to_num(batch_norm, nan=0.0) + np.nan_to_num(ai_norm, nan=0.0))
            if norm_total > 0:
                total = norm_total
                rt_ratio = float(np.nan_to_num(rt_norm, nan=0.0)) / norm_total
                batch_ratio = float(np.nan_to_num(batch_norm, nan=0.0)) / norm_total
            rt_ratio = name_to_val.get("prev_rt_arrival_ratio", rt_ratio)
            batch_ratio = name_to_val.get("prev_batch_arrival_ratio", batch_ratio)
        except Exception:
            pass
    total_norm = float(total) / max(1e-12, float(scale)) if np.isfinite(total) else np.nan
    return {
        "total_arrivals_raw": float(total) if np.isfinite(total) else np.nan,
        "total_arrivals_norm": float(total_norm) if np.isfinite(total_norm) else np.nan,
        "rt_ratio": float(rt_ratio) if np.isfinite(rt_ratio) else np.nan,
        "batch_ratio": float(batch_ratio) if np.isfinite(batch_ratio) else np.nan,
    }


def _cbo_macro_context_from_context(agent, context, records=None, scale=None):
    pseudo = {"context": context}
    if context is not None and getattr(agent, "use_context", False):
        try:
            names = lite_context_feature_names(getattr(agent, "context_mode", "pressure_taskmix_counts"))
            ctx = list(agent._normalize_context(context))
            for i, name in enumerate(names[:len(ctx)]):
                pseudo[str(name)] = float(ctx[i])
        except Exception:
            pass
    return _cbo_macro_context_from_record(agent, pseudo, records=records, scale=scale)


def _cbo_macro_similarity(agent, macro_a, macro_b):
    try:
        vals = []
        for key, ls_name, default in [
            ("total_arrivals_norm", "cbo_macro_lengthscale_total", 1.0),
            ("rt_ratio", "cbo_macro_lengthscale_rt", 0.15),
            ("batch_ratio", "cbo_macro_lengthscale_batch", 0.15),
        ]:
            a = float(macro_a.get(key, np.nan))
            b = float(macro_b.get(key, np.nan))
            if not np.isfinite(a) or not np.isfinite(b):
                vals.append(1e6)
            else:
                ls = max(1e-12, float(getattr(agent, ls_name, default)))
                vals.append(((a - b) / ls) ** 2)
        sim = float(np.exp(-0.5 * float(np.sum(vals))))
        alpha = max(1e-12, float(getattr(agent, "cbo_macro_alpha", 1.0)))
        return float(sim ** alpha)
    except Exception:
        return 0.0


def _cbo_record_identity(agent, rec):
    try:
        theta = _cbo_theta_norm(agent, rec.get("theta", []))
        theta_key = tuple(np.round(theta, 8).tolist())
    except Exception:
        theta_key = tuple()
    try:
        ctx = np.asarray(agent._normalize_context(rec.get("context")), dtype=float) if rec.get("context") is not None and getattr(agent, "use_context", False) else np.zeros(0)
        ctx_key = tuple(np.round(ctx, 8).tolist())
    except Exception:
        ctx_key = tuple()
    return (
        int(rec.get("bo_iter", -1) if rec.get("bo_iter", -1) is not None else -1),
        str(rec.get("group_key", "")),
        theta_key,
        ctx_key,
    )


def _cbo_cluster_records(agent, records, context=None):
    eps = max(1e-9, float(getattr(agent, "cbo_theta_merge_eps", _cfg_cbo_float("CBO_THETA_MERGE_EPS", 0.05))))
    mode = str(getattr(agent, "cbo_robust_score_mode", _cfg_cbo_str("CBO_ROBUST_SCORE_MODE", "none")) or "none").lower()
    std_weight = float(getattr(agent, "cbo_robust_std_weight", _cfg_cbo_float("CBO_ROBUST_STD_WEIGHT", 0.5)))
    clusters = []
    for idx, rec in enumerate(records):
        theta = rec.get("theta")
        if theta is None:
            continue
        tn = _cbo_theta_norm(agent, theta)
        best_j, best_d = None, None
        for j, cluster in enumerate(clusters):
            d = float(np.linalg.norm(tn - cluster["theta_norm"]))
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        if best_j is not None and best_d <= eps:
            clusters[best_j]["records"].append(rec)
            n = len(clusters[best_j]["records"])
            clusters[best_j]["theta_norm"] = ((clusters[best_j]["theta_norm"] * (n - 1)) + tn) / n
        else:
            clusters.append({"theta_norm": tn, "records": [rec]})
    out = []
    for cluster in clusters:
        recs = cluster["records"]
        costs = np.asarray([_cbo_record_cost(r) for r in recs], dtype=float)
        costs = costs[np.isfinite(costs)]
        if costs.size == 0:
            continue
        sims = np.asarray([_cbo_context_similarity(agent, context, r) for r in recs], dtype=float)
        sims = np.clip(np.nan_to_num(sims, nan=0.0), 0.0, None)
        if float(np.sum(sims)) <= 1e-12:
            sims = np.ones(len(recs), dtype=float)
        cost_all = np.asarray([_cbo_record_cost(r) for r in recs], dtype=float)
        valid = np.isfinite(cost_all)
        cost_valid = cost_all[valid]
        sim_valid = sims[valid]
        sim_valid = sim_valid / max(1e-12, float(np.sum(sim_valid)))
        cw_mean = float(np.sum(sim_valid * cost_valid))
        cw_std = float(np.sqrt(np.sum(sim_valid * (cost_valid - cw_mean) ** 2)))
        mean_cost = float(np.mean(cost_valid))
        std_cost = float(np.std(cost_valid))
        if mode == "mean":
            robust_score = mean_cost
        elif mode == "mean_std":
            robust_score = mean_cost + std_weight * std_cost
        elif mode == "context_weighted_mean_std":
            robust_score = cw_mean + std_weight * cw_std
        else:
            robust_score = np.nan
        best_rec = min(recs, key=lambda r: _cbo_record_cost(r) if np.isfinite(_cbo_record_cost(r)) else float("inf"))
        rep_theta = list(best_rec.get("theta", []))
        contexts = [np.asarray(agent._normalize_context(r.get("context")), dtype=float) for r in recs if r.get("context") is not None and getattr(agent, "use_context", False)]
        mean_context = np.mean(np.vstack(contexts), axis=0).tolist() if contexts else None
        out.append({
            "records": recs,
            "eval_count": int(len(recs)),
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "min_cost": float(np.min(cost_valid)),
            "max_cost": float(np.max(cost_valid)),
            "recent_mean_cost": float(np.mean(cost_valid[-min(len(cost_valid), 20):])),
            "context_weighted_mean_cost": cw_mean,
            "context_weighted_std_cost": cw_std,
            "robust_score": float(robust_score) if np.isfinite(robust_score) else np.nan,
            "representative_theta": rep_theta,
            "mean_context": mean_context,
            "context_similarity_to_current": _cbo_context_similarity(agent, context, {"context": mean_context}) if mean_context is not None else (1.0 if context is None else 0.0),
        })
    return out


def _compute_robust_incumbent(self, context=None):
    mode = str(getattr(self, "cbo_robust_score_mode", _cfg_cbo_str("CBO_ROBUST_SCORE_MODE", "none")) or "none").lower()
    if mode == "none":
        return None, {"robust_incumbent_available": False, "robust_incumbent_reason": "robust_score_mode_none"}
    macro_mode = str(getattr(self, "cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")) or "off").strip().lower()
    macro_records = list(getattr(self, "_last_macro_pool_records", []) or [])
    source_pool = "macro_pool" if macro_mode == "hierarchical" and macro_records else "all_records"
    records = macro_records if source_pool == "macro_pool" else _cbo_all_records(self)
    clusters = _cbo_cluster_records(self, records, context=context)
    clusters = [c for c in clusters if np.isfinite(c.get("robust_score", np.nan))]
    if not clusters:
        return None, {"robust_incumbent_available": False, "robust_incumbent_reason": "no_valid_clusters"}
    threshold = float(getattr(self, "cbo_context_sim_threshold", _cfg_cbo_float("CBO_CONTEXT_SIM_THRESHOLD", 0.0)))
    if threshold > 0:
        clusters = [c for c in clusters if float(c.get("context_similarity_to_current", 0.0)) >= threshold]
    if not clusters:
        return None, {"robust_incumbent_available": False, "robust_incumbent_reason": "context_threshold_filtered_all"}
    multi = [c for c in clusters if int(c.get("eval_count", 0)) >= 2]
    pool = multi if multi else clusters
    best = min(pool, key=lambda c: float(c.get("robust_score", float("inf"))))
    info = {
        "robust_incumbent_available": True,
        "robust_incumbent_score": float(best.get("robust_score", np.nan)),
        "robust_incumbent_eval_count": int(best.get("eval_count", 0)),
        "robust_incumbent_mean_cost": float(best.get("mean_cost", np.nan)),
        "robust_incumbent_std_cost": float(best.get("std_cost", np.nan)),
        "robust_incumbent_context_similarity": float(best.get("context_similarity_to_current", np.nan)),
        "robust_incumbent_theta": list(best.get("representative_theta", [])),
        "robust_incumbent_used": False,
        "robust_incumbent_reason": "recommend_only",
        "robust_incumbent_source_pool": source_pool,
    }
    return list(best.get("representative_theta", [])), info


FederatedBOAgent._compute_robust_incumbent = _compute_robust_incumbent


def _refactor_collect_samples(self, state=None):
    records = list(_ORIG_AGENT_COLLECT_SAMPLES(self, state=state))
    if not records:
        return records
    context = getattr(self, "_active_context", None)
    select_mode = str(getattr(self, "cbo_history_select_mode", _cfg_cbo_history_select_mode("recent")) or "recent").strip().lower()
    recent_window = max(2, int(getattr(self, "recent_window", _cfg_recent_window())))

    def set_debug(pool, recent_count=None, context_count=0, elite_count=0, diverse_count=0, sims=None, elite=None,
                  macro_count=0, macro_sims=None, macro_current=None, macro_pool_sims=None,
                  selected_macro_sims=None, selected_from_macro_pool_count=None, selected_outside_macro_pool_count=0,
                  macro_gate_fallback_used=False, macro_gate_fallback_reason="",
                  context_selection_source_pool="all_records", elite_selection_source_pool="all_records",
                  tr_anchor_source_pool="all_records"):
        sims = list(sims or [])
        macro_sims = list(macro_sims or [])
        macro_pool_sims = list(macro_pool_sims or [])
        selected_macro_sims = list(selected_macro_sims if selected_macro_sims is not None else macro_sims[:int(macro_count)])
        elite = elite or {}
        macro_current = macro_current or {}
        if selected_from_macro_pool_count is None:
            selected_from_macro_pool_count = int(macro_count)
        self.last_history_debug = {
            "history_select_mode": select_mode,
            "effective_history_mode": str(getattr(self, "history_mode", _cfg_history_mode("all"))),
            "effective_recent_window": int(recent_window),
            "selected_recent_count": int(recent_count if recent_count is not None else min(len(getattr(self, "local_recent", [])), recent_window)),
            "selected_macro_count": int(macro_count),
            "selected_context_count": int(context_count),
            "selected_elite_count": int(elite_count),
            "selected_diverse_count": int(diverse_count),
            "selected_total_count": int(len(pool)),
            "cbo_macro_gate_mode": str(getattr(self, "cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off"))),
            "macro_total_arrivals_norm": macro_current.get("total_arrivals_norm"),
            "macro_rt_ratio": macro_current.get("rt_ratio"),
            "macro_batch_ratio": macro_current.get("batch_ratio"),
            "macro_similarity_max": float(max(macro_sims)) if macro_sims else np.nan,
            "macro_similarity_mean": float(np.mean(macro_sims)) if macro_sims else np.nan,
            "macro_similarity_p50": float(np.percentile(macro_sims, 50)) if macro_sims else np.nan,
            "macro_similarity_p90": float(np.percentile(macro_sims, 90)) if macro_sims else np.nan,
            "selected_macro_mean_similarity": float(np.mean(selected_macro_sims)) if selected_macro_sims else np.nan,
            "selected_macro_min_similarity": float(np.min(selected_macro_sims)) if selected_macro_sims else np.nan,
            "selected_macro_max_similarity": float(np.max(selected_macro_sims)) if selected_macro_sims else np.nan,
            "macro_pool_count": int(len(macro_pool_sims)),
            "macro_pool_mean_similarity": float(np.mean(macro_pool_sims)) if macro_pool_sims else np.nan,
            "macro_pool_min_similarity": float(np.min(macro_pool_sims)) if macro_pool_sims else np.nan,
            "macro_pool_max_similarity": float(np.max(macro_pool_sims)) if macro_pool_sims else np.nan,
            "macro_pool_p50_similarity": float(np.percentile(macro_pool_sims, 50)) if macro_pool_sims else np.nan,
            "macro_pool_p90_similarity": float(np.percentile(macro_pool_sims, 90)) if macro_pool_sims else np.nan,
            "selected_from_macro_pool_count": int(selected_from_macro_pool_count),
            "selected_outside_macro_pool_count": int(selected_outside_macro_pool_count),
            "macro_gate_fallback_used": bool(macro_gate_fallback_used),
            "macro_gate_fallback_reason": str(macro_gate_fallback_reason),
            "context_selection_source_pool": str(context_selection_source_pool),
            "elite_selection_source_pool": str(elite_selection_source_pool),
            "tr_anchor_source_pool": str(tr_anchor_source_pool),
            "macro_k": int(getattr(self, "cbo_macro_k", _cfg_cbo_int("CBO_MACRO_K", 100))),
            "macro_lengthscale_total": float(getattr(self, "cbo_macro_lengthscale_total", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_TOTAL", 1.0))),
            "macro_lengthscale_rt": float(getattr(self, "cbo_macro_lengthscale_rt", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_RT", 0.15))),
            "macro_lengthscale_batch": float(getattr(self, "cbo_macro_lengthscale_batch", _cfg_cbo_float("CBO_MACRO_LENGTHSCALE_BATCH", 0.15))),
            "context_similarity_max": float(max(sims)) if sims else np.nan,
            "context_similarity_mean": float(np.mean(sims)) if sims else np.nan,
            "elite_best_robust_score": elite.get("robust_score"),
            "elite_best_eval_count": elite.get("eval_count"),
            "elite_best_mean_cost": elite.get("mean_cost"),
            "elite_best_std_cost": elite.get("std_cost"),
        }

    mode = str(getattr(self, "history_mode", _cfg_history_mode("all")) or "all").strip().lower()
    if mode in {"all", "legacy", "none"}:
        set_debug(records)
        return records

    # 按插入顺序近似时间顺序。local_recent 本身已经是时间顺序，archive 在前，recent 在后。
    min_keep = max(2, int(getattr(self, "confidence_min_samples", _cfg_confidence_min_samples())))
    min_conf = float(getattr(self, "confidence_min", _cfg_confidence_min()))

    macro_mode = str(getattr(self, "cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")) or "off").strip().lower()
    if select_mode in {"recent_context", "recent_context_elite", "hybrid"} or macro_mode != "off":
        recent = [self._unpack_sample(s) for s in list(getattr(self, "local_recent", []))[-recent_window:]]
        all_records = _cbo_all_records(self)
        threshold = float(getattr(self, "cbo_context_sim_threshold", _cfg_cbo_float("CBO_CONTEXT_SIM_THRESHOLD", 0.0)))
        context_k = max(0, int(getattr(self, "cbo_context_k", _cfg_cbo_int("CBO_CONTEXT_K", 50))))
        elite_k = max(0, int(getattr(self, "cbo_elite_k", _cfg_cbo_int("CBO_ELITE_K", 20))))
        diverse_k = max(0, int(getattr(self, "cbo_diverse_k", _cfg_cbo_int("CBO_DIVERSE_K", 20))))
        macro_k = max(0, int(getattr(self, "cbo_macro_k", _cfg_cbo_int("CBO_MACRO_K", 100))))
        macro_scale = _cbo_macro_total_scale(self, all_records)
        macro_current = _cbo_macro_context_from_context(self, context, records=all_records, scale=macro_scale)
        macro_scored = []
        if macro_mode != "off" and macro_k > 0:
            for rec in all_records:
                rec_macro = _cbo_macro_context_from_record(self, rec, records=all_records, scale=macro_scale)
                sim_m = _cbo_macro_similarity(self, macro_current, rec_macro)
                macro_scored.append((sim_m, rec))
            macro_scored.sort(key=lambda x: x[0], reverse=True)
        macro_records = [rec for _, rec in macro_scored[:macro_k]]
        macro_sims = [float(s) for s, _ in macro_scored]
        macro_pool_sims = [float(s) for s, _ in macro_scored[:macro_k]]
        if macro_mode == "hierarchical":
            macro_pool = list(macro_records)
            macro_pool_keys = {_cbo_record_identity(self, r) for r in macro_pool}
            fallback_records = []
            fallback_used = False
            fallback_reason = ""
            if len(macro_pool) < min_keep:
                needed = max(0, min_keep - len(macro_pool))
                recent_tail = [self._unpack_sample(s) for s in list(getattr(self, "local_recent", []))[-recent_window:]]
                for rec in reversed(recent_tail):
                    if _cbo_record_identity(self, rec) not in macro_pool_keys:
                        fallback_records.append(rec)
                        if len(fallback_records) >= needed:
                            break
                fallback_used = bool(fallback_records)
                fallback_reason = f"macro_pool_below_min_keep added_recent={len(fallback_records)}" if fallback_used else "macro_pool_below_min_keep_no_recent_available"
            self._last_macro_pool_records = list(macro_pool)
            self._last_macro_pool_keys = set(macro_pool_keys)

            macro_recent_keep = min(recent_window, max(min_keep, min(50, max(1, macro_k))))
            recent_scored = []
            for rec in recent:
                sim_m = _cbo_macro_similarity(self, macro_current, _cbo_macro_context_from_record(self, rec, records=all_records, scale=macro_scale))
                if _cbo_record_identity(self, rec) in macro_pool_keys:
                    recent_scored.append((sim_m, rec))
            recent_scored.sort(key=lambda x: (x[0], int(x[1].get("bo_iter", -1) or -1)), reverse=True)
            macro_recent = [rec for _, rec in recent_scored[:macro_recent_keep]]
            if len(macro_recent) < min_keep:
                need = min_keep - len(macro_recent)
                existing = {_cbo_record_identity(self, r) for r in macro_recent}
                for rec in reversed(recent):
                    key = _cbo_record_identity(self, rec)
                    if key in existing:
                        continue
                    macro_recent.append(rec)
                    existing.add(key)
                    fallback_used = True
                    if not fallback_reason:
                        fallback_reason = "recent_keep_fallback"
                    if len(macro_recent) >= min_keep or len(macro_recent) >= macro_recent_keep + need:
                        break

            scored = []
            context_records = []
            if select_mode in {"recent_context", "recent_context_elite", "hybrid"}:
                for rec in macro_pool:
                    sim = _cbo_context_similarity(self, context, rec)
                    if threshold <= 0 or sim >= threshold:
                        scored.append((sim, rec))
                scored.sort(key=lambda x: x[0], reverse=True)
                context_records = [rec for _, rec in scored[:context_k]]

            clusters = _cbo_cluster_records(self, macro_pool, context=context)
            elite_clusters = [c for c in clusters if np.isfinite(c.get("robust_score", np.nan))]
            elite_clusters.sort(key=lambda c: float(c.get("robust_score", float("inf"))))
            elite_records = []
            if select_mode in {"recent_context_elite", "hybrid"}:
                for c in elite_clusters[:elite_k]:
                    if c.get("records"):
                        elite_records.append(c["records"][0])

            diverse_records = []
            if select_mode == "hybrid" and diverse_k > 0:
                used_ids = {_cbo_record_identity(self, r) for r in (macro_recent + context_records + elite_records)}
                candidates = [r for r in macro_pool if _cbo_record_identity(self, r) not in used_ids]
                selected_norms = [_cbo_theta_norm(self, r.get("theta", [])) for r in macro_recent + context_records + elite_records if r.get("theta") is not None]
                while candidates and len(diverse_records) < diverse_k:
                    best_i, best_score = 0, -1.0
                    for idx, rec in enumerate(candidates):
                        tn = _cbo_theta_norm(self, rec.get("theta", []))
                        d_theta = min([float(np.linalg.norm(tn - s)) for s in selected_norms], default=1.0)
                        d_context = 1.0 - _cbo_context_similarity(self, context, rec)
                        score = d_theta + 0.25 * d_context
                        if score > best_score:
                            best_i, best_score = idx, score
                    rec = candidates.pop(best_i)
                    diverse_records.append(rec)
                    selected_norms.append(_cbo_theta_norm(self, rec.get("theta", [])))

            merged = []
            for priority, block in [(0, macro_recent), (1, context_records), (2, elite_records), (3, diverse_records), (4, fallback_records)]:
                for rec in block:
                    rec = dict(rec)
                    rec["_cbo_select_priority"] = priority
                    rec["_cbo_context_similarity"] = _cbo_context_similarity(self, context, rec)
                    rec["_cbo_macro_similarity"] = _cbo_macro_similarity(self, macro_current, _cbo_macro_context_from_record(self, rec, records=all_records, scale=macro_scale))
                    merged.append(rec)
            dedup = {}
            eps = max(1e-9, float(getattr(self, "cbo_theta_merge_eps", _cfg_cbo_float("CBO_THETA_MERGE_EPS", 0.05))))
            for rec in merged:
                tn = _cbo_theta_norm(self, rec.get("theta", []))
                cn = np.asarray(self._normalize_context(rec.get("context")), dtype=float) if rec.get("context") is not None and getattr(self, "use_context", False) else np.zeros(0)
                key = tuple(np.round(np.concatenate([tn / eps, cn / max(eps, 1e-9)]), 0).astype(int).tolist())
                old = dedup.get(key)
                if old is None:
                    dedup[key] = rec
                else:
                    old_score = (old.get("_cbo_select_priority", 9), -float(old.get("_cbo_macro_similarity", 0.0)), -float(old.get("_cbo_context_similarity", 0.0)), -int(old.get("bo_iter", -1) or -1))
                    new_score = (rec.get("_cbo_select_priority", 9), -float(rec.get("_cbo_macro_similarity", 0.0)), -float(rec.get("_cbo_context_similarity", 0.0)), -int(rec.get("bo_iter", -1) or -1))
                    if new_score < old_score:
                        dedup[key] = rec
            pool = list(dedup.values())
            max_train = macro_recent_keep + (context_k if select_mode in {"recent_context", "recent_context_elite", "hybrid"} else 0) + (elite_k if select_mode in {"recent_context_elite", "hybrid"} else 0) + (diverse_k if select_mode == "hybrid" else 0)
            pool.sort(key=lambda r: (int(r.get("_cbo_select_priority", 9)), -float(r.get("_cbo_macro_similarity", 0.0)), -float(r.get("_cbo_context_similarity", 0.0)), -int(r.get("bo_iter", -1) or -1)))
            pool = pool[:max(2, max_train)]
            pool_keys = {_cbo_record_identity(self, r) for r in pool}
            selected_macro_sims = [float(r.get("_cbo_macro_similarity", np.nan)) for r in pool if _cbo_record_identity(self, r) in macro_pool_keys and np.isfinite(float(r.get("_cbo_macro_similarity", np.nan)))]
            outside_count = sum(1 for r in pool if _cbo_record_identity(self, r) not in macro_pool_keys)
            best_elite = elite_clusters[0] if elite_clusters else {}
            set_debug(
                pool,
                recent_count=len(macro_recent),
                macro_count=len(selected_macro_sims),
                context_count=len(context_records),
                elite_count=len(elite_records),
                diverse_count=len(diverse_records),
                sims=[s for s, _ in scored],
                elite=best_elite,
                macro_sims=macro_sims,
                macro_pool_sims=macro_pool_sims,
                selected_macro_sims=selected_macro_sims,
                selected_from_macro_pool_count=sum(1 for k in pool_keys if k in macro_pool_keys),
                selected_outside_macro_pool_count=outside_count,
                macro_gate_fallback_used=fallback_used or outside_count > 0,
                macro_gate_fallback_reason=fallback_reason,
                context_selection_source_pool="macro_pool",
                elite_selection_source_pool="macro_pool",
                tr_anchor_source_pool="macro_pool",
                macro_current=macro_current,
            )
            return list(pool)

        self._last_macro_pool_records = []
        self._last_macro_pool_keys = set()
        scored = []
        context_records = []
        if select_mode in {"recent_context", "recent_context_elite", "hybrid"}:
            for rec in all_records:
                sim = _cbo_context_similarity(self, context, rec)
                if threshold <= 0 or sim >= threshold:
                    scored.append((sim, rec))
            scored.sort(key=lambda x: x[0], reverse=True)
            context_records = [rec for _, rec in scored[:context_k]]
        clusters = _cbo_cluster_records(self, all_records, context=context)
        elite_clusters = [c for c in clusters if np.isfinite(c.get("robust_score", np.nan))]
        elite_clusters.sort(key=lambda c: float(c.get("robust_score", float("inf"))))
        elite_records = []
        if select_mode in {"recent_context_elite", "hybrid"}:
            for c in elite_clusters[:elite_k]:
                if c.get("records"):
                    elite_records.append(c["records"][0])
        diverse_records = []
        if select_mode == "hybrid" and diverse_k > 0:
            used_ids = {id(r) for r in (recent + context_records + elite_records)}
            candidates = [r for r in all_records if id(r) not in used_ids]
            selected_norms = [_cbo_theta_norm(self, r.get("theta", [])) for r in recent + context_records + elite_records if r.get("theta") is not None]
            while candidates and len(diverse_records) < diverse_k:
                best_i, best_score = 0, -1.0
                for idx, rec in enumerate(candidates):
                    tn = _cbo_theta_norm(self, rec.get("theta", []))
                    d_theta = min([float(np.linalg.norm(tn - s)) for s in selected_norms], default=1.0)
                    d_context = 1.0 - _cbo_context_similarity(self, context, rec)
                    score = d_theta + 0.25 * d_context
                    if score > best_score:
                        best_i, best_score = idx, score
                rec = candidates.pop(best_i)
                diverse_records.append(rec)
                selected_norms.append(_cbo_theta_norm(self, rec.get("theta", [])))
        merged = []
        for priority, block in [(0, recent), (1, macro_records), (2, context_records), (3, elite_records), (4, diverse_records)]:
            for rec in block:
                rec = dict(rec)
                rec["_cbo_select_priority"] = priority
                rec["_cbo_context_similarity"] = _cbo_context_similarity(self, context, rec)
                if macro_mode != "off":
                    rec["_cbo_macro_similarity"] = _cbo_macro_similarity(self, macro_current, _cbo_macro_context_from_record(self, rec, records=all_records, scale=macro_scale))
                merged.append(rec)
        dedup = {}
        eps = max(1e-9, float(getattr(self, "cbo_theta_merge_eps", _cfg_cbo_float("CBO_THETA_MERGE_EPS", 0.05))))
        for rec in merged:
            tn = _cbo_theta_norm(self, rec.get("theta", []))
            cn = np.asarray(self._normalize_context(rec.get("context")), dtype=float) if rec.get("context") is not None and getattr(self, "use_context", False) else np.zeros(0)
            key = tuple(np.round(np.concatenate([tn / eps, cn / max(eps, 1e-9)]), 0).astype(int).tolist())
            old = dedup.get(key)
            if old is None:
                dedup[key] = rec
            else:
                old_score = (old.get("_cbo_select_priority", 9), -float(old.get("_cbo_context_similarity", 0.0)), -int(old.get("bo_iter", -1) or -1))
                new_score = (rec.get("_cbo_select_priority", 9), -float(rec.get("_cbo_context_similarity", 0.0)), -int(rec.get("bo_iter", -1) or -1))
                if new_score < old_score:
                    dedup[key] = rec
        pool = list(dedup.values())
        max_train = recent_window + (macro_k if macro_mode != "off" else 0) + (context_k if select_mode in {"recent_context", "recent_context_elite", "hybrid"} else 0) + (elite_k if select_mode in {"recent_context_elite", "hybrid"} else 0) + (diverse_k if select_mode == "hybrid" else 0)
        pool.sort(key=lambda r: (int(r.get("_cbo_select_priority", 9)), -float(r.get("_cbo_context_similarity", 0.0)), -int(r.get("bo_iter", -1) or -1)))
        pool = pool[:max(2, max_train)]
        best_elite = elite_clusters[0] if elite_clusters else {}
        set_debug(
            pool,
            recent_count=len(recent),
            macro_count=len(macro_records),
            context_count=len(context_records),
            elite_count=len(elite_records),
            diverse_count=len(diverse_records),
            sims=[s for s, _ in scored],
            elite=best_elite,
            macro_sims=macro_sims,
            macro_pool_sims=macro_pool_sims,
            selected_from_macro_pool_count=len(macro_records),
            selected_outside_macro_pool_count=0,
            context_selection_source_pool="all_records",
            elite_selection_source_pool="all_records",
            tr_anchor_source_pool="all_records",
            macro_current=macro_current,
        )
        return list(pool)

    pool = records
    if mode in {"recent", "recent_only", "recent_confidence", "confidence_recent", "decay"}:
        pool = pool[-recent_window:]

    if mode in {"confidence", "conf", "recent_confidence", "confidence_recent"}:
        keep = [r for r in pool if float(r.get("feedback_confidence", 1.0)) >= min_conf]
        if len(keep) >= min_keep:
            pool = keep
        else:
            # 样本太少时不能硬过滤，否则 BO 直接失明。保留置信度最高的一批 + 最近样本兜底。
            ranked = sorted(pool, key=lambda r: float(r.get("feedback_confidence", 1.0)), reverse=True)
            top = ranked[:min(len(ranked), min_keep)]
            tail = pool[-min_keep:]
            merged = []
            seen = set()
            for r in top + tail:
                key = id(r)
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            pool = merged

    set_debug(pool)
    return list(pool)


FederatedBOAgent._collect_samples = _refactor_collect_samples


_ORIG_AGENT_TELL_STABILITY = FederatedBOAgent.tell
_ORIG_AGENT_ASK_CONTEXTUAL_STABILITY = FederatedBOAgent._ask_contextual


def _cbo_tr_update_params(agent):
    return {
        "mode": str(getattr(agent, "cbo_tr_update_mode", _cfg_cbo_str("CBO_TR_UPDATE_MODE", "best_so_far")) or "best_so_far").strip().lower(),
        "compare_window": max(1, int(getattr(agent, "cbo_tr_compare_window", _cfg_cbo_int("CBO_TR_COMPARE_WINDOW", 30)))),
        "baseline_window": max(1, int(getattr(agent, "cbo_tr_baseline_window", _cfg_cbo_int("CBO_TR_BASELINE_WINDOW", 60)))),
        "improve_pct": float(getattr(agent, "cbo_tr_improve_pct", _cfg_cbo_float("CBO_TR_IMPROVE_PCT", 0.015))),
        "worsen_pct": float(getattr(agent, "cbo_tr_worsen_pct", _cfg_cbo_float("CBO_TR_WORSEN_PCT", 0.03))),
        "deadband_pct": float(getattr(agent, "cbo_tr_deadband_pct", _cfg_cbo_float("CBO_TR_DEADBAND_PCT", 0.01))),
        "patience": max(1, int(getattr(agent, "cbo_tr_update_patience", _cfg_cbo_int("CBO_TR_UPDATE_PATIENCE", 2)))),
        "r_min": float(getattr(agent, "cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04)))),
        "r_max": float(getattr(agent, "cbo_tr_radius_max", _cfg_cbo_float("CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35)))),
        "grow": float(getattr(agent, "cbo_tr_grow", _cfg_cbo_float("CBO_TR_GROW", getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15)))),
        "shrink": float(getattr(agent, "cbo_tr_shrink", _cfg_cbo_float("CBO_TR_SHRINK", getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92)))),
    }


def _cbo_ewma(values, alpha=None):
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return np.nan
    if alpha is None:
        alpha = 2.0 / (len(vals) + 1.0)
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    cur = vals[0]
    for v in vals[1:]:
        cur = alpha * float(v) + (1.0 - alpha) * cur
    return float(cur)


def _cbo_apply_tr_signal(agent, params, signal, reason, radius_before):
    mode = str(params.get("mode", "best_so_far"))
    patience = max(1, int(params.get("patience", 1)))
    actionable = signal in {"improve", "worse"}
    prev_signal = str(getattr(agent, "cbo_tr_update_last_signal", "") or "")
    if actionable:
        count = int(getattr(agent, "cbo_tr_update_patience_count", 0))
        count = count + 1 if prev_signal == signal else 1
    else:
        count = 0
    agent.cbo_tr_update_last_signal = signal
    agent.cbo_tr_update_patience_count = int(count)

    radius_after = float(radius_before)
    applied = False
    final_reason = reason
    if actionable and count >= patience:
        if signal == "improve":
            radius_after = max(float(params["r_min"]), float(radius_before) * float(params["shrink"]))
            agent.cbo_tr_failure_count = int(getattr(agent, "cbo_tr_failure_count", 0)) + 1
        else:
            radius_after = min(float(params["r_max"]), float(radius_before) * float(params["grow"]))
            agent.cbo_tr_success_count = int(getattr(agent, "cbo_tr_success_count", 0)) + 1
        agent.trust_radius = float(radius_after)
        agent.cbo_tr_update_patience_count = 0
        applied = True
    elif actionable:
        final_reason = f"{reason}_pending_patience"

    return {
        "tr_update_mode": mode,
        "tr_update_signal": signal,
        "tr_update_patience_count": int(getattr(agent, "cbo_tr_update_patience_count", count)),
        "cbo_tr_update_reason": final_reason,
        "cbo_tr_radius_before_update": float(radius_before),
        "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", radius_after)),
        "tr_update_applied": int(applied),
    }


def _cbo_update_tr_radius_after_tell(agent, cost, prev_best_value=None, radius_before=None):
    params = _cbo_tr_update_params(agent)
    mode = params["mode"]
    if mode not in {"best_so_far", "rolling_mean", "ewma_trend"}:
        mode = "best_so_far"
        params["mode"] = mode
    if radius_before is None or not np.isfinite(float(radius_before)):
        radius_before = float(getattr(agent, "trust_radius", np.nan))
    radius_before = float(radius_before)
    agent.trust_radius = float(np.clip(float(getattr(agent, "trust_radius", radius_before)), params["r_min"], params["r_max"]))

    hist = list(getattr(agent, "cbo_tr_cost_history", []))
    if np.isfinite(float(cost)):
        hist.append(float(cost))
    max_len = max(200, (params["compare_window"] + params["baseline_window"]) * 4)
    agent.cbo_tr_cost_history = hist[-max_len:]

    info = {
        "tr_update_mode": mode,
        "tr_baseline_mean": np.nan,
        "tr_current_mean": np.nan,
        "tr_improve_pct": np.nan,
        "tr_worse_pct": np.nan,
        "tr_update_signal": "none",
        "tr_update_patience_count": int(getattr(agent, "cbo_tr_update_patience_count", 0)),
        "cbo_tr_update_reason": "tr_update_not_run",
        "cbo_tr_radius_before_update": float(radius_before),
        "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", radius_before)),
    }

    if mode == "best_so_far":
        y_val = -float(cost)
        improved = prev_best_value is None or (np.isfinite(float(prev_best_value)) and y_val >= float(prev_best_value))
        if improved:
            agent.trust_radius = min(params["r_max"], radius_before * params["grow"])
            agent.cbo_tr_success_count = int(getattr(agent, "cbo_tr_success_count", 0)) + 1
            signal = "best_so_far_improve"
            reason = "best_so_far_improved_grow"
        else:
            agent.trust_radius = max(params["r_min"], radius_before * params["shrink"])
            agent.cbo_tr_failure_count = int(getattr(agent, "cbo_tr_failure_count", 0)) + 1
            signal = "best_so_far_worse"
            reason = "best_so_far_worse_shrink"
        info.update({
            "tr_update_signal": signal,
            "tr_update_patience_count": 0,
            "cbo_tr_update_reason": reason,
            "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", np.nan)),
        })
    else:
        need = params["compare_window"] + params["baseline_window"]
        if len(hist) < need:
            info.update({
                "tr_update_signal": "insufficient_history",
                "cbo_tr_update_reason": f"{mode}_insufficient_history",
                "cbo_tr_radius_after_update": float(getattr(agent, "trust_radius", np.nan)),
            })
        else:
            baseline = hist[-need:-params["compare_window"]]
            current = hist[-params["compare_window"]:]
            if mode == "ewma_trend":
                alpha = 2.0 / (params["compare_window"] + 1.0)
                baseline_mean = _cbo_ewma(baseline, alpha=alpha)
                current_mean = _cbo_ewma(current, alpha=alpha)
            else:
                baseline_mean = float(np.mean(baseline))
                current_mean = float(np.mean(current))
            if np.isfinite(baseline_mean) and abs(float(baseline_mean)) > 1e-12 and np.isfinite(current_mean):
                improve_pct = (baseline_mean - current_mean) / abs(baseline_mean)
                worse_pct = (current_mean - baseline_mean) / abs(baseline_mean)
            else:
                improve_pct = np.nan
                worse_pct = np.nan
            info.update({
                "tr_baseline_mean": float(baseline_mean) if np.isfinite(baseline_mean) else np.nan,
                "tr_current_mean": float(current_mean) if np.isfinite(current_mean) else np.nan,
                "tr_improve_pct": float(improve_pct) if np.isfinite(improve_pct) else np.nan,
                "tr_worse_pct": float(worse_pct) if np.isfinite(worse_pct) else np.nan,
            })
            if not np.isfinite(improve_pct) or not np.isfinite(worse_pct):
                signal, reason = "invalid_trend", f"{mode}_invalid_trend"
            elif improve_pct > params["improve_pct"]:
                signal, reason = "improve", f"{mode}_improved_refine"
            elif worse_pct > params["worsen_pct"]:
                signal, reason = "worse", f"{mode}_worse_expand"
            elif abs(current_mean - baseline_mean) / max(1e-12, abs(baseline_mean)) <= params["deadband_pct"]:
                signal, reason = "deadband", f"{mode}_hold_deadband"
            else:
                signal, reason = "hold", f"{mode}_hold_between_thresholds"
            info.update(_cbo_apply_tr_signal(agent, params, signal, reason, radius_before))
            info.update({
                "tr_baseline_mean": float(baseline_mean) if np.isfinite(baseline_mean) else np.nan,
                "tr_current_mean": float(current_mean) if np.isfinite(current_mean) else np.nan,
                "tr_improve_pct": float(improve_pct) if np.isfinite(improve_pct) else np.nan,
                "tr_worse_pct": float(worse_pct) if np.isfinite(worse_pct) else np.nan,
            })

    agent.cbo_tr_update_reason = str(info.get("cbo_tr_update_reason", "tr_update_unknown"))
    debug = dict(getattr(agent, "last_debug_info", {}) or {})
    debug.update(info)
    agent.last_debug_info = debug
    return info


def _stability_tell(self, theta, cost, state=None, context=None):
    prev_best_value = getattr(self, "prev_best_value", None)
    tr_mode = str(getattr(self, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) or "off").lower()
    managed_tr = tr_mode in {"adaptive", "residual_adaptive", "condition_adaptive"}
    radius_before = float(getattr(self, "trust_radius", np.nan))
    old_use_trust_region = bool(getattr(self, "use_trust_region", False))
    if managed_tr:
        self.use_trust_region = False
    try:
        _ORIG_AGENT_TELL_STABILITY(self, theta, cost, state=state, context=context)
    finally:
        if managed_tr:
            self.use_trust_region = old_use_trust_region
    if tr_mode in {"adaptive", "residual_adaptive", "condition_adaptive"}:
        _cbo_update_tr_radius_after_tell(self, float(cost), prev_best_value=prev_best_value, radius_before=radius_before)
        if tr_mode in {"residual_adaptive", "condition_adaptive"}:
            _cbo_update_residual_condition_state(self, float(cost))
    elif tr_mode == "good_region":
        self.cbo_tr_update_reason = "good_region_fixed_radius"


FederatedBOAgent.tell = _stability_tell


def _cbo_choose_tr_anchor(agent, mode, context, records, candidate_scores=None):
    low = np.asarray(agent.bounds[0].tolist(), dtype=float)
    high = np.asarray(agent.bounds[1].tolist(), dtype=float)
    mode = str(mode or "posterior_mean").lower()
    macro_mode = str(getattr(agent, "cbo_macro_gate_mode", _cfg_cbo_str("CBO_MACRO_GATE_MODE", "off")) or "off").strip().lower()
    macro_records = list(getattr(agent, "_last_macro_pool_records", []) or [])
    anchor_records = macro_records if macro_mode == "hierarchical" and macro_records else list(records or [])
    source_pool = "macro_pool" if macro_mode == "hierarchical" and macro_records else "all_records"
    if mode == "robust_elite":
        theta, info = agent._compute_robust_incumbent(context=context)
        if theta is not None:
            info = dict(info or {})
            info.setdefault("tr_anchor_source_pool", source_pool)
            return list(np.clip(np.asarray(theta, dtype=float), low, high)), "robust_elite", info
    if mode == "recent_best":
        if source_pool == "macro_pool":
            recent = sorted([r for r in anchor_records if r.get("theta") is not None], key=lambda r: int(r.get("bo_iter", -1) or -1))
        else:
            recent = [agent._unpack_sample(s) for s in getattr(agent, "local_recent", [])]
        if recent:
            best = max(recent, key=lambda r: float(r.get("y", -1e300)))
            return list(best.get("theta", [])), "recent_best", {"tr_anchor_source_pool": source_pool}
    if mode == "context_best":
        scored = [(_cbo_context_similarity(agent, context, r), r) for r in anchor_records if r.get("theta") is not None]
        if scored:
            scored.sort(key=lambda x: (x[0], float(x[1].get("y", -1e300))), reverse=True)
            return list(scored[0][1].get("theta", [])), "context_best", {"tr_anchor_source_pool": source_pool}
    if candidate_scores is not None and len(candidate_scores) > 0:
        try:
            idx = int(np.argmax(candidate_scores))
            return None, "posterior_mean", {"posterior_anchor_idx": idx, "tr_anchor_source_pool": source_pool}
        except Exception:
            pass
    pivot = agent._select_pivot_theta(context, anchor_records)
    if pivot is not None:
        return list(pivot), "posterior_mean_pivot", {"tr_anchor_source_pool": source_pool}
    if getattr(agent, "prev_best", None) is not None:
        return list(agent.prev_best), "posterior_mean_prev_best_fallback", {"tr_anchor_source_pool": source_pool}
    return None, "no_anchor", {"tr_anchor_source_pool": source_pool}


def _cbo_theta_distance(agent, a, b):
    try:
        if a is None or b is None:
            return np.nan
        return float(np.linalg.norm(_cbo_theta_norm(agent, a) - _cbo_theta_norm(agent, b)))
    except Exception:
        return np.nan


def _cbo_anchor_for_distance(agent, mode, context, records):
    try:
        theta, source, _ = _cbo_choose_tr_anchor(agent, mode, context, records)
        return theta if str(source) == str(mode) else None
    except Exception:
        return None


def _cbo_resolve_actual_tr_anchor(agent, context, records):
    configured_mode = str(getattr(agent, "cbo_tr_anchor_mode", _cfg_cbo_str("CBO_TR_ANCHOR_MODE", "posterior_mean")) or "posterior_mean").strip().lower()
    override_mode = getattr(agent, "cbo_runtime_anchor_override", None)
    override_mode = None if _is_missing_value(override_mode) else str(override_mode).strip().lower()
    effective_mode = str(override_mode or configured_mode or "posterior_mean").strip().lower()
    anchor, anchor_source, anchor_info = _cbo_choose_tr_anchor(agent, effective_mode, context, records)
    anchor_info = dict(anchor_info or {})
    fallback_used = bool(anchor is None or (effective_mode in {"robust_elite", "context_best", "recent_best"} and str(anchor_source) != effective_mode))
    fallback_reason = "" if not fallback_used else f"{effective_mode}_unavailable_used_{anchor_source}"
    override_used = bool(override_mode)
    reason_parts = []
    if override_used:
        reason_parts.append(f"runtime_override:{override_mode}")
    if fallback_used:
        reason_parts.append(f"fallback:{fallback_reason}")
    if not reason_parts:
        reason_parts.append("configured_anchor")

    prev_theta = getattr(agent, "last_theta", None)
    robust_theta = _cbo_anchor_for_distance(agent, "robust_elite", context, records)
    context_theta = _cbo_anchor_for_distance(agent, "context_best", context, records)
    recent_theta = _cbo_anchor_for_distance(agent, "recent_best", context, records)
    debug = {
        "configured_tr_anchor_mode": configured_mode,
        "effective_tr_anchor_mode": effective_mode,
        "actual_tr_anchor_mode": str(anchor_source),
        "actual_tr_anchor_source": str(anchor_source),
        "actual_tr_anchor_theta": list(anchor) if anchor is not None else None,
        "actual_tr_anchor_reason": ";".join(reason_parts),
        "anchor_override_used": int(override_used),
        "anchor_override_reason": f"runtime_anchor_override={override_mode}" if override_used else "",
        "anchor_fallback_used": int(fallback_used),
        "anchor_fallback_reason": fallback_reason,
        "anchor_theta_distance_to_prev": _cbo_theta_distance(agent, anchor, prev_theta),
        "anchor_theta_distance_to_robust_elite": _cbo_theta_distance(agent, anchor, robust_theta),
        "anchor_theta_distance_to_context_best": _cbo_theta_distance(agent, anchor, context_theta),
        "anchor_theta_distance_to_recent_best": _cbo_theta_distance(agent, anchor, recent_theta),
        "runtime_anchor_override": override_mode,
        "runtime_anchor_override_reason": f"runtime_anchor_override={override_mode}" if override_used else "",
        "tr_anchor_source_pool": str(anchor_info.get("tr_anchor_source_pool", "all_records")),
    }
    try:
        agent.cbo_last_actual_anchor_debug = dict(debug)
    except Exception:
        pass
    return (list(anchor) if anchor is not None else None), debug


def _cbo_radius_norm(agent):
    try:
        r = float(getattr(agent, "trust_radius", np.nan))
        r_min = float(getattr(agent, "cbo_tr_radius_min", _cfg_cbo_float("CBO_TR_RADIUS_MIN", getattr(CFG, "TRUST_RADIUS_MIN", 0.04))))
        r_max = float(getattr(agent, "cbo_tr_radius_max", _cfg_cbo_float("CBO_TR_RADIUS_MAX", getattr(CFG, "TRUST_RADIUS_MAX", 0.35))))
        if not np.isfinite(r) or not np.isfinite(r_min) or not np.isfinite(r_max) or abs(r_max - r_min) <= 1e-12:
            return np.nan
        return float(np.clip((r - r_min) / (r_max - r_min), 0.0, 1.0))
    except Exception:
        return np.nan


def _cbo_beta_eff_info(agent):
    mode = str(getattr(agent, "cbo_acq_beta_mode", _cfg_cbo_str("CBO_ACQ_BETA_MODE", "fixed")) or "fixed").strip().lower()
    base_beta = max(0.0, float(getattr(agent, "cbo_acq_beta", _cfg_cbo_float("CBO_ACQ_BETA", 3.0))))
    beta_min = max(0.0, float(getattr(agent, "cbo_beta_min", _cfg_cbo_float("CBO_BETA_MIN", 0.1))))
    beta_max = max(beta_min, float(getattr(agent, "cbo_beta_max", _cfg_cbo_float("CBO_BETA_MAX", 2.0))))
    power = max(1e-12, float(getattr(agent, "cbo_radius_beta_power", _cfg_cbo_float("CBO_RADIUS_BETA_POWER", 1.0))))
    radius_norm = _cbo_radius_norm(agent)
    if mode == "fixed":
        beta_eff = base_beta
        radius_component = base_beta
        formula = "posterior_mu + cbo_acq_beta * posterior_sigma"
    else:
        rn = 0.0 if not np.isfinite(radius_norm) else float(radius_norm)
        radius_component = float(beta_min + (beta_max - beta_min) * (rn ** power))
        beta_eff = radius_component
        formula = "posterior_mu + beta_eff * posterior_sigma"

    boost_used = False
    boost_reason = "none"
    if mode == "radius_state_adaptive":
        debug = dict(getattr(agent, "last_debug_info", {}) or {})
        reasons = []
        unstable_rebound = float(getattr(agent, "cbo_radius_unstable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04)))
        surprise_thr = float(getattr(agent, "cbo_radius_surprise_boost_threshold", _cfg_cbo_float("CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0)))
        cost_gap = _safe_float(debug.get("cost_gap_pct"), 0.0)
        surprise = _safe_float(debug.get("surprise"), np.nan)
        prediction_error = _safe_float(debug.get("prediction_error"), np.nan)
        predicted_cost = _safe_float(debug.get("predicted_cost"), np.nan)
        if np.isfinite(cost_gap) and cost_gap >= unstable_rebound:
            reasons.append("rebound")
        if np.isfinite(surprise) and surprise >= surprise_thr:
            reasons.append("surprise")
        if bool(int(debug.get("residual_trigger", 0) or 0)):
            reasons.append("residual_trigger")
        if bool(int(debug.get("condition_trigger", 0) or 0)):
            reasons.append("condition_trigger")
        stable_gap = float(getattr(agent, "cbo_radius_stable_rebound_pct", _cfg_cbo_float("CBO_RADIUS_STABLE_REBOUND_PCT", 0.02)))
        if np.isfinite(prediction_error) and np.isfinite(predicted_cost) and prediction_error > stable_gap * max(1.0, abs(predicted_cost)):
            reasons.append("actual_gt_predicted")
        if reasons:
            boost = max(1.0, float(getattr(agent, "cbo_radius_beta_boost", _cfg_cbo_float("CBO_RADIUS_BETA_BOOST", 1.5))))
            cap = max(0.0, float(getattr(agent, "cbo_radius_beta_cap", _cfg_cbo_float("CBO_RADIUS_BETA_CAP", 3.0))))
            beta_eff = min(float(beta_eff) * boost, cap)
            boost_used = True
            boost_reason = "+".join(reasons)
    info = {
        "cbo_acq_beta_mode": mode,
        "beta_eff": float(beta_eff),
        "radius_norm": float(radius_norm) if np.isfinite(radius_norm) else np.nan,
        "radius_beta_component": float(radius_component),
        "state_beta_boost_used": int(boost_used),
        "state_beta_boost_reason": boost_reason,
        "actual_score_formula": formula,
    }
    try:
        agent.cbo_last_beta_eff = float(beta_eff)
        agent.cbo_last_beta_info = dict(info)
        agent.cbo_last_actual_beta_used = float(beta_eff)
    except Exception:
        pass
    return info


def _cbo_service_guard_apply(agent, score):
    mode = str(getattr(agent, "cbo_service_guard_mode", _cfg_cbo_str("CBO_SERVICE_GUARD_MODE", "off")) or "off").strip().lower()
    score = np.asarray(score, dtype=float)
    penalty = np.zeros_like(score, dtype=float)
    reason = "off" if mode == "off" else "not_available"
    info = {
        "service_guard_mode": mode,
        "service_guard_available": False,
        "service_guard_penalty": 0.0,
        "service_guard_reason": reason,
    }
    return score - penalty, penalty, info


def _cbo_update_good_region_memory(agent, iteration, theta, eval_cost, safe_info):
    if agent is None:
        return safe_info
    costs = list(getattr(agent, "cbo_eval_cost_history", []) or [])
    try:
        costs.append(float(eval_cost))
    except Exception:
        costs.append(np.nan)
    agent.cbo_eval_cost_history = costs
    rolling = np.nan
    if len(costs) >= 50:
        recent = np.asarray(costs[-50:], dtype=float)
        if np.isfinite(recent).all():
            rolling = float(np.mean(recent))
            best = getattr(agent, "good_region_best_rolling50_cost", None)
            if best is None or not np.isfinite(float(best)) or rolling < float(best):
                agent.good_region_best_iter = int(iteration)
                agent.good_region_best_rolling50_cost = float(rolling)
                agent.good_region_anchor_theta = list(theta) if theta is not None else None
                agent.good_region_anchor_source = str(safe_info.get("selected_candidate_source", safe_info.get("deploy_source", "selected_theta")))
    best_cost = getattr(agent, "good_region_best_rolling50_cost", None)
    anchor = getattr(agent, "good_region_anchor_theta", None)
    available = bool(anchor is not None and best_cost is not None and np.isfinite(float(best_cost)))
    gap = np.nan
    if available and np.isfinite(float(best_cost)) and abs(float(best_cost)) > 1e-12:
        try:
            gap = (float(eval_cost) - float(best_cost)) / abs(float(best_cost))
        except Exception:
            gap = np.nan
    safe_info.update({
        "good_region_available": int(available),
        "good_region_best_iter": getattr(agent, "good_region_best_iter", None),
        "good_region_best_rolling50_cost": float(best_cost) if available else np.nan,
        "good_region_anchor_theta": list(anchor) if anchor is not None else None,
        "good_region_anchor_source": getattr(agent, "good_region_anchor_source", None),
        "distance_to_good_region_anchor": _cbo_theta_distance(agent, theta, anchor),
        "current_vs_good_region_gap_pct": float(gap) if np.isfinite(gap) else np.nan,
    })
    return safe_info


def _cbo_rank_ascending(values):
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=int)
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = int(rank)
    return ranks


def _cbo_candidate_rows(agent, candidates, sources, mu, sigma, score, selected_idx, selected_reason, deploy_policy=None, deploy_source=None, anchor=None, robust_theta=None, recent_best=None, beta_eff=None, service_penalty=None):
    rows = []
    if candidates is None:
        return rows, {}
    mu = np.asarray(mu if mu is not None else [np.nan] * len(candidates), dtype=float)
    sigma = np.asarray(sigma if sigma is not None else [np.nan] * len(candidates), dtype=float)
    score = np.asarray(score if score is not None else [np.nan] * len(candidates), dtype=float)
    service_penalty = np.asarray(service_penalty if service_penalty is not None else [0.0] * len(candidates), dtype=float)
    ranks_mu = _cbo_rank_ascending(-mu) if len(mu) else []
    ranks_sigma = _cbo_rank_ascending(-sigma) if len(sigma) else []
    ranks_score = _cbo_rank_ascending(-score) if len(score) else []
    ranks_acq = ranks_score
    best_mu_idx = int(np.nanargmax(mu)) if len(mu) and np.isfinite(mu).any() else None
    best_acq_idx = int(np.nanargmax(score)) if len(score) and np.isfinite(score).any() else None
    names = list(getattr(CFG, "FEATURE_NAMES", []))
    selected_idx = int(selected_idx) if selected_idx is not None else -1
    anchor_debug = dict(getattr(agent, "cbo_last_actual_anchor_debug", {}) or {})
    beta_info = dict(getattr(agent, "cbo_last_beta_info", {}) or {})
    beta_value = beta_eff if beta_eff is not None else beta_info.get("beta_eff")
    for i, theta in enumerate(candidates):
        row = {
            "candidate_id": int(i),
            "candidate_source": str(sources[i] if i < len(sources) else "unknown"),
            "theta": _safe_json(list(theta)),
            "control_vector": _safe_json(list(theta)),
            "posterior_mu": float(mu[i]) if i < len(mu) and np.isfinite(mu[i]) else np.nan,
            "posterior_sigma": float(sigma[i]) if i < len(sigma) and np.isfinite(sigma[i]) else np.nan,
            "acquisition_score": float(score[i]) if i < len(score) and np.isfinite(score[i]) else np.nan,
            "beta_eff": float(beta_value) if beta_value is not None and np.isfinite(float(beta_value)) else np.nan,
            "score": float(score[i]) if i < len(score) and np.isfinite(score[i]) else np.nan,
            "rank_by_mu": int(ranks_mu[i]) if len(ranks_mu) else None,
            "rank_by_sigma": int(ranks_sigma[i]) if len(ranks_sigma) else None,
            "rank_by_score": int(ranks_score[i]) if len(ranks_score) else None,
            "rank_by_acq": int(ranks_acq[i]) if len(ranks_acq) else None,
            "is_selected": int(i == selected_idx),
            "selected_reason": str(selected_reason),
            "deploy_policy": deploy_policy,
            "deploy_source": deploy_source,
            "service_guard_available": beta_info.get("service_guard_available", False),
            "service_guard_penalty": float(service_penalty[i]) if i < len(service_penalty) and np.isfinite(service_penalty[i]) else 0.0,
            "service_guard_reason": beta_info.get("service_guard_reason", "off"),
            "actual_tr_anchor_mode": anchor_debug.get("actual_tr_anchor_mode"),
            "actual_tr_anchor_source": anchor_debug.get("actual_tr_anchor_source"),
            "actual_tr_anchor_theta": _safe_json(anchor_debug.get("actual_tr_anchor_theta")),
            "anchor_override_used": anchor_debug.get("anchor_override_used"),
            "anchor_fallback_used": anchor_debug.get("anchor_fallback_used"),
            "distance_to_tr_anchor": _cbo_theta_distance(agent, theta, anchor),
            "distance_to_robust_incumbent": _cbo_theta_distance(agent, theta, robust_theta),
            "distance_to_recent_best": _cbo_theta_distance(agent, theta, recent_best),
            "cbo_tr_mode": str(getattr(agent, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off"))),
            "cbo_tr_radius": float(getattr(agent, "trust_radius", np.nan)),
        }
        for j, name in enumerate(names[:len(theta)]):
            row[str(name)] = float(theta[j])
        rows.append(row)
    summary = {
        "selected_candidate_id": selected_idx if selected_idx >= 0 else None,
        "selected_source": str(sources[selected_idx]) if 0 <= selected_idx < len(sources) else None,
        "selected_mu": float(mu[selected_idx]) if 0 <= selected_idx < len(mu) and np.isfinite(mu[selected_idx]) else np.nan,
        "selected_sigma": float(sigma[selected_idx]) if 0 <= selected_idx < len(sigma) and np.isfinite(sigma[selected_idx]) else np.nan,
        "selected_acq": float(score[selected_idx]) if 0 <= selected_idx < len(score) and np.isfinite(score[selected_idx]) else np.nan,
        "selected_score": float(score[selected_idx]) if 0 <= selected_idx < len(score) and np.isfinite(score[selected_idx]) else np.nan,
        "selected_candidate_score": float(score[selected_idx]) if 0 <= selected_idx < len(score) and np.isfinite(score[selected_idx]) else np.nan,
        "selected_candidate_beta_eff": float(beta_value) if beta_value is not None and np.isfinite(float(beta_value)) else np.nan,
        "best_mu_candidate_source": str(sources[best_mu_idx]) if best_mu_idx is not None and best_mu_idx < len(sources) else None,
        "best_acq_candidate_source": str(sources[best_acq_idx]) if best_acq_idx is not None and best_acq_idx < len(sources) else None,
        "num_candidates": int(len(candidates)),
        "num_tr_candidates": int(sum(1 for s in sources if str(s) == "trust_region")),
        "num_global_candidates": int(sum(1 for s in sources if str(s) == "global_random")),
        "selected_rank_by_mu": int(ranks_mu[selected_idx]) if 0 <= selected_idx < len(ranks_mu) else None,
        "selected_rank_by_sigma": int(ranks_sigma[selected_idx]) if 0 <= selected_idx < len(ranks_sigma) else None,
        "selected_rank_by_score": int(ranks_score[selected_idx]) if 0 <= selected_idx < len(ranks_score) else None,
        "selected_rank_by_acq": int(ranks_acq[selected_idx]) if 0 <= selected_idx < len(ranks_acq) else None,
        "selected_candidate_rank_by_score": int(ranks_score[selected_idx]) if 0 <= selected_idx < len(ranks_score) else None,
        "selected_candidate_rank_by_sigma": int(ranks_sigma[selected_idx]) if 0 <= selected_idx < len(ranks_sigma) else None,
    }
    summary.update(anchor_debug)
    summary.update(beta_info)
    return rows, summary


def _stability_ask_contextual(self, state=None, context=None):
    tr_mode = str(getattr(self, "cbo_tr_mode", _cfg_cbo_str("CBO_TR_MODE", "off")) or "off").lower()
    self._active_context = context
    if tr_mode == "off":
        theta = _ORIG_AGENT_ASK_CONTEXTUAL_STABILITY(self, state=state, context=context)
        hist = dict(getattr(self, "last_history_debug", {}) or {})
        self.last_debug_info = {**getattr(self, "last_debug_info", {}), **hist,
                                "cbo_tr_mode": "off", "cbo_tr_anchor_mode": getattr(self, "cbo_tr_anchor_mode", "posterior_mean"),
                                "cbo_tr_radius": float(getattr(self, "trust_radius", np.nan)),
                                "cbo_tr_anchor_theta": None, "cbo_tr_candidate_count": 0,
                                "cbo_global_candidate_count": int(getattr(self, "last_debug_info", {}).get("candidate_count", 0) or 0),
                                "cbo_tr_update_reason": getattr(self, "cbo_tr_update_reason", "off"),
                                "cbo_tr_success_count": int(getattr(self, "cbo_tr_success_count", 0)),
                                "cbo_tr_failure_count": int(getattr(self, "cbo_tr_failure_count", 0)),
                                "selected_candidate_source": getattr(self, "last_debug_info", {}).get("selected_candidate_source", "acquisition_candidate"),
                                "selected_candidate_mu": getattr(self, "last_debug_info", {}).get("posterior_mu"),
                                "selected_candidate_sigma": getattr(self, "last_debug_info", {}).get("posterior_sigma"),
                                "selected_candidate_acq": getattr(self, "last_debug_info", {}).get("selected_candidate_acq"),
                                "selected_candidate_rank_by_mu": getattr(self, "last_debug_info", {}).get("selected_candidate_rank_by_mu"),
                                "selected_candidate_rank_by_acq": getattr(self, "last_debug_info", {}).get("selected_candidate_rank_by_acq"),
                                "best_mu_candidate_source": getattr(self, "last_debug_info", {}).get("best_mu_candidate_source"),
                                "best_acq_candidate_source": getattr(self, "last_debug_info", {}).get("best_acq_candidate_source"),
                                "num_candidates": int(getattr(self, "last_debug_info", {}).get("candidate_count", 0) or 0),
                                "num_tr_candidates": 0,
                                "num_global_candidates": int(getattr(self, "last_debug_info", {}).get("candidate_count", 0) or 0)}
        return theta

    self.step_count += 1
    low = self.bounds[0].tolist()
    high = self.bounds[1].tolist()
    base_debug = {
        "step": int(self.step_count),
        "state": str(state) if state is not None else None,
        "context": self._normalize_context(context) if context is not None else ([0.0] * self.context_dim if self.use_context else []),
        "training_sample_count": 0,
        "recent_sample_count": int(len(self.local_recent)),
        "archive_sample_count": int(sum(len(v) for v in self.local_archive.values())),
        "neighbor_k": int(self.neighbor_k),
        "topk_history": [],
        "topk_similarity": [],
        "pivot": None,
        "trust_radius": float(self.trust_radius),
        "best_selected": None,
        "candidate_count": 0,
    }
    if self.anchor_points and self.step_count <= len(self.anchor_points):
        theta = list(self.anchor_points[self.step_count - 1])
        self.last_theta = list(theta)
        self.acq_history.append({"step": int(self.step_count), "candidates": [theta], "acq_values": [], "best_selected": theta})
        self.last_debug_info = {**base_debug, **dict(getattr(self, "last_history_debug", {}) or {}), "best_selected": list(theta), "candidate_count": 1}
        return theta
    x, y, records = self._training_data(state=state)
    base_debug["training_sample_count"] = int(len(records))
    model_pack = self.fit_local_gp(state=state)
    if len(x) < 2 or model_pack is None:
        theta = self._sample_in_bounds(low, high)
        self.last_theta = list(theta)
        self.last_debug_info = {**base_debug, **dict(getattr(self, "last_history_debug", {}) or {}), "best_selected": list(theta), "candidate_count": 1}
        return theta
    gp = model_pack["gp"]
    y_mean = model_pack["y_mean"]
    y_std = model_pack["y_std"]
    bounds_full = model_pack["bounds"]
    global_count = 52
    tr_count = 76
    candidates = self._random_candidates(low, high, global_count)
    candidate_sources = ["global_random"] * len(candidates)
    anchor, anchor_debug = _cbo_resolve_actual_tr_anchor(self, context, records)
    anchor_mode = str(anchor_debug.get("effective_tr_anchor_mode", getattr(self, "cbo_tr_anchor_mode", "posterior_mean")) or "posterior_mean").lower()
    anchor_source = str(anchor_debug.get("actual_tr_anchor_source", "no_anchor"))
    anchor_info = {"tr_anchor_source_pool": anchor_debug.get("tr_anchor_source_pool", "all_records")}
    if anchor is None:
        anchor = self._select_pivot_theta(context, records) or getattr(self, "prev_best", None)
    if anchor is not None:
        base_debug["pivot"] = list(anchor)
        for _ in range(tr_count):
            cand = []
            for d in range(self.dim):
                span = (high[d] - low[d]) * float(self.trust_radius)
                cand.append(min(max(anchor[d] + span * (2.0 * self.py_rng.random() - 1.0), low[d]), high[d]))
            candidates.append(cand)
            candidate_sources.append("trust_region")
    unique, seen = [], set()
    unique_sources = []
    for src_idx, c in enumerate(candidates):
        key = tuple(round(float(v), 6) for v in c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        unique_sources.append(candidate_sources[src_idx] if src_idx < len(candidate_sources) else "unknown")
    candidates = unique[:160] or [self._sample_in_bounds(low, high)]
    candidate_sources = unique_sources[:len(candidates)] if unique_sources else ["fallback"] * len(candidates)
    mu, sigma, score = self._contextual_scores(gp, y_mean, y_std, bounds_full, candidates, context)
    beta_info = _cbo_beta_eff_info(self)
    score = np.asarray(mu, dtype=float) + float(beta_info.get("beta_eff", getattr(self, "cbo_acq_beta", self.beta_init))) * np.asarray(sigma, dtype=float)
    score, service_penalty, guard_info = _cbo_service_guard_apply(self, score)
    beta_info.update(guard_info)
    try:
        self.cbo_last_beta_info = dict(beta_info)
    except Exception:
        pass
    best_idx, select_reason = _cbo_select_index_from_scores(self, mu, sigma, score, default_reason="max_acquisition_score")
    best = list(candidates[best_idx])
    if self.last_theta is not None and self.theta_momentum > 0.0:
        best = [self.theta_momentum * self.last_theta[d] + (1.0 - self.theta_momentum) * best[d] for d in range(self.dim)]
    best = [min(max(best[d], low[d]), high[d]) for d in range(self.dim)]
    self.last_theta = list(best)
    self.acq_history.append({
        "step": int(self.step_count),
        "candidates": [list(c) for c in candidates],
        "acq_values": [float(v) for v in score.tolist()],
        "best_selected": list(best),
        "model_state_dict": gp.state_dict(),
    })
    hist = dict(getattr(self, "last_history_debug", {}) or {})
    recent_records = [self._unpack_sample(s) for s in getattr(self, "local_recent", [])]
    recent_best = list(max(recent_records, key=lambda r: float(r.get("y", -1e300))).get("theta", [])) if recent_records else None
    robust_theta = None
    try:
        robust_theta, _robust_info_for_diag = self._compute_robust_incumbent(context=context)
    except Exception:
        robust_theta = None
    candidate_rows, candidate_summary = _cbo_candidate_rows(
        self, candidates, candidate_sources, mu, sigma, score, best_idx,
        selected_reason=str(select_reason),
        deploy_policy=str(tr_mode), deploy_source=str(select_reason),
        anchor=anchor, robust_theta=robust_theta, recent_best=recent_best,
        beta_eff=beta_info.get("beta_eff"), service_penalty=service_penalty,
    )
    self.last_debug_info = {
        **base_debug,
        **hist,
        **candidate_summary,
        **anchor_debug,
        **beta_info,
        "best_selected": list(best),
        "candidate_count": int(len(candidates)),
        "posterior_mu": float(mu[best_idx]),
        "posterior_sigma": float(sigma[best_idx]),
        "selected_candidate_source": str(select_reason if str(select_reason) != "max_acquisition_score" else (candidate_sources[best_idx] if best_idx < len(candidate_sources) else "unknown")),
        "selected_candidate_mu": float(mu[best_idx]),
        "selected_candidate_sigma": float(sigma[best_idx]),
        "selected_candidate_acq": float(score[best_idx]),
        "selected_candidate_score": float(score[best_idx]),
        "selected_candidate_beta_eff": float(beta_info.get("beta_eff", np.nan)),
        "actual_beta_used": float(getattr(self, "cbo_last_actual_beta_used", beta_info.get("beta_eff", np.nan))),
        "cbo_select_mode": str(getattr(self, "cbo_select_mode", "greedy")),
        "selected_reason": str(select_reason),
        "candidate_diagnostic_rows": candidate_rows,
        "cbo_tr_mode": str(tr_mode),
        "cbo_tr_anchor_mode": str(anchor_mode),
        "cbo_tr_radius": float(self.trust_radius),
        "cbo_tr_anchor_theta": list(anchor) if anchor is not None else None,
        "cbo_tr_anchor_source": str(anchor_source),
        "tr_anchor_source_pool": str((anchor_info or {}).get("tr_anchor_source_pool", hist.get("tr_anchor_source_pool", "all_records"))),
        "cbo_tr_candidate_count": int(tr_count if anchor is not None else 0),
        "cbo_global_candidate_count": int(global_count),
        "cbo_tr_update_reason": getattr(self, "cbo_tr_update_reason", "candidate_generation"),
        "cbo_tr_success_count": int(getattr(self, "cbo_tr_success_count", 0)),
        "cbo_tr_failure_count": int(getattr(self, "cbo_tr_failure_count", 0)),
    }
    return list(best)


FederatedBOAgent._ask_contextual = _stability_ask_contextual


def reduced6_lite_context_agent_kwargs(use_trust_region=False, anchor_mode="none", context_mode="lite"):
    kwargs = reduced6_agent_kwargs(use_context=True, use_trust_region=use_trust_region, anchor_mode=anchor_mode)
    kwargs["use_state_partition"] = False
    kwargs["context_dim"] = len(lite_context_feature_names(context_mode))
    kwargs["context_bounds"] = lite_context_bounds(context_mode)
    return kwargs


USER_METHOD_ALIASES = {
    "fixed_mid": "reduced6_fixed_mid",
    "fixed_tuned": "reduced6_fixed_tuned",
    "fixed_queue_high": "reduced6_fixed_queue_high",
    "fixed_risk_high": "reduced6_fixed_risk_high",
    "fixed_edge_safe": "reduced6_fixed_edge_safe",
    "bo-ei": "reduced6_bo_ei",
    "bo_ei": "reduced6_bo_ei",
    "boei": "reduced6_bo_ei",
    "bo-greedy": "reduced6_bo_greedy",
    "bo_greedy": "reduced6_bo_greedy",
    "bogreedy": "reduced6_bo_greedy",
    "bo-greedy-recent": "reduced6_bo_greedy_recent_conf",
    "bo_greedy_recent": "reduced6_bo_greedy_recent_conf",
    "bo-greedy-confidence": "reduced6_bo_greedy_recent_conf",
    "bo_greedy_confidence": "reduced6_bo_greedy_recent_conf",
    "bo-greedy-recent-confidence": "reduced6_bo_greedy_recent_conf",
    "bo_greedy_recent_confidence": "reduced6_bo_greedy_recent_conf",
    "cbo-lite": "reduced6_cbo_lite_recent_conf",
    "cbo_lite": "reduced6_cbo_lite_recent_conf",
    "cbo-lite-greedy": "reduced6_cbo_lite_recent_conf",
    "cbo_lite_greedy": "reduced6_cbo_lite_recent_conf",
    "cbo-lite-recent-confidence": "reduced6_cbo_lite_recent_conf",
    "cbo_lite_recent_confidence": "reduced6_cbo_lite_recent_conf",
    "cbo-full": "reduced6_cbo_lite_full",
    "cbo_full": "reduced6_cbo_lite_full",
    "cbo-lite-full": "reduced6_cbo_lite_full",
    "cbo_lite_full": "reduced6_cbo_lite_full",
    "cbo-load": "reduced6_cbo_lite_load_only",
    "cbo_load": "reduced6_cbo_lite_load_only",
    "cbo-lite-load": "reduced6_cbo_lite_load_only",
    "cbo_lite_load": "reduced6_cbo_lite_load_only",
    "cbo-util": "reduced6_cbo_lite_util_only",
    "cbo_util": "reduced6_cbo_lite_util_only",
    "cbo-lite-util": "reduced6_cbo_lite_util_only",
    "cbo_lite_util": "reduced6_cbo_lite_util_only",
    "cbo-pressure": "reduced6_cbo_lite_pressure_only",
    "cbo_pressure": "reduced6_cbo_lite_pressure_only",
    "cbo-lite-pressure": "reduced6_cbo_lite_pressure_only",
    "cbo_lite_pressure": "reduced6_cbo_lite_pressure_only",
    "cbo-no-cloud": "reduced6_cbo_lite_no_cloud",
    "cbo_no_cloud": "reduced6_cbo_lite_no_cloud",
    "cbo-lite-no-cloud": "reduced6_cbo_lite_no_cloud",
    "cbo_lite_no_cloud": "reduced6_cbo_lite_no_cloud",
    "cbo-no-arrival": "reduced6_cbo_lite_no_arrival",
    "cbo_no_arrival": "reduced6_cbo_lite_no_arrival",
    "cbo-lite-no-arrival": "reduced6_cbo_lite_no_arrival",
    "cbo_lite_no_arrival": "reduced6_cbo_lite_no_arrival",
    "cbo-taskmix": "reduced6_cbo_lite_taskmix",
    "cbo_taskmix": "reduced6_cbo_lite_taskmix",
    "cbo-lite-taskmix": "reduced6_cbo_lite_taskmix",
    "cbo_lite_taskmix": "reduced6_cbo_lite_taskmix",
    "cbo-recent-mix": "reduced6_cbo_lite_recent_mix",
    "cbo_recent_mix": "reduced6_cbo_lite_recent_mix",
    "cbo-prev-counts": "reduced6_cbo_lite_prev_counts",
    "cbo_prev_counts": "reduced6_cbo_lite_prev_counts",
    "cbo-pressure-taskmix": "reduced6_cbo_lite_pressure_taskmix",
    "cbo_pressure_taskmix": "reduced6_cbo_lite_pressure_taskmix",
    "cbo-ptask": "reduced6_cbo_lite_pressure_taskmix",
    "cbo_ptask": "reduced6_cbo_lite_pressure_taskmix",
    "cbo-pressure-recent-mix": "reduced6_cbo_lite_pressure_recent_mix",
    "cbo_pressure_recent_mix": "reduced6_cbo_lite_pressure_recent_mix",
    "cbo-pressure-counts": "reduced6_cbo_lite_pressure_counts",
    "cbo_pressure_counts": "reduced6_cbo_lite_pressure_counts",
    "cbo-pressure-taskmix-counts": "reduced6_cbo_lite_pressure_taskmix_counts",
    "cbo_pressure_taskmix_counts": "reduced6_cbo_lite_pressure_taskmix_counts",
    "cbo-ptc": "reduced6_cbo_lite_pressure_taskmix_counts",
    "cbo_ptc": "reduced6_cbo_lite_pressure_taskmix_counts",
    "cbo-full-taskmix": "reduced6_cbo_lite_full_taskmix",
    "cbo_full_taskmix": "reduced6_cbo_lite_full_taskmix",
    "cbo-full-taskmix-counts": "reduced6_cbo_lite_full_taskmix_counts",
    "cbo_full_taskmix_counts": "reduced6_cbo_lite_full_taskmix_counts",
    "cbo-greedy": "reduced6_cbo_greedy_legacy",
    "cbo_greedy": "reduced6_cbo_greedy_legacy",
    "cbo-tr-greedy": "reduced6_cbo_tr_greedy_legacy",
    "cbo_tr_greedy": "reduced6_cbo_tr_greedy_legacy",
    "round-robin-direct": "direct_round_robin",
    "round_robin_direct": "direct_round_robin",
    "roundrobin-direct": "direct_round_robin",
    "rr-direct": "direct_round_robin",
    "greedy-direct-cost": "direct_greedy_cost",
    "greedy_direct_cost": "direct_greedy_cost",
    "direct-greedy-cost": "direct_greedy_cost",
    "least-load-direct": "direct_least_load",
    "least_load_direct": "direct_least_load",
    "leastload-direct": "direct_least_load",
    "queue-aware-greedy-direct": "direct_queue_aware_greedy",
    "queue_aware_greedy_direct": "direct_queue_aware_greedy",
    "queueaware-greedy-direct": "direct_queue_aware_greedy",
    "dpp-greedy-direct": "direct_queue_aware_greedy",
    "dpp_greedy_direct": "direct_queue_aware_greedy",
    # Backwards-compatible aliases for old names.
    "reduced6_vanilla_bo_anchor": "reduced6_bo_ei",
    "reduced6_context_bo_anchor": "reduced6_cbo_greedy_legacy",
    "reduced6_context_tr_bo_anchor": "reduced6_cbo_tr_greedy_legacy",
}


def normalize_selected_method_keys(selected_keys):
    """支持用户输入短别名，同时保留原有完整 group key。"""
    if selected_keys is None:
        return None
    out = []
    for raw in selected_keys:
        k = str(raw).strip()
        if not k:
            continue
        out.append(USER_METHOD_ALIASES.get(k, USER_METHOD_ALIASES.get(k.lower(), k)))
    # 去重但保持顺序。
    seen = set()
    dedup = []
    for k in out:
        if k not in seen:
            seen.add(k)
            dedup.append(k)
    return dedup


def _dual_mark_last_sample(agent, sample_id, window_cost, state=None, context=None):
    for rec in reversed(list(getattr(agent, "local_recent", []))):
        if isinstance(rec, dict) and rec.get("sample_id") is None:
            rec["sample_id"] = sample_id
            rec["feedback_cost"] = float(window_cost)
            rec["feedback_source"] = "window_provisional"
            rec["feedback_state"] = str(state) if state is not None else None
            rec["feedback_context"] = list(context) if context is not None else None
            return True
    return False


def _cbo_dump_candidate_diagnostics(output_dir, iteration, safe_info, group_key=None):
    rows = list(safe_info.get("candidate_diagnostic_rows") or [])
    if not rows:
        return
    try:
        diag_dir = os.path.join(output_dir, "candidate_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        topn = max(1, int(getattr(CFG, "CBO_DUMP_CANDIDATES_TOPN", 30)))
        selected_id = safe_info.get("selected_candidate_id")
        clean_rows = []
        for row in rows:
            r = dict(row)
            r["iteration"] = int(iteration)
            r["group_key"] = str(group_key) if group_key is not None else None
            r["deploy_policy"] = safe_info.get("deploy_policy", r.get("deploy_policy"))
            r["deploy_source"] = safe_info.get("deploy_source", r.get("deploy_source"))
            clean_rows.append(r)
        selected_rows = [r for r in clean_rows if int(r.get("is_selected", 0) or 0) == 1]
        top_rows = sorted(clean_rows, key=lambda r: int(r.get("rank_by_acq") or 10**9))[:topn]
        if selected_rows:
            sel = dict(selected_rows[0])
            def fill_missing(key, value):
                if _is_missing_value(safe_info.get(key)):
                    safe_info[key] = value
            fill_missing("selected_candidate_id", sel.get("candidate_id"))
            fill_missing("selected_candidate_source", sel.get("candidate_source"))
            fill_missing("selected_source", sel.get("candidate_source"))
            fill_missing("selected_candidate_mu", sel.get("posterior_mu"))
            fill_missing("selected_mu", sel.get("posterior_mu"))
            fill_missing("selected_candidate_sigma", sel.get("posterior_sigma"))
            fill_missing("selected_sigma", sel.get("posterior_sigma"))
            fill_missing("selected_candidate_acq", sel.get("acquisition_score"))
            fill_missing("selected_acq", sel.get("acquisition_score"))
            fill_missing("selected_candidate_score", sel.get("score", sel.get("acquisition_score")))
            fill_missing("selected_score", sel.get("score", sel.get("acquisition_score")))
            fill_missing("selected_candidate_beta_eff", sel.get("beta_eff"))
            fill_missing("selected_candidate_rank_by_mu", sel.get("rank_by_mu"))
            fill_missing("selected_rank_by_mu", sel.get("rank_by_mu"))
            fill_missing("selected_candidate_rank_by_score", sel.get("rank_by_score"))
            fill_missing("selected_rank_by_score", sel.get("rank_by_score"))
            fill_missing("selected_candidate_rank_by_sigma", sel.get("rank_by_sigma"))
            fill_missing("selected_rank_by_sigma", sel.get("rank_by_sigma"))
            fill_missing("selected_candidate_rank_by_acq", sel.get("rank_by_acq"))
            fill_missing("selected_rank_by_acq", sel.get("rank_by_acq"))
        merged = []
        seen = set()
        for r in selected_rows + top_rows:
            cid = int(r.get("candidate_id", -1))
            if cid in seen:
                continue
            seen.add(cid)
            merged.append(r)
        cand_path = os.path.join(diag_dir, f"candidates_iter_{int(iteration):04d}.csv")
        pd.DataFrame(merged).to_csv(cand_path, index=False, encoding="utf-8-sig")

        runtime_override_reason = safe_info.get("runtime_anchor_override_reason")
        anchor_override_reason = safe_info.get("anchor_override_reason")
        runtime_override_mode = safe_info.get("runtime_anchor_override")
        actual_anchor_reason = str(safe_info.get("actual_tr_anchor_reason") or "")
        try:
            override_used = (not _is_missing_value(safe_info.get("anchor_override_used"))) and int(float(safe_info.get("anchor_override_used"))) != 0
        except Exception:
            override_used = False
        if _is_missing_value(runtime_override_reason) and (not _is_missing_value(runtime_override_mode)) and (override_used or "runtime_override" in actual_anchor_reason):
            runtime_override_reason = f"runtime_anchor_override={runtime_override_mode}"
        if _is_missing_value(anchor_override_reason) and not _is_missing_value(runtime_override_reason):
            anchor_override_reason = runtime_override_reason

        summary = {
            "iteration": int(iteration),
            "group_key": str(group_key) if group_key is not None else None,
            "selected_candidate_id": safe_info.get("selected_candidate_id", selected_id),
            "selected_source": safe_info.get("selected_source", safe_info.get("selected_candidate_source")),
            "selected_mu": safe_info.get("selected_mu", safe_info.get("selected_candidate_mu")),
            "selected_sigma": safe_info.get("selected_sigma", safe_info.get("selected_candidate_sigma")),
            "selected_acq": safe_info.get("selected_acq", safe_info.get("selected_candidate_acq")),
            "selected_score": safe_info.get("selected_score", safe_info.get("selected_candidate_score")),
            "selected_beta_eff": safe_info.get("selected_candidate_beta_eff", safe_info.get("beta_eff")),
            "best_mu_candidate_source": safe_info.get("best_mu_candidate_source"),
            "best_acq_candidate_source": safe_info.get("best_acq_candidate_source"),
            "num_candidates": safe_info.get("num_candidates", safe_info.get("candidate_count")),
            "num_tr_candidates": safe_info.get("num_tr_candidates", safe_info.get("cbo_tr_candidate_count")),
            "num_global_candidates": safe_info.get("num_global_candidates", safe_info.get("cbo_global_candidate_count")),
            "selected_rank_by_mu": safe_info.get("selected_rank_by_mu", safe_info.get("selected_candidate_rank_by_mu")),
            "selected_rank_by_score": safe_info.get("selected_rank_by_score", safe_info.get("selected_candidate_rank_by_score")),
            "selected_rank_by_sigma": safe_info.get("selected_rank_by_sigma", safe_info.get("selected_candidate_rank_by_sigma")),
            "selected_rank_by_acq": safe_info.get("selected_rank_by_acq", safe_info.get("selected_candidate_rank_by_acq")),
            "actual_tr_anchor_mode": safe_info.get("actual_tr_anchor_mode"),
            "actual_tr_anchor_source": safe_info.get("actual_tr_anchor_source"),
            "actual_tr_anchor_theta": _safe_json(safe_info.get("actual_tr_anchor_theta")),
            "anchor_override_used": safe_info.get("anchor_override_used"),
            "anchor_override_reason": anchor_override_reason,
            "anchor_fallback_used": safe_info.get("anchor_fallback_used"),
            "anchor_fallback_reason": safe_info.get("anchor_fallback_reason"),
            "runtime_anchor_override_reason": runtime_override_reason,
        }
        summary_path = os.path.join(diag_dir, "candidate_selection_summary.csv")
        summary_df = pd.DataFrame([summary])
        if os.path.exists(summary_path):
            try:
                old_summary = pd.read_csv(summary_path, encoding="utf-8-sig")
                summary_df = pd.concat([old_summary, summary_df], ignore_index=True, sort=False)
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            except Exception:
                # If an older run left a mismatched header/row shape, restart this compact diagnostic file.
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        else:
            summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[WARN] failed to dump candidate diagnostics: {type(exc).__name__}: {exc}", flush=True)


def run_scenario_group(seed, group_key, group_cfg):
    group_cfg["group_key"] = group_key
    fac = ConnectedFactory(fid=0, name=group_cfg["label"], seed=seed, node_config=CFG.NODES_CFG, scheduler_type=group_cfg.get("scheduler_type", "Boltzmann"), norm_mode=group_cfg.get("norm_mode", "rolling"))
    fac.reset(use_batch=False)
    fac.agent = create_scenario_agent(group_cfg, seed)
    configure_refactor_agent(fac.agent, group_cfg)
    fac.perf_log["group_key"] = group_key
    fac.perf_log["group_label"] = group_cfg["label"]
    # v6.2 runtime logging: per-method and per-iteration elapsed time.
    runtime_group_t0 = time.perf_counter()
    runtime_group_wall_t0 = time.time()
    fac.perf_log["runtime_group_start_unix"] = [float(runtime_group_wall_t0)]
    print(
        f"[DEPLOY] method={group_key} deploy_policy={_safebo_policy_name(group_cfg)} "
        f"history_mode={group_cfg.get('history_mode', _cfg_history_mode())} "
        f"recent_window={group_cfg.get('recent_window', _cfg_recent_window())} "
        f"feedback={getattr(CFG, 'BO_TRAINING_FEEDBACK_SCORE', 'window_original')}",
        flush=True,
    )
    print(
        f"[HISTORY] method={group_key} "
        f"history_mode={group_cfg.get('history_mode', _cfg_history_mode())} "
        f"recent_window={group_cfg.get('recent_window', _cfg_recent_window())} "
        f"source={group_cfg.get('history_override_source', 'method_default')}",
        flush=True,
    )
    print(
        f"[SCHED-TRADEOFF] mode={getattr(CFG, 'SCHEDULER_TRADEOFF_MODE', 'legacy')} "
        f"alpha={getattr(CFG, 'SCHEDULER_TRADEOFF_ALPHA', 0.85)} "
        f"alpha_min={getattr(CFG, 'SCHEDULER_ALPHA_MIN', 0.60)} "
        f"alpha_max={getattr(CFG, 'SCHEDULER_ALPHA_MAX', 0.97)} "
        f"service_latency_weight={getattr(CFG, 'SCHEDULER_SERVICE_LATENCY_WEIGHT', 1.0)} "
        f"service_risk_weight={getattr(CFG, 'SCHEDULER_SERVICE_RISK_WEIGHT', 1.0)} "
        f"service_queue_weight={getattr(CFG, 'SCHEDULER_SERVICE_QUEUE_WEIGHT', 1.0)} "
        f"energy_weight={getattr(CFG, 'SCHEDULER_ENERGY_WEIGHT', 1.0)}",
        flush=True,
    )
    print(
        f"[SCHED-NORM] mode={getattr(CFG, 'SCHEDULER_SCORE_NORM_MODE', 'legacy')} "
        f"clip_max={getattr(CFG, 'SCHEDULER_NORM_CLIP_MAX', 3.0)} "
        f"eps={getattr(CFG, 'SCHEDULER_NORM_EPS', 1e-6)} "
        f"ema_alpha={getattr(CFG, 'SCHEDULER_NORM_EMA_ALPHA', 0.995)}",
        flush=True,
    )
    if _is_cbo_method_key(group_key, group_cfg):
        print(
            f"[CBO-STABILITY] method={group_key} "
            f"history_select_mode={group_cfg.get('cbo_history_select_mode', _cfg_cbo_history_select_mode())} "
            f"history_mode={group_cfg.get('history_mode', _cfg_history_mode())} "
            f"recent_window={group_cfg.get('recent_window', _cfg_recent_window())} "
            f"context_k={group_cfg.get('cbo_context_k', _cfg_cbo_int('CBO_CONTEXT_K', 50))} "
            f"elite_k={group_cfg.get('cbo_elite_k', _cfg_cbo_int('CBO_ELITE_K', 20))} "
            f"diverse_k={group_cfg.get('cbo_diverse_k', _cfg_cbo_int('CBO_DIVERSE_K', 20))} "
            f"robust_score_mode={group_cfg.get('cbo_robust_score_mode', _cfg_cbo_str('CBO_ROBUST_SCORE_MODE', 'none'))} "
            f"tr_mode={group_cfg.get('cbo_tr_mode', _cfg_cbo_str('CBO_TR_MODE', 'off'))} "
            f"tr_anchor_mode={group_cfg.get('cbo_tr_anchor_mode', _cfg_cbo_str('CBO_TR_ANCHOR_MODE', 'posterior_mean'))} "
            f"robust_incumbent_mode={group_cfg.get('cbo_robust_incumbent_mode', _cfg_cbo_str('CBO_ROBUST_INCUMBENT_MODE', 'off'))}",
            flush=True,
        )
        print(
            f"[CBO-MACRO-GATE] method={group_key} "
            f"macro_gate_mode={group_cfg.get('cbo_macro_gate_mode', _cfg_cbo_str('CBO_MACRO_GATE_MODE', 'off'))} "
            f"macro_k={group_cfg.get('cbo_macro_k', _cfg_cbo_int('CBO_MACRO_K', 100))} "
            f"lengthscale_total={group_cfg.get('cbo_macro_lengthscale_total', _cfg_cbo_float('CBO_MACRO_LENGTHSCALE_TOTAL', 1.0))} "
            f"lengthscale_rt={group_cfg.get('cbo_macro_lengthscale_rt', _cfg_cbo_float('CBO_MACRO_LENGTHSCALE_RT', 0.15))} "
            f"lengthscale_batch={group_cfg.get('cbo_macro_lengthscale_batch', _cfg_cbo_float('CBO_MACRO_LENGTHSCALE_BATCH', 0.15))} "
            f"history_select_mode={group_cfg.get('cbo_history_select_mode', _cfg_cbo_history_select_mode())} "
            f"recent_window={group_cfg.get('recent_window', _cfg_recent_window())}",
            flush=True,
        )
        print(
            f"[CBO-ACQ-BETA] method={group_key} "
            f"beta_mode={group_cfg.get('cbo_acq_beta_mode', _cfg_cbo_str('CBO_ACQ_BETA_MODE', 'fixed'))} "
            f"beta_min={group_cfg.get('cbo_beta_min', _cfg_cbo_float('CBO_BETA_MIN', 0.1))} "
            f"beta_max={group_cfg.get('cbo_beta_max', _cfg_cbo_float('CBO_BETA_MAX', 2.0))} "
            f"radius_beta_power={group_cfg.get('cbo_radius_beta_power', _cfg_cbo_float('CBO_RADIUS_BETA_POWER', 1.0))} "
            f"select_mode={group_cfg.get('cbo_select_mode', _cfg_cbo_str('CBO_SELECT_MODE', 'greedy'))} "
            f"tr_mode={group_cfg.get('cbo_tr_mode', _cfg_cbo_str('CBO_TR_MODE', 'off'))} "
            f"tr_anchor_mode={group_cfg.get('cbo_tr_anchor_mode', _cfg_cbo_str('CBO_TR_ANCHOR_MODE', 'posterior_mean'))} "
            f"tr_update_mode={group_cfg.get('cbo_tr_update_mode', _cfg_cbo_str('CBO_TR_UPDATE_MODE', 'best_so_far'))}",
            flush=True,
        )

    is_reduced = group_cfg.get("control_mode") in {"reduced4", "reduced6"}
    fac.disable_internal_agent_tell = bool(is_reduced and fac.agent is not None)

    for i in range(CFG.BO_ITERATIONS):
        runtime_iter_t0 = time.perf_counter()
        state, _, _ = fac.scenario_monitor.get_state(fac.current_time)
        base_ctx = fac.scenario_monitor.get_context_vector(fac.current_time)
        ctx = build_context_for_group(fac, group_cfg, base_context=base_ctx)
        safe_info = {"deploy_policy": "fixed", "deploy_source": "fixed_theta", "explore_used": 0, "posterior_mu": None, "posterior_sigma": None, "candidate_count_safe": None}

        if fac.agent is None:
            theta_control = list(group_cfg["fixed_theta"])
            ask_state = state
            ask_ctx = ctx
        else:
            ask_state = state if getattr(fac.agent, "use_state_partition", False) else None
            ask_ctx = ctx if getattr(fac.agent, "use_context", False) else None
            theta_control, safe_info = _safebo_select_theta(fac.agent, state=ask_state, context=ask_ctx, group_cfg=group_cfg)
            if bool(group_cfg.get("cbo_dump_candidates", bool(getattr(CFG, "CBO_DUMP_CANDIDATES", False)))):
                every = max(1, int(group_cfg.get("cbo_dump_candidates_every", getattr(CFG, "CBO_DUMP_CANDIDATES_EVERY", 20))))
                if ((i + 1) % every == 0) or ((i + 1) == int(CFG.BO_ITERATIONS)):
                    _cbo_dump_candidate_diagnostics(SCENARIO_SAVE_DIR, i + 1, safe_info, group_key=group_key)

        theta_full = map_group_theta_to_full(theta_control, group_cfg)
        paired_shadow = None
        paired_window_end = None
        if _paired_delta_enabled():
            paired_window_end = float(fac.current_time) + float(getattr(CFG, "BO_INTERVAL", 40.0))
            try:
                _agent_ref = fac.agent
                fac.agent = None  # avoid copying torch generator / GP state into the shadow
                paired_shadow = copy.deepcopy(fac)
            finally:
                fac.agent = _agent_ref

        fac.current_control_vector = list(theta_full)
        fac.current_control_label = group_cfg.get("label", group_key)
        _, _, _, _, metrics, _ = fac.run_continuous(
            theta_full,
            eval_state=ask_state if fac.agent is not None else state,
            eval_context=ask_ctx if fac.agent is not None else ctx,
            feedback_control=theta_control,
        )
        if paired_shadow is not None:
            try:
                baseline_key, baseline_theta_control, baseline_metrics = _run_paired_shadow_baseline(
                    paired_shadow, group_cfg, ask_state=ask_state, ask_ctx=ask_ctx, window_end=paired_window_end
                )
                _attach_paired_delta_metrics(metrics, baseline_key, baseline_metrics)
            except Exception as e:
                metrics["paired_note"] = "paired_shadow_failed:" + type(e).__name__ + ":" + str(e)
                metrics["paired_baseline_key"] = _paired_baseline_key_for_group(group_cfg)
                metrics["paired_baseline_cost"] = np.nan
                metrics["paired_delta_cost"] = np.nan
                metrics["paired_delta_relative_pct"] = np.nan
        else:
            metrics["paired_note"] = "paired_delta_disabled"
        train_cost, train_feedback_mode, train_feedback_note = select_bo_training_feedback_cost(metrics, fac=fac, group_key=group_key)
        metrics["bo_training_cost"] = float(train_cost)
        metrics["bo_training_feedback_score"] = str(train_feedback_mode)
        metrics["bo_training_feedback_note"] = str(train_feedback_note)
        safe_info["current_candidate_cost"] = float(metrics.get("cost", np.nan))
        safe_info["current_train_cost"] = float(train_cost)
        safe_info["best_so_far_cost"] = -float(fac.agent.prev_best_value) if (fac.agent is not None and getattr(fac.agent, "prev_best_value", None) is not None) else safe_info.get("best_so_far_cost")
        safe_info["best_so_far_iter"] = getattr(fac.agent, "prev_best_iter", safe_info.get("best_so_far_iter")) if fac.agent is not None else safe_info.get("best_so_far_iter")
        feedback_confidence, confidence_parts = compute_feedback_confidence(metrics, group_cfg=group_cfg)
        metrics["feedback_confidence"] = float(feedback_confidence)
        metrics.update(confidence_parts)
        log_bo_training_feedback(fac, metrics, train_cost, train_feedback_mode, train_feedback_note)
        log_feedback_confidence(fac, feedback_confidence, confidence_parts, group_cfg=group_cfg)
        log_paired_delta_feedback(fac, metrics)

        if is_reduced and fac.agent is not None:
            state_arg = ask_state if getattr(fac.agent, "use_state_partition", False) else None
            context_arg = ask_ctx if getattr(fac.agent, "use_context", False) else None
            if _dual_is_enabled():
                # Provisional window tell: fast but noisy. A finalized cohort later replaces this sample.
                sample_id = f"w{int(i)}_c{metrics.get('cohort_id')}"
                agent_tell_with_feedback_meta(fac.agent, theta_control, train_cost, state=state_arg, context=context_arg, metrics=metrics, bo_iter=i, group_key=group_key, group_cfg=group_cfg, confidence=feedback_confidence, parts=confidence_parts)
                _dual_mark_last_sample(fac.agent, sample_id, train_cost, state=state_arg, context=context_arg)
                # Attach sample id to current cohort, so finalized precise feedback can update the same sample.
                try:
                    cur = fac.cohorts.get(metrics.get("cohort_id"))
                    if cur is not None:
                        cur.sample_id = sample_id
                except Exception:
                    pass
                applied = _dual_apply_pending_refinements(fac, fac.agent)
                fac.perf_log.setdefault("dual_window_provisional_cost", []).append(float(train_cost))
                fac.perf_log.setdefault("dual_refinement_applied_count", []).append(int(applied))
                fac.scheduler.update_beta(train_cost)
            elif not fac._use_cohort_feedback():
                agent_tell_with_feedback_meta(fac.agent, theta_control, train_cost, state=state_arg, context=context_arg, metrics=metrics, bo_iter=i, group_key=group_key, group_cfg=group_cfg, confidence=feedback_confidence, parts=confidence_parts)
                fac.scheduler.update_beta(train_cost)

        safe_info["best_so_far_cost"] = -float(fac.agent.prev_best_value) if (fac.agent is not None and getattr(fac.agent, "prev_best_value", None) is not None) else safe_info.get("best_so_far_cost")
        safe_info["best_so_far_iter"] = getattr(fac.agent, "prev_best_iter", safe_info.get("best_so_far_iter")) if fac.agent is not None else safe_info.get("best_so_far_iter")
        if fac.agent is not None and _is_cbo_method_key(group_key, group_cfg):
            tell_debug = dict(getattr(fac.agent, "last_debug_info", {}) or {})
            for post_tell_key in [
                "tr_update_mode", "tr_baseline_mean", "tr_current_mean", "tr_improve_pct",
                "tr_worse_pct", "tr_update_signal", "tr_update_patience_count",
                "cbo_tr_update_reason", "cbo_tr_radius_before_update", "cbo_tr_radius_after_update",
                "cbo_tr_success_count", "cbo_tr_failure_count", "predicted_cost", "actual_cost",
                "prediction_error", "surprise", "cost_gap_pct", "residual_trigger",
                "condition_trigger", "radius_min_stuck_count", "force_explore_countdown",
                "runtime_anchor_override",
            ]:
                if post_tell_key in tell_debug:
                    safe_info[post_tell_key] = tell_debug.get(post_tell_key)
            override_used = False
            try:
                override_used = (not _is_missing_value(safe_info.get("anchor_override_used"))) and int(float(safe_info.get("anchor_override_used"))) != 0
            except Exception:
                override_used = False
            override_mode = safe_info.get("runtime_anchor_override")
            if _is_missing_value(override_mode) and override_used:
                override_mode = safe_info.get("actual_tr_anchor_mode")
            if _is_missing_value(safe_info.get("runtime_anchor_override_reason")) and not _is_missing_value(override_mode):
                safe_info["runtime_anchor_override_reason"] = f"runtime_anchor_override={override_mode}"
            if _is_missing_value(safe_info.get("anchor_override_reason")) and override_used:
                safe_info["anchor_override_reason"] = f"runtime_anchor_override={override_mode}"
        if fac.agent is not None and _is_cbo_method_key(group_key, group_cfg):
            safe_info = _cbo_update_good_region_memory(fac.agent, i + 1, theta_control, float(metrics.get("cost", np.nan)), safe_info)
        for diag_key in [
            "selected_candidate_source", "selected_candidate_mu", "selected_candidate_sigma",
            "selected_candidate_acq", "selected_candidate_score", "selected_candidate_beta_eff",
            "selected_candidate_rank_by_score", "selected_candidate_rank_by_mu",
            "selected_candidate_rank_by_sigma", "selected_candidate_rank_by_acq",
            "best_mu_candidate_source", "best_acq_candidate_source", "num_candidates",
            "num_tr_candidates", "num_global_candidates", "cbo_macro_gate_mode",
            "macro_total_arrivals_norm", "macro_rt_ratio", "macro_batch_ratio",
            "macro_similarity_max", "macro_similarity_mean", "macro_similarity_p50",
            "macro_similarity_p90", "selected_macro_count", "selected_macro_mean_similarity",
            "selected_macro_min_similarity", "selected_macro_max_similarity", "macro_k",
            "macro_lengthscale_total", "macro_lengthscale_rt", "macro_lengthscale_batch",
            "macro_pool_count", "macro_pool_mean_similarity", "macro_pool_min_similarity",
            "macro_pool_max_similarity", "macro_pool_p50_similarity", "macro_pool_p90_similarity",
            "selected_from_macro_pool_count", "selected_outside_macro_pool_count",
            "macro_gate_fallback_used", "macro_gate_fallback_reason",
            "context_selection_source_pool", "elite_selection_source_pool", "tr_anchor_source_pool",
            "cbo_select_mode", "cbo_topk", "cbo_select_temperature", "cbo_epsilon", "cbo_acq_beta",
            "cbo_acq_beta_mode", "beta_eff", "radius_norm", "radius_beta_component",
            "state_beta_boost_used", "state_beta_boost_reason", "actual_score_formula",
            "actual_beta_used", "service_guard_mode", "service_guard_available",
            "service_guard_penalty", "service_guard_reason",
            "actual_tr_anchor_mode", "actual_tr_anchor_source", "actual_tr_anchor_theta",
            "actual_tr_anchor_reason", "anchor_override_used", "anchor_override_reason",
            "anchor_fallback_used", "anchor_fallback_reason", "anchor_theta_distance_to_prev",
            "anchor_theta_distance_to_robust_elite", "anchor_theta_distance_to_context_best",
            "anchor_theta_distance_to_recent_best", "runtime_anchor_override_reason",
            "good_region_available", "good_region_best_iter", "good_region_best_rolling50_cost",
            "good_region_anchor_theta", "good_region_anchor_source",
            "distance_to_good_region_anchor", "current_vs_good_region_gap_pct",
            "tr_update_mode", "tr_baseline_mean", "tr_current_mean", "tr_improve_pct",
            "tr_worse_pct", "tr_update_signal", "tr_update_patience_count",
            "cbo_tr_radius_before_update",
            "predicted_cost", "actual_cost", "prediction_error", "surprise", "cost_gap_pct",
            "residual_trigger", "condition_trigger", "radius_min_stuck_count", "force_explore_countdown",
            "runtime_anchor_override", "cbo_tr_radius_after_update", "selected_reason",
        ]:
            safe_info.setdefault(diag_key, None)
        for k, v in safe_info.items():
            if k == "candidate_diagnostic_rows":
                continue
            fac.perf_log.setdefault(k, []).append(v)
        fac.perf_log.setdefault("theta_control_deployed", []).append(list(theta_control))
        fac.perf_log.setdefault("agent_use_context", []).append(bool(getattr(fac.agent, "use_context", False)) if fac.agent is not None else False)

        # v6.2 runtime logging: append one elapsed-time sample per BO window.
        runtime_iter_elapsed = time.perf_counter() - runtime_iter_t0
        fac.perf_log.setdefault("runtime_iter_elapsed_sec", []).append(float(runtime_iter_elapsed))
        fac.perf_log.setdefault("runtime_iter_elapsed_min", []).append(float(runtime_iter_elapsed / 60.0))

        if (i + 1) % 20 == 0 or (i + 1) == int(CFG.BO_ITERATIONS):
            print(
                f"[SCHED-ROUND] iter={i + 1} "
                f"tradeoff_mode={(fac.perf_log.get('scheduler_tradeoff_mode', [getattr(CFG, 'SCHEDULER_TRADEOFF_MODE', 'legacy')]) or [None])[-1]} "
                f"score_norm_mode={(fac.perf_log.get('scheduler_score_norm_mode', [getattr(CFG, 'SCHEDULER_SCORE_NORM_MODE', 'legacy')]) or [None])[-1]} "
                f"alpha_last={(fac.perf_log.get('scheduler_alpha_last', [None]) or [None])[-1]} "
                f"selected_service_component={(fac.perf_log.get('selected_service_component_last', [None]) or [None])[-1]} "
                f"selected_energy_component={(fac.perf_log.get('selected_energy_component_last', [None]) or [None])[-1]} "
                f"selected_norm_l={(fac.perf_log.get('selected_norm_l_last', [None]) or [None])[-1]} "
                f"selected_norm_e={(fac.perf_log.get('selected_norm_e_last', [None]) or [None])[-1]} "
                f"selected_score={(fac.perf_log.get('selected_score_last', [None]) or [None])[-1]}",
                flush=True,
            )
            print(
                f"[DEPLOY-ROUND] method={group_key} iter={i + 1} "
                f"policy={safe_info.get('deploy_policy')} source={safe_info.get('deploy_source')} "
                f"explore_used={safe_info.get('explore_used')} "
                f"best_so_far_cost={safe_info.get('best_so_far_cost')} current_cost={safe_info.get('current_train_cost')}",
                flush=True,
            )
            if _is_cbo_method_key(group_key, group_cfg):
                print(
                    f"[CBO-STABILITY-ROUND] iter={i + 1} "
                    f"selected_total={safe_info.get('selected_total_count')} "
                    f"recent={safe_info.get('selected_recent_count')} "
                    f"context={safe_info.get('selected_context_count')} "
                    f"elite={safe_info.get('selected_elite_count')} "
                    f"diverse={safe_info.get('selected_diverse_count')} "
                    f"robust_score={safe_info.get('robust_incumbent_score')} "
                    f"robust_eval_count={safe_info.get('robust_incumbent_eval_count')} "
                    f"tr_radius={safe_info.get('cbo_tr_radius')} "
                    f"tr_update_mode={safe_info.get('tr_update_mode')} "
                    f"tr_before={safe_info.get('cbo_tr_radius_before_update')} "
                    f"tr_after={safe_info.get('cbo_tr_radius_after_update')} "
                    f"tr_signal={safe_info.get('tr_update_signal')} "
                    f"tr_reason={safe_info.get('cbo_tr_update_reason')} "
                    f"tr_mode={safe_info.get('cbo_tr_mode')} "
                    f"current_cost={safe_info.get('current_train_cost')} "
                    f"best_so_far_cost={safe_info.get('best_so_far_cost')}",
                    flush=True,
                )
                print(
                    f"[CBO-MACRO-ROUND] iter={i + 1} "
                    f"macro=[{safe_info.get('macro_total_arrivals_norm')},{safe_info.get('macro_rt_ratio')},{safe_info.get('macro_batch_ratio')}] "
                    f"macro_pool={safe_info.get('macro_pool_count')} "
                    f"selected_macro={safe_info.get('selected_macro_count')} "
                    f"sim_mean={safe_info.get('macro_similarity_mean')} "
                    f"sim_max={safe_info.get('macro_similarity_max')} "
                    f"selected_source={safe_info.get('selected_candidate_source', safe_info.get('selected_source'))} "
                    f"rank_mu={safe_info.get('selected_candidate_rank_by_mu', safe_info.get('selected_rank_by_mu'))} "
                    f"rank_acq={safe_info.get('selected_candidate_rank_by_acq', safe_info.get('selected_rank_by_acq'))} "
                    f"current_cost={safe_info.get('current_train_cost')}",
                    flush=True,
                )
                print(
                    f"[CBO-ACQ-ROUND] iter={i + 1} "
                    f"tr_radius={safe_info.get('cbo_tr_radius_after_update', safe_info.get('cbo_tr_radius'))} "
                    f"radius_norm={safe_info.get('radius_norm')} "
                    f"beta_eff={safe_info.get('beta_eff')} "
                    f"boost_used={safe_info.get('state_beta_boost_used')} "
                    f"boost_reason={safe_info.get('state_beta_boost_reason')} "
                    f"actual_anchor_source={safe_info.get('actual_tr_anchor_source')} "
                    f"selected_source={safe_info.get('selected_candidate_source', safe_info.get('selected_source'))} "
                    f"rank_mu={safe_info.get('selected_candidate_rank_by_mu', safe_info.get('selected_rank_by_mu'))} "
                    f"rank_score={safe_info.get('selected_candidate_rank_by_score', safe_info.get('selected_rank_by_score'))} "
                    f"rank_acq={safe_info.get('selected_candidate_rank_by_acq', safe_info.get('selected_rank_by_acq'))} "
                    f"current_cost={safe_info.get('current_train_cost')} "
                    f"surprise={safe_info.get('surprise')}",
                    flush=True,
                )
        if (i + 1) % 10 == 0:
            print(f"  [{group_cfg['label']} | SAFEBO={safe_info.get('deploy_policy')}/{safe_info.get('deploy_source')} | feedback={_dual_feedback_mode()}] Iteration {i + 1}/{CFG.BO_ITERATIONS}")

    if fac._use_cohort_feedback() and bool(getattr(CFG, "COHORT_FORCE_FINALIZE_AT_RUN_END", True)):
        fac._finalize_ready_cohorts(fac.current_time, force=True, reason="run_end")
        if fac.agent is not None and _dual_is_enabled():
            _dual_apply_pending_refinements(fac, fac.agent)
    # v6.2 runtime logging: total method runtime and compact dimension metadata.
    runtime_group_elapsed = time.perf_counter() - runtime_group_t0
    agent_kwargs = group_cfg.get("agent_kwargs", {}) or {}
    try:
        control_dim = int(len(group_cfg.get("fixed_theta", []))) if group_cfg.get("fixed_theta") is not None else int(agent_kwargs.get("dim", 0) or 0)
    except Exception:
        control_dim = int(agent_kwargs.get("dim", 0) or 0)
    try:
        context_dim = int(agent_kwargs.get("context_dim", 0) or 0)
    except Exception:
        context_dim = 0
    total_model_dim = int(control_dim + context_dim)
    fac.perf_log["runtime_group_elapsed_sec"] = [float(runtime_group_elapsed)]
    fac.perf_log["runtime_group_elapsed_min"] = [float(runtime_group_elapsed / 60.0)]
    fac.perf_log["runtime_group_sec_per_iter"] = [float(runtime_group_elapsed / max(1, int(CFG.BO_ITERATIONS)))]
    fac.perf_log["runtime_control_dim"] = [int(control_dim)]
    fac.perf_log["runtime_context_dim"] = [int(context_dim)]
    fac.perf_log["runtime_total_model_dim"] = [int(total_model_dim)]
    fac.perf_log["cohort_feedback_debug_rows"] = list(getattr(fac, "cohort_feedback_rows", []))
    return fac.perf_log



# ===============================================================
# OFFLINE WINDOW NOISE DIAGNOSTIC
# ===============================================================
def _noise_numeric_array(values):
    out = []
    for v in values or []:
        try:
            x = float(v)
            if np.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return np.asarray(out, dtype=float)


def _offline_default_fixed_keys():
    return [
        "reduced6_fixed_mid",
        "reduced6_fixed_tuned",
        "reduced6_fixed_queue_high",
        "reduced6_fixed_risk_high",
        "reduced6_fixed_edge_safe",
    ]


def _extract_cost_series_from_log(log):
    """Return per-window evaluation cost.

    Preferred source is eval_cost if the refactor logging path filled it.
    Fallback is -reward, which is equivalent to metrics["cost"] in the
    original window feedback path.
    """
    eval_cost = log.get("eval_cost", []) if isinstance(log, dict) else []
    arr = _noise_numeric_array(eval_cost)
    if len(arr) > 0:
        return arr
    reward = log.get("reward", []) if isinstance(log, dict) else []
    r = _noise_numeric_array(reward)
    return -r


def run_offline_window_noise_diagnostic(repeat_runs=10, selected_keys=None, output_dir=None):
    """Offline diagnostic: estimate whether window-level feedback noise is large.

    This is NOT an online training method. It repeatedly evaluates fixed policies
    under the same scenario configuration and compares:
      1. within-policy per-window cost volatility;
      2. between-policy mean gaps;
      3. paired same-seed/same-iteration deltas, approximating a CRN-style check;
      4. rank stability across seeds.

    Use this to answer: are differences among fixed policies larger than the
    per-window noise BO receives as one observation?
    """
    global SCENARIO_SAVE_DIR
    old_save_dir = SCENARIO_SAVE_DIR
    root = output_dir or os.path.abspath("offline_window_noise_diagnostic")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    SCENARIO_SAVE_DIR = root

    groups = build_scenario_method_groups()
    if selected_keys is None:
        selected_keys = _offline_default_fixed_keys()
    selected_keys = normalize_selected_method_keys([str(k).strip() for k in selected_keys if str(k).strip()])
    selected_keys = [k for k in selected_keys if k in groups and groups[k].get("agent") is None and "fixed_theta" in groups[k]]
    if not selected_keys:
        raise ValueError("offline_noise requires fixed policy keys. Example: --selected-keys fixed_mid,fixed_tuned,fixed_queue_high,fixed_risk_high,fixed_edge_safe")

    config_payload = {
        "diagnostic": "offline_window_noise",
        "refactor_version": REFACTOR_VERSION,
        "selected_keys": selected_keys,
        "repeat_runs": int(max(1, repeat_runs)),
        "bo_iterations": int(CFG.BO_ITERATIONS),
        "bo_interval": float(CFG.BO_INTERVAL),
        "session_duration": float(CFG.SESSION_DURATION),
        "lambda_schedule": list(getattr(CFG, "LAMBDA_SCHEDULE", [])),
        "task_type_probs": dict(getattr(CFG, "TASK_TYPE_PROBS", {})),
        "use_task_type_adaptation": bool(getattr(CFG, "USE_TASK_TYPE_ADAPTATION", False)),
        "cloud_delay_mult": float(getattr(CFG, "CLOUD_DELAY_MULT", 1.0)),
        "cloud_energy_mult": float(getattr(CFG, "CLOUD_ENERGY_MULT", 1.0)),
        "cloud_speed_mult": float(getattr(CFG, "CLOUD_SPEED_MULT", 1.0)),
        "note": "This diagnostic does not train BO. It quantifies fixed-policy window-cost variance and policy gap/noise ratios.",
    }
    with open(os.path.join(root, "offline_window_noise_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    sample_rows = []
    run_mean_rows = []
    print("=== Offline Window Noise Diagnostic ===")
    print(f"methods={selected_keys}, repeats={repeat_runs}, output={root}")
    try:
        for run_idx in range(max(1, repeat_runs)):
            seed = CFG.BASE_SEED + run_idx
            print(f"[Noise repeat {run_idx + 1}/{max(1, repeat_runs)}] seed={seed}")
            for key in selected_keys:
                cfg = groups[key]
                log = run_scenario_group(seed, key, cfg)
                costs = _extract_cost_series_from_log(log)
                rewards = _noise_numeric_array(log.get("reward", []))
                arrivals = _noise_numeric_array(log.get("arrivals_total", []))
                completed = _noise_numeric_array(log.get("completed_total", []))
                backlog = _noise_numeric_array(log.get("backlog", []))
                feedback_conf = _noise_numeric_array(log.get("feedback_confidence", []))
                n = int(len(costs))
                for i in range(n):
                    sample_rows.append({
                        "Seed": int(seed),
                        "Repeat_Index": int(run_idx),
                        "Method_Key": key,
                        "Method_Label": cfg.get("label", key),
                        "Iteration": int(i + 1),
                        "Window_Cost": float(costs[i]),
                        "Reward": float(rewards[i]) if i < len(rewards) else np.nan,
                        "Arrivals": float(arrivals[i]) if i < len(arrivals) else np.nan,
                        "Completed": float(completed[i]) if i < len(completed) else np.nan,
                        "Backlog": float(backlog[i]) if i < len(backlog) else np.nan,
                        "Feedback_Confidence": float(feedback_conf[i]) if i < len(feedback_conf) else np.nan,
                    })
                if n > 0:
                    run_mean_rows.append({
                        "Seed": int(seed),
                        "Repeat_Index": int(run_idx),
                        "Method_Key": key,
                        "Method_Label": cfg.get("label", key),
                        "Run_Mean_Cost": float(np.nanmean(costs)),
                        "Run_Std_Window_Cost": float(np.nanstd(costs, ddof=1)) if n > 1 else 0.0,
                        "Run_CV_Window_Cost": float(np.nanstd(costs, ddof=1) / max(1e-12, abs(np.nanmean(costs)))) if n > 1 else 0.0,
                    })

        sample_df = pd.DataFrame(sample_rows)
        run_df = pd.DataFrame(run_mean_rows)
        sample_path = os.path.join(root, "offline_window_noise_samples.csv")
        run_path = os.path.join(root, "offline_window_noise_run_means.csv")
        sample_df.to_csv(sample_path, index=False)
        run_df.to_csv(run_path, index=False)

        method_rows = []
        for key, sub in sample_df.groupby("Method_Key"):
            costs = sub["Window_Cost"].astype(float).to_numpy()
            run_sub = run_df[run_df["Method_Key"] == key]
            mean_cost = float(np.nanmean(costs)) if len(costs) else np.nan
            std_window = float(np.nanstd(costs, ddof=1)) if len(costs) > 1 else 0.0
            std_run_mean = float(np.nanstd(run_sub["Run_Mean_Cost"].astype(float), ddof=1)) if len(run_sub) > 1 else 0.0
            method_rows.append({
                "Method_Key": key,
                "Method_Label": str(sub["Method_Label"].iloc[0]),
                "Mean_Window_Cost": mean_cost,
                "Std_Window_Cost": std_window,
                "CV_Window_Cost": std_window / max(1e-12, abs(mean_cost)) if np.isfinite(mean_cost) else np.nan,
                "Mean_Run_Cost": float(np.nanmean(run_sub["Run_Mean_Cost"].astype(float))) if len(run_sub) else np.nan,
                "Std_Run_Mean_Cost": std_run_mean,
                "Num_Windows": int(len(costs)),
                "Num_Repeats": int(len(run_sub)),
            })
        method_df = pd.DataFrame(method_rows).sort_values("Mean_Window_Cost")
        method_path = os.path.join(root, "offline_window_noise_method_summary.csv")
        method_df.to_csv(method_path, index=False)

        # Paired same-seed/same-iteration policy deltas.
        pair_rows = []
        for i, key_a in enumerate(selected_keys):
            for key_b in selected_keys[i + 1:]:
                a = sample_df[sample_df["Method_Key"] == key_a][["Seed", "Iteration", "Window_Cost"]].rename(columns={"Window_Cost": "Cost_A"})
                b = sample_df[sample_df["Method_Key"] == key_b][["Seed", "Iteration", "Window_Cost"]].rename(columns={"Window_Cost": "Cost_B"})
                merged = pd.merge(a, b, on=["Seed", "Iteration"], how="inner")
                if merged.empty:
                    continue
                delta = merged["Cost_A"].astype(float) - merged["Cost_B"].astype(float)
                mean_a = float(np.nanmean(merged["Cost_A"].astype(float)))
                mean_b = float(np.nanmean(merged["Cost_B"].astype(float)))
                mean_delta = float(np.nanmean(delta))
                std_delta = float(np.nanstd(delta, ddof=1)) if len(delta) > 1 else 0.0
                pooled_std = float(np.sqrt(0.5 * (np.nanvar(merged["Cost_A"].astype(float), ddof=1) + np.nanvar(merged["Cost_B"].astype(float), ddof=1)))) if len(delta) > 1 else 0.0
                best_mean = min(mean_a, mean_b)
                pair_rows.append({
                    "Method_A": key_a,
                    "Method_B": key_b,
                    "Mean_Cost_A": mean_a,
                    "Mean_Cost_B": mean_b,
                    "Mean_Delta_A_minus_B": mean_delta,
                    "Std_Paired_Delta": std_delta,
                    "Abs_Delta_Over_StdDelta": abs(mean_delta) / max(1e-12, std_delta),
                    "Abs_Delta_Over_PooledWindowStd": abs(mean_delta) / max(1e-12, pooled_std),
                    "Relative_Gap_Pct_of_Best": 100.0 * abs(mean_delta) / max(1e-12, abs(best_mean)),
                    "Paired_Win_Rate_A_Lower": float(np.nanmean(delta < 0.0)),
                    "Num_Paired_Windows": int(len(delta)),
                })
        pair_df = pd.DataFrame(pair_rows)
        pair_path = os.path.join(root, "offline_window_noise_pairwise.csv")
        pair_df.to_csv(pair_path, index=False)

        # Rank stability by repeat/run mean.
        rank_rows = []
        if not run_df.empty:
            for seed, sub in run_df.groupby("Seed"):
                sub_sorted = sub.sort_values("Run_Mean_Cost")
                for rank, (_, row) in enumerate(sub_sorted.iterrows(), start=1):
                    rank_rows.append({
                        "Seed": int(seed),
                        "Method_Key": row["Method_Key"],
                        "Run_Mean_Cost": float(row["Run_Mean_Cost"]),
                        "Rank_LowerCostBetter": int(rank),
                        "Is_Top1": int(rank == 1),
                    })
        rank_df = pd.DataFrame(rank_rows)
        rank_path = os.path.join(root, "offline_window_noise_rank_by_seed.csv")
        rank_df.to_csv(rank_path, index=False)
        top1_df = pd.DataFrame()
        if not rank_df.empty:
            top1_df = rank_df.groupby("Method_Key", as_index=False).agg(
                Mean_Rank=("Rank_LowerCostBetter", "mean"),
                Std_Rank=("Rank_LowerCostBetter", "std"),
                Top1_Count=("Is_Top1", "sum"),
                Repeat_Count=("Is_Top1", "count"),
            )
            top1_df["Top1_Rate"] = top1_df["Top1_Count"] / top1_df["Repeat_Count"].clip(lower=1)
            top1_df = top1_df.sort_values(["Mean_Rank", "Top1_Rate"], ascending=[True, False])
            top1_df.to_csv(os.path.join(root, "offline_window_noise_rank_stability.csv"), index=False)

        # A compact textual report.
        report_lines = []
        report_lines.append("Offline window-noise diagnostic")
        report_lines.append("================================")
        report_lines.append(f"Output directory: {root}")
        report_lines.append(f"Methods: {', '.join(selected_keys)}")
        report_lines.append(f"Repeats: {int(max(1, repeat_runs))}, BO windows per repeat: {int(CFG.BO_ITERATIONS)}")
        if not method_df.empty:
            best_row = method_df.iloc[0]
            worst_row = method_df.iloc[-1]
            spread = float(worst_row["Mean_Window_Cost"] - best_row["Mean_Window_Cost"])
            typical_std = float(method_df["Std_Window_Cost"].median())
            report_lines.append("")
            report_lines.append(f"Best mean fixed policy: {best_row['Method_Key']} cost={best_row['Mean_Window_Cost']:.4f}")
            report_lines.append(f"Worst mean fixed policy: {worst_row['Method_Key']} cost={worst_row['Mean_Window_Cost']:.4f}")
            report_lines.append(f"Fixed-policy mean spread: {spread:.4f}")
            report_lines.append(f"Median per-window std within policy: {typical_std:.4f}")
            report_lines.append(f"Spread / median per-window std: {spread / max(1e-12, typical_std):.4f}")
            if spread / max(1e-12, typical_std) < 1.0:
                report_lines.append("Interpretation: policy gaps are smaller than a typical single-window fluctuation; one-window BO observations are likely noisy.")
            elif spread / max(1e-12, typical_std) < 2.0:
                report_lines.append("Interpretation: policy gaps are visible but close to the single-window noise scale; confidence/recent filtering may help.")
            else:
                report_lines.append("Interpretation: policy gaps are larger than typical single-window noise; BO should be able to learn if feedback attribution is otherwise clean.")
        report_lines.append("")
        report_lines.append("Key files:")
        report_lines.append("- offline_window_noise_samples.csv: one row per method/seed/window")
        report_lines.append("- offline_window_noise_method_summary.csv: per-method mean/std/CV")
        report_lines.append("- offline_window_noise_pairwise.csv: paired same-seed/same-iteration deltas")
        report_lines.append("- offline_window_noise_rank_stability.csv: rank stability across seeds")
        with open(os.path.join(root, "offline_window_noise_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print("\n".join(report_lines))
        return {
            "samples": sample_df,
            "method_summary": method_df,
            "pairwise": pair_df,
            "rank": rank_df,
            "top1": top1_df,
        }
    finally:
        SCENARIO_SAVE_DIR = old_save_dir


# ===============================================================
# v6.1: short filename export helper
# ---------------------------------------------------------------
# 不改原始结果文件；在 output_dir/_short_export 里复制一份英文短名，并写 filename_mapping.csv。
# 用法：命令行加 --export-short-names。
# ===============================================================

def _short_clean_ascii_name(s):
    repl = {
        "核心指标统计": "key_metrics", "实验汇总": "summary", "轮次汇总": "round",
        "情景调试": "ctx", "节点分配调试": "alloc_node",
        "任务类型节点分配汇总": "alloc_type_sum", "任务类型节点分配调试": "alloc_type_dbg",
        "任务类型节点堆叠图": "alloc_type_stack", "任务类型云边占比": "alloc_cloud_ratio",
        "方法云占比对比": "cloud_ratio_cmp", "批次反馈学习曲线": "learn_curve",
        "每轮平均能耗时延评分": "round_metrics", "控制参数轨迹": "theta_traj",
        "全部方法任务类型分配汇总": "alloc_type_all",
    }
    for k, v in repl.items():
        s = str(s).replace(k, v)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "file"


def _short_scene_part(part):
    part = str(part)
    if part == "window_original":
        return "win"
    if part == "paired_fixed_mid_delta":
        return "paired"
    m = re.match(r"lam(\d+)p(\d+)_RT(\d+)_Batch(\d+)_AI(\d+)", part)
    if m:
        return f"l{m.group(1)}{m.group(2)}_r{m.group(3)}_b{m.group(4)}_a{m.group(5)}"
    m = re.match(r"RT(\d+)_Batch(\d+)_AI(\d+)", part)
    if m:
        return f"r{m.group(1)}_b{m.group(2)}_a{m.group(3)}"
    return _short_clean_ascii_name(part)[:80]


def _short_file_name_for_export(filename):
    p = os.path.basename(str(filename))
    stem, ext = os.path.splitext(p)
    method_map = {
        "reduced6_fixed_mid": "fm", "reduced6_fixed_tuned": "ft",
        "reduced6_fixed_queue_high": "fq", "reduced6_fixed_risk_high": "fr", "reduced6_fixed_edge_safe": "fe",
        "reduced6_bo_greedy": "bo", "reduced6_bo_ei": "boei",
        "reduced6_cbo_lite_full": "cbo_full", "reduced6_cbo_lite_pressure_only": "cbo_p",
        "reduced6_cbo_lite_load_only": "cbo_l", "reduced6_cbo_lite_util_only": "cbo_u",
        "reduced6_cbo_lite_no_cloud": "cbo_nc", "reduced6_cbo_lite_no_arrival": "cbo_na",
        "reduced6_cbo_lite_taskmix": "cbo_tm", "reduced6_cbo_lite_recent_mix": "cbo_rm",
        "reduced6_cbo_lite_prev_counts": "cbo_cnt", "reduced6_cbo_lite_pressure_taskmix": "cbo_pt",
        "reduced6_cbo_lite_pressure_recent_mix": "cbo_prm", "reduced6_cbo_lite_pressure_counts": "cbo_pc",
        "reduced6_cbo_lite_pressure_taskmix_counts": "cbo_ptc", "reduced6_cbo_lite_full_taskmix": "cbo_ftm",
        "reduced6_cbo_lite_full_taskmix_counts": "cbo_ftc",
        "direct_round_robin": "rr_direct",
        "direct_greedy_cost": "greedy_direct",
        "direct_least_load": "leastload_direct",
        "direct_queue_aware_greedy": "qaware_direct",
    }
    suffix_map = {
        "round_summary_轮次汇总": "round", "context_debug_情景调试": "ctx",
        "alloc_debug_节点分配调试": "alloc_node", "alloc_by_type_summary_任务类型节点分配汇总": "alloc_type_sum",
        "alloc_by_type_debug_任务类型节点分配调试": "alloc_type_dbg",
        "alloc_by_type_stacked_任务类型节点堆叠图": "alloc_type_stack",
        "alloc_by_type_cloud_ratio_任务类型云边占比": "alloc_cloud_ratio",
    }
    exact_map = {
        "key_metric_summary_核心指标统计": "key_metrics", "scenario_experiment_summary_实验汇总": "scene_summary",
        "scenario_phase_summary": "phase_summary", "refactor_run_config": "run_config",
        "alloc_by_type_all_methods_summary_全部方法任务类型分配汇总": "alloc_type_all",
        "alloc_by_type_method_cloud_ratio_compare_方法云占比对比": "cloud_ratio_cmp",
        "scenario_theta_trajectory_控制参数轨迹": "theta_traj", "scenario_convergence": "scene_conv",
        "scenario_best_so_far": "scene_best", "scenario_alloc_heatmaps": "scene_alloc_heatmap",
        "scenario_task_delay_bars": "scene_task_delay",
        "scenario_round_mean_energy_delay_score_每轮平均能耗时延评分": "scene_round_metrics",
        "scenario_cohort_learning_curves_批次反馈学习曲线": "scene_learn",
    }
    if stem in exact_map:
        return exact_map[stem] + ext
    for long_m, short_m in method_map.items():
        if stem.startswith(long_m + "_"):
            rest = stem[len(long_m) + 1:]
            for long_suf, short_suf in suffix_map.items():
                if rest == long_suf:
                    return f"{short_m}_{short_suf}{ext}"
            return f"{short_m}_{_short_clean_ascii_name(rest)[:60]}{ext}"
    return _short_clean_ascii_name(stem)[:90] + ext


def export_short_named_results(output_dir, export_dir=None, make_tar=True):
    """复制 output_dir 中的常用结果文件到短文件名目录，避免 Windows/网盘下载丢长路径。"""
    import shutil
    import csv
    import tarfile
    root = os.path.abspath(output_dir or SCENARIO_SAVE_DIR)
    if not os.path.exists(root):
        print(f"[WARN] short export skipped; missing output_dir={root}")
        return None
    export_dir = os.path.abspath(export_dir or os.path.join(root, "_short_export"))
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)
    keep_ext = {".csv", ".png", ".json", ".log", ".txt"}
    mapping = []
    for base, _, files in os.walk(root):
        # 不把导出目录再递归导出一遍
        if os.path.abspath(base).startswith(os.path.abspath(export_dir)):
            continue
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in keep_ext:
                continue
            src = os.path.join(base, name)
            rel = os.path.relpath(src, root)
            parts = rel.split(os.sep)
            short_parts = [_short_scene_part(x) for x in parts[:-1]]
            short_name = _short_file_name_for_export(parts[-1])
            dst_dir = os.path.join(export_dir, *short_parts)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, short_name)
            stem, dst_ext = os.path.splitext(dst)
            k = 2
            while os.path.exists(dst):
                dst = f"{stem}_{k}{dst_ext}"
                k += 1
            shutil.copy2(src, dst)
            mapping.append({"original_relpath": rel, "short_relpath": os.path.relpath(dst, export_dir)})
    mapping_path = os.path.join(export_dir, "filename_mapping.csv")
    with open(mapping_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original_relpath", "short_relpath"])
        writer.writeheader()
        writer.writerows(mapping)
    tar_path = None
    if make_tar:
        tar_path = export_dir.rstrip(os.sep) + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(export_dir, arcname=os.path.basename(export_dir))
    print(f"[OK] short export files={len(mapping)} dir={export_dir}" + (f" tar={tar_path}" if tar_path else ""))
    return export_dir

# ===============================================================
# END SAFE BO + CBO GREEDY + DUAL FEEDBACK PATCH V3
# ===============================================================


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "param", "extreme", "scan", "sensitivity", "scenario", "ratio_grid", "pressure_scan", "offline_noise"], default="all")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--local-delta", type=float, default=0.08)
    parser.add_argument("--dim", type=str, default="W_RT_Latency")
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--fixed-rng", action="store_true")
    parser.add_argument("--fixed-seed", type=int, default=None)
    parser.add_argument("--ratio-step", type=int, default=10, help="任务比例网格步长，默认10，即10%%")
    parser.add_argument("--ratio-min", type=int, default=10, help="每类任务最低比例，默认10，即至少10%%")
    parser.add_argument("--output-root", type=str, default=None, help="ratio_grid输出根目录")
    parser.add_argument("--selected-keys", type=str, default=None, help="逗号分隔的方法键；默认使用备份版 no-RoundRobin 方法集")
    parser.add_argument("--sensitivity-scenarios", type=str, default="default,rt_high,batch_high,ai_high", help="敏感度场景：default,rt_high,batch_high,ai_high 或 name:RT,Batch,AI")
    parser.add_argument("--sens-seeds", type=int, default=2, help="敏感度分析每个参数点重复的随机种子数")
    parser.add_argument("--sens-windows", type=int, default=3, help="敏感度分析每个 theta 连续评价的窗口数")
    parser.add_argument("--sens-greedy", action="store_true", help="敏感度分析时关闭 Boltzmann 随机，使用确定性机会集合选择")
    parser.add_argument("--pref-norm-mode", choices=["rolling", "fixed"], default="rolling", help="静态节点偏好诊断使用的归一化模式")
    parser.add_argument("--task-adaptation", action="store_true", help="启用 type_speed_factor，节点对不同任务类型有差异化速度")
    parser.add_argument("--no-task-adaptation", action="store_true", help="关闭 type_speed_factor，所有任务只使用节点基础 speed")
    parser.add_argument("--rt-deadline-factor", type=float, default=None, help="覆盖 RT 任务 deadline_factor，例如 2.5")
    parser.add_argument("--bo-iterations", type=int, default=None, help="覆盖 BO_ITERATIONS，便于快速测试")
    parser.add_argument("--bo-interval", type=float, default=None, help="覆盖 BO_INTERVAL")
    parser.add_argument("--session-duration", type=float, default=None, help="覆盖 SESSION_DURATION")
    parser.add_argument("--feedback-mode", choices=["window", "cohort_complete", "dual"], default="window", help="BO反馈模式：window为旧窗口级反馈；cohort_complete为任务批次完成后反馈；dual为窗口快反馈+批次/分类精反馈替换")
    parser.add_argument("--feedback-score", choices=["window_original", "task_effective", "task_effective_backlog", "task_effective_backlog_violation", "paired_fixed_mid_delta", "legacy_dual", "legacy_cohort"], default=getattr(CFG, "DEFAULT_SCENARIO_FEEDBACK_SCORE", "task_effective_backlog_violation"), help="BO tell 使用的训练反馈；备份版默认 task_effective_backlog_violation。paired_fixed_mid_delta 为仿真专用：同窗口 shadow fixed_mid 的 delta cost。")
    parser.add_argument("--cbo-reference-mode", choices=["off", "calibrate", "load", "auto_macro"], default="off", help="Scenario reference baseline mode for normalized metrics")
    parser.add_argument("--cbo-reference-calibration-rounds", type=int, default=30, help="Rounds used to build/freeze scenario reference")
    parser.add_argument("--cbo-reference-min-rounds", type=int, default=5, help="Minimum rounds before reference is considered available")
    parser.add_argument("--cbo-reference-stat", choices=["median", "trimmed_mean", "mean"], default="median", help="Statistic used for reference calibration")
    parser.add_argument("--cbo-reference-trim-pct", type=float, default=0.1, help="Trim percent for trimmed_mean reference")
    parser.add_argument("--cbo-reference-freeze-after-calibration", action="store_true", default=True, help="Freeze reference after calibration window")
    parser.add_argument("--cbo-reference-file", type=str, default="", help="JSON reference file to load")
    parser.add_argument("--cbo-reference-output-file", type=str, default="", help="JSON reference output file")
    parser.add_argument("--cbo-objective-mode", choices=["eval_cost", "diagnostic_only", "normalized_tradeoff"], default="eval_cost", help="BO training objective mode")
    parser.add_argument("--cbo-tradeoff-alpha", type=float, default=0.8, help="alpha in alpha*service_norm + (1-alpha)*energy_norm")
    parser.add_argument("--cbo-alpha-min", type=float, default=0.6, help="minimum clipped alpha")
    parser.add_argument("--cbo-alpha-max", type=float, default=0.95, help="maximum clipped alpha")
    parser.add_argument("--cbo-target-success-rate", type=float, default=0.995, help="target SLA success rate for normalized service score")
    parser.add_argument("--cbo-unfinished-penalty-weight", type=float, default=5.0, help="unfinished_rate penalty in service_norm")
    parser.add_argument("--cbo-success-shortfall-weight", type=float, default=2.0, help="success shortfall normalized penalty in service_norm")
    parser.add_argument("--cbo-backlog-growth-penalty-weight", type=float, default=2.0, help="backlog growth rate penalty in service_norm")
    parser.add_argument("--cbo-class-imbalance-weight", type=float, default=0.0, help="class completion imbalance penalty in service_norm")
    parser.add_argument("--cbo-normalized-ratio-clip-min", type=float, default=0.2, help="min clip for normalized ratios")
    parser.add_argument("--cbo-normalized-ratio-clip-max", type=float, default=5.0, help="max clip for normalized ratios")
    parser.add_argument("--scheduler-tradeoff-mode", choices=["legacy", "alpha_fixed", "alpha_from_ratio"], default="legacy", help="底层调度器节点 score 的 service-energy tradeoff 模式；默认 legacy 保持旧逻辑")
    parser.add_argument("--scheduler-tradeoff-alpha", type=float, default=0.85, help="alpha_fixed 模式下的 service 权重 alpha")
    parser.add_argument("--scheduler-alpha-min", type=float, default=0.60, help="scheduler alpha 下限")
    parser.add_argument("--scheduler-alpha-max", type=float, default=0.97, help="scheduler alpha 上限")
    parser.add_argument("--scheduler-service-latency-weight", type=float, default=1.0, help="alpha tradeoff 中 norm_l 的系数")
    parser.add_argument("--scheduler-service-risk-weight", type=float, default=1.0, help="alpha 外部 risk_w*norm_risk 惩罚项的额外系数")
    parser.add_argument("--scheduler-service-queue-weight", type=float, default=1.0, help="alpha 外部 queue_w*norm_queue 惩罚项的额外系数")
    parser.add_argument("--scheduler-energy-weight", type=float, default=1.0, help="energy_component 中 norm_e 的系数")
    parser.add_argument("--scheduler-score-norm-mode", choices=["legacy", "candidate_median", "candidate_iqr", "rolling_ema"], default="legacy", help="底层调度器 score 的 energy/latency 归一化模式；默认 legacy 保持 norm_mode 行为")
    parser.add_argument("--scheduler-norm-clip-max", type=float, default=3.0, help="scheduler score normalization clip max")
    parser.add_argument("--scheduler-norm-eps", type=float, default=1e-6, help="scheduler score normalization epsilon")
    parser.add_argument("--scheduler-norm-ema-alpha", type=float, default=0.995, help="rolling_ema scheduler normalization alpha")
    parser.add_argument("--paired-baseline-key", default="reduced6_fixed_mid", help="paired_fixed_mid_delta 使用的 shadow baseline fixed policy key/alias，默认 reduced6_fixed_mid。")
    parser.add_argument("--deploy-policy", choices=["ei", "greedy", "incumbent", "incumbent_safe", "safe", "safe_bo"], default=None, help="SAFEBO 部署策略别名：ei=BO-EI；greedy=BO-greedy posterior mean；incumbent/safe=安全 incumbent。命令行值只覆盖 CBO 类方法，fixed/direct baseline 不受影响。")
    parser.add_argument("--dual-refined-source", choices=["class", "class_weighted", "class_equal", "class_actual", "class_worst", "cohort"], default=None, help="dual模式下延后精反馈的合成方式：class/class_weighted=按设定任务比例合成；class_equal=三类等权；class_actual=按本批实际到达比例；class_worst=最差类别；cohort=原始整体cohort cost")
    parser.add_argument("--dual-class-metric", choices=["completed", "effective", "censored"], default=None, help="dual分类精反馈内部每类指标使用方式：completed=只用已完成任务均值+完成率惩罚；effective/censored=把未完成任务按截尾等待和当前违约计入")
    parser.add_argument("--task-probs", type=str, default=None, help="固定任务比例 RT,Batch,AI，例如 0.2,0.4,0.4 或 20,40,40")
    parser.add_argument("--task-prob-schedule", type=str, default=None, help="分段任务比例，例如 0:4000:20,40,40;4000:9000:40,10,50")
    parser.add_argument("--lambda-schedule", type=str, default=None, help="分段泊松强度，例如 0:4000:1.0,4000:9000:2.2,9000:12000:1.2")
    parser.add_argument("--lambda-values", type=str, default=None, help="pressure_scan 用，逗号分隔，例如 1.0,1.4,1.8,2.2,2.6,3.0")
    parser.add_argument("--bo-history-mode", choices=["all", "recent", "confidence", "recent_confidence"], default=getattr(CFG, "DEFAULT_BO_HISTORY_MODE", "recent"), help="BO GP训练历史使用方式；备份版默认 recent。all=全部；recent=最近N个；confidence=过滤低可信反馈；recent_confidence=最近+可信度过滤")
    parser.add_argument("--bo-recent-window", type=int, default=getattr(CFG, "DEFAULT_BO_RECENT_WINDOW", 80), help="recent/recent_confidence 模式保留最近多少个BO样本；备份版默认80")
    parser.add_argument("--bo-confidence-min", type=float, default=None, help="confidence/recent_confidence 模式下保留样本的最低反馈可信度")
    parser.add_argument("--bo-confidence-min-samples", type=int, default=None, help="可信过滤后至少保留多少样本，不足时自动使用高可信+最近样本兜底")
    parser.add_argument("--cbo-history-select-mode", choices=["recent", "recent_context", "recent_context_elite", "hybrid"], default=getattr(CFG, "DEFAULT_CBO_HISTORY_SELECT_MODE", "recent"), help="CBO-only stability history selection mode")
    parser.add_argument("--cbo-context-k", type=int, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_K", 50), help="CBO context-nearest historical sample count")
    parser.add_argument("--cbo-elite-k", type=int, default=getattr(CFG, "DEFAULT_CBO_ELITE_K", 20), help="CBO robust elite theta-region count")
    parser.add_argument("--cbo-diverse-k", type=int, default=getattr(CFG, "DEFAULT_CBO_DIVERSE_K", 20), help="CBO diversity sample count for hybrid history selection")
    parser.add_argument("--cbo-robust-score-mode", choices=["none", "mean", "mean_std", "context_weighted_mean_std"], default=getattr(CFG, "DEFAULT_CBO_ROBUST_SCORE_MODE", "none"), help="Robust elite/incumbent score mode")
    parser.add_argument("--cbo-robust-std-weight", type=float, default=getattr(CFG, "DEFAULT_CBO_ROBUST_STD_WEIGHT", 0.5), help="Std penalty coefficient in robust score")
    parser.add_argument("--cbo-theta-merge-eps", type=float, default=getattr(CFG, "DEFAULT_CBO_THETA_MERGE_EPS", 0.05), help="Theta-region merge epsilon in normalized theta space")
    parser.add_argument("--cbo-context-sim-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_CONTEXT_SIM_THRESHOLD", 0.0), help="Minimum context similarity for context-aware history/robust incumbent; 0 disables hard filtering")
    parser.add_argument("--cbo-tr-mode", choices=["off", "good_region", "adaptive", "residual_adaptive", "condition_adaptive"], default=getattr(CFG, "DEFAULT_CBO_TR_MODE", "off"), help="CBO trust-region mode")
    parser.add_argument("--cbo-tr-radius-init", type=float, default=getattr(CFG, "TRUST_RADIUS_INIT", 0.10), help="CBO TR initial radius")
    parser.add_argument("--cbo-tr-radius-min", type=float, default=getattr(CFG, "TRUST_RADIUS_MIN", 0.04), help="CBO TR min radius")
    parser.add_argument("--cbo-tr-radius-max", type=float, default=getattr(CFG, "TRUST_RADIUS_MAX", 0.35), help="CBO TR max radius")
    parser.add_argument("--cbo-tr-grow", type=float, default=getattr(CFG, "TRUST_RADIUS_GROWTH", 1.15), help="CBO TR grow factor")
    parser.add_argument("--cbo-tr-shrink", type=float, default=getattr(CFG, "TRUST_RADIUS_SHRINK", 0.92), help="CBO TR shrink factor")
    parser.add_argument("--cbo-tr-update-mode", choices=["best_so_far", "rolling_mean", "ewma_trend"], default=getattr(CFG, "DEFAULT_CBO_TR_UPDATE_MODE", "best_so_far"), help="CBO TR radius update rule")
    parser.add_argument("--cbo-tr-compare-window", type=int, default=getattr(CFG, "DEFAULT_CBO_TR_COMPARE_WINDOW", 30), help="Recent window for rolling/ewma TR trend comparison")
    parser.add_argument("--cbo-tr-baseline-window", type=int, default=getattr(CFG, "DEFAULT_CBO_TR_BASELINE_WINDOW", 60), help="Baseline window before the compare window for TR trend comparison")
    parser.add_argument("--cbo-tr-improve-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_TR_IMPROVE_PCT", 0.015), help="Rolling improvement threshold for TR refine/shrink")
    parser.add_argument("--cbo-tr-worsen-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_TR_WORSEN_PCT", 0.03), help="Rolling worsening threshold for TR expand/grow")
    parser.add_argument("--cbo-tr-deadband-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_TR_DEADBAND_PCT", 0.01), help="Deadband threshold for holding TR radius")
    parser.add_argument("--cbo-tr-update-patience", type=int, default=getattr(CFG, "DEFAULT_CBO_TR_UPDATE_PATIENCE", 2), help="Consecutive trend signals needed before changing TR radius")
    parser.add_argument("--cbo-tr-anchor-mode", choices=["posterior_mean", "robust_elite", "recent_best", "context_best"], default=getattr(CFG, "DEFAULT_CBO_TR_ANCHOR_MODE", "posterior_mean"), help="CBO TR anchor selection mode")
    parser.add_argument("--cbo-robust-incumbent-mode", choices=["off", "recommend_only", "deploy"], default=getattr(CFG, "DEFAULT_CBO_ROBUST_INCUMBENT_MODE", "off"), help="Robust incumbent diagnostic/deploy mode; default off")
    parser.add_argument("--cbo-macro-gate-mode", choices=["off", "topk", "weighted_topk", "hierarchical"], default=getattr(CFG, "DEFAULT_CBO_MACRO_GATE_MODE", "off"), help="CBO macro workload gate mode")
    parser.add_argument("--cbo-macro-k", type=int, default=getattr(CFG, "DEFAULT_CBO_MACRO_K", 100), help="CBO macro workload topK historical samples")
    parser.add_argument("--cbo-macro-total-scale", type=str, default=getattr(CFG, "DEFAULT_CBO_MACRO_TOTAL_SCALE", "auto"), help="Macro total arrivals scale; auto or numeric")
    parser.add_argument("--cbo-macro-lengthscale-total", type=float, default=getattr(CFG, "DEFAULT_CBO_MACRO_LENGTHSCALE_TOTAL", 1.0), help="RBF lengthscale for normalized total arrivals")
    parser.add_argument("--cbo-macro-lengthscale-rt", type=float, default=getattr(CFG, "DEFAULT_CBO_MACRO_LENGTHSCALE_RT", 0.15), help="RBF lengthscale for RT ratio")
    parser.add_argument("--cbo-macro-lengthscale-batch", type=float, default=getattr(CFG, "DEFAULT_CBO_MACRO_LENGTHSCALE_BATCH", 0.15), help="RBF lengthscale for Batch ratio")
    parser.add_argument("--cbo-macro-alpha", type=float, default=getattr(CFG, "DEFAULT_CBO_MACRO_ALPHA", 1.0), help="Macro similarity exponent")
    parser.add_argument("--cbo-dump-candidates", action="store_true", default=bool(getattr(CFG, "DEFAULT_CBO_DUMP_CANDIDATES", False)), help="Dump CBO candidate theta diagnostics")
    parser.add_argument("--cbo-dump-candidates-every", type=int, default=getattr(CFG, "DEFAULT_CBO_DUMP_CANDIDATES_EVERY", 20), help="Dump candidate diagnostics every N iterations")
    parser.add_argument("--cbo-dump-candidates-topn", type=int, default=getattr(CFG, "DEFAULT_CBO_DUMP_CANDIDATES_TOPN", 30), help="Candidate diagnostic rows per dump, ranked by acquisition plus selected")
    parser.add_argument("--cbo-select-mode", choices=["greedy", "topk_stochastic", "epsilon_greedy", "randomized_ucb"], default=getattr(CFG, "DEFAULT_CBO_SELECT_MODE", "greedy"), help="CBO candidate selection mode; greedy preserves old behavior")
    parser.add_argument("--cbo-topk", type=int, default=getattr(CFG, "DEFAULT_CBO_TOPK", 5), help="Top-K size for stochastic CBO selection")
    parser.add_argument("--cbo-select-temperature", type=float, default=getattr(CFG, "DEFAULT_CBO_SELECT_TEMPERATURE", 0.20), help="Softmax temperature for topK stochastic selection")
    parser.add_argument("--cbo-epsilon", type=float, default=getattr(CFG, "DEFAULT_CBO_EPSILON", 0.10), help="Epsilon for epsilon-greedy selection under triggered exploration")
    parser.add_argument("--cbo-acq-beta", type=float, default=getattr(CFG, "DEFAULT_CBO_ACQ_BETA", 3.0), help="Acquisition beta for mu + beta*sigma scoring")
    parser.add_argument("--cbo-acq-beta-mode", choices=["fixed", "radius_adaptive", "radius_state_adaptive"], default=getattr(CFG, "DEFAULT_CBO_ACQ_BETA_MODE", "fixed"), help="CBO acquisition beta schedule")
    parser.add_argument("--cbo-beta-min", type=float, default=getattr(CFG, "DEFAULT_CBO_BETA_MIN", 0.1), help="Minimum beta for radius-adaptive acquisition")
    parser.add_argument("--cbo-beta-max", type=float, default=getattr(CFG, "DEFAULT_CBO_BETA_MAX", 2.0), help="Maximum beta for radius-adaptive acquisition")
    parser.add_argument("--cbo-radius-beta-power", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_BETA_POWER", 1.0), help="Power on normalized TR radius when computing adaptive beta")
    parser.add_argument("--cbo-radius-stable-rebound-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_STABLE_REBOUND_PCT", 0.02), help="Stable rebound threshold for beta diagnostics")
    parser.add_argument("--cbo-radius-unstable-rebound-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_UNSTABLE_REBOUND_PCT", 0.04), help="Rebound threshold that boosts radius_state_adaptive beta")
    parser.add_argument("--cbo-radius-surprise-boost-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_SURPRISE_BOOST_THRESHOLD", 2.0), help="Surprise threshold that boosts radius_state_adaptive beta")
    parser.add_argument("--cbo-radius-beta-boost", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_BETA_BOOST", 1.5), help="Multiplicative beta boost for unstable radius_state_adaptive state")
    parser.add_argument("--cbo-radius-beta-cap", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_BETA_CAP", 3.0), help="Upper cap after radius_state_adaptive beta boost")
    parser.add_argument("--cbo-service-guard-mode", choices=["off", "soft"], default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_MODE", "off"), help="Optional service-aware exploration score guard")
    parser.add_argument("--cbo-service-guard-delay-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_DELAY_PCT", 0.03), help="Delay degradation threshold for service guard")
    parser.add_argument("--cbo-service-guard-backlog-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SERVICE_GUARD_BACKLOG_PCT", 0.03), help="Backlog degradation threshold for service guard")
    parser.add_argument("--cbo-surprise-window", type=int, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_WINDOW", 10), help="Window length for residual/surprise diagnostics")
    parser.add_argument("--cbo-surprise-z-threshold", type=float, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_Z_THRESHOLD", 2.0), help="Standardized prediction-error threshold for residual adaptive TR")
    parser.add_argument("--cbo-surprise-cost-gap-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_SURPRISE_COST_GAP_PCT", 0.03), help="Cost gap over recent best required to trigger residual adaptive TR")
    parser.add_argument("--cbo-sigma-floor", type=float, default=getattr(CFG, "DEFAULT_CBO_SIGMA_FLOOR", 1e-6), help="Sigma floor for surprise calculation")
    parser.add_argument("--cbo-radius-reset", type=float, default=getattr(CFG, "DEFAULT_CBO_RADIUS_RESET", 0.12), help="TR radius after residual/condition soft reset")
    parser.add_argument("--cbo-radius-min-stuck-rounds", type=int, default=getattr(CFG, "DEFAULT_CBO_RADIUS_MIN_STUCK_ROUNDS", 10), help="Rounds stuck near min radius before condition trigger")
    parser.add_argument("--cbo-rebound-window", type=int, default=getattr(CFG, "DEFAULT_CBO_REBOUND_WINDOW", 20), help="Recent window for cost rebound trigger")
    parser.add_argument("--cbo-rebound-threshold-pct", type=float, default=getattr(CFG, "DEFAULT_CBO_REBOUND_THRESHOLD_PCT", 0.03), help="Relative rebound threshold for condition adaptive TR")
    parser.add_argument("--cbo-selection-cooldown", type=int, default=getattr(CFG, "DEFAULT_CBO_SELECTION_COOLDOWN", 5), help="Number of future selections using exploratory selection after a trigger")
    parser.add_argument("--cbo-condition-anchor-switch", choices=["off", "recent_best", "context_best", "robust_elite"], default=getattr(CFG, "DEFAULT_CBO_CONDITION_ANCHOR_SWITCH", "context_best"), help="Temporary anchor override after residual/condition trigger")
    parser.add_argument("--cloud-delay-mult", type=float, default=1.0, help="只作用于云目标节点的传输时延倍率，>1 表示上云更慢")
    parser.add_argument("--cloud-energy-mult", type=float, default=1.0, help="只作用于云目标节点的传输能耗倍率，>1 表示上云更耗能")
    parser.add_argument("--cloud-speed-mult", type=float, default=1.0, help="只作用于云节点算力的速度倍率，<1 表示云算力被削弱")
    parser.add_argument("--export-short-names", action="store_true", help="实验结束后复制一份短英文文件名结果到 output_root/_short_export，并打包 tar.gz")
    args = parser.parse_args()
    print("[BackupDefaults] no RoundRobin default methods = " + ",".join(DEFAULT_SCENARIO_KEYS))
    print(f"[BackupDefaults] BO_ITERATIONS={CFG.BO_ITERATIONS}, BO_INTERVAL={CFG.BO_INTERVAL}, SESSION_DURATION={CFG.SESSION_DURATION}")
    print(f"[BackupDefaults] feedback_score={args.feedback_score}, bo_history_mode={args.bo_history_mode}, bo_recent_window={args.bo_recent_window}, deploy_policy={args.deploy_policy}")
    print(f"[BackupDefaults] cbo_history_select_mode={args.cbo_history_select_mode}, cbo_robust_score_mode={args.cbo_robust_score_mode}, cbo_tr_mode={args.cbo_tr_mode}, cbo_robust_incumbent_mode={args.cbo_robust_incumbent_mode}")
    print(f"[BackupDefaults] cbo_macro_gate_mode={args.cbo_macro_gate_mode}, cbo_macro_k={args.cbo_macro_k}, cbo_dump_candidates={args.cbo_dump_candidates}")
    print(f"[BackupDefaults] cbo_select_mode={args.cbo_select_mode}, cbo_topk={args.cbo_topk}, cbo_surprise_window={args.cbo_surprise_window}, cbo_radius_reset={args.cbo_radius_reset}")
    if args.task_adaptation and args.no_task_adaptation:
        parser.error("--task-adaptation 和 --no-task-adaptation 不能同时使用")
    if args.task_adaptation:
        CFG.USE_TASK_TYPE_ADAPTATION = True
    if args.no_task_adaptation:
        CFG.USE_TASK_TYPE_ADAPTATION = False
    if args.fixed_rng:
        CFG.USE_FIXED_RNG = True
    if args.fixed_seed is not None:
        CFG.FIXED_RNG_SEED = args.fixed_seed
    if args.rt_deadline_factor is not None:
        CFG.TASK_PROPS["RT"]["deadline_factor"] = float(args.rt_deadline_factor)
    if args.bo_iterations is not None:
        CFG.BO_ITERATIONS = int(args.bo_iterations)
    if args.bo_interval is not None:
        CFG.BO_INTERVAL = float(args.bo_interval)
    if args.session_duration is not None:
        CFG.SESSION_DURATION = float(args.session_duration)
    CFG.FEEDBACK_MODE = str(args.feedback_mode)
    CFG.BO_TRAINING_FEEDBACK_SCORE = str(args.feedback_score)
    CFG.CBO_REFERENCE_MODE = str(args.cbo_reference_mode)
    CFG.CBO_REFERENCE_CALIBRATION_ROUNDS = int(args.cbo_reference_calibration_rounds)
    CFG.CBO_REFERENCE_MIN_ROUNDS = int(args.cbo_reference_min_rounds)
    CFG.CBO_REFERENCE_STAT = str(args.cbo_reference_stat)
    CFG.CBO_REFERENCE_TRIM_PCT = float(args.cbo_reference_trim_pct)
    CFG.CBO_REFERENCE_FREEZE_AFTER_CALIBRATION = bool(args.cbo_reference_freeze_after_calibration)
    CFG.CBO_REFERENCE_FILE = str(args.cbo_reference_file)
    CFG.CBO_REFERENCE_OUTPUT_FILE = str(args.cbo_reference_output_file)
    CFG.CBO_OBJECTIVE_MODE = str(args.cbo_objective_mode)
    CFG.CBO_TRADEOFF_ALPHA = float(args.cbo_tradeoff_alpha)
    CFG.CBO_ALPHA_MIN = float(args.cbo_alpha_min)
    CFG.CBO_ALPHA_MAX = float(args.cbo_alpha_max)
    CFG.CBO_TARGET_SUCCESS_RATE = float(args.cbo_target_success_rate)
    CFG.CBO_UNFINISHED_PENALTY_WEIGHT = float(args.cbo_unfinished_penalty_weight)
    CFG.CBO_SUCCESS_SHORTFALL_WEIGHT = float(args.cbo_success_shortfall_weight)
    CFG.CBO_BACKLOG_GROWTH_PENALTY_WEIGHT = float(args.cbo_backlog_growth_penalty_weight)
    CFG.CBO_CLASS_IMBALANCE_WEIGHT = float(args.cbo_class_imbalance_weight)
    CFG.CBO_NORMALIZED_RATIO_CLIP_MIN = float(args.cbo_normalized_ratio_clip_min)
    CFG.CBO_NORMALIZED_RATIO_CLIP_MAX = float(args.cbo_normalized_ratio_clip_max)
    CFG.SCHEDULER_TRADEOFF_MODE = str(args.scheduler_tradeoff_mode)
    CFG.SCHEDULER_TRADEOFF_ALPHA = float(args.scheduler_tradeoff_alpha)
    CFG.SCHEDULER_ALPHA_MIN = float(args.scheduler_alpha_min)
    CFG.SCHEDULER_ALPHA_MAX = float(args.scheduler_alpha_max)
    CFG.SCHEDULER_SERVICE_LATENCY_WEIGHT = float(args.scheduler_service_latency_weight)
    CFG.SCHEDULER_SERVICE_RISK_WEIGHT = float(args.scheduler_service_risk_weight)
    CFG.SCHEDULER_SERVICE_QUEUE_WEIGHT = float(args.scheduler_service_queue_weight)
    CFG.SCHEDULER_ENERGY_WEIGHT = float(args.scheduler_energy_weight)
    CFG.SCHEDULER_SCORE_NORM_MODE = str(args.scheduler_score_norm_mode)
    CFG.SCHEDULER_NORM_CLIP_MAX = float(args.scheduler_norm_clip_max)
    CFG.SCHEDULER_NORM_EPS = float(args.scheduler_norm_eps)
    CFG.SCHEDULER_NORM_EMA_ALPHA = float(args.scheduler_norm_ema_alpha)
    print(
        f"[SCHED-TRADEOFF] mode={CFG.SCHEDULER_TRADEOFF_MODE} "
        f"alpha={CFG.SCHEDULER_TRADEOFF_ALPHA} alpha_min={CFG.SCHEDULER_ALPHA_MIN} alpha_max={CFG.SCHEDULER_ALPHA_MAX} "
        f"service_latency_weight={CFG.SCHEDULER_SERVICE_LATENCY_WEIGHT} "
        f"service_risk_weight={CFG.SCHEDULER_SERVICE_RISK_WEIGHT} "
        f"service_queue_weight={CFG.SCHEDULER_SERVICE_QUEUE_WEIGHT} "
        f"energy_weight={CFG.SCHEDULER_ENERGY_WEIGHT}",
        flush=True,
    )
    print(
        f"[SCHED-NORM] mode={CFG.SCHEDULER_SCORE_NORM_MODE} "
        f"clip_max={CFG.SCHEDULER_NORM_CLIP_MAX} eps={CFG.SCHEDULER_NORM_EPS} "
        f"ema_alpha={CFG.SCHEDULER_NORM_EMA_ALPHA}",
        flush=True,
    )
    CFG.PAIRED_BASELINE_KEY = USER_METHOD_ALIASES.get(str(args.paired_baseline_key), USER_METHOD_ALIASES.get(str(args.paired_baseline_key).lower(), str(args.paired_baseline_key)))
    os.environ["BO_TRAINING_FEEDBACK_SCORE"] = str(args.feedback_score)
    os.environ["PAIRED_BASELINE_KEY"] = str(CFG.PAIRED_BASELINE_KEY)
    if args.bo_history_mode is not None:
        CFG.BO_HISTORY_MODE = str(args.bo_history_mode)
        os.environ["BO_HISTORY_MODE"] = str(args.bo_history_mode)
    if args.bo_recent_window is not None:
        CFG.BO_RECENT_WINDOW = int(args.bo_recent_window)
        os.environ["BO_RECENT_WINDOW"] = str(int(args.bo_recent_window))
    if args.bo_confidence_min is not None:
        CFG.BO_CONFIDENCE_MIN = float(args.bo_confidence_min)
        os.environ["BO_CONFIDENCE_MIN"] = str(float(args.bo_confidence_min))
    if args.bo_confidence_min_samples is not None:
        CFG.BO_CONFIDENCE_MIN_SAMPLES = int(args.bo_confidence_min_samples)
        os.environ["BO_CONFIDENCE_MIN_SAMPLES"] = str(int(args.bo_confidence_min_samples))
    CFG.CBO_HISTORY_SELECT_MODE = str(args.cbo_history_select_mode)
    CFG.CBO_CONTEXT_K = int(args.cbo_context_k)
    CFG.CBO_ELITE_K = int(args.cbo_elite_k)
    CFG.CBO_DIVERSE_K = int(args.cbo_diverse_k)
    CFG.CBO_ROBUST_SCORE_MODE = str(args.cbo_robust_score_mode)
    CFG.CBO_ROBUST_STD_WEIGHT = float(args.cbo_robust_std_weight)
    CFG.CBO_THETA_MERGE_EPS = float(args.cbo_theta_merge_eps)
    CFG.CBO_CONTEXT_SIM_THRESHOLD = float(args.cbo_context_sim_threshold)
    CFG.CBO_TR_MODE = str(args.cbo_tr_mode)
    CFG.CBO_TR_RADIUS_INIT = float(args.cbo_tr_radius_init)
    CFG.CBO_TR_RADIUS_MIN = float(args.cbo_tr_radius_min)
    CFG.CBO_TR_RADIUS_MAX = float(args.cbo_tr_radius_max)
    CFG.CBO_TR_GROW = float(args.cbo_tr_grow)
    CFG.CBO_TR_SHRINK = float(args.cbo_tr_shrink)
    CFG.CBO_TR_UPDATE_MODE = str(args.cbo_tr_update_mode)
    CFG.CBO_TR_COMPARE_WINDOW = int(args.cbo_tr_compare_window)
    CFG.CBO_TR_BASELINE_WINDOW = int(args.cbo_tr_baseline_window)
    CFG.CBO_TR_IMPROVE_PCT = float(args.cbo_tr_improve_pct)
    CFG.CBO_TR_WORSEN_PCT = float(args.cbo_tr_worsen_pct)
    CFG.CBO_TR_DEADBAND_PCT = float(args.cbo_tr_deadband_pct)
    CFG.CBO_TR_UPDATE_PATIENCE = int(args.cbo_tr_update_patience)
    CFG.CBO_TR_ANCHOR_MODE = str(args.cbo_tr_anchor_mode)
    CFG.CBO_ROBUST_INCUMBENT_MODE = str(args.cbo_robust_incumbent_mode)
    CFG.CBO_MACRO_GATE_MODE = str(args.cbo_macro_gate_mode)
    CFG.CBO_MACRO_K = int(args.cbo_macro_k)
    CFG.CBO_MACRO_TOTAL_SCALE = str(args.cbo_macro_total_scale)
    CFG.CBO_MACRO_LENGTHSCALE_TOTAL = float(args.cbo_macro_lengthscale_total)
    CFG.CBO_MACRO_LENGTHSCALE_RT = float(args.cbo_macro_lengthscale_rt)
    CFG.CBO_MACRO_LENGTHSCALE_BATCH = float(args.cbo_macro_lengthscale_batch)
    CFG.CBO_MACRO_ALPHA = float(args.cbo_macro_alpha)
    CFG.CBO_DUMP_CANDIDATES = bool(args.cbo_dump_candidates)
    CFG.CBO_DUMP_CANDIDATES_EVERY = int(args.cbo_dump_candidates_every)
    CFG.CBO_DUMP_CANDIDATES_TOPN = int(args.cbo_dump_candidates_topn)
    CFG.CBO_SELECT_MODE = str(args.cbo_select_mode)
    CFG.CBO_TOPK = int(args.cbo_topk)
    CFG.CBO_SELECT_TEMPERATURE = float(args.cbo_select_temperature)
    CFG.CBO_EPSILON = float(args.cbo_epsilon)
    CFG.CBO_ACQ_BETA = float(args.cbo_acq_beta)
    CFG.CBO_ACQ_BETA_MODE = str(args.cbo_acq_beta_mode)
    CFG.CBO_BETA_MIN = float(args.cbo_beta_min)
    CFG.CBO_BETA_MAX = float(args.cbo_beta_max)
    CFG.CBO_RADIUS_BETA_POWER = float(args.cbo_radius_beta_power)
    CFG.CBO_RADIUS_STABLE_REBOUND_PCT = float(args.cbo_radius_stable_rebound_pct)
    CFG.CBO_RADIUS_UNSTABLE_REBOUND_PCT = float(args.cbo_radius_unstable_rebound_pct)
    CFG.CBO_RADIUS_SURPRISE_BOOST_THRESHOLD = float(args.cbo_radius_surprise_boost_threshold)
    CFG.CBO_RADIUS_BETA_BOOST = float(args.cbo_radius_beta_boost)
    CFG.CBO_RADIUS_BETA_CAP = float(args.cbo_radius_beta_cap)
    CFG.CBO_SERVICE_GUARD_MODE = str(args.cbo_service_guard_mode)
    CFG.CBO_SERVICE_GUARD_DELAY_PCT = float(args.cbo_service_guard_delay_pct)
    CFG.CBO_SERVICE_GUARD_BACKLOG_PCT = float(args.cbo_service_guard_backlog_pct)
    CFG.CBO_SURPRISE_WINDOW = int(args.cbo_surprise_window)
    CFG.CBO_SURPRISE_Z_THRESHOLD = float(args.cbo_surprise_z_threshold)
    CFG.CBO_SURPRISE_COST_GAP_PCT = float(args.cbo_surprise_cost_gap_pct)
    CFG.CBO_SIGMA_FLOOR = float(args.cbo_sigma_floor)
    CFG.CBO_RADIUS_RESET = float(args.cbo_radius_reset)
    CFG.CBO_RADIUS_MIN_STUCK_ROUNDS = int(args.cbo_radius_min_stuck_rounds)
    CFG.CBO_REBOUND_WINDOW = int(args.cbo_rebound_window)
    CFG.CBO_REBOUND_THRESHOLD_PCT = float(args.cbo_rebound_threshold_pct)
    CFG.CBO_SELECTION_COOLDOWN = int(args.cbo_selection_cooldown)
    CFG.CBO_CONDITION_ANCHOR_SWITCH = str(args.cbo_condition_anchor_switch)
    if _argv_has_option("--cbo-tr-radius-init"):
        CFG.TRUST_RADIUS_INIT = float(args.cbo_tr_radius_init)
    if _argv_has_option("--cbo-tr-radius-min"):
        CFG.TRUST_RADIUS_MIN = float(args.cbo_tr_radius_min)
    if _argv_has_option("--cbo-tr-radius-max"):
        CFG.TRUST_RADIUS_MAX = float(args.cbo_tr_radius_max)
    if _argv_has_option("--cbo-tr-grow"):
        CFG.TRUST_RADIUS_GROWTH = float(args.cbo_tr_grow)
    if _argv_has_option("--cbo-tr-shrink"):
        CFG.TRUST_RADIUS_SHRINK = float(args.cbo_tr_shrink)
    if args.deploy_policy is not None:
        CFG.DEPLOY_POLICY_ARG = str(args.deploy_policy)
        os.environ["SAFEBO_POLICY"] = str(args.deploy_policy)
        os.environ["SAFEBO_POLICY_ARG"] = str(args.deploy_policy)
    if args.dual_refined_source is not None:
        CFG.DUAL_REFINED_SOURCE = str(args.dual_refined_source)
        os.environ["DUAL_REFINED_SOURCE"] = str(args.dual_refined_source)
    if args.dual_class_metric is not None:
        CFG.DUAL_CLASS_METRIC = str(args.dual_class_metric)
        os.environ["DUAL_CLASS_METRIC"] = str(args.dual_class_metric)
    CFG.CLOUD_DELAY_MULT = float(args.cloud_delay_mult)
    CFG.CLOUD_ENERGY_MULT = float(args.cloud_energy_mult)
    CFG.CLOUD_SPEED_MULT = float(args.cloud_speed_mult)
    fixed_task_probs = parse_task_probs_arg(args.task_probs)
    if fixed_task_probs is not None:
        CFG.TASK_TYPE_PROBS = fixed_task_probs
    task_prob_schedule = parse_task_prob_schedule_arg(args.task_prob_schedule)
    if task_prob_schedule is not None:
        CFG.TASK_TYPE_PROB_SCHEDULE = task_prob_schedule
    lambda_schedule = parse_lambda_schedule_arg(args.lambda_schedule)
    if lambda_schedule is not None:
        CFG.LAMBDA_SCHEDULE = lambda_schedule
        CFG.ARRIVAL_THRESHOLDS = infer_arrival_thresholds(CFG.LAMBDA_SCHEDULE)
        # 如果用户没手动覆盖 session_duration，则自动扩展到 schedule 末尾。
        if args.session_duration is None:
            CFG.SESSION_DURATION = max(float(x[1]) for x in CFG.LAMBDA_SCHEDULE)
    CFG.REPEAT_RUNS = max(1, args.repeat)
    if args.mode == "param":
        run_param_analysis(samples=args.samples, local_delta=args.local_delta)
    elif args.mode == "extreme":
        for _ in range(max(1, args.repeat)):
            run_extreme_param_test()
    elif args.mode == "scan":
        run_param_scan(dim_name=args.dim, points=args.points)
    elif args.mode == "sensitivity":
        run_full_sensitivity_analysis(
            points=max(2, args.points),
            seeds=max(1, args.sens_seeds),
            windows=max(1, args.sens_windows),
            scenario_spec=args.sensitivity_scenarios,
            output_dir=args.output_root,
            greedy=bool(args.sens_greedy),
            pref_norm_mode=args.pref_norm_mode,
        )
    elif args.mode == "scenario":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_scenario_method_experiments(repeat_runs=max(1, args.repeat), selected_keys=selected_keys, output_dir=args.output_root)
    elif args.mode == "pressure_scan":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_pressure_scan_experiments(
            repeat_runs=max(1, args.repeat),
            lambda_values=parse_lambda_values_arg(args.lambda_values),
            output_root=args.output_root,
            selected_keys=selected_keys,
            task_probs=parse_task_probs_arg(args.task_probs),
        )
    elif args.mode == "offline_noise":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_offline_window_noise_diagnostic(
            repeat_runs=max(1, args.repeat),
            selected_keys=selected_keys,
            output_dir=args.output_root,
        )
    elif args.mode == "ratio_grid":
        selected_keys = None
        if args.selected_keys:
            selected_keys = normalize_selected_method_keys([x.strip() for x in args.selected_keys.split(",") if x.strip()])
        run_ratio_grid_experiments(
            repeat_runs=max(1, args.repeat),
            step=args.ratio_step,
            min_ratio=args.ratio_min,
            output_root=args.output_root,
            selected_keys=selected_keys,
        )
    else:
        main()

    if getattr(args, "export_short_names", False):
        export_root = args.output_root or SCENARIO_SAVE_DIR
        export_short_named_results(export_root)
