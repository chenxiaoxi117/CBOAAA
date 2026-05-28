#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 5510-7136.
# Scenario method definitions, experiment runners, CLI-independent batch runners.

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
        "reduced6_cbo_lite_pressure_prev_unfinished": cbo_lite_group(
        "Pressure+PrevUnfinished Context",
         "pressure_prev_unfinished_5d",
        ),

        "reduced6_cbo_lite_pressure_transition": cbo_lite_group(
        "Pressure+Transition Context",
        "pressure_transition_6d",
        ),
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
        "reduced6_cbo_alpha_direct": {
            "label": "Reduced6 CBO Alpha-Direct",
            "norm_mode": "fixed",
            "control_mode": "alpha_direct",
            "scheduler_tradeoff_mode": "alpha_direct",
            "deploy_policy": "greedy",
            "method_family": "cbo_alpha_direct",
            "context_mode": "pressure_taskmix_counts",
            "history_mode": "recent_confidence",
            "recent_window": 80,
            "confidence_min": 0.35,
            "confidence_min_samples": 12,
            "agent_kwargs": alpha_direct_agent_kwargs(use_context=True, use_trust_region=False, anchor_mode="none", context_mode="pressure_taskmix_counts"),
        },
        "reduced6_cbo_alpha_direct_no_risk": {
            "label": "Reduced6 CBO Alpha-Direct No-Risk",
            "norm_mode": "fixed",
            "control_mode": "alpha_direct",
            "scheduler_tradeoff_mode": "alpha_direct",
            "scheduler_use_score_risk": False,
            "deploy_policy": "greedy",
            "method_family": "cbo_alpha_direct_no_risk",
            "context_mode": "pressure_taskmix_counts",
            "history_mode": "recent_confidence",
            "recent_window": 80,
            "confidence_min": 0.35,
            "confidence_min_samples": 12,
            "agent_kwargs": alpha_direct_agent_kwargs(use_context=True, use_trust_region=False, anchor_mode="none", context_mode="pressure_taskmix_counts"),
        },
        "reduced6_cbo_alpha_direct_unfinished_context": {
            "label": "Reduced6 CBO Alpha-Direct Unfinished Context",
            "norm_mode": "fixed",
            "control_mode": "alpha_direct",
            "scheduler_tradeoff_mode": "alpha_direct",
            "deploy_policy": "greedy",
            "method_family": "cbo_alpha_direct_unfinished_context",
            "context_mode": "pressure_unfinished_context",
            "history_mode": "recent_confidence",
            "recent_window": 80,
            "confidence_min": 0.35,
            "confidence_min_samples": 12,
            "agent_kwargs": alpha_direct_agent_kwargs(use_context=True, use_trust_region=False, anchor_mode="none", context_mode="pressure_unfinished_context"),
        },
        "reduced6_cbo_alpha_direct_prev_unfinished_context": {
            "label": "Reduced6 CBO Alpha-Direct Prev-Unfinished Context Risk0",
            "norm_mode": "fixed",
            "control_mode": "alpha_direct",
            "alpha_direct_control_variant": "risk_fixed_5d",
            "alpha_direct_fixed_risk_scale": 0.0,
            "scheduler_tradeoff_mode": "alpha_direct",
            "deploy_policy": "greedy",
            "method_family": "cbo_alpha_direct_prev_unfinished_context_risk0",
            "context_mode": "pressure_prev_unfinished_context",
            "history_mode": "recent_confidence",
            "recent_window": 80,
            "confidence_min": 0.35,
            "confidence_min_samples": 12,
            "agent_kwargs": alpha_direct_risk_fixed_agent_kwargs(use_context=True, use_trust_region=False, anchor_mode="none", context_mode="pressure_prev_unfinished_context", fixed_risk_scale=0.0),
        },
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
    if group_cfg.get("control_mode") == "alpha_direct":
        return alpha_direct_to_full_theta(theta, group_cfg=group_cfg)
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


ALPHA_DIRECT_FEATURE_NAMES = [
    "Alpha_RT", "Alpha_Batch", "Alpha_AI",
    "W_Queue", "W_Risk_Scale", "Cloud_Gate"
]
ALPHA_DIRECT_RISK_FIXED_FEATURE_NAMES = [
    "Alpha_RT", "Alpha_Batch", "Alpha_AI",
    "W_Queue", "Cloud_Gate"
]


def _coerce_bounds_pair(value, fallback):
    pair = fallback
    if value is not None:
        try:
            pair = list(value)
        except Exception:
            pair = fallback
    if pair is None or len(pair) < 2:
        pair = fallback
    lo = float(pair[0])
    hi = float(pair[1])
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def get_alpha_direct_task_bounds(task_type):
    task_key = str(task_type or "").strip().upper()
    default_pair = _coerce_bounds_pair(
        getattr(CFG, "ALPHA_DIRECT_BOUNDS", None),
        (float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60)), float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97))),
    )
    attr_map = {
        "RT": "ALPHA_DIRECT_RT_BOUNDS",
        "BATCH": "ALPHA_DIRECT_BATCH_BOUNDS",
        "AI": "ALPHA_DIRECT_AI_BOUNDS",
    }
    return _coerce_bounds_pair(getattr(CFG, attr_map.get(task_key, ""), None), default_pair)


def get_alpha_direct_control_bounds():
    rt_lo, rt_hi = get_alpha_direct_task_bounds("RT")
    batch_lo, batch_hi = get_alpha_direct_task_bounds("Batch")
    ai_lo, ai_hi = get_alpha_direct_task_bounds("AI")
    return [
        [rt_lo, batch_lo, ai_lo,
         float(CFG.CONTROL_QUEUE_BOUNDS[0]), float(CFG.CONTROL_RISK_SCALE_BOUNDS[0]), float(REDUCED6_CLOUD_GATE_BOUNDS[0])],
        [rt_hi, batch_hi, ai_hi,
         float(CFG.CONTROL_QUEUE_BOUNDS[1]), float(CFG.CONTROL_RISK_SCALE_BOUNDS[1]), float(REDUCED6_CLOUD_GATE_BOUNDS[1])],
    ]


def get_alpha_direct_risk_fixed_control_bounds():
    rt_lo, rt_hi = get_alpha_direct_task_bounds("RT")
    batch_lo, batch_hi = get_alpha_direct_task_bounds("Batch")
    ai_lo, ai_hi = get_alpha_direct_task_bounds("AI")
    return [
        [rt_lo, batch_lo, ai_lo,
         float(CFG.CONTROL_QUEUE_BOUNDS[0]), float(REDUCED6_CLOUD_GATE_BOUNDS[0])],
        [rt_hi, batch_hi, ai_hi,
         float(CFG.CONTROL_QUEUE_BOUNDS[1]), float(REDUCED6_CLOUD_GATE_BOUNDS[1])],
    ]


def clip_alpha_direct_control_vector(theta6):
    base = [0.85, 0.85, 0.85, 1.0, 1.0, 0.30]
    t = list(theta6)
    if len(t) < 6:
        t = t + base[len(t):]
    rt_lo, rt_hi = get_alpha_direct_task_bounds("RT")
    batch_lo, batch_hi = get_alpha_direct_task_bounds("Batch")
    ai_lo, ai_hi = get_alpha_direct_task_bounds("AI")
    return [
        float(np.clip(float(t[0]), rt_lo, rt_hi)),
        float(np.clip(float(t[1]), batch_lo, batch_hi)),
        float(np.clip(float(t[2]), ai_lo, ai_hi)),
        float(np.clip(float(t[3]), *CFG.CONTROL_QUEUE_BOUNDS)),
        float(np.clip(float(t[4]), *CFG.CONTROL_RISK_SCALE_BOUNDS)),
        float(np.clip(float(t[5]), float(REDUCED6_CLOUD_GATE_BOUNDS[0]), float(REDUCED6_CLOUD_GATE_BOUNDS[1]))),
    ]


def clip_alpha_direct_risk_fixed_control_vector(theta5):
    base = [0.85, 0.85, 0.85, 1.0, 0.30]
    t = list(theta5)
    if len(t) < 5:
        t = t + base[len(t):]
    rt_lo, rt_hi = get_alpha_direct_task_bounds("RT")
    batch_lo, batch_hi = get_alpha_direct_task_bounds("Batch")
    ai_lo, ai_hi = get_alpha_direct_task_bounds("AI")
    return [
        float(np.clip(float(t[0]), rt_lo, rt_hi)),
        float(np.clip(float(t[1]), batch_lo, batch_hi)),
        float(np.clip(float(t[2]), ai_lo, ai_hi)),
        float(np.clip(float(t[3]), *CFG.CONTROL_QUEUE_BOUNDS)),
        float(np.clip(float(t[4]), float(REDUCED6_CLOUD_GATE_BOUNDS[0]), float(REDUCED6_CLOUD_GATE_BOUNDS[1]))),
    ]


def expand_alpha_direct_control_vector(theta, group_cfg=None):
    """Return scheduler-facing 6D alpha-direct controls."""
    fixed_risk = None
    if isinstance(group_cfg, dict):
        fixed_risk = group_cfg.get("alpha_direct_fixed_risk_scale", None)
    t = list(theta)
    if fixed_risk is not None and len(t) == 5:
        alpha_rt, alpha_batch, alpha_ai, queue_w, cloud_gate = clip_alpha_direct_risk_fixed_control_vector(t)
        return [
            float(alpha_rt), float(alpha_batch), float(alpha_ai),
            float(queue_w), float(fixed_risk), float(cloud_gate),
        ]
    return clip_alpha_direct_control_vector(t)


ALPHA_DIRECT_BOUNDS = get_alpha_direct_control_bounds()


def alpha_direct_to_full_theta(theta6, group_cfg=None):
    """Map alpha-direct controls into the full scheduler theta."""
    alpha_rt, alpha_batch, alpha_ai, queue_w, risk_scale, cloud_gate = expand_alpha_direct_control_vector(theta6, group_cfg=group_cfg)
    full = default_control_vector(fill=1.5)
    names = list(CFG.FEATURE_NAMES)

    def set_name(name, value):
        if name in names:
            full[names.index(name)] = float(value)

    set_name("W_RT_Latency", alpha_rt)
    set_name("W_Batch_Latency", alpha_batch)
    set_name("W_AI_Latency", alpha_ai)
    set_name("W_RT_Energy", 1.0)
    set_name("W_Batch_Energy", 1.0)
    set_name("W_AI_Energy", 1.0)
    set_name("W_Queue", queue_w)
    set_name("W_Risk_Scale", risk_scale)
    set_name("Cloud_Gate", cloud_gate)
    set_name("Beta_Control", 8.0)
    set_name("Opportunity_Rho", 0.0)
    return full


def alpha_direct_anchor_points(anchor_mode="none"):
    mode = str(anchor_mode or "none").strip().lower()
    if mode in {"none", "cold", "cold_start", "no_anchor", "off"}:
        return []
    points = [
        [0.85, 0.85, 0.85, 1.0, 1.0, 0.50],
        [0.92, 0.90, 0.90, 1.0, 1.0, 0.05],
        [0.78, 0.78, 0.78, 5.0, 1.0, 0.30],
    ]
    return [clip_alpha_direct_control_vector(p) for p in points]


def alpha_direct_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="none", context_mode=None):
    context_dim = len(lite_context_feature_names(context_mode)) if context_mode else (len(CFG.CONTEXT_FEATURE_NAMES) if use_context else 0)
    context_bounds = lite_context_bounds(context_mode) if context_mode else (CFG.CONTEXT_BOUNDS if use_context else None)
    return {
        "dim": len(ALPHA_DIRECT_FEATURE_NAMES),
        "bounds": get_alpha_direct_control_bounds(),
        "feature_names": list(ALPHA_DIRECT_FEATURE_NAMES),
        "use_context": bool(use_context),
        "use_state_partition": bool(use_context and not context_mode),
        "use_trust_region": bool(use_trust_region),
        "context_dim": context_dim,
        "context_bounds": context_bounds,
        "anchor_points": alpha_direct_anchor_points(anchor_mode),
    }


def alpha_direct_risk_fixed_agent_kwargs(use_context=False, use_trust_region=False, anchor_mode="none", context_mode=None, fixed_risk_scale=0.0):
    context_dim = len(lite_context_feature_names(context_mode)) if context_mode else (len(CFG.CONTEXT_FEATURE_NAMES) if use_context else 0)
    context_bounds = lite_context_bounds(context_mode) if context_mode else (CFG.CONTEXT_BOUNDS if use_context else None)
    return {
        "dim": len(ALPHA_DIRECT_RISK_FIXED_FEATURE_NAMES),
        "bounds": get_alpha_direct_risk_fixed_control_bounds(),
        "feature_names": list(ALPHA_DIRECT_RISK_FIXED_FEATURE_NAMES),
        "use_context": bool(use_context),
        "use_state_partition": bool(use_context and not context_mode),
        "use_trust_region": bool(use_trust_region),
        "context_dim": context_dim,
        "context_bounds": context_bounds,
        "anchor_points": [],
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
    old_scheduler_tradeoff_mode = str(getattr(CFG, "SCHEDULER_TRADEOFF_MODE", "legacy"))
    method_scheduler_tradeoff_mode = group_cfg.get("scheduler_tradeoff_mode")
    if method_scheduler_tradeoff_mode:
        CFG.SCHEDULER_TRADEOFF_MODE = str(method_scheduler_tradeoff_mode)
    old_use_score_risk = bool(getattr(CFG, "USE_SCORE_RISK", True))
    method_use_score_risk = group_cfg.get("scheduler_use_score_risk", None)
    if method_use_score_risk is not None:
        CFG.USE_SCORE_RISK = bool(method_use_score_risk)
    is_reduced = group_cfg.get("control_mode") in {"reduced4", "reduced6", "alpha_direct"}
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
    if method_scheduler_tradeoff_mode:
        CFG.SCHEDULER_TRADEOFF_MODE = old_scheduler_tradeoff_mode
    if method_use_score_risk is not None:
        CFG.USE_SCORE_RISK = old_use_score_risk
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
        if len(info.get("logs", [])) == 1 and not round_df.empty:
            raw_log = info["logs"][0]
            warm_meta_cols = [
                "cbo_warm_start_enabled", "cbo_warm_start_mode", "cbo_warm_start_loaded_rows",
                "cbo_warm_start_used_rows", "selected_warm_rows_count", "selected_local_rows_count",
                "cbo_warm_start_history_path",
            ]
            for col in warm_meta_cols:
                vals = list(raw_log.get(col, [])) if isinstance(raw_log, dict) else []
                if vals:
                    fill = vals[-1]
                    while len(vals) < len(round_df):
                        vals.append(fill)
                    round_df[col] = vals[:len(round_df)]
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
            "scheduler_use_score_risk": bool(getattr(CFG, "USE_SCORE_RISK", True)),
            "scheduler_tradeoff_alpha": float(getattr(CFG, "SCHEDULER_TRADEOFF_ALPHA", 0.85)),
            "scheduler_alpha_min": float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60)),
            "scheduler_alpha_max": float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97)),
            "scheduler_le_scale": float(getattr(CFG, "SCHEDULER_LE_SCALE", 1.0)),
            "alpha_direct_bounds": getattr(CFG, "ALPHA_DIRECT_BOUNDS", None),
            "alpha_direct_rt_bounds": getattr(CFG, "ALPHA_DIRECT_RT_BOUNDS", None),
            "alpha_direct_batch_bounds": getattr(CFG, "ALPHA_DIRECT_BATCH_BOUNDS", None),
            "alpha_direct_ai_bounds": getattr(CFG, "ALPHA_DIRECT_AI_BOUNDS", None),
            "alpha_direct_fixed_theta": getattr(CFG, "ALPHA_DIRECT_FIXED_THETA", None),
            "alpha_direct_effective_bounds": get_alpha_direct_control_bounds(),
            "scheduler_service_latency_weight": float(getattr(CFG, "SCHEDULER_SERVICE_LATENCY_WEIGHT", 1.0)),
            "scheduler_service_risk_weight": float(getattr(CFG, "SCHEDULER_SERVICE_RISK_WEIGHT", 1.0)),
            "scheduler_service_queue_weight": float(getattr(CFG, "SCHEDULER_SERVICE_QUEUE_WEIGHT", 1.0)),
            "scheduler_energy_weight": float(getattr(CFG, "SCHEDULER_ENERGY_WEIGHT", 1.0)),
            "scheduler_score_norm_mode": str(getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")),
            "scheduler_norm_clip_max": float(getattr(CFG, "SCHEDULER_NORM_CLIP_MAX", 3.0)),
            "scheduler_norm_eps": float(getattr(CFG, "SCHEDULER_NORM_EPS", 1e-6)),
            "scheduler_norm_ema_alpha": float(getattr(CFG, "SCHEDULER_NORM_EMA_ALPHA", 0.995)),
            "use_boltzmann_random": bool(getattr(CFG, "USE_BOLTZMANN_RANDOM", True)),
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
            "cbo_good_region_guard": str(getattr(CFG, "CBO_GOOD_REGION_GUARD", "off")),
            "cbo_good_region_window": int(getattr(CFG, "CBO_GOOD_REGION_WINDOW", 50)),
            "cbo_good_region_worse_pct": float(getattr(CFG, "CBO_GOOD_REGION_WORSE_PCT", 0.03)),
            "cbo_good_region_distance_threshold": float(getattr(CFG, "CBO_GOOD_REGION_DISTANCE_THRESHOLD", 0.35)),
            "cbo_good_region_tr_radius_threshold": float(getattr(CFG, "CBO_GOOD_REGION_TR_RADIUS_THRESHOLD", 0.15)),
            "cbo_good_region_beta_threshold": float(getattr(CFG, "CBO_GOOD_REGION_BETA_THRESHOLD", 0.5)),
            "cbo_good_region_guard_mode": str(getattr(CFG, "CBO_GOOD_REGION_GUARD_MODE", "conservative")),
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
            "pressure_unfinished_context_names": list(globals().get("PRESSURE_UNFINISHED_CONTEXT_NAMES", [])),
            "pressure_prev_unfinished_context_names": list(globals().get("PRESSURE_PREV_UNFINISHED_CONTEXT_NAMES", [])),
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
    groups = apply_alpha_direct_fixed_theta_override(groups)
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
    export_warm_fn = globals().get("export_bo_warm_history_csv")
    if callable(export_warm_fn):
        try:
            export_warm_fn(group_logs, output_dir=SCENARIO_SAVE_DIR, selected_keys=selected_keys, groups=groups)
        except Exception as e:
            print(f"[WARN] failed to export bo_warm_history.csv: {e}", flush=True)
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
                "scheduler_le_scale": float(getattr(CFG, "SCHEDULER_LE_SCALE", 1.0)),
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
                    "scheduler_le_scale": float(getattr(CFG, "SCHEDULER_LE_SCALE", 1.0)),
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
