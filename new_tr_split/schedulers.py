#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 1130-1890.
# Scheduling algorithms and node-selection policies.

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

    def _boltzmann_distribution_debug(self, scores, probs, choice):
        scores = np.asarray(scores, dtype=float)
        probs = np.asarray(probs, dtype=float)
        finite = np.isfinite(scores)
        if len(scores) == 0 or not finite.any():
            return {
                "score_gap_best_2nd": np.nan,
                "score_gap_min_max": np.nan,
                "boltzmann_top1_prob": np.nan,
                "boltzmann_selected_prob": np.nan,
                "boltzmann_entropy": np.nan,
                "boltzmann_entropy_norm": np.nan,
            }
        valid_scores = scores[finite]
        valid_indices = np.where(finite)[0]
        order = np.argsort(valid_scores)
        best_i = int(valid_indices[order[0]])
        if len(valid_scores) >= 2:
            gap_best_2nd = float(valid_scores[order[1]] - valid_scores[order[0]])
        else:
            gap_best_2nd = 0.0
        gap_min_max = float(np.max(valid_scores) - np.min(valid_scores))
        top1_prob = float(probs[best_i]) if 0 <= best_i < len(probs) else np.nan
        selected_prob = float(probs[int(choice)]) if 0 <= int(choice) < len(probs) else np.nan
        p = probs[np.isfinite(probs) & (probs > 0.0)]
        entropy = float(-np.sum(p * np.log(p))) if len(p) else 0.0
        entropy_norm = float(entropy / np.log(len(probs))) if len(probs) > 1 else 0.0
        return {
            "score_gap_best_2nd": gap_best_2nd,
            "score_gap_min_max": gap_min_max,
            "boltzmann_top1_prob": top1_prob,
            "boltzmann_selected_prob": selected_prob,
            "boltzmann_entropy": entropy,
            "boltzmann_entropy_norm": entropy_norm,
        }

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

    def _resolve_scheduler_alpha(self, task_type, latency_w, energy_w, theta_full=None, controls=None):
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
        if mode in {"alpha_direct", "direct_alpha"}:
            alpha_raw = float(latency_w)
            task_bounds_fn = globals().get("get_alpha_direct_task_bounds")
            if callable(task_bounds_fn):
                alpha_min, alpha_max = task_bounds_fn(task_type)
            return float(np.clip(alpha_raw, alpha_min, alpha_max)), f"Alpha_{task_type}", "alpha_direct"
        if mode != "legacy":
            raise ValueError(f"Unknown scheduler tradeoff mode={mode}")
        return None, "legacy_linear_weights", mode

    def _score_candidate_components(self, c, latency_w, energy_w, risk_w, queue_w=0.0, alpha=None, tradeoff_mode="legacy"):
        norm_l = float(c.get("norm_l", 0.0))
        norm_e = float(c.get("norm_e", 0.0))
        norm_r = float(c.get("norm_risk", 0.0)) if getattr(CFG, "USE_SCORE_RISK", True) else 0.0
        norm_q = float(c.get("norm_queue", 0.0)) if getattr(CFG, "USE_QUEUE_PRESSURE_SCORE", True) else 0.0
        mode = str(tradeoff_mode).lower()
        if mode == "legacy":
            service_component = float(latency_w) * norm_l + float(risk_w) * norm_r + float(queue_w) * norm_q
            energy_component = float(energy_w) * norm_e
            score = energy_component + service_component
            latency_energy_component = energy_component + float(latency_w) * norm_l
            latency_energy_component_unscaled = latency_energy_component
            latency_energy_component_scaled = latency_energy_component
            scheduler_le_scale = 1.0
        else:
            a = float(alpha if alpha is not None else getattr(CFG, "SCHEDULER_TRADEOFF_ALPHA", 0.85))
            latency_component = norm_l
            energy_component = norm_e
            risk_penalty = float(risk_w) * norm_r
            queue_penalty = float(queue_w) * norm_q
            latency_energy_component_unscaled = a * latency_component + (1.0 - a) * energy_component
            scheduler_le_scale = float(getattr(CFG, "SCHEDULER_LE_SCALE", 1.0)) if mode == "alpha_direct" else 1.0
            latency_energy_component_scaled = scheduler_le_scale * latency_energy_component_unscaled
            latency_energy_component = latency_energy_component_scaled
            service_component = latency_energy_component  # Deprecated alias kept for old analysis scripts.
            score = latency_energy_component + risk_penalty + queue_penalty
            c["latency_component"] = float(latency_component)
            c["latency_energy_component"] = float(latency_energy_component)
            c["base_latency_energy_score"] = float(latency_energy_component)
            c["risk_penalty"] = float(risk_penalty)
            c["queue_penalty"] = float(queue_penalty)
            c["service_component_deprecated"] = float(service_component)
        c["scheduler_le_scale"] = float(scheduler_le_scale)
        c["latency_energy_component_unscaled"] = float(latency_energy_component_unscaled)
        c["latency_energy_component_scaled"] = float(latency_energy_component_scaled)
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
        alpha, alpha_source, tradeoff_mode = self._resolve_scheduler_alpha(task.task_type, latency_w, energy_w, theta_full=theta_full)
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
        score_dist_debug = self._boltzmann_distribution_debug(candidate_scores, probs, choice)
        all_scores = np.array([c.get("score", np.nan) for c in candidates], dtype=float)
        self.last_score_debug = {
            "norm_mode": self.norm_mode,
            "task_type": task.task_type,
            "scheduler_tradeoff_mode": tradeoff_mode,
            "scheduler_score_norm_mode": norm_debug.get("scheduler_score_norm_mode", getattr(CFG, "SCHEDULER_SCORE_NORM_MODE", "legacy")),
            "scheduler_alpha": float(alpha) if alpha is not None else None,
            "scheduler_alpha_source": alpha_source,
            "scheduler_alpha_min": float(getattr(CFG, "SCHEDULER_ALPHA_MIN", 0.60)),
            "scheduler_alpha_max": float(getattr(CFG, "SCHEDULER_ALPHA_MAX", 0.97)),
            "scheduler_le_scale": float(selected.get("scheduler_le_scale", 1.0)),
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
            "opportunity_candidate_count": int(len(candidates)),
            "selected_node": int(selected_idx),
            "selected_norm_e": float(selected.get("norm_e", 0.0)),
            "selected_norm_l": float(selected.get("norm_l", 0.0)),
            "selected_norm_risk": float(selected.get("norm_risk", 0.0)),
            "selected_norm_queue": float(selected.get("norm_queue", 0.0)),
            "selected_latency_component": float(selected.get("latency_component", np.nan)),
            "selected_risk_penalty": float(selected.get("risk_penalty", 0.0)),
            "selected_queue_penalty": float(selected.get("queue_penalty", 0.0)),
            "selected_latency_energy_component": float(selected.get("latency_energy_component", np.nan)),
            "selected_latency_energy_component_unscaled": float(selected.get("latency_energy_component_unscaled", np.nan)),
            "selected_latency_energy_component_scaled": float(selected.get("latency_energy_component_scaled", np.nan)),
            "selected_base_latency_energy_score": float(selected.get("base_latency_energy_score", np.nan)),
            "selected_service_component": float(selected.get("service_component", np.nan)),
            "selected_service_component_deprecated": float(selected.get("service_component", np.nan)),
            "selected_energy_component": float(selected.get("energy_component", np.nan)),
            "selected_score": float(selected.get("score", np.nan)),
            "score_min": float(np.nanmin(all_scores)) if len(all_scores) else 0.0,
            "score_max": float(np.nanmax(all_scores)) if len(all_scores) else 0.0,
            **score_dist_debug,
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
        alpha, alpha_source, tradeoff_mode = self._resolve_scheduler_alpha(task.task_type, latency_w, energy_w, theta_full=theta_full, controls=controls)

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

        score_dist_debug = self._boltzmann_distribution_debug(candidate_scores, probs, choice)
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
            "scheduler_le_scale": float(selected.get("scheduler_le_scale", 1.0)),
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
            "selected_latency_energy_component_unscaled": float(selected.get("latency_energy_component_unscaled", np.nan)),
            "selected_latency_energy_component_scaled": float(selected.get("latency_energy_component_scaled", np.nan)),
            "selected_base_latency_energy_score": float(selected.get("base_latency_energy_score", np.nan)),
            "selected_service_component": float(selected.get("service_component", np.nan)),
            "selected_service_component_deprecated": float(selected.get("service_component", np.nan)),
            "selected_energy_component": float(selected.get("energy_component", np.nan)),
            "score_min": float(np.nanmin(all_scores)) if len(all_scores) else 0.0,
            "score_max": float(np.nanmax(all_scores)) if len(all_scores) else 0.0,
            **score_dist_debug,
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
