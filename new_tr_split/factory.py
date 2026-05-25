#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 2921-3617.
# ConnectedFactory and event-driven window simulation.

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
