#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 1893-2919.
# BO agent, federated aggregator, and scenario monitor.

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
    arrived_tasks: Dict[str, Any] = field(default_factory=dict)

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

    def completion_ratio(self) -> float:
        return self.completed / max(1, self.arrivals)

    def sla_success_rate(self) -> float:
        return 1.0 - self.vio_rate()

    def _arrival_window_effective_terms(self, backlog: int, prev_backlog: Optional[int] = None) -> Dict[str, Any]:
        """Compute the main window objective from tasks that arrived in this window.

        Completed-only delay/lateness remain exported as diagnostics, but the
        main objective uses censored response time for every arrival.
        """
        eps = max(float(getattr(CFG, "SCHEDULER_NORM_EPS", 1e-6)), 1e-12)
        tasks = list(getattr(self, "arrived_tasks", {}).values())
        n_arr = max(1, int(self.arrivals))
        window_end = float(self.end_time)
        delay_norm_sum = 0.0
        unfinished = 0
        rt_arr = 0
        rt_violated = 0
        censored_delay_sum = 0.0
        effective_lateness_sum = 0.0
        for task in tasks:
            create_t = float(getattr(task, "arrival_time", getattr(task, "create_time", self.start_time)))
            deadline_abs = float(getattr(task, "deadline", create_t))
            budget = max(deadline_abs - create_t, eps)
            finish_t = float(getattr(task, "finish_time", -1.0))
            completed_by_window_end = finish_t >= 0.0 and finish_t <= window_end + eps
            observed_end = finish_t if completed_by_window_end else window_end
            response = max(0.0, observed_end - create_t)
            censored_delay_sum += response
            delay_norm_sum += response / budget
            if not completed_by_window_end:
                unfinished += 1
            lateness = max(0.0, observed_end - deadline_abs)
            effective_lateness_sum += lateness
            if getattr(task, "task_type", "Batch") == "RT":
                rt_arr += 1
                if observed_end > deadline_abs + eps:
                    rt_violated += 1

        window_delay_deadline_norm = delay_norm_sum / n_arr if self.arrivals > 0 else 0.0
        window_censored_avg_delay = censored_delay_sum / n_arr if self.arrivals > 0 else 0.0
        window_effective_avg_lateness = effective_lateness_sum / n_arr if self.arrivals > 0 else 0.0
        window_energy_per_arrival = float(self.total_energy) / max(float(self.arrivals), eps) if self.arrivals > 0 else 0.0
        energy_scale = max(float(getattr(CFG, "WINDOW_ENERGY_SCALE", 1000.0)), eps)
        window_energy_norm = window_energy_per_arrival / energy_scale
        rt_violation_rate = float(rt_violated) / max(float(rt_arr), eps) if rt_arr > 0 else 0.0
        unfinished_rate = float(unfinished) / max(float(self.arrivals), eps) if self.arrivals > 0 else 0.0
        if prev_backlog is None:
            backlog_growth = 0
        else:
            backlog_growth = max(0, int(backlog) - int(prev_backlog))
        backlog_growth_rate = float(backlog_growth) / max(float(self.arrivals), eps) if self.arrivals > 0 else 0.0

        violation_excess = max(0.0, rt_violation_rate - float(getattr(CFG, "WINDOW_RT_VIOLATION_EPS", 0.02)))
        cost = (
            float(getattr(CFG, "WINDOW_DELAY_WEIGHT", 1.0)) * window_delay_deadline_norm
            + float(getattr(CFG, "WINDOW_ENERGY_WEIGHT", 1.0)) * window_energy_norm
            + float(getattr(CFG, "WINDOW_RT_VIOLATION_WEIGHT", 4.0)) * violation_excess
            + float(getattr(CFG, "WINDOW_UNFINISHED_WEIGHT", 6.0)) * unfinished_rate
            + float(getattr(CFG, "WINDOW_BACKLOG_GROWTH_WEIGHT", 2.0)) * max(0.0, backlog_growth_rate)
        )
        return {
            "cost": float(cost),
            "window_delay_deadline_norm": float(window_delay_deadline_norm),
            "window_censored_avg_delay": float(window_censored_avg_delay),
            "window_energy_per_arrival": float(window_energy_per_arrival),
            "window_energy_norm": float(window_energy_norm),
            "window_rt_violation_rate": float(rt_violation_rate),
            "window_rt_violation_excess": float(violation_excess),
            "window_unfinished_arrivals": int(unfinished),
            "window_unfinished_rate": float(unfinished_rate),
            "window_backlog_growth": int(backlog_growth),
            "window_backlog_growth_rate": float(backlog_growth_rate),
            "window_effective_avg_lateness": float(window_effective_avg_lateness),
            "window_cost_formula": "arrival_censored_delay_energy_rt_violation_unfinished_backlog_growth",
            "window_cost_delay_weight": float(getattr(CFG, "WINDOW_DELAY_WEIGHT", 1.0)),
            "window_cost_energy_weight": float(getattr(CFG, "WINDOW_ENERGY_WEIGHT", 1.0)),
            "window_cost_rt_violation_weight": float(getattr(CFG, "WINDOW_RT_VIOLATION_WEIGHT", 4.0)),
            "window_cost_unfinished_weight": float(getattr(CFG, "WINDOW_UNFINISHED_WEIGHT", 6.0)),
            "window_cost_backlog_growth_weight": float(getattr(CFG, "WINDOW_BACKLOG_GROWTH_WEIGHT", 2.0)),
        }

    def to_metrics(self, cumulative_energy: float, backlog: int = 0, cumulative_energy_real: Optional[float] = None, prev_backlog: Optional[int] = None) -> Dict[str, Any]:
        """把窗口统计转换成 cost / reward 和各种监控指标。"""
        n = max(1, self.completed)
        backlog = max(0, int(backlog))
        avg_delay = self.avg_delay()
        vio_rate = self.vio_rate()
        avg_early = self.avg_earliness()
        avg_late = self.avg_lateness()
        zero_completion_penalty = CFG.ZERO_COMPLETION_PENALTY if self.arrivals > 0 and self.completed == 0 else 0.0
        objective_terms = self._arrival_window_effective_terms(backlog=backlog, prev_backlog=prev_backlog)

        early_bonus = 0.0
        if getattr(CFG, "USE_EARLY_BONUS", False):
            # 提前完成奖励封顶，避免 Batch/AI 的长 deadline slack 无限放大 reward。
            early_bonus = CFG.EARLY_BONUS_WEIGHT * min(avg_early, float(getattr(CFG, "EARLY_BONUS_CAP", 5.0)))

        cost = float(objective_terms["cost"])

        metrics = {
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
            "completion_ratio": self.completion_ratio(),
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
        metrics.update(objective_terms)
        return metrics



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
            prepare_fn = globals().get("prepare_gp_training_data")
            x_fit, y_fit, records_fit = x, y, records
            if callable(prepare_fn):
                x_fit, y_fit, records_fit, _denoise_stats = prepare_fn(self, x, y, records)
            if len(x_fit) < 2:
                return None
            y_mean = y_fit.mean(dim=0)
            y_std = y_fit.std(dim=0, unbiased=False)
            y_std = torch.where(y_std == 0, torch.tensor(1.0, dtype=y_std.dtype), y_std)
            y_std_vals = (y_fit - y_mean) / y_std
            bounds_full = self._combined_bounds()
            x_norm = torch.clamp(normalize(x_fit, bounds_full), 0.0, 1.0)
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
                "records": records_fit,
                "raw_y": y.detach().clone(),
                "used_y": y_fit.detach().clone(),
                "denoise_stats": dict(getattr(self, "cbo_last_history_denoise_stats", {}) or {}),
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
        train_x, train_y, train_records = self._training_data(state=state)
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
                    prepare_fn = globals().get("prepare_gp_training_data")
                    train_x_fit, train_y_fit, train_records_fit = train_x, train_y, train_records
                    if callable(prepare_fn):
                        train_x_fit, train_y_fit, train_records_fit, _denoise_stats = prepare_fn(self, train_x, train_y, train_records)
                    if len(train_x_fit) < 2:
                        raise RuntimeError("insufficient training rows after history denoise/filter")
                    train_y_std = standardize(train_y_fit)
                    train_x_norm = torch.clamp(normalize(train_x_fit, self.bounds), 0.0, 1.0)
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
            "completion_ratio": 0.0,
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
            "completion_ratio": float(self.window_feedback.get("completion_ratio", 0.0)),
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
            "completion_ratio": float(metrics.get("completion_ratio", 0.0)),
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
