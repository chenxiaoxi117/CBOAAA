#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Split from new_TR.py lines 820-1128.
# Event, Task, Node, and workload generators.

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

