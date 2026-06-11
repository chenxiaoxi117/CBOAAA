#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static batch scheduling and multi-client BO/FBO experiments.

This module is intentionally separate from ConnectedFactory.run_continuous().
The old path models streaming windows with carry-over backlog.  The code here
models a client receiving a finite task batch, dispatching that batch locally,
and returning one consistent batch-level objective to BO/CBO.
"""

STATIC_BATCH_CONTEXT_FEATURES = [
    "task_count",
    "rt_ratio",
    "batch_ratio",
    "ai_ratio",
    "total_workload_gcycles",
    "mean_workload_gcycles",
    "mean_upload_size",
    "mean_download_size",
    "mean_deadline",
    "deadline_tightness",
    "node_count",
    "edge_count",
    "cloud_count",
    "cloud_ratio",
    "total_capacity_slots",
    "mean_capacity_slots",
    "total_service_gips",
    "mean_service_gips",
    "service_cv",
    "total_idle_power",
    "total_max_power",
    "mean_link_delay",
    "p95_link_delay",
    "mean_link_bw",
    "min_link_bw",
    "capacity_pressure",
    "data_pressure",
]

STATIC_BATCH_CONTEXT_BOUNDS = {
    "task_count": (0.0, 10000.0),
    "rt_ratio": (0.0, 1.0),
    "batch_ratio": (0.0, 1.0),
    "ai_ratio": (0.0, 1.0),
    "total_workload_gcycles": (0.0, 500000.0),
    "mean_workload_gcycles": (0.0, 1000.0),
    "mean_upload_size": (0.0, 1000.0),
    "mean_download_size": (0.0, 200.0),
    "mean_deadline": (0.0, 1000.0),
    "deadline_tightness": (0.0, 10.0),
    "node_count": (1.0, 256.0),
    "edge_count": (0.0, 256.0),
    "cloud_count": (0.0, 64.0),
    "cloud_ratio": (0.0, 1.0),
    "total_capacity_slots": (1.0, 20000.0),
    "mean_capacity_slots": (1.0, 1000.0),
    "total_service_gips": (0.1, 5000.0),
    "mean_service_gips": (0.1, 500.0),
    "service_cv": (0.0, 5.0),
    "total_idle_power": (0.0, 100000.0),
    "total_max_power": (0.0, 200000.0),
    "mean_link_delay": (0.0, 50.0),
    "p95_link_delay": (0.0, 100.0),
    "mean_link_bw": (0.1, 5000.0),
    "min_link_bw": (0.1, 5000.0),
    "capacity_pressure": (0.0, 20.0),
    "data_pressure": (0.0, 20.0),
}

DEFAULT_STATIC_BATCH_CONTEXT = [
    "rt_ratio",
    "batch_ratio",
    "ai_ratio",
    "capacity_pressure",
    "data_pressure",
    "cloud_ratio",
    "mean_link_delay",
]

DEFAULT_STATIC_BATCH_OBJECTIVE_WEIGHTS = {
    "delay": 1.0,
    "energy": 0.25,
    "lateness": 2.0,
    "violation": 6.0,
    "makespan": 0.5,
    "unfinished": 20.0,
}


@dataclass
class StaticBatchClientConfig:
    client_id: int
    name: str
    node_config: List[Dict[str, Any]]
    task_probs: Dict[str, float]
    task_count: int
    network_delay_scale: float = 1.0
    bandwidth_scale: float = 1.0
    task_scale: float = 1.0
    workload_scale: float = 1.0
    deadline_scale: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _parse_static_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def parse_static_batch_int_list(spec, fallback=None):
    if spec is None or str(spec).strip() == "":
        return list(fallback or [])
    vals = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(float(part)))
    return vals


def parse_static_batch_float_list(spec, fallback=None):
    if spec is None or str(spec).strip() == "":
        return list(fallback or [])
    vals = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def parse_static_batch_task_prob_clients(spec, n_clients, fallback=None):
    if spec is None or str(spec).strip() == "":
        base = _normalize_task_probs(fallback or getattr(CFG, "TASK_TYPE_PROBS", None))
        return [dict(base) for _ in range(max(1, int(n_clients)))]
    out = []
    groups = [g.strip() for g in str(spec).split(";") if g.strip()]
    for group in groups:
        vals = [float(x.strip()) for x in group.split(",") if x.strip()]
        out.append(_normalize_task_probs(vals))
    if not out:
        out.append(_normalize_task_probs(fallback or getattr(CFG, "TASK_TYPE_PROBS", None)))
    while len(out) < int(n_clients):
        out.append(dict(out[-1]))
    return out[:int(n_clients)]


def parse_static_batch_context_features(spec):
    if spec is None or str(spec).strip() == "":
        return list(DEFAULT_STATIC_BATCH_CONTEXT)
    val = str(spec).strip().lower()
    if val in {"none", "off", "no"}:
        return []
    if val == "all":
        return list(STATIC_BATCH_CONTEXT_FEATURES)
    aliases = {
        "rt": "rt_ratio",
        "batch": "batch_ratio",
        "ai": "ai_ratio",
        "nodes": "node_count",
        "cloud": "cloud_ratio",
        "pressure": "capacity_pressure",
        "data": "data_pressure",
        "delay": "mean_link_delay",
    }
    names = []
    for raw in str(spec).split(","):
        key = raw.strip()
        if not key:
            continue
        key = aliases.get(key.lower(), key)
        if key not in STATIC_BATCH_CONTEXT_FEATURES:
            raise ValueError(
                f"Unknown batch context feature '{raw}'. "
                f"Use one of: {', '.join(STATIC_BATCH_CONTEXT_FEATURES)}"
            )
        names.append(key)
    return names


def parse_static_batch_objective_weights(spec):
    weights = dict(DEFAULT_STATIC_BATCH_OBJECTIVE_WEIGHTS)
    if spec is None or str(spec).strip() == "":
        return weights
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--batch-objective-weights must use name=value items")
        name, value = item.split("=", 1)
        name = name.strip().lower()
        if name not in weights:
            raise ValueError(
                f"Unknown objective weight '{name}'. "
                f"Use any of: {', '.join(sorted(weights))}"
            )
        weights[name] = float(value)
    return weights


def _deepcopy_jsonable(obj):
    return copy.deepcopy(obj)


def _scaled_node_cfg(node_cfg, service_rate_scale=1.0, capacity_scale=1.0, power_scale=1.0, bandwidth_scale=1.0, speed_scale=None):
    cfg = _deepcopy_jsonable(node_cfg)
    if speed_scale is not None:
        service_rate_scale = speed_scale
    if "service_rate_gips" in cfg:
        cfg["service_rate_gips"] = max(0.05, float(cfg["service_rate_gips"]) * float(service_rate_scale))
    if "service_rate_mips" in cfg:
        cfg["service_rate_mips"] = max(1.0, float(cfg["service_rate_mips"]) * float(service_rate_scale))
    if "speed" in cfg:
        cfg["speed"] = max(0.05, float(cfg["speed"]) * float(service_rate_scale))
    if "capacity_slots" in cfg:
        cfg["capacity_slots"] = max(1, int(round(float(cfg["capacity_slots"]) * float(capacity_scale))))
    if "num_cores" in cfg:
        cfg["num_cores"] = max(1, int(round(float(cfg["num_cores"]) * float(capacity_scale))))
    for key in ("idle_power", "max_power", "p_idle", "p_max"):
        if key in cfg:
            cfg[key] = max(0.0, float(cfg[key]) * float(power_scale))
    for key in ("uplink_bandwidth", "downlink_bandwidth"):
        if key in cfg:
            cfg[key] = max(0.1, float(cfg[key]) * float(bandwidth_scale))
    return cfg


def _client_rng(seed, client_id, stream=0):
    return random.Random(resolve_base_seed(int(seed), stream=9000 + int(client_id) * 37 + int(stream)))


def _select_client_nodes(base_nodes, rng, node_count, profile):
    profile = str(profile or "heterogeneous").lower()
    edges = [cfg for cfg in base_nodes if not _node_is_cloud(cfg)]
    clouds = [cfg for cfg in base_nodes if _node_is_cloud(cfg)]
    if not edges:
        edges = list(base_nodes)
    if not clouds:
        clouds = []

    node_count = max(1, int(node_count))
    if profile == "edge_small":
        cloud_n = 1 if node_count >= 4 and clouds else 0
    elif profile == "cloud_heavy":
        cloud_n = min(len(clouds), max(1, node_count // 3))
    elif profile == "edge_large":
        cloud_n = 1 if clouds else 0
    else:
        cloud_n = min(len(clouds), max(1, int(round(node_count * 0.18)))) if node_count >= 5 else (1 if clouds else 0)
    edge_n = max(0, node_count - cloud_n)

    if edge_n > len(edges):
        chosen_edges = [rng.choice(edges) for _ in range(edge_n)]
    else:
        chosen_edges = rng.sample(edges, edge_n)
    if cloud_n > len(clouds):
        chosen_clouds = [rng.choice(clouds) for _ in range(cloud_n)]
    else:
        chosen_clouds = rng.sample(clouds, cloud_n)
    selected = chosen_edges + chosen_clouds
    for new_id, cfg in enumerate(selected):
        cfg = _deepcopy_jsonable(cfg)
        cfg["id"] = new_id
        selected[new_id] = cfg
    return selected


def load_static_batch_client_config(path, n_clients, default_task_count):
    if path is None or str(path).strip() == "":
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    clients = payload.get("clients", payload if isinstance(payload, list) else [])
    out = []
    for idx, raw in enumerate(clients[: int(n_clients)]):
        nodes = raw.get("node_config", raw.get("nodes", None))
        if nodes is None:
            ids = raw.get("node_ids", None)
            if ids is None:
                nodes = CFG.NODES_CFG
            else:
                nodes = [CFG.NODES_CFG[int(i)] for i in ids]
        nodes = [_deepcopy_jsonable(cfg) for cfg in nodes]
        for new_id, cfg in enumerate(nodes):
            cfg["id"] = new_id
        task_probs = _normalize_task_probs(raw.get("task_probs", raw.get("task_mix", getattr(CFG, "TASK_TYPE_PROBS", None))))
        out.append(StaticBatchClientConfig(
            client_id=idx,
            name=str(raw.get("name", f"client_{idx}")),
            node_config=nodes,
            task_probs=task_probs,
            task_count=int(raw.get("task_count", default_task_count)),
            network_delay_scale=float(raw.get("network_delay_scale", raw.get("delay_scale", 1.0))),
            bandwidth_scale=float(raw.get("bandwidth_scale", 1.0)),
            task_scale=float(raw.get("task_scale", 1.0)),
            workload_scale=float(raw.get("workload_scale", 1.0)),
            deadline_scale=float(raw.get("deadline_scale", 1.0)),
            metadata={k: v for k, v in raw.items() if k not in {"node_config", "nodes"}},
        ))
    return out


def build_static_batch_clients(
    n_clients=3,
    task_count=120,
    task_counts=None,
    seed=42,
    topology_profile="heterogeneous",
    node_counts=None,
    task_probs_by_client=None,
    client_config_path=None,
):
    loaded = load_static_batch_client_config(client_config_path, n_clients, task_count)
    if loaded is not None:
        return loaded

    n_clients = max(1, int(n_clients))
    if task_counts is None:
        task_counts = [int(task_count) for _ in range(n_clients)]
    else:
        task_counts = [int(x) for x in task_counts]
        while len(task_counts) < n_clients:
            task_counts.append(task_counts[-1])
    base_nodes = [_deepcopy_jsonable(cfg) for cfg in CFG.NODES_CFG]
    if not node_counts:
        default_counts = [6, 8, 10, 12, 7, 9]
        node_counts = [default_counts[i % len(default_counts)] for i in range(n_clients)]
    while len(node_counts) < n_clients:
        node_counts.append(node_counts[-1])

    if task_probs_by_client is None:
        task_probs_by_client = []
        mixes = [
            [0.20, 0.40, 0.40],
            [0.45, 0.25, 0.30],
            [0.10, 0.60, 0.30],
            [0.15, 0.20, 0.65],
            [0.30, 0.30, 0.40],
        ]
        for i in range(n_clients):
            task_probs_by_client.append(_normalize_task_probs(mixes[i % len(mixes)]))
    while len(task_probs_by_client) < n_clients:
        task_probs_by_client.append(dict(task_probs_by_client[-1]))

    clients = []
    for cid in range(n_clients):
        rng = _client_rng(seed, cid, stream=11)
        selected = _select_client_nodes(base_nodes, rng, int(node_counts[cid]), topology_profile)
        service_rate_scale = 0.85 + 0.30 * rng.random()
        capacity_scale = 0.85 + 0.35 * rng.random()
        power_scale = 0.90 + 0.30 * rng.random()
        bandwidth_scale = 0.80 + 0.45 * rng.random()
        delay_scale = 0.85 + 0.50 * rng.random()
        scaled = [
            _scaled_node_cfg(
                cfg,
                service_rate_scale=service_rate_scale,
                capacity_scale=capacity_scale,
                power_scale=power_scale,
                bandwidth_scale=bandwidth_scale,
            )
            for cfg in selected
        ]
        clients.append(StaticBatchClientConfig(
            client_id=cid,
            name=f"client_{cid}",
            node_config=scaled,
            task_probs=dict(task_probs_by_client[cid]),
            task_count=int(task_counts[cid]),
            network_delay_scale=float(delay_scale),
            bandwidth_scale=float(bandwidth_scale),
            task_scale=1.0,
            workload_scale=1.0,
            deadline_scale=1.0,
            metadata={
                "topology_profile": str(topology_profile),
                "service_rate_scale": float(service_rate_scale),
                "capacity_scale": float(capacity_scale),
                "power_scale": float(power_scale),
                "bandwidth_scale": float(bandwidth_scale),
                "delay_scale": float(delay_scale),
            },
        ))
    return clients


class _StaticClientRuntime:
    def __init__(self, client_cfg):
        self.client_cfg = client_cfg
        self.saved = None

    def __enter__(self):
        self.saved = {
            "NODES_CFG": getattr(CFG, "NODES_CFG", None),
            "TRANS_DELAY_MATRIX": getattr(CFG, "TRANS_DELAY_MATRIX", None),
            "TRANS_BW_MATRIX": getattr(CFG, "TRANS_BW_MATRIX", None),
        }
        CFG.NODES_CFG = [_deepcopy_jsonable(cfg) for cfg in self.client_cfg.node_config]
        CFG.TRANS_DELAY_MATRIX = build_topology_matrix(len(CFG.NODES_CFG))
        CFG.TRANS_BW_MATRIX = build_bandwidth_matrix(len(CFG.NODES_CFG))
        delay_scale = float(getattr(self.client_cfg, "network_delay_scale", 1.0))
        bw_scale = float(getattr(self.client_cfg, "bandwidth_scale", 1.0))
        if delay_scale != 1.0:
            CFG.TRANS_DELAY_MATRIX = [
                [float(x) * delay_scale for x in row]
                for row in CFG.TRANS_DELAY_MATRIX
            ]
        if bw_scale != 1.0:
            CFG.TRANS_BW_MATRIX = [
                [max(1e-6, float(x) * bw_scale) for x in row]
                for row in CFG.TRANS_BW_MATRIX
            ]
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.saved is not None:
            CFG.NODES_CFG = self.saved["NODES_CFG"]
            CFG.TRANS_DELAY_MATRIX = self.saved["TRANS_DELAY_MATRIX"]
            CFG.TRANS_BW_MATRIX = self.saved["TRANS_BW_MATRIX"]
        return False


def _task_props_for_client(task_type, client_cfg):
    props = _deepcopy_jsonable(CFG.TASK_PROPS[task_type])
    task_scale = float(getattr(client_cfg, "task_scale", 1.0))
    workload_scale = float(getattr(client_cfg, "workload_scale", 1.0))
    deadline_scale = float(getattr(client_cfg, "deadline_scale", 1.0))
    for key in ("upload_size", "download_size", "data"):
        if key in props:
            props[key] = max(0.0, float(props[key]) * task_scale)
    if "workload_cycles" in props:
        props["workload_cycles"] = max(1.0, float(props["workload_cycles"]) * workload_scale)
    if "dur" in props:
        props["dur"] = max(1e-9, float(props["dur"]) * workload_scale)
    if "deadline" in props:
        props["deadline"] = max(1e-9, float(props["deadline"]) * deadline_scale)
    if "deadline_factor" in props:
        props["deadline_factor"] = max(1e-9, float(props["deadline_factor"]) * deadline_scale)
    return props


def _sample_static_client_origin(rng, client_cfg):
    local_nodes = list(getattr(client_cfg, "node_config", []) or [])
    edge_nodes = [cfg for cfg in local_nodes if not _node_is_cloud(cfg)]
    if not edge_nodes:
        edge_nodes = list(local_nodes)
    if not edge_nodes:
        return 0
    by_workshop = collections.defaultdict(list)
    for cfg in edge_nodes:
        by_workshop[_get_node_site_from_cfg(cfg, default_site=0)].append(int(cfg.get("id", 0)))
    weights = list(getattr(CFG, "WORKSHOP_ORIGIN_BIAS", []))
    available = sorted(by_workshop)
    weighted_sites = []
    weighted_values = []
    for site in available:
        weighted_sites.append(site)
        weighted_values.append(float(weights[site]) if 0 <= int(site) < len(weights) else 1.0)
    total = sum(max(0.0, w) for w in weighted_values)
    if total <= 0.0:
        site = rng.choice(available)
    else:
        r = rng.random() * total
        cum = 0.0
        site = available[-1]
        for s, w in zip(weighted_sites, weighted_values):
            cum += max(0.0, w)
            if r <= cum:
                site = s
                break
    return int(rng.choice(by_workshop[site]))


def generate_static_batch_tasks(client_cfg, batch_index=0, seed=42, reuse_mode="new_each_round", order="generated"):
    stream = 1000 if str(reuse_mode) == "fixed_per_client" else 1000 + int(batch_index)
    rng = _client_rng(seed, int(client_cfg.client_id), stream=stream)
    n = int(client_cfg.task_count)
    tasks = []
    probs = _normalize_task_probs(client_cfg.task_probs)
    saved_probs = getattr(CFG, "TASK_TYPE_PROBS", None)
    try:
        CFG.TASK_TYPE_PROBS = probs
        for i in range(n):
            t_type = sample_task_type(rng, current_time=0.0)
            props = _task_props_for_client(t_type, client_cfg)
            origin = _sample_static_client_origin(rng, client_cfg)
            task = Task(
                id=f"c{client_cfg.client_id}-b{batch_index}-t{i}",
                **build_task_kwargs_from_props(t_type, props, 0.0, origin_node_id=origin),
            )
            tasks.append(task)
    finally:
        CFG.TASK_TYPE_PROBS = saved_probs

    order = str(order or "generated").lower()
    if order == "deadline":
        tasks.sort(key=lambda t: (float(t.deadline), str(t.task_type), str(t.id)))
    elif order == "type_deadline":
        type_rank = {"RT": 0, "Batch": 1, "AI": 2}
        tasks.sort(key=lambda t: (type_rank.get(t.task_type, 9), float(t.deadline), str(t.id)))
    elif order == "largest_workload":
        tasks.sort(key=lambda t: (-float(t.cpu_cycles), float(t.deadline), str(t.id)))
    elif order == "random":
        rng.shuffle(tasks)
    return tasks


def _try_start_static(node, current_time, event_heap, event_counter):
    for task in list(node.ready_queue):
        if node.allocate(task):
            node.ready_queue.remove(task)
            service_rate = node.effective_speed(task)
            task.compute_finish_time = float(current_time) + float(task.cpu_cycles) / (service_rate + 1e-9)
            heapq.heappush(event_heap, (task.compute_finish_time, event_counter, "task_finish", task))
            event_counter += 1
    return event_counter


def _accumulate_static_power(nodes, start_time, end_time, energy_acc):
    dt = float(end_time) - float(start_time)
    if dt <= 1e-12:
        return
    obj_energy = 0.0
    real_energy = 0.0
    dyn_energy = 0.0
    idle_obj_energy = 0.0
    for node in nodes:
        idle_obj, dyn_power, total_obj_power = node.power_components(objective=True)
        _idle_real, _dyn_power_real, total_real_power = node.power_components(objective=False)
        obj_energy += total_obj_power * dt
        real_energy += total_real_power * dt
        dyn_energy += dyn_power * dt
        idle_obj_energy += idle_obj * dt
    energy_acc["total_energy"] += obj_energy
    energy_acc["total_energy_real"] += real_energy
    energy_acc["compute_dynamic_energy"] += dyn_energy
    energy_acc["compute_idle_energy"] += idle_obj_energy
    energy_acc["compute_energy_real"] += real_energy


def _batch_type_counts(tasks):
    counts = {t: 0 for t in TASK_TYPE_ORDER}
    for task in tasks:
        counts[task.task_type] = counts.get(task.task_type, 0) + 1
    return counts


def compute_static_batch_reference(tasks, client_cfg, normalization="batch_reference"):
    n = max(1, len(tasks))
    if str(normalization).lower() in {"none", "raw", "off"}:
        return {
            "reference_delay": 1.0,
            "reference_energy": 1.0,
            "reference_lateness": 1.0,
            "reference_makespan": 1.0,
            "normalization_mode": "none",
        }
    with _StaticClientRuntime(client_cfg):
        nodes = [Node(cfg) for cfg in CFG.NODES_CFG]
        median_latencies = []
        median_energies = []
        min_service_times = []
        for task in tasks:
            origin = task.origin_node_id if task.origin_node_id >= 0 else 0
            per_node = []
            service_times = []
            for node in nodes:
                try:
                    e, l = node.estimate_metrics(task, 0.0, origin_node_idx=origin)
                except Exception:
                    e, l = 0.0, float(task.deadline)
                per_node.append((float(e), float(l)))
                try:
                    service_times.append(float(task.cpu_cycles) / (float(node.effective_speed(task)) + 1e-9))
                except Exception:
                    pass
            if per_node:
                median_energies.append(float(np.median([x[0] for x in per_node])))
                median_latencies.append(float(np.median([x[1] for x in per_node])))
            if service_times:
                min_service_times.append(float(np.min(service_times)))
    deadline_budgets = [max(1e-9, float(t.deadline) - float(t.create_time)) for t in tasks]
    mean_deadline = float(np.mean(deadline_budgets)) if deadline_budgets else 1.0
    ref_delay = max(1e-6, 0.5 * mean_deadline + 0.5 * (float(np.mean(median_latencies)) if median_latencies else mean_deadline))
    ref_energy = max(1e-6, float(np.mean(median_energies)) if median_energies else 1.0)
    ref_lateness = max(1e-6, mean_deadline)
    ref_makespan = max(
        1e-6,
        float(np.percentile(deadline_budgets, 90)) if deadline_budgets else mean_deadline,
        float(np.sum(min_service_times)) / max(1.0, float(len(client_cfg.node_config))) if min_service_times else 1.0,
    )
    return {
        "reference_delay": float(ref_delay),
        "reference_energy": float(ref_energy),
        "reference_lateness": float(ref_lateness),
        "reference_makespan": float(ref_makespan),
        "normalization_mode": "batch_reference",
    }


def compute_static_batch_objective(metrics, reference, weights=None, clip=10.0):
    weights = dict(DEFAULT_STATIC_BATCH_OBJECTIVE_WEIGHTS if weights is None else weights)
    clip = float(clip)

    def ratio(value, ref):
        ref = max(1e-9, float(ref))
        val = float(value) / ref
        if clip > 0:
            val = min(max(val, 0.0), clip)
        return float(val)

    delay_norm = ratio(metrics.get("effective_avg_delay", metrics.get("avg_delay", 0.0)), reference.get("reference_delay", 1.0))
    energy_norm = ratio(metrics.get("avg_energy", 0.0), reference.get("reference_energy", 1.0))
    lateness_norm = ratio(metrics.get("effective_avg_lateness", metrics.get("avg_lateness", 0.0)), reference.get("reference_lateness", 1.0))
    makespan_norm = ratio(metrics.get("makespan", 0.0), reference.get("reference_makespan", 1.0))
    violation = float(metrics.get("effective_vio_rate", metrics.get("vio_rate", 0.0)))
    unfinished = float(metrics.get("unfinished_rate", 0.0))
    components = {
        "delay_norm": delay_norm,
        "energy_norm": energy_norm,
        "lateness_norm": lateness_norm,
        "violation": violation,
        "makespan_norm": makespan_norm,
        "unfinished": unfinished,
    }
    cost = (
        float(weights.get("delay", 0.0)) * delay_norm
        + float(weights.get("energy", 0.0)) * energy_norm
        + float(weights.get("lateness", 0.0)) * lateness_norm
        + float(weights.get("violation", 0.0)) * violation
        + float(weights.get("makespan", 0.0)) * makespan_norm
        + float(weights.get("unfinished", 0.0)) * unfinished
    )
    return float(cost), components


def simulate_static_batch_schedule(
    client_cfg,
    tasks,
    theta,
    eval_seed=42,
    scheduler_type="Boltzmann",
    norm_mode="rolling",
    deterministic_scheduler=True,
    dispatch_gap=0.01,
    max_time=None,
    objective_weights=None,
    normalization="batch_reference",
    objective_clip=10.0,
):
    reference = compute_static_batch_reference(tasks, client_cfg, normalization=normalization)
    norm_mode = str(norm_mode or "rolling").lower()
    if norm_mode not in {"fixed", "rolling"}:
        norm_mode = "rolling"
    saved_random = getattr(CFG, "USE_BOLTZMANN_RANDOM", True)
    if deterministic_scheduler:
        CFG.USE_BOLTZMANN_RANDOM = False
    try:
        with _StaticClientRuntime(client_cfg):
            nodes = [Node(cfg) for cfg in CFG.NODES_CFG]
            scheduler_type_norm = str(scheduler_type or "Boltzmann").strip().lower().replace("-", "_")
            if scheduler_type_norm == "boltzmann":
                scheduler_cls = ConstrainedBoltzmannScheduler if getattr(CFG, "USE_CONSTRAINED_BOLTZMANN", True) else BoltzmannScheduler
                scheduler = scheduler_cls(np_rng=np.random.default_rng(int(eval_seed) + 1), norm_mode=norm_mode)
            elif scheduler_type_norm in {"roundrobin", "round_robin", "round_robin_direct", "rr"}:
                scheduler = RoundRobinScheduler()
            else:
                scheduler = DirectHeuristicScheduler(mode=scheduler_type_norm, np_rng=np.random.default_rng(int(eval_seed) + 1))

            event_heap = []
            counter = 0
            dispatch_gap = max(0.0, float(dispatch_gap))
            for rank, task in enumerate(tasks):
                dispatch_time = float(rank) * dispatch_gap
                heapq.heappush(event_heap, (dispatch_time, counter, "arrival", task))
                counter += 1

            energy_acc = {
                "total_energy": 0.0,
                "total_energy_real": 0.0,
                "compute_dynamic_energy": 0.0,
                "compute_idle_energy": 0.0,
                "compute_energy_real": 0.0,
                "transmission_energy": 0.0,
            }
            current_time = 0.0
            completed = []
            completed_ids = set()
            alloc_counts = [0 for _ in nodes]
            alloc_by_type = {t: [0 for _ in nodes] for t in TASK_TYPE_ORDER}
            scheduler_debugs = []
            link_busy_until = {}
            max_time = None if max_time is None or float(max_time) <= 0 else float(max_time)

            while event_heap:
                next_time, event_counter, event_type, task = heapq.heappop(event_heap)
                if max_time is not None and float(next_time) > max_time:
                    _accumulate_static_power(nodes, current_time, max_time, energy_acc)
                    current_time = max_time
                    break
                _accumulate_static_power(nodes, current_time, next_time, energy_acc)
                current_time = float(next_time)
                try:
                    scheduler.current_time = current_time
                except Exception:
                    pass

                if event_type == "arrival":
                    node_idx, _probs, _score = scheduler.select_node(task, nodes, theta)
                    task.arrival_node_idx = int(node_idx)
                    alloc_counts[int(node_idx)] += 1
                    if task.task_type in alloc_by_type:
                        alloc_by_type[task.task_type][int(node_idx)] += 1
                    debug = getattr(scheduler, "last_score_debug", {}) or {}
                    if debug:
                        scheduler_debugs.append(dict(debug))
                    origin = task.origin_node_id if task.origin_node_id >= 0 else int(node_idx)
                    upload_delay = get_transmission_delay(origin, int(node_idx), getattr(task, "upload_size", task.data_size), include_local=True, direction="upload")
                    if getattr(CFG, "USE_LINK_QUEUE", False):
                        link_key = (int(origin), int(node_idx))
                        trans_start = max(current_time, float(link_busy_until.get(link_key, current_time)))
                        trans_finish = trans_start + upload_delay
                        link_busy_until[link_key] = trans_finish
                    else:
                        trans_finish = current_time + upload_delay
                    task.upload_finish_time = float(trans_finish)
                    trans_energy = get_transmission_energy(origin, int(node_idx), getattr(task, "upload_size", task.data_size))
                    task.transmission_energy = float(trans_energy)
                    energy_acc["transmission_energy"] += float(trans_energy)
                    energy_acc["total_energy"] += float(trans_energy)
                    energy_acc["total_energy_real"] += float(trans_energy)
                    heapq.heappush(event_heap, (trans_finish, counter, "transfer_finish", task))
                    counter += 1

                elif event_type == "transfer_finish":
                    node = nodes[int(task.arrival_node_idx)]
                    node.enqueue_task(task)
                    counter = _try_start_static(node, current_time, event_heap, counter)

                elif event_type == "task_finish":
                    node = nodes[int(task.arrival_node_idx)]
                    node.release(task)
                    counter = _try_start_static(node, current_time, event_heap, counter)
                    task.compute_finish_time = float(current_time)
                    origin = task.origin_node_id if task.origin_node_id >= 0 else int(task.arrival_node_idx)
                    download_delay = get_transmission_delay(int(task.arrival_node_idx), origin, getattr(task, "download_size", 0.0), include_local=True, direction="download")
                    if getattr(CFG, "USE_LINK_QUEUE", False):
                        link_key = (int(task.arrival_node_idx), int(origin))
                        download_start = max(current_time, float(link_busy_until.get(link_key, current_time)))
                        finish_time = download_start + download_delay
                        link_busy_until[link_key] = finish_time
                    else:
                        finish_time = current_time + download_delay
                    download_energy = get_transmission_energy(int(task.arrival_node_idx), origin, getattr(task, "download_size", 0.0))
                    task.transmission_energy = float(getattr(task, "transmission_energy", 0.0)) + float(download_energy)
                    energy_acc["transmission_energy"] += float(download_energy)
                    energy_acc["total_energy"] += float(download_energy)
                    energy_acc["total_energy_real"] += float(download_energy)
                    task.finish_time = float(finish_time)
                    heapq.heappush(event_heap, (finish_time, counter, "download_finish", task))
                    counter += 1

                elif event_type == "download_finish":
                    task.finish_time = float(current_time)
                    task.energy_consumed = float(getattr(task, "transmission_energy", 0.0))
                    completed.append(task)
                    completed_ids.add(task.id)

            makespan = float(current_time)
            n = max(1, len(tasks))
            completed_count = len(completed)
            unfinished_tasks = [t for t in tasks if t.id not in completed_ids]
            unfinished_count = len(unfinished_tasks)
            delay_vals = [max(0.0, float(t.finish_time) - float(t.create_time)) for t in completed]
            late_vals = [max(0.0, float(t.finish_time) - float(t.deadline)) for t in completed]
            vio_vals = [1.0 if float(t.finish_time) > float(t.deadline) else 0.0 for t in completed]
            censored_delays = list(delay_vals)
            censored_late = list(late_vals)
            censored_vio = list(vio_vals)
            for task in unfinished_tasks:
                elapsed = max(0.0, makespan - float(task.create_time))
                censored_delays.append(elapsed)
                censored_late.append(max(0.0, makespan - float(task.deadline)))
                censored_vio.append(1.0 if makespan > float(task.deadline) else 0.0)

            type_counts = _batch_type_counts(tasks)
            metrics = {
                "client_id": int(client_cfg.client_id),
                "client_name": str(client_cfg.name),
                "task_count": int(len(tasks)),
                "completed_total": int(completed_count),
                "unfinished_end": int(unfinished_count),
                "completion_ratio": float(completed_count) / n,
                "unfinished_rate": float(unfinished_count) / n,
                "makespan": float(makespan),
                "avg_delay": float(np.mean(delay_vals)) if delay_vals else 0.0,
                "avg_lateness": float(np.mean(late_vals)) if late_vals else 0.0,
                "vio_rate": float(np.mean(vio_vals)) if vio_vals else 0.0,
                "effective_avg_delay": float(np.mean(censored_delays)) if censored_delays else 0.0,
                "effective_avg_lateness": float(np.mean(censored_late)) if censored_late else 0.0,
                "effective_vio_rate": float(np.mean(censored_vio)) if censored_vio else 0.0,
                "total_energy": float(energy_acc["total_energy"]),
                "total_energy_real": float(energy_acc["total_energy_real"]),
                "compute_dynamic_energy": float(energy_acc["compute_dynamic_energy"]),
                "compute_idle_energy": float(energy_acc["compute_idle_energy"]),
                "compute_energy_real": float(energy_acc["compute_energy_real"]),
                "transmission_energy": float(energy_acc["transmission_energy"]),
                "avg_energy": float(energy_acc["total_energy"]) / n,
                "alloc": list(alloc_counts),
                "alloc_by_type": {k: list(v) for k, v in alloc_by_type.items()},
                "rt_arrivals": int(type_counts.get("RT", 0)),
                "batch_arrivals": int(type_counts.get("Batch", 0)),
                "ai_arrivals": int(type_counts.get("AI", 0)),
                "rt_ratio": float(type_counts.get("RT", 0)) / n,
                "batch_ratio": float(type_counts.get("Batch", 0)) / n,
                "ai_ratio": float(type_counts.get("AI", 0)) / n,
            }

            for task_type in TASK_TYPE_ORDER:
                done = [t for t in completed if t.task_type == task_type]
                prefix = task_type.lower() if task_type != "Batch" else "batch"
                metrics[f"{prefix}_completed"] = int(len(done))
                metrics[f"{prefix}_completion_ratio"] = float(len(done)) / max(1, int(type_counts.get(task_type, 0)))
                metrics[f"{prefix}_avg_delay"] = float(np.mean([t.finish_time - t.create_time for t in done])) if done else 0.0
                metrics[f"{prefix}_vio_rate"] = float(np.mean([1.0 if t.finish_time > t.deadline else 0.0 for t in done])) if done else 0.0

            cost, components = compute_static_batch_objective(
                metrics,
                reference,
                weights=objective_weights,
                clip=objective_clip,
            )
            metrics["cost"] = float(cost)
            metrics["reward"] = -float(cost)
            metrics.update(reference)
            metrics.update(components)
            if scheduler_debugs:
                last_debug = dict(scheduler_debugs[-1])
                metrics["scheduler_debug_last"] = last_debug
                alpha_vals = []
                for debug in scheduler_debugs:
                    try:
                        val = debug.get("scheduler_alpha", None)
                        if val is not None and np.isfinite(float(val)):
                            alpha_vals.append(float(val))
                    except Exception:
                        pass
                metrics["scheduler_alpha_mean"] = float(np.mean(alpha_vals)) if alpha_vals else np.nan
            else:
                metrics["scheduler_debug_last"] = {}
                metrics["scheduler_alpha_mean"] = np.nan
            return metrics
    finally:
        CFG.USE_BOLTZMANN_RANDOM = saved_random


def compute_static_batch_context(client_cfg, tasks, feature_names=None):
    feature_names = list(STATIC_BATCH_CONTEXT_FEATURES if feature_names is None else feature_names)
    n = max(1, len(tasks))
    type_counts = _batch_type_counts(tasks)
    workload_g = [float(t.cpu_cycles) / 1e9 for t in tasks]
    uploads = [float(getattr(t, "upload_size", 0.0)) for t in tasks]
    downloads = [float(getattr(t, "download_size", 0.0)) for t in tasks]
    deadlines = [max(1e-9, float(t.deadline) - float(t.create_time)) for t in tasks]
    duration_refs = [max(1e-9, float(getattr(t, "duration_base", 0.0))) for t in tasks]
    node_cfg = list(client_cfg.node_config)
    node_count = max(1, len(node_cfg))
    cloud_count = sum(1 for cfg in node_cfg if _node_is_cloud(cfg))
    edge_count = node_count - cloud_count
    capacities = [float(get_node_capacity_slots(cfg)) for cfg in node_cfg]
    services = [float(get_node_service_rate(cfg, None)) / 1e9 for cfg in node_cfg]
    idle_power = [float(get_node_idle_power(cfg)) for cfg in node_cfg]
    max_power = [float(get_node_max_power(cfg)) for cfg in node_cfg]
    with _StaticClientRuntime(client_cfg):
        delays = np.array(CFG.TRANS_DELAY_MATRIX, dtype=float)
        bws = np.array(CFG.TRANS_BW_MATRIX, dtype=float)
    finite_delays = delays[np.isfinite(delays)] if delays.size else np.array([0.0])
    finite_bw = bws[np.isfinite(bws)] if bws.size else np.array([1.0])
    total_work = float(np.sum([float(t.cpu_cycles) for t in tasks]))
    mean_deadline = float(np.mean(deadlines)) if deadlines else 1.0
    total_service = max(1e-9, float(np.sum(services)) * 1e9)
    total_data = float(np.sum(uploads) + np.sum(downloads))
    mean_bw = max(1e-9, float(np.mean(finite_bw)))
    all_features = {
        "task_count": float(n),
        "rt_ratio": float(type_counts.get("RT", 0)) / n,
        "batch_ratio": float(type_counts.get("Batch", 0)) / n,
        "ai_ratio": float(type_counts.get("AI", 0)) / n,
        "total_workload_gcycles": float(np.sum(workload_g)) if workload_g else 0.0,
        "mean_workload_gcycles": float(np.mean(workload_g)) if workload_g else 0.0,
        "mean_upload_size": float(np.mean(uploads)) if uploads else 0.0,
        "mean_download_size": float(np.mean(downloads)) if downloads else 0.0,
        "mean_deadline": float(mean_deadline),
        "deadline_tightness": float(np.mean([duration_refs[i] / deadlines[i] for i in range(len(deadlines))])) if deadlines else 0.0,
        "node_count": float(node_count),
        "edge_count": float(edge_count),
        "cloud_count": float(cloud_count),
        "cloud_ratio": float(cloud_count) / node_count,
        "total_capacity_slots": float(np.sum(capacities)) if capacities else 0.0,
        "mean_capacity_slots": float(np.mean(capacities)) if capacities else 0.0,
        "total_service_gips": float(np.sum(services)) if services else 0.0,
        "mean_service_gips": float(np.mean(services)) if services else 0.0,
        "service_cv": float(np.std(services) / max(1e-9, np.mean(services))) if services else 0.0,
        "total_idle_power": float(np.sum(idle_power)) if idle_power else 0.0,
        "total_max_power": float(np.sum(max_power)) if max_power else 0.0,
        "mean_link_delay": float(np.mean(finite_delays)) if finite_delays.size else 0.0,
        "p95_link_delay": float(np.percentile(finite_delays, 95)) if finite_delays.size else 0.0,
        "mean_link_bw": float(mean_bw),
        "min_link_bw": float(np.min(finite_bw)) if finite_bw.size else 0.0,
        "capacity_pressure": float(total_work / max(1e-9, total_service * mean_deadline)),
        "data_pressure": float(total_data / max(1e-9, mean_bw * mean_deadline)),
    }
    return {name: float(all_features.get(name, 0.0)) for name in feature_names}


def static_batch_context_bounds(feature_names):
    lows = []
    highs = []
    for name in feature_names:
        lo, hi = STATIC_BATCH_CONTEXT_BOUNDS.get(name, (0.0, 1.0))
        lows.append(float(lo))
        highs.append(float(hi))
    return [lows, highs]


def static_batch_context_vector(context_dict, feature_names):
    return [float(context_dict.get(name, 0.0)) for name in feature_names]


def _make_static_batch_agent(seed, context_features, use_trust_region=True):
    context_features = list(context_features or [])
    use_context = bool(context_features)
    return FederatedBOAgent(
        dim=CFG.DIM_THETA,
        py_rng=random.Random(resolve_base_seed(int(seed), stream=7100)),
        torch_gen=torch.Generator().manual_seed(resolve_base_seed(int(seed), stream=7101)),
        bounds=get_control_bounds(CFG.DIM_THETA),
        feature_names=list(CFG.FEATURE_NAMES),
        use_context=use_context,
        use_state_partition=False,
        use_trust_region=bool(use_trust_region),
        context_dim=len(context_features),
        context_bounds=static_batch_context_bounds(context_features) if use_context else None,
        anchor_points=default_scenario_anchor_points(),
    )


def _theta_to_prefixed_dict(theta, prefix="theta"):
    theta = normalize_theta_vector(theta, dim=CFG.DIM_THETA)
    out = {}
    for idx, name in enumerate(list(CFG.FEATURE_NAMES)[: len(theta)]):
        out[f"{prefix}_{name}"] = float(theta[idx])
    return out


def _sample_theta_candidates(rng, n, include=None):
    low, high = get_control_bounds(CFG.DIM_THETA)
    candidates = []
    include = include or []
    for theta in include:
        if theta is not None:
            candidates.append(normalize_theta_vector(theta, dim=CFG.DIM_THETA))
    for _ in range(max(0, int(n))):
        candidates.append([float(low[d]) + (float(high[d]) - float(low[d])) * rng.random() for d in range(CFG.DIM_THETA)])
    unique = []
    seen = set()
    for theta in candidates:
        key = tuple(round(float(v), 8) for v in theta)
        if key in seen:
            continue
        seen.add(key)
        unique.append(theta)
    return unique


def _append_external_summary(agent, theta, cost, context=None):
    if agent is None:
        return
    theta = agent._normalize_theta(theta)
    sample = agent._pack_sample(theta, -float(cost), state=None, context=context)
    if len(agent.local_recent) == agent.local_recent.maxlen and agent.local_recent:
        old = agent.local_recent[0]
        old_rec = agent._unpack_sample(old)
        agent._archive_state_sample(old_rec.get("state"), old_rec)
    agent.local_recent.append(sample)


def _experience_share(round_rows, agents, context_features, topk=3):
    if not round_rows:
        return
    ranked = sorted(round_rows, key=lambda r: float(r.get("cost", float("inf"))))
    for row in ranked[: max(1, int(topk))]:
        source_client = int(row.get("client_id", -1))
        theta = [float(row.get(f"theta_{name}", np.nan)) for name in CFG.FEATURE_NAMES]
        if not np.all(np.isfinite(theta)):
            continue
        context = [float(row.get(f"ctx_{name}", 0.0)) for name in context_features] if context_features else None
        for cid, agent in agents.items():
            if int(cid) == source_client:
                continue
            _append_external_summary(agent, theta, float(row.get("cost", 0.0)), context=context)


def _choose_federated_theta(target_cid, agents, context_vec, candidate_pool, aggregator, fed_beta=None):
    packets = []
    for source_cid, agent in agents.items():
        preds = agent.predict_candidates(candidate_pool, state=None, context=context_vec if agent.use_context else None)
        if preds:
            packets.append({
                "factory_id": int(source_cid),
                "client_id": int(source_cid),
                "target_client_id": int(target_cid),
                "predictions": preds,
            })
    aggregated = aggregator.aggregate_predictions(packets, beta_cloud=fed_beta)
    if aggregated:
        best = max(aggregated, key=lambda row: float(row.get("score_fed", -float("inf"))))
        return list(best["theta"]), {
            "fed_packet_count": int(len(packets)),
            "fed_candidate_count": int(len(aggregated)),
            "fed_mu": float(best.get("mu_fed", np.nan)),
            "fed_sigma": float(best.get("sigma_fed", np.nan)),
            "fed_score": float(best.get("score_fed", np.nan)),
            "fed_fallback": False,
        }
    agent = agents[target_cid]
    theta = agent.ask(context=context_vec if agent.use_context else None)
    return list(theta), {
        "fed_packet_count": int(len(packets)),
        "fed_candidate_count": 0,
        "fed_mu": np.nan,
        "fed_sigma": np.nan,
        "fed_score": np.nan,
        "fed_fallback": True,
    }


def run_static_batch_federated_experiment(
    method="independent_bo",
    n_clients=3,
    rounds=20,
    task_count=120,
    task_counts=None,
    seed=42,
    output_root=None,
    topology_profile="heterogeneous",
    node_counts=None,
    task_probs_by_client=None,
    client_config_path=None,
    context_features=None,
    objective_weights=None,
    normalization="batch_reference",
    objective_clip=10.0,
    batch_reuse_mode="new_each_round",
    batch_order="deadline",
    scheduler_type="Boltzmann",
    norm_mode="rolling",
    deterministic_scheduler=True,
    dispatch_gap=0.01,
    fed_share_mode="surrogate",
    fed_candidate_count=96,
    fed_beta=None,
):
    method = str(method or "independent_bo").lower()
    valid_methods = {"fixed", "local_bo", "independent_bo", "centralized_bo", "federated_bo"}
    if method not in valid_methods:
        raise ValueError(f"Unknown batch method={method}. Use one of: {', '.join(sorted(valid_methods))}")

    context_features = parse_static_batch_context_features(
        ",".join(context_features) if isinstance(context_features, (list, tuple)) else context_features
    )
    objective_weights = parse_static_batch_objective_weights(objective_weights) if not isinstance(objective_weights, dict) else dict(objective_weights)
    node_counts = parse_static_batch_int_list(node_counts) if isinstance(node_counts, str) else (list(node_counts) if node_counts else None)
    task_counts = parse_static_batch_int_list(task_counts) if isinstance(task_counts, str) else (list(task_counts) if task_counts else None)
    if isinstance(task_probs_by_client, str) or task_probs_by_client is None:
        task_probs_by_client = parse_static_batch_task_prob_clients(task_probs_by_client, n_clients, fallback=getattr(CFG, "TASK_TYPE_PROBS", None))

    clients = build_static_batch_clients(
        n_clients=n_clients,
        task_count=task_count,
        task_counts=task_counts,
        seed=seed,
        topology_profile=topology_profile,
        node_counts=node_counts,
        task_probs_by_client=task_probs_by_client,
        client_config_path=client_config_path,
    )
    if method == "local_bo":
        clients = clients[:1]

    out_dir = output_root or os.path.join("results", f"static_batch_{method}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    use_trust_region = method in {"independent_bo", "centralized_bo", "federated_bo", "local_bo"}
    agents = {
        int(c.client_id): _make_static_batch_agent(seed + 100 * int(c.client_id), context_features, use_trust_region=use_trust_region)
        for c in clients
    }
    central_agent = _make_static_batch_agent(seed + 9999, context_features, use_trust_region=True) if method == "centralized_bo" else None
    aggregator = FederatedAggregator()
    rng = random.Random(resolve_base_seed(int(seed), stream=7300))
    fixed_theta = default_control_vector(fill=1.5)

    rows = []
    client_rows = []
    for client in clients:
        type_probs = _normalize_task_probs(client.task_probs)
        client_rows.append({
            "client_id": int(client.client_id),
            "name": str(client.name),
            "task_count": int(client.task_count),
            "node_count": int(len(client.node_config)),
            "task_probs_rt": float(type_probs["RT"]),
            "task_probs_batch": float(type_probs["Batch"]),
            "task_probs_ai": float(type_probs["AI"]),
            "network_delay_scale": float(client.network_delay_scale),
            "bandwidth_scale": float(client.bandwidth_scale),
            "task_scale": float(client.task_scale),
            "workload_scale": float(client.workload_scale),
            "deadline_scale": float(client.deadline_scale),
            **{f"meta_{k}": v for k, v in (client.metadata or {}).items() if isinstance(v, (str, int, float, bool))},
        })

    for r in range(int(rounds)):
        round_rows = []
        for client in clients:
            tasks = generate_static_batch_tasks(
                client,
                batch_index=r,
                seed=seed,
                reuse_mode=batch_reuse_mode,
                order=batch_order,
            )
            context_dict = compute_static_batch_context(client, tasks, feature_names=context_features)
            context_vec = static_batch_context_vector(context_dict, context_features)
            cid = int(client.client_id)
            if method == "fixed":
                theta = list(fixed_theta)
                choose_debug = {}
            elif method == "centralized_bo":
                theta = central_agent.ask(context=context_vec if central_agent.use_context else None)
                choose_debug = dict(getattr(central_agent, "last_debug_info", {}) or {})
            elif method == "federated_bo":
                include = [agents[cid].prev_best]
                include.extend(agent.prev_best for agent in agents.values() if agent.prev_best is not None)
                candidate_pool = _sample_theta_candidates(rng, fed_candidate_count, include=include)
                theta, choose_debug = _choose_federated_theta(
                    cid,
                    agents,
                    context_vec,
                    candidate_pool,
                    aggregator,
                    fed_beta=fed_beta,
                )
            else:
                agent = agents[cid]
                theta = agent.ask(context=context_vec if agent.use_context else None)
                choose_debug = dict(getattr(agent, "last_debug_info", {}) or {})

            eval_seed = resolve_base_seed(int(seed), stream=8000 + int(r) * 101 + cid)
            metrics = simulate_static_batch_schedule(
                client,
                tasks,
                theta,
                eval_seed=eval_seed,
                scheduler_type=scheduler_type,
                norm_mode=norm_mode,
                deterministic_scheduler=deterministic_scheduler,
                dispatch_gap=dispatch_gap,
                max_time=None,
                objective_weights=objective_weights,
                normalization=normalization,
                objective_clip=objective_clip,
            )
            cost = float(metrics["cost"])

            if method == "centralized_bo":
                central_agent.tell(theta, cost, context=context_vec if central_agent.use_context else None)
            elif method != "fixed":
                agents[cid].tell(theta, cost, context=context_vec if agents[cid].use_context else None)

            row = {
                "method": method,
                "round": int(r),
                "client_id": cid,
                "client_name": str(client.name),
                "batch_reuse_mode": str(batch_reuse_mode),
                "batch_order": str(batch_order),
                "context_features": ",".join(context_features),
                "objective_weights": json.dumps(objective_weights, sort_keys=True),
                "normalization": str(normalization),
                "deterministic_scheduler": bool(deterministic_scheduler),
                "scheduler_type": str(scheduler_type),
                "scheduler_norm_mode": str(norm_mode),
                "dispatch_gap": float(dispatch_gap),
                "cost": cost,
                **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
                **_theta_to_prefixed_dict(theta, prefix="theta"),
                **{f"ctx_{k}": float(v) for k, v in context_dict.items()},
                "agent_training_sample_count": int(choose_debug.get("training_sample_count", np.nan)) if str(choose_debug.get("training_sample_count", "")).strip() not in {"", "nan"} else np.nan,
                "agent_candidate_count": int(choose_debug.get("candidate_count", np.nan)) if str(choose_debug.get("candidate_count", "")).strip() not in {"", "nan"} else np.nan,
                "fed_packet_count": choose_debug.get("fed_packet_count", np.nan),
                "fed_candidate_count": choose_debug.get("fed_candidate_count", np.nan),
                "fed_mu": choose_debug.get("fed_mu", np.nan),
                "fed_sigma": choose_debug.get("fed_sigma", np.nan),
                "fed_score": choose_debug.get("fed_score", np.nan),
                "fed_fallback": choose_debug.get("fed_fallback", np.nan),
            }
            rows.append(row)
            round_rows.append(row)

        if method == "federated_bo" and str(fed_share_mode).lower() in {"experience", "hybrid"}:
            _experience_share(round_rows, agents, context_features, topk=max(1, min(3, len(round_rows))))

    detail_path = os.path.join(out_dir, "batch_detail.csv")
    summary_path = os.path.join(out_dir, "batch_summary.csv")
    client_path = os.path.join(out_dir, "client_configs.csv")
    config_path = os.path.join(out_dir, "run_config.json")
    df = pd.DataFrame(rows)
    df.to_csv(detail_path, index=False)
    pd.DataFrame(client_rows).to_csv(client_path, index=False)
    if len(df):
        summary = (
            df.groupby(["method", "client_id", "client_name"], dropna=False)
            .agg(
                mean_cost=("cost", "mean"),
                best_cost=("cost", "min"),
                final_cost=("cost", "last"),
                mean_delay=("effective_avg_delay", "mean"),
                mean_energy=("avg_energy", "mean"),
                mean_violation=("effective_vio_rate", "mean"),
                mean_completion=("completion_ratio", "mean"),
                rounds=("round", "count"),
            )
            .reset_index()
        )
        overall = pd.DataFrame([{
            "method": method,
            "client_id": -1,
            "client_name": "ALL",
            "mean_cost": float(df["cost"].mean()),
            "best_cost": float(df["cost"].min()),
            "final_cost": float(df.sort_values(["round", "client_id"]).groupby("client_id")["cost"].last().mean()),
            "mean_delay": float(df["effective_avg_delay"].mean()),
            "mean_energy": float(df["avg_energy"].mean()),
            "mean_violation": float(df["effective_vio_rate"].mean()),
            "mean_completion": float(df["completion_ratio"].mean()),
            "rounds": int(len(df)),
        }])
        summary = pd.concat([summary, overall], ignore_index=True)
    else:
        summary = pd.DataFrame()
    summary.to_csv(summary_path, index=False)
    run_config = {
        "method": method,
        "n_clients": int(len(clients)),
        "rounds": int(rounds),
        "task_count": int(task_count),
        "task_counts": task_counts,
        "seed": int(seed),
        "topology_profile": str(topology_profile),
        "node_counts": node_counts,
        "context_features": context_features,
        "available_context_features": list(STATIC_BATCH_CONTEXT_FEATURES),
        "objective_weights": objective_weights,
        "normalization": str(normalization),
        "normalization_note": "batch_reference uses task/client references only; it does not use method outcomes.",
        "batch_reuse_mode": str(batch_reuse_mode),
        "batch_order": str(batch_order),
        "scheduler_type": str(scheduler_type),
        "scheduler_norm_mode": str(norm_mode),
        "deterministic_scheduler": bool(deterministic_scheduler),
        "dispatch_gap": float(dispatch_gap),
        "fed_share_mode": str(fed_share_mode),
        "fed_candidate_count": int(fed_candidate_count),
        "fed_beta": fed_beta,
        "outputs": {
            "detail": detail_path,
            "summary": summary_path,
            "clients": client_path,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"[StaticBatch] method={method} clients={len(clients)} rounds={rounds} output={out_dir}", flush=True)
    if len(summary):
        print(summary.tail(min(8, len(summary))).to_string(index=False), flush=True)
    return {
        "output_dir": out_dir,
        "detail_path": detail_path,
        "summary_path": summary_path,
        "client_path": client_path,
        "config_path": config_path,
        "rows": rows,
        "summary": summary,
    }
