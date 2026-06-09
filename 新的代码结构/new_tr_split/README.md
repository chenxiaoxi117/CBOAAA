# new_tr_split

This folder contains the split version of `new_TR.py`. The original
monolithic script is kept unchanged at the project root.

The split keeps runtime behavior compatible with the original file. Several
late sections of `new_TR.py` patch classes and replace functions at import
time, so `runtime.py` loads the feature files in the same order as the original
script.

## File Layout

- `core_config.py`: imports, global helpers, `ExperimentConfig`, `CFG`, control
  vector helpers, topology and task-probability helpers.
- `simulation_entities.py`: `Event`, `Task`, `Node`, and workload generators.
- `schedulers.py`: Boltzmann, constrained Boltzmann, direct heuristic, and
  RoundRobin schedulers.
- `agents.py`: `FederatedBOAgent`, federated aggregation, state signatures,
  window snapshots, cohort records, and scenario monitor.
- `factory.py`: `ConnectedFactory` and the event-driven window simulation.
- `basic_outputs.py`: legacy plotting, log aggregation, metric summaries, and
  parameter-scan helpers.
- `sensitivity.py`: sensitivity-analysis evaluation and report helpers.
- `diagnostics.py`: log-to-table conversion, allocation diagnostics, key metric
  summaries, and diagnostic plots.
- `scenario_experiments.py`: method groups, scenario experiments, pressure
  scans, ratio-grid experiments, and batch runners.
- `runtime_patches.py`: deploy/history policy overrides, dual feedback,
  CBO/TR stability patches, and the final `run_scenario_group` override.
- `offline_export.py`: offline noise diagnostics and short-name result export.
- `cli.py`: original command-line argument block.
- `runtime.py`: compatibility loader that executes the split files in the
  original order and exposes the shared namespace.

## Run

From the project root:

```powershell
python -m new_tr_split --help
python -m new_tr_split --mode scenario
```

## Import

```python
from new_tr_split import CFG, ConnectedFactory, FederatedBOAgent
```

## Notes For Editing

Prefer editing the feature file that owns the code you are changing. For
example, scheduler scoring belongs in `schedulers.py`, simulation flow belongs
in `factory.py`, and CBO/TR runtime overrides belong in `runtime_patches.py`.

Because compatibility still depends on shared globals, avoid changing the load
order in `runtime.py` unless you are also untangling the corresponding runtime
patches.
