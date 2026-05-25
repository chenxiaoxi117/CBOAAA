#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime loader for the split new_TR implementation.

The original file relies on top-level execution order and late monkey patches.
These functional slices are therefore executed in one shared namespace, in the
same order as new_TR.py, so behavior stays compatible while the code is easier
to browse and edit by area.
"""

from pathlib import Path

_PARTS = [
    "core_config.py",
    "simulation_entities.py",
    "schedulers.py",
    "agents.py",
    "factory.py",
    "basic_outputs.py",
    "sensitivity.py",
    "diagnostics.py",
    "scenario_experiments.py",
    "runtime_patches.py",
    "offline_export.py",
]

_BASE_DIR = Path(__file__).resolve().parent


def _exec_part(filename):
    path = _BASE_DIR / filename
    code = path.read_text(encoding="utf-8")
    old_file = globals().get("__file__")
    globals()["__file__"] = str(path)
    try:
        exec(compile(code, str(path), "exec"), globals())
    finally:
        if old_file is None:
            globals().pop("__file__", None)
        else:
            globals()["__file__"] = old_file


for _part in _PARTS:
    _exec_part(_part)

del _part

__all__ = [name for name in globals() if not name.startswith("_")]
