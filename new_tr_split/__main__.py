#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry point for the split new_TR implementation."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from new_tr_split import runtime
else:
    from . import runtime

_cli_path = Path(__file__).resolve().parent / "cli.py"
_ns = runtime.__dict__
_ns["__name__"] = "__main__"
_ns["__file__"] = str(_cli_path)
exec(compile(_cli_path.read_text(encoding="utf-8"), str(_cli_path), "exec"), _ns)
