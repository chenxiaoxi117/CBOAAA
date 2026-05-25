#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split package for new_TR.py.

Importing this package loads the functional slices through runtime.py and
exports the same public names that the original monolithic script defined.
"""

from .runtime import *  # noqa: F401,F403
