# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Declared ``FEATHER_PGS_*`` environment variables of SolverFeatherPGS: none, since the solver reads none.

``newton/tests/test_environment_flags.py`` keeps the declarations equal to the read sites. The solver records its
environment once at construction; a name nothing reads, a removed switch among them, is undeclared and recorded as
unknown, not rejected.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from ...core.environment_flags import declare, record_environment

PREFIXES = ("FEATHER_PGS_", "IL_NEWTON_FPGS_")

FLAGS = declare()


def record(environ: Mapping[str, str] | None = None) -> dict:
    """Record the solver's environment once at construction."""
    return record_environment(FLAGS, PREFIXES, os.environ if environ is None else environ)
