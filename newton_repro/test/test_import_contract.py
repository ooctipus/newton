# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-boundary checks for the standalone Newton repro modules."""

from __future__ import annotations

import ast
import pathlib


def test_only_capture_imports_isaaclab() -> None:
    """Only the ``capture/`` package may import Isaac Lab; everything else must be standalone.

    Scans:

    * Top-level ``scripts/newton_repro/*.py`` (modules)
    * ``scripts/newton_repro/envs/mdp/**.py`` (shared Newton-only MDP helpers)

    Per-bundle ``tasks/<name>/mdp.py`` files are NOT scanned -- those are
    user-authored task scripts that may import whatever helpers they want,
    including Isaac Lab if the user chooses (though it would be unusual).
    """
    repro_dir = pathlib.Path(__file__).resolve().parents[1]
    paths: list[pathlib.Path] = []
    paths.extend(repro_dir.glob("*.py"))
    paths.extend((repro_dir / "envs" / "mdp").rglob("*.py"))
    paths.extend((repro_dir / "envs" / "math").rglob("*.py"))

    offenders: list[str] = []
    for path in paths:
        rel = path.relative_to(repro_dir)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "isaaclab" or alias.name.startswith("isaaclab_"):
                        offenders.append(f"{rel}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "isaaclab" or module.startswith("isaaclab_"):
                    offenders.append(f"{rel}: from {module} import ...")
    assert offenders == []
