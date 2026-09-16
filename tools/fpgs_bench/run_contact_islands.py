# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse the fixed paired protocol with the explicit contact-sleep initializer."""

import importlib.util
from pathlib import Path


def main():
    """Keep the established timing, budgets, imports and source guards unchanged."""
    fixed_path = Path("/tmp/fpgs-kinetic-fixed-variants-TZRkIPYw/run.py")
    spec = importlib.util.spec_from_file_location("fixed_contact_island_variants", fixed_path)
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    fixed.TOOLS = Path(__file__).resolve().parent
    fixed.EXTRA.append(Path(__file__).resolve())
    return fixed.main()


if __name__ == "__main__":
    raise SystemExit(main())
