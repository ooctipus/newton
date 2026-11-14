# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Best-effort disk cache of the collision tables a model derives from each mesh.

Finalizing a model derives from every collision mesh its cell index (:mod:`.mesh_cells`), its welded topology
(:mod:`.mesh_topology`), its open-boundary bits and its edge clusters (:mod:`.sdf_contact`) and whether it is watertight
(:mod:`.types`), and from every small hull its support direction cells (:mod:`.convex_cells_build`): about 13 s for a
1.2 million triangle rough terrain, in every process that builds the scene; importing an asset approximates collision
meshes by their convex hulls (:mod:`.convex_hull`). Each table is a pure function of the mesh's host buffers and a few
parameters, so it is stored under a content hash of those inputs, and a later build of the same mesh loads it instead.

Entries are uncompressed ``*.npz`` files of arrays (no pickles) in a directory named by the code that computes the
tables (the source of the modules above, NumPy's and Warp's versions), so a change to any of them starts a fresh
directory; directories unused for :data:`KEEP_DAYS` days are removed when a new one starts. A missing, unreadable or
partial entry is a miss, and a failed write is ignored. :attr:`newton.config.cache_dir_collision` is the location;
``None`` disables the cache.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

KEEP_DAYS = 7
"""Days a code version's directory survives without use."""

_SOURCES = (
    "geometry/convex_hull.py",
    "geometry/mesh_cells.py",
    "geometry/mesh_topology.py",
    "geometry/sdf_contact.py",
    "geometry/convex_cells_build.py",
    "geometry/types.py",
    "core/host_unique.py",
)


@functools.cache
def _code_version() -> str:
    import warp as wp  # noqa: PLC0415

    digest = hashlib.sha256(f"numpy {np.__version__} warp {wp.config.version}".encode())
    root = Path(__file__).resolve().parents[1]
    for name in (*_SOURCES, "geometry/_collision_cache.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:16]


def _directory() -> Path | None:
    from ... import config  # noqa: PLC0415

    location = config.cache_dir_collision
    return None if location is None else Path(location) / _code_version()


def _key(kind: str, inputs: tuple) -> str:
    digest = hashlib.sha256(kind.encode())
    for value in inputs:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(f"array {array.dtype.str} {array.shape}".encode())
            digest.update(array.data)
        else:
            digest.update(f"value {type(value).__name__} {value!r}".encode())
    return digest.hexdigest()


def _pack(result) -> dict[str, np.ndarray]:
    """``result`` (an array, a tuple of arrays, or a dict of arrays, NumPy scalars and JSON values) as npz arrays."""
    if isinstance(result, np.ndarray):
        return {"layout": np.array("array"), "a.value": result}
    if isinstance(result, tuple):
        return {"layout": np.array("tuple"), **{f"a.{i}": np.asarray(v) for i, v in enumerate(result)}}
    arrays, plain = {"layout": np.array("dict")}, {}
    for name, value in result.items():
        if isinstance(value, np.ndarray):
            arrays[f"a.{name}"] = value
        elif isinstance(value, np.generic):
            arrays[f"s.{name}"] = np.asarray(value)
        else:
            plain[name] = value
    arrays["json"] = np.array(json.dumps(plain, sort_keys=True))
    return arrays


def _unpack(data) -> Any:
    layout = str(data["layout"][()])
    if layout == "array":
        return data["a.value"]
    if layout == "tuple":
        count = sum(1 for name in data.files if name.startswith("a."))
        return tuple(data[f"a.{i}"] for i in range(count))
    result = json.loads(str(data["json"][()]))
    for name in data.files:
        if name.startswith("a."):
            result[name[2:]] = data[name]
        elif name.startswith("s."):
            result[name[2:]] = data[name][()]
    return result


def _prune(root: Path, keep: Path) -> None:
    """Remove the code versions' directories beside ``keep`` unused for :data:`KEEP_DAYS` days."""
    cutoff = time.time() - KEEP_DAYS * 86400
    for other in root.iterdir():
        with contextlib.suppress(OSError):
            if other != keep and other.is_dir() and other.stat().st_mtime < cutoff:
                shutil.rmtree(other, ignore_errors=True)


def cached(kind: str, inputs: tuple, compute: Callable[[], Any]) -> Any:
    """``compute()``, or its stored result for the same ``kind`` and ``inputs`` (arrays and plain values).

    Args:
        kind: The table's name; with ``inputs`` it keys the entry.
        inputs: Everything ``compute`` reads: host arrays (by dtype, shape and bytes) and plain values (by ``repr``).
        compute: Builds the table: an array, a tuple of arrays, or a dict of arrays, NumPy scalars and JSON values.

    Returns:
        The table, equal to what ``compute()`` returns.
    """
    directory = _directory()
    if directory is None:
        return compute()
    path = directory / f"{_key(kind, inputs)}.{kind}.npz"
    with contextlib.suppress(Exception), path.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
        result = _unpack(data)
        with contextlib.suppress(OSError):
            os.utime(directory)
        return result
    result = compute()
    with contextlib.suppress(Exception):
        fresh = not directory.exists()
        directory.mkdir(parents=True, exist_ok=True)
        if fresh:
            _prune(directory.parent, directory)
        # Written aside and renamed into place, so a reader sees a whole entry or none.
        stream = tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False)
        try:
            with stream:
                np.savez(stream, **_pack(result))
            os.replace(stream.name, path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(stream.name)
    return result
