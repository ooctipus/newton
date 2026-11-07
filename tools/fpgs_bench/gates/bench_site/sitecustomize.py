# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# The timing hook run_suite_par.py puts first on its workers' PYTHONPATH.
"""Timing isolation for parallel suite workers: setups overlap everything, GPU-heavy phases never meet a timed loop.

check_suite's worker calls run_profiled.sample_random_actions once per environment step: `warmup` times, then once
per timed step, then for the reset check. With FPGS_GPU_TOKEN=<lock file> set in a perf run:

* When it enters Newton's ModelBuilder.finalize (FPGS_FREEZE=finalize, the default) a worker waits until no timed
  loop is running (a brief exclusive flock on the gate file <token>.gate) and marks itself GPU-heavy
  (<workers>/<pid>.gpu): finalize, solver construction, graph capture, warmup, timed loop and reset check. Before
  that it only imports and builds the scene on the CPU.
* At its first timed step it takes the token (exclusive flock), holds the gate (no new warmup starts), SIGSTOPs every
  GPU-heavy worker (FPGS_FREEZE=all stops every worker; =gpu marks workers only at warmup, which measured biased), waits
  FPGS_DRAIN_S (default 0.5) for their queued GPU work, and times alone; the first call after the timed loop resumes
  them and releases the gate and the token.
* The solver's construction-time lane timing (TreeKineticState._choose_lanes, a wall-clock race between kernel
  layouts) runs alone too: token, gate and every other worker stopped.

Workers register in FPGS_WORKERS_DIR; events go to FPGS_GPU_TOKEN_LOG (JSON lines). Without FPGS_GPU_TOKEN this file
does nothing.
"""

import os
import sys

_TOKEN = os.environ.get("FPGS_GPU_TOKEN")

if _TOKEN:
    import atexit
    import fcntl
    import importlib.abc
    import importlib.util
    import json
    import signal
    import time

    _WORKERS = os.environ.get("FPGS_WORKERS_DIR")
    _LOG = os.environ.get("FPGS_GPU_TOKEN_LOG")
    _DRAIN = float(os.environ.get("FPGS_DRAIN_S", "0.5"))
    _GATE = _TOKEN + ".gate"
    # What a timed loop stops. "all": every other worker. "gpu": workers past their first warmup step (measured
    # 2026-10-23: setups on the GPU bias timings by +1.1..1.7%, g1 +9.6%). "finalize" (default): workers that have
    # entered Newton's ModelBuilder.finalize; before it a worker only imports and builds the scene on the CPU.
    _FREEZE = os.environ.get("FPGS_FREEZE", "finalize")
    _state = {"calls": 0, "start": None, "end": None, "held": []}

    def _job():
        path = sys.argv[-1] if sys.argv and sys.argv[-1].endswith("job.json") else None
        if not path:
            return None
        job = json.load(open(path))
        if job.get("mode") != "perf":
            return None
        p = job["protocol"]
        return int(p["warmup"]), int(p["steps"]) * int(p["blocks"])

    _plan = _job()
    if _plan is not None:
        _state["start"] = _plan[0] + 1
        _state["end"] = _plan[0] + _plan[1] + 1

    def _log(**fields):
        if _LOG:
            with open(_LOG, "a") as f:
                f.write(json.dumps({"pid": os.getpid(), "job": sys.argv[-1], "t": time.time(), **fields}) + "\n")

    if _WORKERS:
        os.makedirs(_WORKERS, exist_ok=True)
        open(os.path.join(_WORKERS, str(os.getpid())), "w").close()

    def _others(gpu_only):
        if not _WORKERS:
            return []
        names = set(os.listdir(_WORKERS))
        pids = []
        for name in names:
            if name.isdigit() and int(name) != os.getpid() and (not gpu_only or f"{name}.gpu" in names):
                pids.append(int(name))
        return pids

    def _lock(path):
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def _unlock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def _isolate(gpu_only, label):
        """Token, gate, stop the other workers (all, or only the GPU-heavy ones), drain; returns the hold."""
        started = time.time()
        token = _lock(_TOKEN)
        gate = _lock(_GATE)
        stopped = []
        for pid in _others(gpu_only):
            try:
                os.killpg(pid, signal.SIGSTOP)
                stopped.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(_DRAIN)
        hold = {"token": token, "gate": gate, "stopped": stopped, "t0": time.time(), "label": label}
        _log(event=f"{label}_begin", wait_s=hold["t0"] - started - _DRAIN, stopped=len(stopped))
        return hold

    def _release(hold):
        for pid in hold["stopped"]:
            try:
                os.killpg(pid, signal.SIGCONT)
            except (ProcessLookupError, PermissionError):
                pass
        _log(event=f"{hold['label']}_end", held_s=time.time() - hold["t0"], resumed=len(hold["stopped"]))
        _unlock(hold["gate"])
        _unlock(hold["token"])

    def _enter_gpu_phase():
        # Wait out a running timed loop and mark this worker GPU-heavy while holding the gate, so a timer that takes
        # the gate next always sees the mark.
        gate = _lock(_GATE)
        if _WORKERS:
            open(os.path.join(_WORKERS, f"{os.getpid()}.gpu"), "w").close()
        _unlock(gate)

    def _cleanup():
        while _state["held"]:
            _release(_state["held"].pop())
        if _WORKERS:
            for name in (str(os.getpid()), f"{os.getpid()}.gpu"):
                try:
                    os.unlink(os.path.join(_WORKERS, name))
                except FileNotFoundError:
                    pass

    atexit.register(_cleanup)

    def _patch_actions(module):
        original = getattr(module, "sample_random_actions", None)
        if original is None or getattr(original, "_token_wrapped", False) or _state["start"] is None:
            return

        def wrapped(*args, **kwargs):
            _state["calls"] += 1
            if _state["calls"] == 1 and _FREEZE == "gpu":
                _enter_gpu_phase()
            if _state["calls"] == _state["start"]:
                _state["held"].append(_isolate(gpu_only=_FREEZE != "all", label="timed"))
            elif _state["calls"] == _state["end"] and _state["held"]:
                _release(_state["held"].pop())
            return original(*args, **kwargs)

        wrapped._token_wrapped = True
        module.sample_random_actions = wrapped

    def _patch_lanes(module):
        cls = getattr(module, "TreeKineticState", None)
        original = getattr(cls, "_choose_lanes", None) if cls is not None else None
        if original is None or getattr(original, "_token_wrapped", False):
            return

        def wrapped(self, *args, **kwargs):
            hold = _isolate(gpu_only=False, label="lanes")
            try:
                return original(self, *args, **kwargs)
            finally:
                _release(hold)

        wrapped._token_wrapped = True
        cls._choose_lanes = wrapped

    def _patch_builder(module):
        cls = getattr(module, "ModelBuilder", None)
        original = getattr(cls, "finalize", None) if cls is not None else None
        if original is None or getattr(original, "_token_wrapped", False) or _FREEZE != "finalize":
            return

        def wrapped(self, *args, **kwargs):
            if not _state.get("gpu_marked"):
                _enter_gpu_phase()
                _state["gpu_marked"] = True
            return original(self, *args, **kwargs)

        wrapped._token_wrapped = True
        cls.finalize = wrapped

    _HOOKS = {
        "run_profiled": _patch_actions,
        "newton._src.solvers.feather_pgs.tree_kinetic_owner": _patch_lanes,
        "newton._src.sim.builder": _patch_builder,
    }

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _HOOKS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def patched(mod, _exec=exec_module, _name=name):
                _exec(mod)
                _HOOKS[_name](mod)

            spec.loader.exec_module = patched
            return spec

    sys.meta_path.insert(0, _Finder())
