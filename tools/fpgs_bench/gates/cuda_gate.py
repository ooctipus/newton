#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA gate for Newton: run a frozen list of newton/tests modules on one GPU and compare with known failures.

The selection is a list of runner-relative test modules (``sel_pro.txt`` and ``sel_gb300.txt`` beside this file, one
per GPU half; lines starting with ``base``/``head``/``tree`` are ignored). Listed modules missing from the tree are
reported and skipped. A test module that the gate should cover is added to a list by hand.

With ``--run`` it refuses to start while the GPU has a foreign compute process or a benchmark is alive (unless
``--shared-gpu`` / ``--overlap-timing``), runs exactly the listed modules through the newton.tests parallel runner
(``--no-cache-clear``, private Warp kernel cache, JUnit report) and compares the failing ``Class.test`` keys with a
known-failure list; ``--baseline-report`` (the same list run at the base) classifies a failure that also fails there
as pre-existing. Without ``--run`` it prints the selection; ``--report`` only compares an existing JUnit report.

    cuda_gate.py --repo TREE --base BASE --half pro --run --gpu UUID --shared-gpu --out-dir DIR

Exit status: 0 no new failure (or selection only), 1 new failures, 2 refused or bad arguments, 3 the runner produced
no usable report.
"""

import argparse
import datetime
import fnmatch
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.environ.get("FPGS_OUT", "/home/octi/Projects/IsaacLab.wt/output/fpgs-ws-20261015")
DEFAULT_GPU = os.environ.get("FPGS_PRO_UUID", "GPU-883586b6-3100-0610-81e5-3b4c26f45639")
DEFAULT_KNOWN = os.path.join(OUTPUT_ROOT, "tests-status", "known_failures.txt")
DEFAULT_CACHE = os.path.join(OUTPUT_ROOT, "cuda-gate-cache")
DEFAULT_RUNS = os.path.join(OUTPUT_ROOT, "cuda-gate-runs")
TESTS_DIR = "newton/tests"
# Benchmark processes that must not share the GPU with the gate: the suite driver and its
# workers, Nsight Systems, the nsys scene driver and the measurement queue scripts.
DEFAULT_BLOCK = (
    r"run_suite\.py",
    r"check_suite(?:\.py)?",
    r"nsys",
    r"nsys_scene\.py",
    r"measure_[\w.-]*\.sh",
    r"queue\d*\.sh",
)
# Variables that select solver routes; the gate runs the tests at their defaults.
STRIPPED_ENV_PREFIXES = ("FEATHER_PGS_", "IL_NEWTON_FPGS_", "NEWTON_")
# Asset locations, not route selectors: tests read them to find local copies, so they pass through.
KEPT_ENV = ("NEWTON_CACHE_PATH", "NEWTON_MENAGERIE_PATH", "NEWTON_TEST_ALLEGRO_USD")
# Above this many test modules the runner gets a longer --parallel-timeout.
LARGE_SELECTION = 150
LARGE_SELECTION_TIMEOUT = 7200


class GateError(RuntimeError):
    """A git or configuration problem that stops the gate."""


def git(repo, *args, data=None, binary=False):
    result = subprocess.run(["git", "-C", repo, *args], input=data, capture_output=True, check=False)
    if result.returncode != 0:
        raise GateError(f"git {' '.join(args)}: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout if binary else result.stdout.decode()


def git_lines(repo, *args):
    return {line for line in git(repo, *args).splitlines() if line}


def read_selection(path):
    """Runner-relative test modules listed in a selection file (after its base/head/tree lines)."""
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    return [line for line in lines if line and not line.startswith(("base ", "head ", "tree ", "#"))]


def discovered_tests(repo):
    """Runner-relative names of the test modules in HEAD's newton/tests (``test*.py`` in packages)."""
    paths = git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", TESTS_DIR).splitlines()
    packages = {os.path.dirname(path) for path in paths if path.endswith("/__init__.py")}
    found = set()
    for path in paths:
        directory, name = os.path.split(path)
        if not fnmatch.fnmatchcase(name, "test*.py"):
            continue
        walk = directory
        while walk != TESTS_DIR and walk in packages:
            walk = os.path.dirname(walk)
        if walk == TESTS_DIR:
            found.add(os.path.relpath(path, TESTS_DIR)[: -len(".py")].replace("/", "."))
    return found


class Selection:
    def __init__(self, repo, base, listed_from):
        self.base_sha = git(repo, "rev-parse", "--verify", f"{base}^{{commit}}").strip()
        self.head_sha = git(repo, "rev-parse", "--verify", "HEAD").strip()
        self.subject = git(repo, "log", "-1", "--format=%h %s", "HEAD").strip()
        self.tree = "head"
        self.listed_from = listed_from
        present = discovered_tests(repo)
        listed = list(dict.fromkeys(read_selection(listed_from)))
        self.total = len(present)
        self.missing = sorted(relative for relative in listed if relative not in present)
        self.selected = sorted(relative for relative in listed if relative in present)
        self.tests = {relative: relative for relative in self.selected}

    def patterns(self):
        """Runner ``-k`` patterns that select exactly the listed modules' test methods."""
        result = []
        for relative in self.selected:
            result += [f"{relative}.*", f"*.{relative}.*"]
        return result

    def report(self):
        print(f"HEAD:      {self.subject} ({self.head_sha[:12]})")
        print(f"base:      {self.base_sha[:12]}")
        print(f"listed:    {len(self.selected)} modules from {self.listed_from}")
        for relative in self.missing:
            print(f"  not in this tree: {relative}")
        print(f"selected:  {len(self.selected)} of {self.total} test modules")
        for relative in self.selected:
            print(f"  {relative}")


# ---------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------


# Owner, 2026-10-31: two pass/fail gates may share the GB300, and gpu_locks.sh admits them, so a busy card isn't a refusal.
SHARED_GPUS = {"GPU-ebfac9e8-02d5-d8a9-3bfc-bac64c62ffd4"}


def gpu_users(uuid):
    """PIDs of compute processes on the GPU, or None with a reason when it cannot be checked."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"nvidia-smi unavailable: {exc}"
    if result.returncode != 0:
        return None, f"nvidia-smi failed: {result.stderr.strip() or result.stdout.strip()}"
    pids = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] == uuid:
            pids.append(fields[1])
    return pids, None


def ancestors():
    """This process and its ancestors, so the scan never matches the gate's own command."""
    seen, pid = set(), os.getpid()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                pid = int(handle.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return seen


def blocking_processes(patterns):
    """(pid, command) of live processes with an argument matching a block pattern."""
    token = re.compile(r"(?:^|[\s/=])(?:" + "|".join(patterns) + r")(?=\s|$)")
    skip, found = ancestors(), []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in skip:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                argv = [part.decode(errors="replace") for part in handle.read().split(b"\0") if part]
        except OSError:
            continue
        if any(token.search(arg) for arg in argv):
            found.append((int(entry), " ".join(argv)[:200]))
    return sorted(found)


def junit_results(path):
    """(failing {key: message}, passing keys) from the runner's JUnit report."""
    failing, passing = {}, set()
    for case in ET.parse(path).getroot().iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        if classname == "_ErrorHolder":  # setUpClass/setUpModule errors: "setUpClass (module.Class)"
            match = re.match(r"(\w+) \((?:[\w.]*\.)?(\w+)\)", name)
            if match:
                classname, name = match.group(2), match.group(1)
        key = f"{classname}.{name}"
        problem = case.find("failure")
        problem = problem if problem is not None else case.find("error")
        if problem is not None:
            text = (problem.text or "").strip().splitlines()
            failing.setdefault(key, problem.get("message") or (text[-1] if text else ""))
        elif case.find("skipped") is None:
            passing.add(key)
    return failing, passing - set(failing)


def load_known(path):
    patterns = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            key = line.split("#", 1)[0].strip()
            if key:
                patterns.append(key)
    return patterns


def run(selection, args):
    if not selection.selected:
        print("nothing selected; no tests run")
        return 0
    users, problem = gpu_users(args.gpu)
    busy = blocking_processes(list(DEFAULT_BLOCK) + list(args.block_pattern))
    refused = False
    if users is None:
        print(f"REFUSED: cannot verify that {args.gpu} is idle ({problem})")
        refused = True
    elif users and not (args.shared_gpu or args.gpu in SHARED_GPUS):
        print(f"REFUSED: {args.gpu} has compute processes: pids {', '.join(users)}")
        refused = True
    elif users:
        print(f"note: sharing {args.gpu} with test processes (--shared-gpu): pids {', '.join(users)}")
    if busy and args.overlap_timing:
        # Owner, 2026-10-26: the tests may share the host with timing runs on another GPU, at the lowest CPU and I/O
        # priority (inherited by the runner and its workers); a little timing noise is accepted.
        print(f"note: overlapping {len(busy)} benchmark processes (--overlap-timing): running at nice 19, idle I/O")
        os.nice(19 - os.nice(0))
        subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], check=False)
    elif busy:
        print("REFUSED: benchmark processes are alive:")
        for pid, command in busy:
            print(f"  {pid}: {command}")
        refused = True
    if refused:
        return 2
    known = load_known(args.known_failures)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or os.path.join(DEFAULT_RUNS, f"{stamp}-{selection.head_sha[:8]}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    junit = os.path.join(out_dir, "junit.xml")
    log_path = os.path.join(out_dir, "runner.log")
    command = ["uv", "run", "--extra", "dev", "python", "-m", "newton.tests", "--no-cache-clear"]
    command += ["--junit-report-xml", junit]
    if args.jobs:
        command += ["--jobs", str(args.jobs)]
    if args.maxjobs:
        command += ["--maxjobs", str(args.maxjobs)]
    # The runner discards every result when its 3600 s default expires; a large selection gets room.
    timeout = args.parallel_timeout or (LARGE_SELECTION_TIMEOUT if len(selection.selected) > LARGE_SELECTION else None)
    if timeout:
        command += ["--parallel-timeout", str(timeout)]
    for pattern in selection.patterns():
        command += ["-k", pattern]
    env = dict(os.environ)
    stripped = (
        [] if args.keep_env else sorted(k for k in env if k.startswith(STRIPPED_ENV_PREFIXES) and k not in KEPT_ENV)
    )
    for key in stripped:
        del env[key]
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["WARP_CACHE_ROOT"] = env["WARP_CACHE_PATH"] = args.cache_dir
    with open(os.path.join(out_dir, "selection.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"base {selection.base_sha}\nhead {selection.head_sha}\ntree {selection.tree}\n")
        handle.writelines(f"{selection.tests[module]}\n" for module in selection.selected)
    if stripped:
        print("unset for the run: " + ", ".join(stripped))
    print(f"running {len(selection.selected)} test modules on {args.gpu}; log {log_path}")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        try:
            code = subprocess.run(
                command, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            ).returncode
        except OSError as exc:
            print(f"ERROR: cannot start the runner: {exc}")
            return 3
    with open(log_path, encoding="utf-8", errors="replace") as log:
        tail = [line.rstrip() for line in log if line.strip()][-3:]
    print(f"runner exit status {code}: " + " | ".join(tail))
    if not os.path.exists(junit):
        print(f"ERROR: no JUnit report at {junit}; see {log_path}")
        return 3
    return compare(junit, known, args.known_failures, args.baseline_report)


def compare(junit, known, known_path, baseline=None):
    """Print the failures of a JUnit report against the known list; 1 when any is new.

    With ``baseline``, a JUnit report of the same selection at the base, a failure outside the
    known list that also fails there is pre-existing, not new.
    """
    failing, passing = junit_results(junit)
    unknown = sorted(key for key in failing if not any(fnmatch.fnmatchcase(key, p) for p in known))
    still = sorted(key for key in failing if key not in unknown)
    base_failing, base_passing = junit_results(baseline) if baseline else ({}, set())
    preexisting = [key for key in unknown if key in base_failing]
    new = [key for key in unknown if key not in base_failing]
    fixed = sorted(
        p
        for p in known
        if any(fnmatch.fnmatchcase(key, p) for key in passing)
        and not any(fnmatch.fnmatchcase(key, p) for key in failing)
    )
    print(
        f"report {junit}: passing {len(passing)}, failing {len(failing)} "
        f"(known {len(still)}, pre-existing {len(preexisting)}, new {len(new)})"
    )
    if baseline:
        print(f"baseline {baseline}: passing {len(base_passing)}, failing {len(base_failing)}")
    for key in still:
        print(f"  known failure: {key}")
    for key in preexisting:
        print(f"  pre-existing at base: {key} (add it to {known_path})")
    for pattern in fixed:
        print(f"  known failure now passes: {pattern} (consider removing it from {known_path})")
    if new:
        print("NEW FAILURES:")
        for key in new:
            status = "" if not baseline else " [passes at base]" if key in base_passing else " [not run at base]"
            print(f"  {key}{status}: {' '.join(failing[key].split())[:160]}")
        return 1
    print("PASS: no new failures")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--repo", required=True, help="Newton checkout (a clean frozen tree)")
    parser.add_argument("--base", default="HEAD~1", help="base revision, recorded with the run (default: %(default)s)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="run the listed modules on the GPU")
    mode.add_argument("--report", help="only compare an existing JUnit report with the known failures (no run)")
    parser.add_argument("--half", choices=("pro", "gb300"), default="pro", help="the list beside this file to run")
    parser.add_argument("--modules-from", help="another selection file (replaces --half's list)")
    parser.add_argument("--known-failures", default=DEFAULT_KNOWN, help="known-failure keys (default: %(default)s)")
    parser.add_argument(
        "--baseline-report",
        help="JUnit report of the same selection at the base (--run or --report): its failures are pre-existing",
    )
    parser.add_argument(
        "--parallel-timeout",
        type=int,
        help=f"runner --parallel-timeout in seconds (default: {LARGE_SELECTION_TIMEOUT} above "
        f"{LARGE_SELECTION} modules, else the runner's)",
    )
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="GPU UUID for the run (default: %(default)s)")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, help="private Warp kernel cache (default: %(default)s)")
    parser.add_argument("--out-dir", help=f"run directory (default: {DEFAULT_RUNS}/<time>-<head>)")
    parser.add_argument("--jobs", type=int, help="runner --jobs")
    parser.add_argument("--maxjobs", type=int, help="runner --maxjobs (runner default 8)")
    parser.add_argument("--block-pattern", action="append", default=[], help="extra process regex that refuses a run")
    parser.add_argument(
        "--shared-gpu",
        action="store_true",
        help="run although other processes use the GPU (the caller holds a gpu_locks.sh test slot); "
        "benchmark processes still refuse the run",
    )
    parser.add_argument(
        "--overlap-timing",
        action="store_true",
        help="run although benchmark processes are alive (timing on another GPU), at nice 19 and idle I/O priority",
    )
    parser.add_argument(
        "--keep-env",
        action="store_true",
        help="keep FEATHER_PGS_*/IL_NEWTON_FPGS_*/NEWTON_* variables for the run (asset locations are always kept)",
    )
    args = parser.parse_args(argv)
    args.repo, args.cache_dir = os.path.abspath(args.repo), os.path.abspath(args.cache_dir)
    if args.out_dir:
        args.out_dir = os.path.abspath(args.out_dir)
    if (args.run or args.report) and not os.path.isfile(args.known_failures):
        parser.error(f"known-failure file not found: {args.known_failures}")
    listed = args.modules_from or os.path.join(HERE, f"sel_{args.half}.txt")
    for path in (args.baseline_report, listed):
        if path and not os.path.isfile(path):
            parser.error(f"file not found: {path}")
    if args.report:
        return compare(args.report, load_known(args.known_failures), args.known_failures, args.baseline_report)
    try:
        print("== CUDA gate selection")
        selection = Selection(args.repo, args.base, listed)
        selection.report()
    except GateError as exc:
        print(f"ERROR: {exc}")
        return 2
    if not args.run:
        return 0
    return run(selection, args)


if __name__ == "__main__":
    sys.exit(main())
