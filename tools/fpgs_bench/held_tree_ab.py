# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record every array the held-tree kernels write, and compare two records bitwise.

``record`` builds held-tree owners on small scenes: single trees on kinetic
and chain-scan schedules with fixed and free roots and ordinary or prescaled
repair motion, and per-size and forest groups with a free body, a short arm, a
diagonal-mass family, mimic rows, velocity limits and stage clocks. It drives
each through two steps of begin, predict, finish and a repeated begin; then
two steps whose caller copies the published bank back into the input bank, one
coordinate changed, so that the repair compares the two banks, first in the
one-pass repair and then classified on a one-block grid (classification,
comparison pass and listed repair); then an invalidated begin. A group of
alternate motion validates by identity alone and is never classified, so both
copy-back steps run its one-pass repair. After each call
it stores every array of the publication the kernels received and of the
owner's caches, with the state, response and spring arrays around them, and
the held-tree kernels the call launched, in launch order
(``<call>/launched``). Each scene also stores its groups' layouts
(``<scene>/layout``: bodies, lanes, trees per block, forest, chain scan, bank
comparison and alternate motion per group). ``compare`` lists the arrays that
differ bitwise, so both arms must also have launched the same kernels on the
same layouts; stage clocks (cycle counts) and the cached state-bank pointers
are not compared.

On CPU, owners build their kernels with static scratch and no release, as this
Warp's CPU JIT cannot resolve ``malloc`` and ``free``, and keep one tree per
block. Lane packing defaults to one tree per block, so construction times no
lane width; on CUDA ``--lane-pack`` fixes the width of every tree it covers
and skips the scenes with a wider tree. A group's trees must fill whole
blocks, or the width doubles until they do
(:func:`~newton._src.solvers.feather_pgs.tree_kinetic_plan.lanes_for`): a
group of one tree per world packs 4 lanes when the world count is a multiple
of 8, and 8 lanes when it is a multiple of 4. ``--worlds`` sets the mixed
scenes' world count (default 2); the single trees' fixtures replicate two
worlds in every checkout, so none runs narrower than 16 lanes. Each scene
prints its groups' widths. ``--scene`` records only the named scenes.
``--poison`` fills freed device memory with a value before each scene, so a
read of an array no kernel wrote shows in both arms' records.

Run with the import-time switches (``FEATHER_PGS_*``) unset::

    python tools/fpgs_bench/held_tree_ab.py record child.npz --device cuda:0
    python tools/fpgs_bench/held_tree_ab.py record parent.npz --root /path/to/parent --device cuda:0
    python tools/fpgs_bench/held_tree_ab.py record child-l4.npz --device cuda:0 --lane-pack 4 --worlds 8
    python tools/fpgs_bench/held_tree_ab.py compare parent.npz child.npz

``--root`` imports Newton and the scenes' fixtures from another checkout, as
in ``kernel_identity.py``. ``compare`` exits 1 when an array differs or a
record lacks an array the other holds.
"""

import argparse
import contextlib
import os
import sys
import zlib
from pathlib import Path

DT = 1.0 / 240.0
# The cache stamps hold the addresses of the bound state banks, which differ between processes.
_POINTERS = ("source_q", "source_qd")
_CLOCKS = ".stage_clocks"
# Augmented arrays the kernels write in part: zeroed so that both arms compare kernel writes only.
_PARTIAL = ("body_q_com", "joint_S_s", "body_v_s", "body_a_s", "body_f_s", "body_I_s", "joint_qdd", "joint_tau")
# Key prefix of every held-tree kernel: finishes, repairs, predictors, classifications and comparison passes.
_HELD_TREE = "tree_kinetic_"
# Mixed scenes: (_build options, owner options, one forest group per world).
MIXED_SCENES = {
    "arm-box": ({}, {}, False),
    # A two-joint arm: a fixed-root tree of three bodies and two coordinates, which packs four lanes.
    "short-arm-box": ({"links": 2}, {}, False),
    "arm-box-tray": ({"prescribed": True}, {}, False),
    "quadruped": ({"quadruped": True}, {}, False),
    "cabinet": ({"cabinet": True}, {}, False),
    "speed-limit": ({"speed_limit": 1.0}, {"motion": True}, False),
    "mimic": ({"mimic": True}, {}, False),
    "forest-arm-box": ({}, {}, True),
    "forest-quadruped": ({"quadruped": True}, {}, True),
    "forest-cabinet": ({"cabinet": True}, {}, True),
    "forest-speed-limit": ({"speed_limit": 1.0}, {"motion": True}, True),
    "clocked-arm-box": ({}, {"clocked": True}, False),
    "clocked-forest-arm-box": ({}, {"clocked": True}, True),
}


def _import_newton(root):
    """Import Newton, and the fixtures with it, from ``root`` (default: the checkout holding this script)."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    sys.path.insert(0, str(root))
    import newton  # noqa: PLC0415

    package = Path(newton.__file__).resolve().parent
    if package.parent != root:
        raise SystemExit(f"Imported Newton from {package}, not from {root}")
    return root


@contextlib.contextmanager
def _environment(**values):
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def single_scene_names():
    """Tags of the single-tree scenes, in recording order."""
    return [
        f"{'prescaled' if motion else 'single'}-floating{int(floating)}-chains{int(chains)}"
        for motion in (False, True)
        for floating in (False, True)
        for chains in (False, True)
    ]


def _describe(groups):
    """One line naming each group's bodies, lane width and kernel forms."""
    parts = []
    for group in groups:
        layout = group.layout
        forms = [
            name
            for name, present in (
                ("forest", layout.forest),
                ("chain scan", group.chain_scan),
                ("compares banks", group.bank_compare),
                ("alternate motion", group.motion_velocity),
            )
            if present
        ]
        parts.append(f"{layout.bodies} bodies on {layout.lanes} lanes" + (f" ({', '.join(forms)})" if forms else ""))
    return "; ".join(parts)


class Recorder:
    """Drive held-tree owners and keep the arrays their kernels can write after each call.

    Args:
        device: The Warp device the scenes are built on.
        poison: The value left in freed device memory before each scene (0: none).
        worlds: The mixed scenes' world count.
        scenes: The scenes to record, by single-tree tag or mixed scene name (``None``: every scene).
    """

    def __init__(self, device, poison=0.0, *, worlds=2, scenes=None):
        from newton._src.solvers.feather_pgs import tree_kinetic_state as native  # noqa: PLC0415

        self.device, self.poison, self.worlds = device, poison, worlds
        self.scenes = None if scenes is None else set(scenes)
        self.records, self.published, self.launched = {}, [], []
        self.native = native

    def selected(self, name):
        """Whether to record the scene ``name``."""
        return self.scenes is None or name in self.scenes

    @contextlib.contextmanager
    def hooks(self):
        """Keep the publications the owners bind and the held-tree kernels they launch; restore both after."""
        import warp as wp  # noqa: PLC0415

        from newton._src.solvers.feather_pgs import tree_kinetic_owner  # noqa: PLC0415

        owner = tree_kinetic_owner.TreeKineticState
        publication = owner._publication
        launches = {name: getattr(wp, name) for name in ("launch", "launch_tiled")}

        def capture(state_owner, *args):
            data = publication(state_owner, *args)
            self.published.append(data)
            return data

        def logged(launch):
            def launch_and_log(kernel, *args, **kwargs):
                if kernel.key.startswith(_HELD_TREE):
                    self.launched.append(kernel.key)
                return launch(kernel, *args, **kwargs)

            return launch_and_log

        owner._publication = capture
        for name, launch in launches.items():
            setattr(wp, name, logged(launch))
        try:
            yield self
        finally:
            owner._publication = publication
            for name, launch in launches.items():
                setattr(wp, name, launch)

    def _struct_arrays(self, instance, struct, prefix):
        import warp as wp  # noqa: PLC0415

        for name in struct.vars:
            value = getattr(instance, name)
            if isinstance(value, wp.array) and name not in _POINTERS:
                yield f"{prefix}.{name}", value

    def snapshot(self, tag, groups, extra):
        import numpy as np  # noqa: PLC0415

        from newton._src.solvers.feather_pgs.publication_data import PublicationData  # noqa: PLC0415

        arrays = dict(extra)
        for index, data in enumerate(self.published):
            arrays.update(self._struct_arrays(data, PublicationData, f"publication{index}"))
        for index, group in enumerate(groups):
            arrays.update(self._struct_arrays(group.data, self.native.KineticData, f"cache{index}"))
        for name, array in arrays.items():
            self.records[f"{tag}/{name}"] = array.numpy().copy()
        self.records[f"{tag}/launched"] = np.array(self.launched, dtype=str)
        self.published.clear()
        self.launched.clear()

    def poison_heap(self):
        """Leave freed blocks of the poison value for the next scene's arrays to reuse."""
        import warp as wp  # noqa: PLC0415

        if self.poison:
            sizes = (4, 8, 16, 32, 64, 128, 256, 1024, 4096)
            blocks = [wp.full(n, self.poison, dtype=float, device=self.device) for n in sizes for _ in range(64)]
            del blocks

    def drive(self, tag, model, solver, owner, groups, refresh, *, steps=2, motion=False):
        """Run ``steps`` begin, predict, finish and repeat calls, two copy-back steps, then an invalidated begin."""
        import numpy as np  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415

        print(f"{tag}: {_describe(groups)}", flush=True)
        self.records[f"{tag}/layout"] = np.array(
            [
                (
                    group.layout.bodies,
                    group.layout.lanes,
                    group.layout.pack,
                    group.layout.forest,
                    group.chain_scan,
                    group.bank_compare,
                    group.motion_velocity,
                )
                for group in groups
            ],
            dtype=np.int64,
        )
        # Launches at construction are not this scene's calls.
        self.launched.clear()
        rng = np.random.default_rng(zlib.crc32(tag.encode()))
        state, output, control = model.state(), model.state(), model.control()
        state.joint_qd.assign(rng.uniform(-0.4, 0.4, model.joint_dof_count).astype(np.float32))
        state.body_f.assign(rng.normal(0, 0.1, (model.body_count, 6)).astype(np.float32))
        control.joint_f.assign(rng.uniform(-0.1, 0.1, model.joint_dof_count).astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver._last_step_dt = DT
        augmented = solver._prepare_augmented_state(state, output, control)
        for name in _PARTIAL:
            getattr(augmented, name).zero_()
        for array in (solver.articulation_origin, getattr(solver, "_body_inertia_terms", None)):
            if isinstance(array, wp.array):
                array.zero_()
        for group in groups:
            for name, array in self._struct_arrays(group.data, self.native.KineticData, "cache"):
                if not name.endswith(("current_valid", "generation", "geometry_valid", "status", _CLOCKS)):
                    array.zero_()
        # An alternate repair motion reads a separate prescaled velocity bank.
        scaled = wp.array(state.joint_qd.numpy() * 0.5, dtype=float, device=self.device) if motion else None

        def extra():
            arrays = {
                "state.body_q": state.body_q,
                "state.body_qd": state.body_qd,
                "output.joint_q": output.joint_q,
                "output.joint_qd": output.joint_qd,
                "output.body_q": output.body_q,
                "output.body_qd": output.body_qd,
                "v_hat": solver.v_hat,
                "joint_qdd": augmented.joint_qdd,
                "joint_tau": augmented.joint_tau,
                "stiffness": solver._passive_spring_stiffness,
                "spring_ref": solver._passive_spring_ref,
                "damping": solver._passive_joint_damping,
                "kinematic_joint": solver._kinematic_joint_mask,
            }
            if getattr(solver, "_sparse_Linv", None) is not None:
                arrays["sparse_Linv"] = solver._sparse_Linv
            return arrays

        for step in range(steps):
            velocity = scaled if motion else state.joint_qd
            owner.begin(state, augmented, velocity, DT, step == 0)
            self.snapshot(f"{tag}/{step}/begin", groups, extra())
            solver.mass_update_mask.fill_(1)
            refresh(augmented)
            owner.predict(state, augmented, control, velocity, DT)
            self.snapshot(f"{tag}/{step}/predict", groups, extra())
            solver.v_out.assign(solver.v_hat)
            owner.finish(state, augmented, output, DT, step % 2 == 0)
            self.snapshot(f"{tag}/{step}/finish", groups, extra())
            # The next step starts from the published state; a repeated begin keeps the cache.
            owner.begin(output, augmented, scaled if motion else output.joint_qd, DT, False)
            self.snapshot(f"{tag}/{step}/repeat", groups, extra())
            state, output = output, state
            state.body_f.assign(rng.normal(0, 0.1, (model.body_count, 6)).astype(np.float32))
        for form in ("copied", "copied-classified"):
            if form == "copied-classified":
                # Classify every repair launch on a one-block grid, whatever its size and device (owners keep
                # the one-pass repair of a motion-velocity group).
                for group in groups:
                    group._classify_switch, group._repair_grid, group._classify_above = True, 1, 0
                    group._bind_kernels()
            velocity = scaled if motion else state.joint_qd
            owner.begin(state, augmented, velocity, DT, False)
            solver.mass_update_mask.fill_(1)
            refresh(augmented)
            owner.predict(state, augmented, control, velocity, DT)
            solver.v_out.assign(solver.v_hat)
            owner.finish(state, augmented, output, DT, False)
            # The caller copies the published bank back into the input bank and changes one coordinate: the
            # stamps name the output bank, so the repair compares the banks (identity alone for prescaled motion).
            state.assign(output)
            coordinates = state.joint_q.numpy()
            coordinates[0] += np.float32(0.01)
            state.joint_q.assign(coordinates)
            owner.begin(state, augmented, scaled if motion else state.joint_qd, DT, False, state_out=output)
            self.snapshot(f"{tag}/{form}/begin", groups, extra())
        world_mask = wp.array([True] + [False] * (solver.world_count - 1), dtype=wp.bool, device=self.device)
        owner.invalidate(world_mask)
        owner.begin(state, augmented, scaled if motion else state.joint_qd, DT, False)
        self.snapshot(f"{tag}/reset/begin", groups, extra())

    def single_scenes(self):
        """Sparse single trees: kinetic and chain schedules, fixed and free roots, ordinary and prescaled motion."""
        from tools.fpgs_bench import test_tree_kinetic_state as single  # noqa: PLC0415
        from tools.fpgs_bench import test_tree_prescaled_state as prescaled  # noqa: PLC0415

        names = iter(single_scene_names())
        for module, motion in ((single, False), (prescaled, True)):
            for floating in (False, True):
                for chains in (False, True):
                    tag = next(names)
                    if not self.selected(tag):
                        continue
                    self.poison_heap()
                    with _environment(FEATHER_PGS_TREE_CHAIN_SCAN=int(chains)):
                        try:
                            model, solver, _, operator, owner, _ = module.fixture(floating, self.device)
                        except ValueError as error:
                            print(f"skipped {tag}: {error}", flush=True)
                            continue
                        self.drive(tag, model, solver, owner, [owner], operator.refresh, motion=motion)

    def mixed_scenes(self):
        """Held factor groups: per-size trees and forests over the mixed free-body scenes."""
        import warp as wp  # noqa: PLC0415

        from newton._src.solvers.feather_pgs import mixed_tree_kinetic  # noqa: PLC0415
        from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS  # noqa: PLC0415
        from newton.tests.test_feather_pgs_mixed_tree_free_body import _build  # noqa: PLC0415

        for tag, (options, owner_options, forest) in MIXED_SCENES.items():
            if not self.selected(tag):
                continue
            self.poison_heap()
            model = _build(wp.get_device(self.device), worlds=self.worlds, **options)
            solver = SolverFeatherPGS(model, pgs_mode="split", friction_anchor_beta=0.0)
            # Velocity limits are solved by matrix-free sweeps; the owner's motion contract reads only the flag.
            solver.enable_joint_velocity_limits = owner_options.get("motion", False)
            supported = mixed_tree_kinetic.supported
            if forest:
                # Admission is CUDA-only; both arms lift it to build the forest group on any device.
                mixed_tree_kinetic.supported = lambda solver: True
            switches = {
                "FEATHER_PGS_TREE_FOREST": int(forest),
                "FEATHER_PGS_STAGE_CLOCKS": int(owner_options.get("clocked", False)),
            }
            try:
                with _environment(**switches):
                    owner = mixed_tree_kinetic.MixedTreeKineticState(solver)
            except ValueError as error:
                print(f"skipped {tag}: {error}", flush=True)
                continue
            finally:
                mixed_tree_kinetic.supported = supported
            if forest and not any(group.layout.forest for group in owner.groups):
                print(f"skipped {tag}: no forest group", flush=True)
                continue
            motion = any(group.motion_velocity for group in owner.groups)
            self.drive(f"mixed-{tag}", model, solver, owner, owner.groups, owner.refresh, motion=motion)


def record(path, *, root=None, device="cpu", lane_pack=0, poison=0.0, worlds=2, scenes=None):
    """Record the scenes' arrays to ``path`` (see the module docstring) and return them.

    Args:
        path: The ``.npz`` file to write.
        root: The Newton checkout to import (default: the one holding this script).
        device: The Warp device.
        lane_pack: ``FEATHER_PGS_TREE_LANE_PACK``: 0 for one tree per block, or a fixed lane width.
        poison: The value left in freed device memory before each scene (0: none).
        worlds: The mixed scenes' world count.
        scenes: The scenes to record, by single-tree tag or mixed scene name (default: every scene).
    """
    switches = sorted(name for name in os.environ if name.startswith("FEATHER_PGS_"))
    if switches:
        raise SystemExit(f"Unset the import-time switches: {', '.join(switches)}")
    if worlds < 1:
        raise SystemExit("--worlds must be positive")
    unknown = sorted(set(scenes or ()) - set(single_scene_names()) - set(MIXED_SCENES))
    if unknown:
        raise SystemExit(f"Unknown scenes: {', '.join(unknown)}")
    # One tree per block, or one fixed width: construction times no lane width. The
    # scenes build their own owners; the solvers build none.
    with _environment(FEATHER_PGS_TREE_LANE_PACK=lane_pack, FEATHER_PGS_TREE_KINETIC_STATE=0):
        _import_newton(root)
        import numpy as np  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        log_level, wp.config.log_level = wp.config.log_level, wp.LOG_WARNING
        recorder = Recorder(device, poison, worlds=worlds, scenes=scenes)
        try:
            with recorder.hooks():
                recorder.single_scenes()
                recorder.mixed_scenes()
        finally:
            wp.config.log_level = log_level
    np.savez_compressed(path, **recorder.records)
    print(f"Recorded {len(recorder.records)} arrays: {path}")
    return recorder.records


def compare(parent, child):
    """Return ``(differing, only_parent, only_child)`` array names of two records, stage clocks skipped."""
    shared = sorted(name for name in parent.keys() & child.keys() if not name.endswith(_CLOCKS))
    differing = [
        name
        for name in shared
        if parent[name].dtype != child[name].dtype
        or parent[name].shape != child[name].shape
        or parent[name].tobytes() != child[name].tobytes()
    ]
    return differing, sorted(parent.keys() - child.keys()), sorted(child.keys() - parent.keys())


def report(parent, child):
    """Print the comparison; return 1 when an array differs or only one record holds it."""
    import numpy as np  # noqa: PLC0415

    differing, only_parent, only_child = compare(parent, child)
    shared = len(parent.keys() & child.keys())
    print(f"{shared} shared arrays: {len(differing)} differ bitwise")
    for name in differing:
        a, b = parent[name], child[name]
        if a.shape == b.shape and a.dtype.kind == "f" and b.dtype.kind == "f":
            gap = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print(f"  {name}: max abs {np.nanmax(gap) if np.isfinite(gap).any() else float('nan'):.3e}")
        elif a.dtype.kind == "U" and b.dtype.kind == "U":
            # Launched kernels: name the launches one arm made and the other did not.
            print(
                f"  {name}: {sorted(set(a.tolist()) - set(b.tolist()))} against {sorted(set(b.tolist()) - set(a.tolist()))}"
            )
        else:
            print(f"  {name}: {a.dtype}{a.shape} against {b.dtype}{b.shape}")
    for label, names in (("only in parent", only_parent), ("only in child", only_child)):
        print(f"{len(names)} {label}")
        for name in names:
            print(f"  {label}: {name}")
    return int(bool(differing or only_parent or only_child))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="record the arrays of one checkout")
    record_parser.add_argument("output", help=".npz file to write")
    record_parser.add_argument("--root", help="Newton checkout to import (default: the one holding this script)")
    record_parser.add_argument("--device", default="cpu", help="Warp device (default: cpu)")
    record_parser.add_argument(
        "--lane-pack", type=int, default=0, choices=(0, 4, 8, 16, 32), help="FEATHER_PGS_TREE_LANE_PACK (default 0)"
    )
    record_parser.add_argument("--poison", type=float, default=0.0, help="value left in freed memory before each scene")
    record_parser.add_argument("--worlds", type=int, default=2, help="world count of the mixed scenes (default 2)")
    record_parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        metavar="NAME",
        help="record only this scene (a single-tree tag or a mixed scene name); repeatable",
    )
    compare_parser = commands.add_parser("compare", help="compare two records bitwise")
    compare_parser.add_argument("parent")
    compare_parser.add_argument("child")
    arguments = parser.parse_args(argv)
    if arguments.command == "record":
        record(
            arguments.output,
            root=arguments.root,
            device=arguments.device,
            lane_pack=arguments.lane_pack,
            poison=arguments.poison,
            worlds=arguments.worlds,
            scenes=arguments.scenes,
        )
        return 0
    import numpy as np  # noqa: PLC0415

    with np.load(arguments.parent) as parent, np.load(arguments.child) as child:
        return report(dict(parent), dict(child))


if __name__ == "__main__":
    sys.exit(main())
