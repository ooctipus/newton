# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test boundary-only checked capture without simulator, Warp, or GPU imports."""

import contextlib
import enum
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import checked_capture as checked


@contextlib.contextmanager
def fresh_modules(modules):
    """Hold only the given Newton and Lab modules, as a capture process does before selection.

    The import guard rejects every other loaded Newton or Lab module, and a test
    process that imported the real packages for other tests holds some.
    """
    with patch.dict(sys.modules, modules):
        for name in list(sys.modules):
            package = name.partition(".")[0]
            if (package in ("newton", "isaaclab") or package.startswith("isaaclab_")) and name not in modules:
                del sys.modules[name]
        yield


def source_modules(lab, newton):
    """Provide source-backed inert packages for capture integration tests."""
    modules = {}
    for name in checked.REQUIRED_PACKAGES:
        root = newton / name if name == "newton" else lab / "source" / name / name
        root.mkdir(parents=True, exist_ok=True)
        path = root / "__init__.py"
        path.write_text("# inert source package\n")
        modules[name] = SimpleNamespace(__file__=str(path), __path__=[str(root)])
    physics = lab / "source/isaaclab_newton/isaaclab_newton/physics.py"
    physics.write_text("# inert physics module\n")
    modules["isaaclab_newton.physics"] = SimpleNamespace(__file__=str(physics))
    return modules


class TestSelectedImports(unittest.TestCase):
    def setUp(self):
        """Build selected and shadow packages without importing a simulator."""
        temporary = tempfile.TemporaryDirectory(prefix="selected-imports-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.lab, self.newton = self.root / "lab", self.root / "newton"
        self.names = ("isaaclab", "isaaclab_newton", "isaaclab_tasks", "isaaclab_assets", "isaaclab_contrib")
        self.packages = {name: self.lab / "source" / name / name for name in self.names}
        self.packages["newton"] = self.newton / "newton"
        for name, directory in self.packages.items():
            directory.mkdir(parents=True)
            (directory / "__init__.py").write_text("selected = True\n")
            shadow = self.root / "shadow" / name
            shadow.mkdir(parents=True)
            (shadow / "__init__.py").write_text("selected = False\n")
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in list(sys.modules):
            if name.split(".")[0] in self.packages:
                del sys.modules[name]
        self.paths = patch.object(sys, "path", [str(self.root / "shadow"), *sys.path])
        self.paths.start()
        self.addCleanup(self.paths.stop)

    def test_selected_roots_override_shadow_and_record_actual_imports(self):
        """Select every Lab source package instead of the old editable environment."""
        selection = checked.select_imports(self.lab, self.newton)
        for name in self.packages:
            self.assertTrue(importlib.import_module(name).selected)
        provenance = checked.import_provenance(selection, required=self.packages)
        for name, directory in self.packages.items():
            self.assertEqual(provenance["loaded"][name]["file"], str(directory / "__init__.py"))
            self.assertEqual(provenance["selected"][name]["root"], str(directory))

    def test_preloaded_foreign_submodule_is_not_repaired_silently(self):
        """Reject a stale backend already imported before source selection."""
        sys.modules["isaaclab_newton.physics"] = SimpleNamespace(
            __file__=str(self.root / "shadow/isaaclab_newton/physics.py")
        )
        with self.assertRaisesRegex(RuntimeError, "isaaclab_newton.physics"):
            checked.select_imports(self.lab, self.newton)

    def test_unknown_lab_package_and_namespace_escape_fail_closed(self):
        """Reject unselected Lab modules and externally extended package paths."""
        with patch.dict(sys.modules, {"isaaclab_unknown": SimpleNamespace(__file__=__file__)}):
            with self.assertRaisesRegex(RuntimeError, "unselected"):
                checked.select_imports(self.lab, self.newton)
        selected = checked.select_imports(self.lab, self.newton)
        module = importlib.import_module("isaaclab_newton")
        module.__path__.append(str(self.root / "shadow/isaaclab_newton"))
        with self.assertRaisesRegex(RuntimeError, "isaaclab_newton"):
            checked.import_provenance(selected)

    def test_selected_source_drift_is_detected(self):
        """Reject source changes even when the module still reports the same path."""
        selected = checked.select_imports(self.lab, self.newton)
        importlib.import_module("isaaclab_newton")
        before = checked.import_provenance(selected)
        (self.packages["isaaclab_newton"] / "__init__.py").write_text("selected = 'changed'\n")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            checked.import_provenance(selected, previous=before)

    def test_source_owned_generated_namespace_does_not_trigger_lazy_imports(self):
        """Trace generated namespaces to their source owner without executing lazy loaders."""
        selected = checked.select_imports(self.lab, self.newton)
        owner = importlib.import_module("newton")
        generated = ModuleType("newton.experimental")
        generated.__getattr__ = Mock(side_effect=AssertionError("lazy import executed"))
        owner.experimental = generated
        sys.modules[generated.__name__] = generated
        info = checked.import_provenance(selected)
        record = info["loaded"][generated.__name__]
        self.assertEqual(record["generated_by"], "newton")
        self.assertEqual(record["file"], owner.__file__)
        generated.__getattr__.assert_not_called()
        del owner.experimental
        with self.assertRaisesRegex(RuntimeError, "newton.experimental"):
            checked.import_provenance(selected)


class FlagArray:
    """Provide the tiny host-read interface without a device runtime."""

    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)
        self.device = SimpleNamespace(is_capturing=False)

    def numpy(self):
        """Return the immutable fake host-copy wrapper."""
        return self

    def tolist(self):
        """Return a copy so test helpers cannot modify the source flags."""
        return self.values.copy()


def fpgs(flags=None):
    """Create a duck-typed current FPGS checker with observable method calls."""
    solver = type("SolverFeatherPGS", (), {})()
    flags = flags or dict.fromkeys(checked.FPGS_FLAGS, False)
    solver.constraint_capacity_status = Mock(return_value=flags)
    solver.check_constraint_capacity = Mock()
    solver.placement = SimpleNamespace(dense_max_constraints=704, mf_max_constraints=64)
    return solver


class Bits(enum.IntFlag):
    """Name representative official overflow bits while rejecting every bit."""

    NEFC = 1
    ITERATIONS = 512
    LS_ITERATIONS = 1024


def mjwarp(values):
    """Create a full-DOF nonsleeping MJWarp owner with sticky test flags."""
    solver = type("SolverMuJoCo", (), {})()
    solver.use_mujoco_cpu, solver.enable_sleeping = False, False
    solver.mjw_data = SimpleNamespace(
        overflow=FlagArray(values), nworld=len(values), njmax=300, naconmax=2048, njmax_nnz=600
    )
    solver.mjw_model = SimpleNamespace(
        opt=SimpleNamespace(warn_overflow=True, solver=2, integrator=3, cone=0, iterations=100, ls_iterations=50),
        is_sparse=False,
    )
    solver._use_mujoco_contacts = False
    solver.contact_capacity_status = Mock(
        return_value={"negative_count": False, "source_contacts": False, "mjwarp_contacts": False}
    )
    solver.check_contact_capacity = Mock()
    return solver


def collision_pipeline(flags=None):
    """Provide a current rigid-only narrow-phase checker with observable method calls."""
    narrow = SimpleNamespace(
        verify_buffers=True,
        hydroelastic_sdf=None,
        buffer_capacity_status=Mock(return_value=flags or dict.fromkeys(checked.NARROW_FLAGS, False)),
        check_buffer_capacity=Mock(),
    )
    return SimpleNamespace(
        model=SimpleNamespace(particle_count=0),
        narrow_phase=narrow,
        contact_reduction_config=SimpleNamespace(body_pairs=False),
        _body_pair_reducer=None,
        _contact_matcher=None,
        contact_matching="disabled",
        _contact_sorter=None,
        deterministic=False,
    )


def legacy_manager():
    """Build the audited legacy row flags and reset-each-call collision counters."""
    solver = fpgs()
    del solver.constraint_capacity_status, solver.check_constraint_capacity
    solver.placement.track_row_capacity = True
    solver.options = SimpleNamespace(warn_constraint_overflow=True, contact_compliance=False)
    solver.laws = SimpleNamespace(friction_patches=SimpleNamespace(capacity=8), friction_anchor_beta=0.2)
    solver.placement.row_overflow_warning_emitted = FlagArray([0, 0, 0])
    solver.placement.max_contacts_alloc = 8
    solver.contact_torsion_radius = 0.0
    pipeline = collision_pipeline()
    narrow = pipeline.narrow_phase
    del narrow.buffer_capacity_status, narrow.check_buffer_capacity
    pipeline.broad_phase_pair_count = FlagArray([2])
    pipeline.broad_phase_shape_pairs = SimpleNamespace(shape=(8,))
    narrow.gjk_candidate_pairs_count = FlagArray([2])
    narrow.gjk_candidate_pairs = SimpleNamespace(shape=(8,))
    narrow.split_gjk_mpr = False
    narrow.global_contact_reducer = None
    for name in (
        "shape_pairs_mesh",
        "triangle_pairs",
        "shape_pairs_mesh_plane",
        "shape_pairs_mesh_mesh",
        "shape_pairs_sdf_sdf",
    ):
        setattr(narrow, name, None)
        setattr(narrow, name + "_count", None)
    narrow.shape_pairs_sdf_sdf_count = FlagArray([0])  # Legacy always allocates this counter, not its buffer.
    contacts = SimpleNamespace(rigid_contact_count=FlagArray([3]), rigid_contact_max=8)
    return SimpleNamespace(_solver=solver, _collision_pipeline=pipeline, _contacts=contacts)


class TestLegacyCapacitySurvey(unittest.TestCase):
    def test_legacy_replication_routes_labels_into_existing_merge(self):
        """Preserve transforms, labels and world accounting through the baseline merger."""

        class Builder:
            def __init__(self):
                self.current_world, self.world_count, self.world_gravity = -1, 3, []
                self._merge_builder_copies = Mock()

            def replicate(self, builder, world_count, spacing=(0, 0, 0), *, xforms=None):
                return "original"

        original, report = Builder.replicate, {}
        checked.install_legacy_replication(Builder, report)
        try:
            builder = Builder()
            source = SimpleNamespace(_gravity_as_vector=lambda: (0, 0, -9.81))
            transforms, prefixes = ["transform0", "transform1"], ["env0", "env1"]
            builder.replicate(source, 2, xforms=transforms, label_prefixes=prefixes)
            builder._merge_builder_copies.assert_called_once_with(source, [3, 4], transforms, prefixes)
            self.assertEqual(builder.world_count, 5)
            self.assertEqual(builder.world_gravity, [(0, 0, -9.81)] * 2)
            self.assertEqual(builder.replicate(source, 2), "original")
            with self.assertRaisesRegex(ValueError, "label_prefixes"):
                builder.replicate(source, 2, xforms=transforms, label_prefixes=["wrong"])
        finally:
            Builder.replicate = original

    def test_storage_override_materializes_only_missing_collision_config(self):
        """Retain unrelated overrides and physics while materializing requested storage."""

        class Config:
            def __init__(self):
                self.broad_phase = "explicit"
                self.rigid_contact_max = None
                self.max_triangle_pairs = 1_000_000

            def to_pipeline_args(self):
                return vars(self).copy()

        class Collision:
            def __init__(self, *, broad_phase=None, rigid_contact_max=None, max_triangle_pairs=1_000_000):
                pass

        for existing in (None, Config()):
            physics = SimpleNamespace(collision_cfg=existing, solver_cfg=SimpleNamespace(solver_type="feather_pgs"))
            cfg = SimpleNamespace(sim=SimpleNamespace(physics=physics))
            parser = Mock(return_value=cfg)
            harness, report = SimpleNamespace(parse_env_cfg=parser), {}
            checked.install_collision_storage_overrides(harness, Collision, report, lambda: None)
            module = SimpleNamespace(NewtonCollisionPipelineCfg=Config)
            overrides = ["physics=feather_pgs", "env.sim.physics.collision_cfg.rigid_contact_max=72", "env.other=3"]
            with patch.dict(sys.modules, {"isaaclab_newton.physics.newton_collision_cfg": module}):
                result = harness.parse_env_cfg("Task", overrides=overrides)
            parser.assert_called_once_with("Task", overrides=["physics=feather_pgs", "env.other=3"])
            self.assertIs(result, cfg)
            self.assertEqual(physics.collision_cfg.rigid_contact_max, 72)
            self.assertEqual(physics.collision_cfg.max_triangle_pairs, 1_000_000)
            if existing is not None:
                self.assertIs(physics.collision_cfg, existing)
            self.assertEqual(len(overrides), 3)

    def test_opt_in_preserves_sticky_rows_and_discloses_collision_limits(self):
        """Read existing state without inventing whole-run collision coverage."""
        manager, entry = legacy_manager(), {}
        with patch.dict(os.environ, {"FPGS_BENCH_LEGACY_CAPACITY": "1"}):
            checked.check_solver("feather_pgs", manager._solver, entry)
            checked.check_collision("feather_pgs", manager, entry)
        self.assertEqual(set(entry["flags"]), {"dense", "matrix_free", "propagation"})
        self.assertEqual(manager._solver.placement.row_overflow_warning_emitted.values, [0, 0, 0])
        self.assertEqual(entry["collision"]["status"], "legacy_warning_screened_boundary_snapshots")
        self.assertFalse(entry["collision"]["whole_run_collision_capacity_verified"])
        self.assertTrue(entry["collision"]["log_warning_screen_required"])
        self.assertEqual(entry["collision"]["snapshots"]["contacts"], {"count": 3, "capacity": 8})
        self.assertEqual(entry["contact_law"]["friction_anchor_beta"], 0.2)

    def test_legacy_capacity_failure_is_not_clean(self):
        """Reject row history, boundary overflows, disabled warnings and patch mismatch."""
        for failure in ("rows", "disabled", "broad", "contacts", "patch"):
            manager, entry = legacy_manager(), {}
            if failure == "rows":
                manager._solver.placement.row_overflow_warning_emitted.values[1] = 1
            elif failure == "disabled":
                manager._collision_pipeline.narrow_phase.verify_buffers = False
            elif failure == "broad":
                manager._collision_pipeline.broad_phase_pair_count.values[0] = 9
            elif failure == "contacts":
                manager._contacts.rigid_contact_count.values[0] = 9
            else:
                manager._solver.laws.friction_patches.capacity = 4
            with self.subTest(failure=failure), patch.dict(os.environ, {"FPGS_BENCH_LEGACY_CAPACITY": "1"}):
                with self.assertRaises(RuntimeError):
                    checked.check_solver("feather_pgs", manager._solver, entry)
                    checked.check_collision("feather_pgs", manager, entry)

    def test_legacy_flag_does_not_replace_native_checkers(self):
        """Keep public native checks even when the opt-in variable is present."""
        manager, entry = SimpleNamespace(_solver=fpgs(), _collision_pipeline=collision_pipeline()), {}
        with patch.dict(os.environ, {"FPGS_BENCH_LEGACY_CAPACITY": "1"}):
            checked.check_solver("feather_pgs", manager._solver, entry)
            checked.check_collision("feather_pgs", manager, entry)
        manager._solver.check_constraint_capacity.assert_called_once()
        self.assertEqual(entry["collision"]["status"], "checked")

    def test_legacy_source_requires_exact_clean_pin(self):
        """Reject other revisions and dirty baseline sources before importing physics."""
        for head, status in (("wrong", ""), ("65ab4b7973fafa81f6e17615600d2aae708b94e6", " M runtime.py")):
            with (
                self.subTest(head=head, status=status),
                patch.object(checked.subprocess, "check_output", side_effect=[head, status]),
            ):
                with self.assertRaisesRegex(RuntimeError, "clean.*65ab"):
                    checked.validate_legacy_source(Path("/inert"))

    def test_boundary_memory_snapshot_is_not_a_peak_claim(self):
        """Read already initialized CUDA memory only at the existing metadata boundary."""
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_initialized=lambda: True, mem_get_info=Mock(return_value=(8, 16)))
        )
        harness = SimpleNamespace(_model_meta=lambda physics: {"state_finite": True})
        manager = SimpleNamespace(_solver=fpgs(), _collision_pipeline=collision_pipeline())
        report = {"boundaries": [], "boundary_count": 0}
        checked.install_boundary_check(harness, report, lambda: None, lambda: manager)
        with patch.dict(sys.modules, {"torch": torch}):
            harness._model_meta("feather_pgs")
        memory = report["boundaries"][0]["device_memory_snapshot"]
        self.assertEqual((memory["free_bytes"], memory["total_bytes"]), (8, 16))
        self.assertIn("not process allocation", memory["scope"])
        torch.cuda.mem_get_info.assert_called_once_with()


class Pipeline:
    """Exercise constructor signature preservation and original mode rejection."""

    def __init__(self, model=None, *, broad_phase_output_max=None, broad_phase="explicit"):
        if broad_phase != "explicit":
            raise ValueError("original constructor rejects unsupported mode")
        self.shape_pairs_filtered = list(range(100))
        self.shape_pairs_max = min(100, broad_phase_output_max or 100)
        self.broad_phase_mode = broad_phase
        self.narrow_phase = SimpleNamespace(
            split_gjk_mpr=False,
            sparse_gjk_pairs=True,
            max_candidate_pairs=self.shape_pairs_max,
            block_dim=64,
            total_num_threads=128,
        )


class TestCheckedCapture(unittest.TestCase):
    def test_heightfield_csr_capacity_extension_matches_actual_owner(self):
        """Require the known CSR flag only when its actual collision owner is active."""
        flags = dict.fromkeys(checked.NARROW_FLAGS, False)
        flags["heightfield_pair_csr_raw_or_membership"] = False
        pipeline = collision_pipeline(flags)
        manager = SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline)
        with self.assertRaisesRegex(RuntimeError, "Unrecognized"):
            checked.check_collision("feather_pgs", manager, {})
        pipeline.narrow_phase._heightfield_pair_csr = True
        entry = {}
        checked.check_collision("feather_pgs", manager, entry)
        self.assertTrue(entry["collision"]["check_pass"])
        flags["heightfield_pair_csr_raw_or_membership"] = True
        with self.assertRaisesRegex(RuntimeError, "nonzero"):
            checked.check_collision("feather_pgs", manager, {})

    def test_compat_install_model_admission_and_source_guard(self):
        """Install before execution and reject unsupported models or changed helper bytes."""
        for supported, drift in ((True, False), (False, False), (True, True)):
            with self.subTest(supported=supported, drift=drift), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lab, newton, output = root / "lab", root / "newton", root / "out"
                path, helper = lab / checked.HARNESS, root / "compat.py"
                path.parent.mkdir(parents=True)
                helper.write_text(
                    "installed=False\n"
                    "def install():\n"
                    "    global installed\n    installed=True\n"
                    "    return dict(installed_package_modified=False, gradient_tolerance_changed=False, iteration_budget_changed=False)\n"
                    "def supports_model(model): return model.supported\n"
                )
                original_hash = checked.sha(helper)
                path.write_text(
                    "import sys\nfrom pathlib import Path\n"
                    "def _model_meta(physics): return {'state_finite': True}\n"
                    "def main():\n"
                    "    assert sys.modules['_newton_checked_mjwarp_linesearch'].installed\n"
                    "    for _ in range(2): _model_meta('newton_mjwarp')\n"
                    "    Path(sys.argv[sys.argv.index('--output')+1]).write_text('{}')\n"
                    + (f"    Path({str(helper)!r}).write_text('# changed')\n" if drift else "")
                )
                solver = mjwarp([0])
                solver.mjw_model.supported = supported
                modules = source_modules(lab, newton)
                modules["mujoco_warp"] = SimpleNamespace(OverflowType=Bits)
                modules["isaaclab_newton.physics"].NewtonManager = SimpleNamespace(
                    _solver=solver, _collision_pipeline=collision_pipeline()
                )
                args = [
                    "--isaaclab",
                    str(lab),
                    "--newton",
                    str(newton),
                    "--expected-run-sha256",
                    checked.sha(path),
                    "--checks-output",
                    str(output / "checks.json"),
                    "--mjwarp-linesearch-fix",
                    "--",
                    "--physics",
                    "newton_mjwarp",
                    "--output",
                    str(output / "capture.json"),
                ]
                with fresh_modules(modules), patch.object(checked, "COMPAT_PATH", helper):
                    if supported and not drift:
                        self.assertEqual(checked.main(args), 0)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "actual model|helper changed"):
                            checked.main(args)
                report = json.loads((output / "checks.json").read_text())
                self.assertEqual(report["complete"], supported and not drift)
                self.assertIs(report["physics_work_modified"], True)
                self.assertIs(report["physics_budgets_modified"], False)
                self.assertEqual(report["mjwarp_linesearch_compat"]["helper_sha256"], original_hash)
                self.assertEqual(report["boundaries"][0]["mjwarp_linesearch_model_supported"], supported)

    def test_narrow_public_checker_and_no_clear(self):
        """Execute the real checker method after retaining its immutable flag snapshot."""
        pipeline, entry = collision_pipeline(), {}
        checked.check_collision("feather_pgs", SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline), entry)
        pipeline.narrow_phase.buffer_capacity_status.assert_called_once_with()
        pipeline.narrow_phase.check_buffer_capacity.assert_called_once_with()
        self.assertTrue(entry["collision"]["check_pass"])

    def test_narrow_failure_is_saved_by_boundary_before_raise(self):
        """Persist a genuine narrow checker failure even when the FPGS rows fit."""
        flags = dict.fromkeys(checked.NARROW_FLAGS, False)
        flags["broad_phase"] = True
        pipeline = collision_pipeline(flags)
        pipeline.narrow_phase.check_buffer_capacity.side_effect = RuntimeError("lost broad pairs")
        manager = SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline)
        report, saved = {"boundaries": [], "boundary_count": 0}, []
        harness = SimpleNamespace(_model_meta=lambda physics: {"state_finite": True})
        checked.install_boundary_check(
            harness, report, lambda: saved.append(json.loads(json.dumps(report))), lambda: manager
        )
        with self.assertRaisesRegex(RuntimeError, "lost broad pairs"):
            harness._model_meta("feather_pgs")
        self.assertTrue(saved[0]["boundaries"][0]["collision"]["flags"]["broad_phase"])
        self.assertFalse(saved[0]["boundaries"][0]["check_pass"])

    def test_narrow_missing_disabled_and_uncovered_owners_rejected(self):
        """Refuse unavailable history and exact pipeline owners outside this verifier's scope."""
        for failure in ("missing", "disabled", "particles", "hydro", "body_pairs", "matching", "sort"):
            pipeline = collision_pipeline()
            if failure == "missing":
                del pipeline.narrow_phase.check_buffer_capacity
            elif failure == "disabled":
                pipeline.narrow_phase.verify_buffers = False
            elif failure == "particles":
                pipeline.model.particle_count = 1
            elif failure == "hydro":
                pipeline.narrow_phase.hydroelastic_sdf = object()
            elif failure == "body_pairs":
                pipeline._body_pair_reducer = object()
            elif failure == "matching":
                pipeline.contact_matching = "latest"
            else:
                pipeline._contact_sorter = object()
            entry = {}
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                checked.check_collision(
                    "feather_pgs", SimpleNamespace(_solver=fpgs(), _collision_pipeline=pipeline), entry
                )
            self.assertFalse(entry["collision"]["check_pass"])

    def test_only_native_mj_collision_can_have_no_newton_pipeline(self):
        """Distinguish an internal MJ owner from an accidentally missing external pipeline."""
        solver, entry = mjwarp([0]), {}
        solver._use_mujoco_contacts = True
        solver.mjw_model.opt.run_collision_detection = True
        manager = SimpleNamespace(_solver=solver, _collision_pipeline=None)
        checked.check_collision("newton_mjwarp", manager, entry)
        self.assertEqual(entry["collision"]["status"], "not_applicable")
        solver._use_mujoco_contacts = False
        with self.assertRaises(RuntimeError):
            checked.check_collision("newton_mjwarp", manager, {})
        with self.assertRaises(RuntimeError):
            checked.check_collision("feather_pgs", SimpleNamespace(_solver=fpgs(), _collision_pipeline=None), {})

    def test_constructor_injection_preserves_signature_and_rejects_conflicts(self):
        """Keep original validation and full input ownership, recording only constructed output bounds."""
        original, report = Pipeline.__init__, {}
        saved = []
        try:
            self.assertIs(checked.install_broad_phase_limit(Pipeline, 32, report, lambda: saved.append(True)), original)
            self.assertEqual(inspect.signature(Pipeline.__init__), inspect.signature(original))
            result = Pipeline()
            self.assertEqual(result.shape_pairs_max, 32)
            self.assertEqual(len(result.shape_pairs_filtered), 100)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                Pipeline(broad_phase_output_max=64)
            with self.assertRaisesRegex(ValueError, "original constructor"):
                Pipeline(broad_phase="sap")
            self.assertEqual(len(saved), 3)
            records = report["collision_capacity"]["pipelines"]
            self.assertTrue(records[0]["construction_pass"])
            self.assertEqual(records[0]["narrow_phase"]["max_candidate_pairs"], 32)
            self.assertFalse(records[1]["construction_pass"])
            self.assertFalse(records[2]["construction_pass"])
        finally:
            Pipeline.__init__ = original

    def test_actual_harness_execution_preserves_argv_and_requires_two_boundaries(self):
        """Exercise loading, forwarding, output, and strict boundary completion without a simulator."""
        for boundary_count in (0, 1, 2, 3):
            with self.subTest(boundary_count=boundary_count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                lab, newton, output = root / "lab", root / "newton", root / "out"
                path = lab / checked.HARNESS
                path.parent.mkdir(parents=True)
                path.write_text(
                    "import json, sys, newton\nfrom pathlib import Path\n"
                    "def _model_meta(physics): return {'state_finite': True}\n"
                    "def main():\n"
                    "    newton.CollisionPipeline()\n"
                    f"    for _ in range({boundary_count}): _model_meta('feather_pgs')\n"
                    "    p = Path(sys.argv[sys.argv.index('--output') + 1])\n"
                    "    p.write_text(json.dumps({'argv': sys.argv[1:]}))\n"
                )
                solver = fpgs()
                modules = source_modules(lab, newton)
                modules["newton"].CollisionPipeline = Pipeline
                modules["isaaclab_newton.physics"].NewtonManager = SimpleNamespace(
                    _solver=solver, _collision_pipeline=collision_pipeline()
                )
                forwarded = ["--output", str(output / "capture.json"), "--solver-attr", "pgs_iterations=8"]
                argv = [
                    "--isaaclab",
                    str(lab),
                    "--newton",
                    str(newton),
                    "--expected-run-sha256",
                    checked.sha(path),
                    "--checks-output",
                    str(output / "checks.json"),
                    "--broad-phase-output-max",
                    "32",
                    "--",
                    *forwarded,
                ]
                original_argv = sys.argv
                original_constructor = Pipeline.__init__
                with fresh_modules(modules):
                    if boundary_count == 2:
                        self.assertEqual(checked.main(argv), 0)
                        self.assertEqual(json.loads((output / "capture.json").read_text())["argv"], forwarded)
                        solver.check_constraint_capacity.assert_has_calls([unittest.mock.call(), unittest.mock.call()])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "boundary|boundaries"):
                            checked.main(argv)
                self.assertIs(sys.argv, original_argv)
                self.assertIs(Pipeline.__init__, original_constructor)
                report = json.loads((output / "checks.json").read_text())
                self.assertEqual(report["complete"], boundary_count == 2)
                self.assertEqual(report["check_pass"], boundary_count == 2)
                self.assertEqual(report["boundary_count"], boundary_count)
                self.assertEqual(report["collision_capacity"]["pipelines"][0]["effective"], 32)

    def test_shell_forwards_profile_budget_and_uses_unchanged_analyzer(self):
        """Run the real shell against inert command stubs and inspect its exact argv."""
        for options in (
            ["--broad-phase-output-max", "32768", "--mjwarp-linesearch-fix"],
            ["--mjwarp-linesearch-fix", "--broad-phase-output-max", "32768"],
        ):
            with self.subTest(options=options):
                self._run_shell_forwarding(options)

    def _run_shell_forwarding(self, options):
        """Check either wrapper-flag ordering without invoking a GPU or simulator."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir, lab, output = root / "bin", root / "lab", root / "out"
            bin_dir.mkdir()
            lab.mkdir()
            record = root / "calls.jsonl"
            source = (
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['RECORD'], 'a') as f: f.write(json.dumps(sys.argv) + '\\n')\n"
                "print('RESULT inert-command-stub')\n"
            )
            for command in ("nsys", "uv"):
                path = bin_dir / command
                path.write_text(source)
                path.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "RECORD": str(record),
                "FPGS_BENCH_ISAACLAB": str(lab),
                "FPGS_BENCH_NEWTON": str(root / "newton"),
                "FPGS_BENCH_RUN_SHA256": "a" * 64,
                "OUT_DIR": str(output),
                "GPU": "test-uuid",
                "STEPS": "40",
                "PROFILE_STEPS": "17",
                "REPEATS": "3",
                "FPGS_NSYS_TRACE_MODE": "graph",
            }
            subprocess.run(
                [
                    "bash",
                    str(Path(checked.__file__).with_name("nsys_checked.sh")),
                    "capture",
                    "newton_mjwarp",
                    "Task",
                    *options,
                    "--solver-attr",
                    "pgs_iterations=8",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            calls = [json.loads(line) for line in record.read_text().splitlines()]
            self.assertEqual(len(calls), 3)
            profile, export, analyzer = calls
            self.assertEqual(profile[1:4], ["profile", "--capture-range=cudaProfilerApi", "--capture-range-end=stop"])
            self.assertIn("--cuda-graph-trace=graph:host-only", profile)
            self.assertEqual(profile[profile.index("--checks-output") + 1], str(output / "capture_checks.json"))
            self.assertEqual(profile[profile.index("--broad-phase-output-max") + 1], "32768")
            self.assertLess(profile.index("--broad-phase-output-max"), profile.index("--"))
            self.assertLess(profile.index("--mjwarp-linesearch-fix"), profile.index("--"))
            forwarded = profile[profile.index("--") + 1 :]
            self.assertEqual(
                forwarded,
                [
                    "--physics",
                    "newton_mjwarp",
                    "--task",
                    "Task",
                    "--device",
                    "cuda:0",
                    "--repeats",
                    "3",
                    "--steps",
                    "40",
                    "--profile-steps",
                    "17",
                    "--output",
                    str(output / "capture.json"),
                    "--solver-attr",
                    "pgs_iterations=8",
                ],
            )
            self.assertEqual(export[1:4], ["export", "--type", "sqlite"])
            self.assertEqual(analyzer[1:4], ["run", "python", str(lab / checked.HARNESS.parent / "analyze_nsys.py")])
            self.assertIn("--require-graph-trace", analyzer)

    def test_fpgs_public_checker_is_executed_and_never_cleared(self):
        """Require the actual public method even when the snapshot is clean."""
        solver, entry = fpgs(), {}
        checked.check_solver("feather_pgs", solver, entry)
        solver.constraint_capacity_status.assert_called_once_with()
        solver.check_constraint_capacity.assert_called_once_with()
        self.assertTrue(entry["check_pass"])

    def test_fpgs_actual_paths_are_recorded(self):
        """Record the actual solver paths without changing state."""
        solver, entry = fpgs(), {}
        solver.placement.grouped_response = True
        checked.check_solver("feather_pgs", solver, entry)
        self.assertNotIn("solver_recipe", entry)
        self.assertTrue(entry["solver_paths"]["grouped_response"])

    def test_fpgs_failure_records_flags_before_raising(self):
        """Retain overflow evidence and reject even a defective no-op checker."""
        flags = dict.fromkeys(checked.FPGS_FLAGS, False)
        flags["dense"] = True
        solver, entry = fpgs(flags), {}
        with self.assertRaises(RuntimeError):
            checked.check_solver("feather_pgs", solver, entry)
        self.assertTrue(entry["flags"]["dense"])
        solver.check_constraint_capacity.assert_called_once_with()

    def test_missing_legacy_api_is_not_clean(self):
        """Fail checked mode explicitly on a revision without sticky visibility."""
        solver = fpgs()
        del solver.check_constraint_capacity
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            checked.check_solver("feather_pgs", solver, {})

    def test_all_mj_bits_including_unknown_are_rejected(self):
        """Reject capacity, outer-iteration, line-search, and future unknown bits."""
        with patch.dict("sys.modules", {"mujoco_warp": SimpleNamespace(OverflowType=Bits)}):
            for bit in (1, 512, 1024, 1 << 20):
                entry = {}
                with self.subTest(bit=bit), self.assertRaises(RuntimeError):
                    checked.check_solver("newton_mjwarp", mjwarp([0, bit]), entry)
                self.assertEqual(entry["flags"]["mask"], bit)
                self.assertEqual(entry["flags"]["worlds_nonzero"], 1)
            entry = {}
            checked.check_solver("newton_mjwarp", mjwarp([0, 0]), entry)
            self.assertTrue(entry["check_pass"])

    def test_mj_actual_solver_options_are_recorded_without_mutation(self):
        """Record live MJWarp budgets and modes rather than assumed task defaults."""
        with patch.dict("sys.modules", {"mujoco_warp": SimpleNamespace(OverflowType=Bits)}):
            for native in (False, True):
                for sparse in (False, True):
                    solver, entry = mjwarp([0]), {}
                    solver._use_mujoco_contacts = native
                    solver.mjw_model.is_sparse = sparse
                    options = solver.mjw_model.opt
                    options.iterations, options.ls_iterations = 17, 7
                    before = vars(options).copy()
                    with self.subTest(native=native, sparse=sparse):
                        checked.check_solver("newton_mjwarp", solver, entry)
                        self.assertEqual(
                            entry["solver_options"],
                            {
                                "solver": 2,
                                "integrator": 3,
                                "cone": 0,
                                "iterations": 17,
                                "ls_iterations": 7,
                                "native_contacts": native,
                                "is_sparse": sparse,
                                "enable_sleeping": False,
                            },
                        )
                        self.assertEqual(vars(options), before)

    def test_mj_suppression_and_resettable_history_are_rejected(self):
        """Do not label disabled warnings or resettable sticky history clean."""
        for field in ("warnings", "sleeping"):
            solver = mjwarp([0])
            if field == "warnings":
                solver.mjw_model.opt.warn_overflow = False
            else:
                solver.enable_sleeping = True
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                checked.check_solver("newton_mjwarp", solver, {})

    def test_external_mj_prefix_loss_rejected_with_zero_native_bits(self):
        """Require the converter checker, including native-not-applicable records."""
        with patch.dict("sys.modules", {"mujoco_warp": SimpleNamespace(OverflowType=Bits)}):
            for flag in ("negative_count", "source_contacts", "mjwarp_contacts"):
                solver = mjwarp([0])
                solver.contact_capacity_status.return_value[flag] = True
                entry = {}
                with self.subTest(flag=flag), self.assertRaisesRegex(RuntimeError, "contact"):
                    checked.check_solver("newton_mjwarp", solver, entry)
                self.assertTrue(entry["external_contact_conversion"]["flags"][flag])
                solver.check_contact_capacity.assert_called_once_with()
            for missing in ("contact_capacity_status", "check_contact_capacity"):
                solver = mjwarp([0])
                delattr(solver, missing)
                with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, "lacks"):
                    checked.check_solver("newton_mjwarp", solver, {})
            for native in (False, True):
                solver, entry = mjwarp([0]), {}
                solver._use_mujoco_contacts = native
                checked.check_solver("newton_mjwarp", solver, entry)
                solver.check_contact_capacity.assert_called_once_with()
                self.assertEqual(
                    entry["external_contact_conversion"]["status"], "not_applicable" if native else "checked"
                )

    def test_boundary_wrapper_saves_failure_outside_swallowing_lab_helper(self):
        """Persist the second-boundary failure even when Lab metadata succeeds."""
        solver, report, saved = fpgs(), {"boundaries": [], "boundary_count": 0}, []
        harness = SimpleNamespace(_model_meta=lambda physics: {"state_finite": True})
        manager = SimpleNamespace(_solver=solver, _collision_pipeline=collision_pipeline())
        checked.install_boundary_check(
            harness, report, lambda: saved.append(json.loads(json.dumps(report))), lambda: manager
        )
        self.assertTrue(harness._model_meta("feather_pgs")["overflow_check"]["check_pass"])
        solver.check_constraint_capacity.side_effect = RuntimeError("capacity omitted constraints")
        with self.assertRaisesRegex(RuntimeError, "omitted"):
            harness._model_meta("feather_pgs")
        self.assertEqual(report["boundary_count"], 2)
        self.assertEqual(len(saved), 2)
        self.assertFalse(saved[-1]["boundaries"][1]["check_pass"])

    def test_source_pin_failure_writes_unaccepted_report(self):
        """Preserve a failure artifact before importing any simulation module."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lab = root / "lab"
            path = lab / checked.HARNESS
            path.parent.mkdir(parents=True)
            path.write_text("raise AssertionError('must not execute wrong bytes')\n")
            output = root / "out"
            args = [
                "--isaaclab",
                str(lab),
                "--newton",
                str(root / "newton"),
                "--expected-run-sha256",
                "0" * 64,
                "--checks-output",
                str(output / "checks.json"),
                "--",
                "--output",
                str(output / "capture.json"),
            ]
            with self.assertRaisesRegex(RuntimeError, "parent-pinned"):
                checked.main(args)
            report = json.loads((output / "checks.json").read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report["check_pass"])
            self.assertEqual(report["boundary_count"], 0)


if __name__ == "__main__":
    unittest.main()
