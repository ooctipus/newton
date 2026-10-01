# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import io
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from newton import worlds

try:
    from sphinx import addnodes
    from sphinx.application import Sphinx
    from sphinx.pycode import ModuleAnalyzer
except ModuleNotFoundError as exc:
    if exc.name != "sphinx":
        raise
    Sphinx = None

try:
    from docs import generate_api
except ModuleNotFoundError as exc:
    # The ``docs`` package lives at the repository root and is not included in
    # installed/wheel builds, so these tests only run from a source checkout.
    # Re-raise anything other than a missing top-level ``docs`` package so that
    # genuine import failures (e.g. a broken ``generate_api``) are not masked.
    if exc.name != "docs":
        raise
    generate_api = None


@unittest.skipUnless(generate_api is not None, "requires the docs/ package (source checkout only)")
class TestGenerateApiPublicSymbols(unittest.TestCase):
    def test_public_symbols_rejects_module_without_all(self):
        """Reject a module that does not declare its public API."""
        module = ModuleType("newton.missing_all")
        module.undeclared_symbol = object()

        with self.assertRaisesRegex(ValueError, r"newton\.missing_all must define __all__"):
            generate_api.public_symbols(module)

    def test_public_symbols_rejects_non_string_entry(self):
        """Reject a public API declaration containing a non-string entry."""
        module = ModuleType("newton.invalid_all_entry")
        module.__all__ = ["declared_symbol", 1]

        with self.assertRaisesRegex(
            ValueError,
            r"newton\.invalid_all_entry\.__all__ must contain only strings; got 1",
        ):
            generate_api.public_symbols(module)


@unittest.skipUnless(generate_api is not None, "requires the docs/ package (source checkout only)")
class TestGenerateApiCopyright(unittest.TestCase):
    def tearDown(self):
        generate_api._COPYRIGHT_LINES.clear()

    def test_copyright_line_preserves_existing_generated_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            api_page = output_dir / "newton_existing.rst"
            existing_line = ".. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers"
            api_page.write_text(
                "\n".join(
                    [
                        existing_line,
                        ".. SPDX-License-Identifier: CC-BY-4.0",
                        "",
                        "newton.existing",
                        "===============",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(generate_api, "OUTPUT_DIR", output_dir):
                generate_api._snapshot_copyright_lines()
            api_page.unlink()

            self.assertEqual(generate_api.copyright_line(api_page), existing_line)

    def test_copyright_line_uses_current_year_for_new_generated_file(self):
        class FakeDateTime:
            @classmethod
            def now(cls):
                return SimpleNamespace(year=2042)

        with tempfile.TemporaryDirectory() as tmp:
            api_page = Path(tmp) / "newton_new.rst"

            with mock.patch.object(generate_api, "datetime", FakeDateTime):
                self.assertEqual(
                    generate_api.copyright_line(api_page),
                    ".. SPDX-FileCopyrightText: Copyright (c) 2042 The Newton Developers",
                )


@unittest.skipUnless(generate_api is not None, "requires the docs/ package (source checkout only)")
class TestGenerateApiDeprecatedSymbols(unittest.TestCase):
    def test_deprecated_symbols_render_without_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with (
                mock.patch.object(generate_api, "OUTPUT_DIR", output_dir),
                mock.patch.object(generate_api, "REPO_ROOT", output_dir.parent),
            ):
                generate_api.write_module_page("newton.geometry", api_toctree_modules=set())

            page = (output_dir / "newton_geometry.rst").read_text(encoding="utf-8")
            self.assertIn("MATCH_BROKEN", page)
            self.assertIn("MATCH_NOT_FOUND", page)
            self.assertIn("Do not rely on this value", page)
            self.assertNotIn("``-1``", page)
            self.assertNotIn("``-2``", page)


@unittest.skipUnless(generate_api is not None, "requires the docs/ package (source checkout only)")
@unittest.skipUnless(Sphinx is not None, "requires the docs Sphinx dependency")
class TestWarpAutodoc(unittest.TestCase):
    def test_struct_fields_render_with_source_docs_and_types(self):
        """Build real autosummary templates; fields must retain their single source of documentation."""
        root = Path(__file__).resolve().parents[2]
        records = (
            "WorldCommands",
            "WorldResults",
            "WorldBatchResult",
            "WorldDirectoryData",
            "WorldTransaction",
            "WorldCompaction",
        )
        snapshots = {name: vars(getattr(worlds, name)).copy() for name in records}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "conf.py").write_text(
                f"import sys\nsys.path.insert(0, {str(root / 'docs/_ext')!r})\n"
                "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'autodoc_filter', 'autodoc_warp']\n"
                f"templates_path = [{str(root / 'docs/_templates')!r}]\n"
                "autodoc_inherit_docstrings = False\nautosummary_generate = True\nhtml_theme = 'alabaster'\n"
            )
            (source / "index.rst").write_text(
                "Warp API\n========\n\n.. autosummary::\n   :toctree: generated\n"
                "   :template: autosummary/class.rst\n\n"
                + "\n".join(f"   newton.worlds.{name}" for name in (*records, "FieldSpec", "WorldDirectory"))
                + "\n\n.. autosummary::\n   :toctree: generated\n\n   newton.worlds.world_location\n"
            )
            warnings = io.StringIO()
            app = Sphinx(
                str(source),
                str(source),
                str(Path(tmp) / "html"),
                str(Path(tmp) / "doctrees"),
                buildername="html",
                status=io.StringIO(),
                warning=warnings,
                freshenv=True,
                warningiserror=True,
            )
            app.build(force_all=True)
            self.assertEqual(app.statuscode, 0, warnings.getvalue())
            for name in records:
                record = getattr(worlds, name)
                page = f"generated/newton.worlds.{name}"
                descriptions = {
                    node[0]["ids"][0]: (node[0].astext(), node[1].astext())
                    for node in app.env.get_doctree(page).findall(addnodes.desc)
                    if node[0].get("ids")
                }
                self.assertEqual(
                    set(descriptions),
                    {f"newton.worlds.{name}", *(f"newton.worlds.{name}.{field}" for field in record.vars)},
                )
                source_docs = ModuleAnalyzer.for_module(record.cls.__module__).find_attr_docs()
                html = (Path(tmp) / "html" / f"{page}.html").read_text()
                for field in record.vars:
                    key = f"newton.worlds.{name}.{field}"
                    with self.subTest(field=key):
                        self.assertIn(f'id="{key}"', html)
                        signature, description = descriptions[key]
                        self.assertIn("wp.array", signature)
                        expected = " ".join(source_docs[(record.cls.__qualname__, field)])
                        self.assertEqual(" ".join(description.split()), " ".join(expected.split()))
                # Sphinx must not attach annotations or other documentation state to Warp's wrapper.
                self.assertEqual(vars(record).keys(), snapshots[name].keys())
                for key, value in snapshots[name].items():
                    self.assertIs(vars(record)[key], value)
            ordinary = app.env.get_doctree("generated/newton.worlds.WorldDirectory").astext()
            self.assertIn("begin(commands:", ordinary)
            self.assertIn("Record batch validation", ordinary)
            fields = app.env.get_doctree("generated/newton.worlds.FieldSpec").astext()
            self.assertIn("alignment_bytes", fields)
            function = app.env.get_doctree("generated/newton.worlds.world_location").astext()
            self.assertIn("identity", function)
            self.assertIn("Resolve a live handle", function)
            self.assertFalse((root / "docs/_ext/autodoc_wpfunc.py").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
