# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sphinx extension to document Warp functions and structured records.

This extension registers a custom *autodoc* documenter that recognises
`warp.types.Function` objects (created by the :pyfunc:`warp.func` decorator),
unwraps them to their original Python function (stored in the ``.func``
attribute) and then delegates all further processing to the standard
:class:`sphinx.ext.autodoc.FunctionDocumenter`.

Warp structs are documented from their Python ``.cls`` definition through the
ordinary ``autoclass`` directive. Field annotations and source docstrings remain
with that definition; no parallel field schema or modified Warp metadata is needed.
"""

from __future__ import annotations

import inspect
from typing import Any

from sphinx.ext.autodoc import AttributeDocumenter, ClassDocumenter, FunctionDocumenter

# Detect Warp wrappers without importing Warp, so loading this extension does
# not add runtime dependencies to Sphinx projects that do not document them.


class WarpFunctionDocumenter(FunctionDocumenter):
    """Autodoc documenter that unwraps :pyclass:`warp.types.Function`."""

    objtype = "warpfunc"
    directivetype = "function"
    # Ensure we run *before* the builtin FunctionDocumenter (higher priority)
    priority = FunctionDocumenter.priority + 10

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    @staticmethod
    def _looks_like_warp_function(obj: Any) -> bool:
        """Return *True* if *obj* appears to be a `warp.types.Function`."""
        cls = obj.__class__
        return getattr(cls, "__name__", "") == "Function" and hasattr(obj, "func")

    @classmethod
    def can_document_member(
        cls,
        member: Any,
        member_name: str,
        isattr: bool,
        parent,
    ) -> bool:
        """Return *True* when *member* is a Warp function we can handle."""
        return cls._looks_like_warp_function(member)

    # ------------------------------------------------------------------
    # Autodoc overrides - we proxy to the underlying Python function.
    # ------------------------------------------------------------------
    def _unwrap(self):
        """Return the original Python function or *self.object* as fallback."""
        orig = getattr(self.object, "func", None)
        if orig and inspect.isfunction(orig):
            return orig
        return self.object

    # Each of these hooks replaces *self.object* with the unwrapped function
    # *before* delegating to the base implementation.
    def format_args(self):
        self.object = self._unwrap()
        return super().format_args()

    def get_doc(self, *args: Any, **kwargs: Any) -> list[list[str]]:
        self.object = self._unwrap()
        return super().get_doc(*args, **kwargs)

    def add_directive_header(self, sig: str) -> None:
        self.object = self._unwrap()
        super().add_directive_header(sig)


class WarpClassDocumenter(ClassDocumenter):
    """Use a Warp struct's Python definition for ordinary ``autoclass`` processing."""

    priority = ClassDocumenter.priority + 10

    @staticmethod
    def _looks_like_warp_struct(obj: Any) -> bool:
        return type(obj).__name__ == "Struct" and inspect.isclass(getattr(obj, "cls", None))

    @classmethod
    def can_document_member(cls, member, member_name, isattr, parent):
        return cls._looks_like_warp_struct(member) or super().can_document_member(member, member_name, isattr, parent)

    def import_object(self, raiseerror: bool = False) -> bool:
        imported = super().import_object(raiseerror)
        if imported and self._looks_like_warp_struct(self.object):
            self.object = self.object.cls
            self.doc_as_attr = self.objpath[-1] != self.object.__name__
        return imported


class WarpAttributeDocumenter(AttributeDocumenter):
    """Find field comments on the Python definition without modifying Warp metadata."""

    def update_annotations(self, parent: Any) -> None:
        if WarpClassDocumenter._looks_like_warp_struct(parent):
            self.parent = parent.cls
        else:
            super().update_annotations(parent)


# ----------------------------------------------------------------------------
# Sphinx extension entry point
# ----------------------------------------------------------------------------


def setup(app):  # type: ignore[override]
    """Register Warp functions and class definitions with *app*."""

    app.add_autodocumenter(WarpFunctionDocumenter, override=True)
    app.add_autodocumenter(WarpClassDocumenter, override=True)
    app.add_autodocumenter(WarpAttributeDocumenter, override=True)
    # Declare the extension safe for parallel reading/writing
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
