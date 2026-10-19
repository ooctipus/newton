# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resolve existing FPGS implementation options before topology admission."""

_KERNEL_CHOICES = {
    "cholesky_kernel": ("auto", "tiled", "loop"),
    "trisolve_kernel": ("auto", "tiled", "loop"),
    "hinv_jt_kernel": ("auto", "tiled", "par_row"),
    "delassus_kernel": ("auto", "tiled", "par_row_col"),
    "pgs_kernel": ("auto", "loop", "tiled_row", "tiled_contact", "streaming"),
    "sparse_mass_matrix": ("auto", "on", "off"),
    "cached_dynamics": ("auto", "on", "off"),
}
_CPU_KERNELS = {
    "cholesky_kernel": "loop",
    "trisolve_kernel": "loop",
    "hinv_jt_kernel": "par_row",
    "delassus_kernel": "par_row_col",
    "pgs_kernel": "loop",
    "sparse_mass_matrix": "off",
    "cached_dynamics": "off",
}
_DENSE_KERNELS = ("cholesky_kernel", "trisolve_kernel", "hinv_jt_kernel", "delassus_kernel")


def resolve_kernel_choices(
    requested: dict[str, str],
    overrides: dict[str, str | bool],
    *,
    is_cpu: bool,
    pgs_mode: str,
    enable_joint_limits: bool,
) -> tuple[dict[str, str], bool, dict[str, str], dict[str, str]]:
    """Return effective choices, explicit dense ownership, construction-time reasons and ignored requests.

    Public choices take precedence over private test hooks. A private hook is
    validated before CPU fallback, but does not acquire explicit dense ownership.
    Topology and resource admission remain the responsibility of execution planning.
    Matrix-free mode owns its sweep kernels, so a valid explicit ``pgs_kernel``
    there resolves like ``auto`` on any device; the last mapping holds the warning
    the caller emits for it.
    """
    pgs_kernel = requested["pgs_kernel"]
    matrix_free = pgs_mode == "matrix_free"
    if not matrix_free and enable_joint_limits and pgs_kernel in ("tiled_contact", "streaming"):
        raise ValueError(f"pgs_kernel={pgs_kernel!r} is contact-only and cannot solve joint-limit rows")
    ignored = {}
    for name, value in requested.items():
        if value not in _KERNEL_CHOICES[name]:
            raise ValueError(f"{name} must be one of {_KERNEL_CHOICES[name]}, got {value!r}")
        if matrix_free and name == "pgs_kernel" and value != "auto":
            ignored[name] = (
                f"pgs_kernel={value!r} is ignored in matrix_free mode, which owns its sweep kernels; "
                "omit it or use pgs_mode='split'."
            )
        elif is_cpu and value not in ("auto", _CPU_KERNELS[name]):
            raise ValueError(f"{name}={value!r} requires CUDA; use 'auto' or {_CPU_KERNELS[name]!r}")
    if matrix_free and requested["delassus_kernel"] != "auto":
        raise ValueError("delassus_kernel is unused in matrix_free mode; use 'auto' or pgs_mode='split'")
    explicit_dense = any(requested[name] != "auto" for name in _DENSE_KERNELS)
    for name in ("sparse_mass_matrix", "cached_dynamics"):
        if requested[name] == "on" and explicit_dense:
            raise ValueError(f"{name}='on' conflicts with explicit dense kernel choices; use 'auto'")

    resolved = dict(requested)
    reasons = {
        name: f"Explicit {name}={value!r}." if value != "auto" else "Automatic selection."
        for name, value in requested.items()
    }
    for name in _DENSE_KERNELS:
        if requested[name] == "auto" and name in overrides:
            value = overrides[name]
            if value not in _KERNEL_CHOICES[name]:
                raise ValueError(f"{name} must be one of {sorted(_KERNEL_CHOICES[name])}")
            resolved[name] = value
            reasons[name] = f"Private override selects {name}={value!r}."
    if pgs_kernel == "auto" or "pgs_kernel" in ignored:
        resolved["pgs_kernel"] = "loop" if is_cpu else "tiled_row"
        reasons["pgs_kernel"] = (
            f"Explicit pgs_kernel={pgs_kernel!r} is ignored in matrix_free mode."
            if "pgs_kernel" in ignored
            else f"Automatic {'CPU' if is_cpu else 'CUDA'} default: {resolved['pgs_kernel']}."
        )
    if is_cpu:
        for name in (*_DENSE_KERNELS, "pgs_kernel"):
            resolved[name] = _CPU_KERNELS[name]
            reasons[name] += f" CPU uses {_CPU_KERNELS[name]}."
    return resolved, explicit_dense, reasons, ignored
