# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental paired16 workspace lifetime replacement; unchanged state law.

Sixteen body slots hold pose7/velocity6/acceleration6 during the scans. Once
each body has loaded its own values, its slot becomes wrench6/moments13.
The three padded slots later hold the 54 inertia-axis actions. Their separate
location avoids overwriting another lane's still-live subtree terms.
"""

import functools
import hashlib
import inspect
import linecache
import textwrap

import warp as wp

from . import franka_kinetic_state as original

KineticPlan = original.KineticPlan
KineticData = original.KineticData
PublicationData = original.PublicationData

FLOATS_PER_WORLD = 376
BODY_STRIDE = 19
ACTION_OFFSET = 247
AXIS_OFFSET = 304
ORIGIN_OFFSET = 358
SHIFT_OFFSET = 367


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    __shared__ float values[2*376];
    return reinterpret_cast<uint64_t>(values+376*((threadIdx.x&31)>>4));
#else
    return reinterpret_cast<uint64_t>(malloc(376*sizeof(float)));
#endif
""")
def _storage() -> wp.uint64: ...


@wp.func_native("return 16;")
def _scan_width() -> int: ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+19*lane; for(int k=0;k<3;++k)s[k]=pose.p[k]; for(int k=0;k<4;++k)s[3+k]=pose.q[k];"
)
def _store_pose(address: wp.uint64, lane: int, pose: wp.transform): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+19*lane; return wp::transform(wp::vec3(s[0],s[1],s[2]),wp::quat(s[3],s[4],s[5],s[6]));"
)
def _load_pose(address: wp.uint64, lane: int) -> wp.transform: ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+19*lane+7; for(int k=0;k<6;++k){s[k]=velocity[k];s[6+k]=0.0f;}"
)
def _store_motion(address: wp.uint64, lane: int, velocity: wp.spatial_vector): ...


@wp.func_native(
    "float* s=reinterpret_cast<float*>(address)+19*lane+7+6*acceleration; wp::spatial_vector result;for(int k=0;k<6;++k)result[k]=s[k];return result;"
)
def _load_motion(address: wp.uint64, lane: int, acceleration: int) -> wp.spatial_vector: ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+358+3*root;for(int k=0;k<3;++k)s[k]=origin[k];")
def _store_origin(address: wp.uint64, root: int, origin: wp.vec3): ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+358+3*root;return wp::vec3(s[0],s[1],s[2]);")
def _load_origin(address: wp.uint64, root: int) -> wp.vec3: ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+367+3*root;for(int k=0;k<3;++k)s[k]=shift[k];")
def _store_shift(address: wp.uint64, root: int, shift: wp.vec3): ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+367+3*root;return wp::vec3(s[0],s[1],s[2]);")
def _load_shift(address: wp.uint64, root: int) -> wp.vec3: ...


@wp.func_native("float* s=reinterpret_cast<float*>(address)+304+6*dof;for(int k=0;k<6;++k)s[k]=axis[k];")
def _store_axis(address: wp.uint64, dof: int, axis: wp.spatial_vector): ...


def scan_source(motion):
    """Change addressing/padding only, retaining all four original scan rounds."""
    source = original._paired_scan(
        original.world_scan_publication._MOTION_SCAN if motion else original.world_scan_publication._POSE_SCAN
    )
    replace = original._replace
    source = replace(source, "[32]", "[16]", 6 if motion else 4)
    source = replace(source, "lane<32", "lane<16", 4)
    if motion:
        source = replace(source, "reinterpret_cast<float*>(address)+224", "reinterpret_cast<float*>(address)+7")
        source = replace(source, "6*lane", "19*lane", 6)
        source = replace(source, "s[192+", "s[6+", 2)
    else:
        source = replace(source, "7*lane", "19*lane", 4)
    return source


@wp.func_native(scan_source(False))
def _scan_poses(address: wp.uint64, world: int, plan: KineticPlan): ...


@wp.func_native(scan_source(True))
def _scan_motion(address: wp.uint64, world: int, plan: KineticPlan): ...


def terms_source():
    """Overwrite only the calling body's already-consumed pose/motion slot."""
    source = original._store_terms.native_snippet
    source = original._replace(source, "s[686+6*body+k]", "s[19*body+k]")
    return original._replace(source, "s+752+13*body", "s+19*body+6")


@wp.func_native(terms_source())
def _store_terms(
    address: wp.uint64,
    body: int,
    mass: float,
    radius: wp.vec3,
    inertia: wp.mat33,
    wrench: wp.spatial_vector,
    refresh: int,
): ...


def collect_source():
    """Preserve reduction/projection order and fences with nonaliasing actions."""
    source = original._collect.native_snippet
    for old, new, count in (
        ("const int width=component<6?6:13;", "const int width=19;", 1),
        ("s+(component<6?686+component:752+component-6)", "s+component", 1),
        ("s+632+6*dof", "s+304+6*dof", 1),
        ("s[686+6*body+k]", "s[19*body+k]", 1),
        ("s+752+13*body", "s+19*body+6", 1),
        ("s[6*dof+k]", "s[247+6*dof+k]", 1),
        ("s[6*dof+3+k]", "s[247+6*dof+3+k]", 1),
        ("s[632+6*projection+k]", "s[304+6*projection+k]", 1),
        ("s[6*src+k]", "s[247+6*src+k]", 1),
    ):
        source = original._replace(source, old, new, count)
    return source


@wp.func_native(collect_source())
def _collect(
    address: wp.uint64, world: int, plan: KineticPlan, data: PublicationData, cache: KineticData, refresh: int
) -> int: ...


@functools.cache
def get_state_kernel(finish):
    """Bind the original state law to compact helpers; preserve its five arguments."""
    terms = textwrap.dedent(inspect.getsource(original._primary_terms.func))
    terms = original._replace(terms, "def _primary_terms(", "def _compact_primary_terms(")
    source = original.state_source()
    source = original._replace(source, '"franka_kinetic_finish13_h81_p16"', '"franka_kinetic_finish13_h81_c376_p16"')
    source = original._replace(source, '"franka_kinetic_repair13_h81_p16"', '"franka_kinetic_repair13_h81_c376_p16"')
    source = terms + "\n_primary_terms = _compact_primary_terms\n" + source
    filename = f"<franka-compact-workspace-{hashlib.sha256(source.encode()).hexdigest()}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = dict(original.__dict__)
    namespace.update(globals())
    exec(compile(source, filename, "exec"), namespace)
    return namespace["_factory"](finish)
