# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental paired16 workspace lifetime replacement; unchanged state law.

Sixteen body slots hold pose7/velocity6/acceleration6 during the scans. Once
each body has loaded its own values, its slot becomes wrench6/moments13.
Dimension-derived storage places inertia-axis actions after all live body
terms. A fence separates action writes from projection reads.
"""

import functools

import warp as wp

from . import franka_kinetic_state as original
from .forest_kinetic_layout import DEFAULT_LAYOUT

KineticPlan = original.KineticPlan
KineticData = original.KineticData
PublicationData = original.PublicationData

FLOATS_PER_WORLD = DEFAULT_LAYOUT.state_floats
BODY_STRIDE = 19
ACTION_OFFSET = DEFAULT_LAYOUT.actions
AXIS_OFFSET = DEFAULT_LAYOUT.axes
ORIGIN_OFFSET = DEFAULT_LAYOUT.origins
SHIFT_OFFSET = DEFAULT_LAYOUT.shifts


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


def scan_source(motion):
    """Bind the common scan to interleaved slots without rewriting CUDA source."""
    return original.scan_source(
        motion,
        cuda_width=16,
        cpu_width=16,
        pose_stride=19,
        motion_offset=7,
        motion_stride=19,
        acceleration_offset=6,
    )


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


def collect_source(layout=DEFAULT_LAYOUT):
    """Preserve reduction/projection order and fences with nonaliasing actions."""
    source = original.collect_source(layout)
    for old, new, count in (
        ("const int width=component<6?6:13;", "const int width=19;", 1),
        ("s+(component<6?686+component:752+component-6)", "s+component", 1),
        ("s+632+6*dof", f"s+{layout.axes}+6*dof", 1),
        ("s[686+6*body+k]", "s[19*body+k]", 1),
        ("s+752+13*body", "s+19*body+6", 1),
        ("s[6*dof+k]", f"s[{layout.actions}+6*dof+k]", 1),
        ("s[6*dof+3+k]", f"s[{layout.actions}+6*dof+3+k]", 1),
        ("s[632+6*projection+k]", f"s[{layout.axes}+6*projection+k]", 1),
        ("s[6*src+k]", f"s[{layout.actions}+6*src+k]", 1),
    ):
        source = original._replace(source, old, new, count)
    return source


@functools.cache
def layout_namespace(layout):
    """Bind dimension-derived offsets to the shared compact body and scan law."""
    storage_source = f"""
#if defined(__CUDA_ARCH__)
        __shared__ float values[2*{layout.state_floats}];
        return reinterpret_cast<uint64_t>(values+{layout.state_floats}*((threadIdx.x&31)>>4));
#else
        return reinterpret_cast<uint64_t>(malloc({layout.state_floats}*sizeof(float)));
#endif
    """

    @wp.func_native(storage_source)
    def storage() -> wp.uint64: ...

    @wp.func_native(
        f"float* s=reinterpret_cast<float*>(address)+{layout.origins}+3*root;for(int k=0;k<3;++k)s[k]=origin[k];"
    )
    def store_origin(address: wp.uint64, root: int, origin: wp.vec3): ...

    @wp.func_native(
        f"float* s=reinterpret_cast<float*>(address)+{layout.origins}+3*root;return wp::vec3(s[0],s[1],s[2]);"
    )
    def load_origin(address: wp.uint64, root: int) -> wp.vec3: ...

    @wp.func_native(
        f"float* s=reinterpret_cast<float*>(address)+{layout.shifts}+3*root;for(int k=0;k<3;++k)s[k]=shift[k];"
    )
    def store_shift(address: wp.uint64, root: int, shift: wp.vec3): ...

    @wp.func_native(
        f"float* s=reinterpret_cast<float*>(address)+{layout.shifts}+3*root;return wp::vec3(s[0],s[1],s[2]);"
    )
    def load_shift(address: wp.uint64, root: int) -> wp.vec3: ...

    @wp.func_native(f"float* s=reinterpret_cast<float*>(address)+{layout.axes}+6*dof;for(int k=0;k<6;++k)s[k]=axis[k];")
    def store_axis(address: wp.uint64, dof: int, axis: wp.spatial_vector): ...

    @wp.func_native(collect_source(layout))
    def collect(
        address: wp.uint64, world: int, plan: KineticPlan, data: PublicationData, cache: KineticData, refresh: int
    ) -> int: ...

    namespace = dict(original.__dict__)
    namespace.update(globals())
    namespace.update(
        _storage=storage,
        _store_origin=store_origin,
        _load_origin=load_origin,
        _store_shift=store_shift,
        _load_shift=load_shift,
        _store_axis=store_axis,
        _collect=collect,
        _primary_terms=original.make_primary_terms(_store_terms),
    )
    return namespace


def get_state_kernel(finish, layout=None):
    return _get_state_kernel(finish, DEFAULT_LAYOUT if layout is None else layout)


@functools.cache
def _get_state_kernel(finish, layout):
    """Bind shared state arithmetic to dimension-derived compact storage."""
    return original.make_state_kernel(
        finish, layout_namespace(layout), suffix=f"_c{layout.state_floats}", layout=layout
    )
