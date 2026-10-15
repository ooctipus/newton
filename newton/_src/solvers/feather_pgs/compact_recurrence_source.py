# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind compact arena views and current-MF phase to the shared paired law."""

from .paired_factor_source import get_source as paired_source

_VIEWS = r"""
    float* arena=reinterpret_cast<float*>(arena_address);
    float* s_lam=arena; float* s_rhs=arena+192; float* s_diag=arena+384;
    unsigned char* s_type=reinterpret_cast<unsigned char*>(arena+576);
    float* s_contact_mu=arena+624; float* s_lam_mf=arena+688;
    float* s_v=arena+752; float* hybrid_base=arena+781;
"""

_ENTRY = r"""
    const int po=primary_offset,so=secondary_offset;
    const bool split_component=false;
    const int m_mf=mf_constraint_count.data[world];
    int mf_contact_end=mf_contact_rows_end.data[world];
    const int off_mf=world*64,off_meta=off_mf*4,mf6_base=off_mf*6;
    if(m_mf<0||m_mf>64||mf_contact_end<0||mf_contact_end>m_mf){if(lane==0)solve.status.data[world]=1;return;}
    for(int i=lane;i<m_mf;i+=32)s_lam_mf[i]=mf_impulses.data[off_mf+i];
    if(lane<6)hybrid_base[lane]=state.v_hat.data[plan.dof_ids.data[world*35+23+lane]];
    auto dense_j=[&](int row,float y){return lane>=po&&lane<po+23?y:J_world.data[(world*192+row)*29+lane];};
"""

_BEFORE_MF = r"""
        // Only six physical increments cross this phase boundary; primary du
        // stays in registers. The original MF and rigid-limit source is exact.
        if(lane>=so&&lane<so+6)s_v[lane]=factor_velocity;
        __syncwarp(MASK);
"""

_AFTER_MF = r"""
        __syncwarp(MASK);
        if(lane>=so&&lane<so+6)factor_velocity=s_v[lane];
"""

_STORE_MF = r"""
    for(int i=lane;i<m_mf;i+=32)mf_impulses.data[off_mf+i]=s_lam_mf[i];
"""


def get_source(mf_phase, *, contact_triples=True):
    """Keep dense, current-MF and rigid-limit ownership in their original order."""
    if not mf_phase:
        raise ValueError("Compact recurrence requires the complete current-MF phase")
    return paired_source(
        192,
        29,
        23,
        6,
        contact_triples=contact_triples,
        kinetic=True,
        compact=True,
        compact_views=_VIEWS,
        compact_entry=_ENTRY,
        compact_sweep=_BEFORE_MF + mf_phase + _AFTER_MF,
        compact_store=_STORE_MF,
    )
