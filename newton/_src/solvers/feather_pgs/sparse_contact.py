# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse factor rows' contact record layout."""

CONTACT_RECORD_FLOATS = 8
"""Floats of a contact's block record, its friction metric formed once per solve. A contact unit of three:
``W_nt1, W_nt2, W_t1t2, 1/W_nn``, then the pair law's tangent eigenbasis ``largest, smallest, vx, vy``
(:func:`.projection_laws.friction_pair_metric`). A contact with a spin row: the friction triple's metric
(:func:`.projection_laws.friction_ball_metric`), its three eigenvalues and a zero, then its eigenbasis as a rotation
quaternion ``(x, y, z, w)``."""
CONTACT_RING_FLOATS = 4
"""Floats a linked sparse sweep appends to each record: the anchor's other ring members, as int bits."""
