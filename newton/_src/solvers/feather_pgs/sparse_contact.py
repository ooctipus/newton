# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sparse factor rows' contact record layout."""

CONTACT_RECORD_FLOATS = 8
"""Floats of a contact triplet's block record: ``W_nt1, W_nt2, W_t1t2, 1/W_nn``, then the pair
law's tangent eigenbasis ``largest, smallest, vx, vy`` relative to the larger tangent diagonal."""
CONTACT_RING_FLOATS = 4
"""Floats a linked sparse sweep appends to each record: the anchor's other ring members, as int bits."""
