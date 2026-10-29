# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve one contact unit's Coulomb problem exactly against its own Delassus block.

A contact unit is every set of rows that shares one Coulomb law: a point
contact (a normal row and its tangent pair), a patch region (its ring of normal
rows and the tangent pairs of at most two anchors, whose disk radius is the
anchor's coefficient times the ring's total normal impulse) or a frictionless
ring. A Gauss-Seidel visit of a unit solves the unit's local complementarity
problem with every other impulse held (non-smooth Gauss-Seidel, Jourdan, Alart
and Jean 1998). Every solution of the whole problem solves each unit's problem,
so the visit returns the unit's impulses whenever they solve it, also where the
problem has several solutions (a sliding contact whose friction presses its
normal down, a Painleve-type non-uniqueness): the visit keeps the branch the
unit holds.

* take-off: no normal impulse, the friction of an anchor whose ring keeps a load
  ``L`` on the disk of radius ``mu * L``; an unloaded unit tries it first where
  it may solve the problem (a contact that separates at zero impulse, or an
  anchor holding friction, which its ring's load may let lift it), a loaded unit
  after its loaded branches, keeping whichever of the two solutions lies nearer
  its impulses in the W-norm;
* the friction's own slide root: a friction on its disk's boundary resumes it
  first, by Newton on ``h(alpha) = |s| + mu g.s - mu S0`` from the root it
  implies. Every root lies in ``(0, |c| (1 + mu |g|) / (mu |S0|) - e_min)`` for
  either sign of ``S0`` (the friction presses the normal where ``S0 <= 0``); the
  iteration stops where it leaves that interval and does not bracket, so it
  continues the root the friction holds and may miss others. A root whose normal
  impulse is negative (an anchor's load below ``L``) is not taken;
* stick: the Schur complement of the normal row, ``A = W_TT - w w^T / W_nn``,
  solved in its eigenbasis;
* slide, where ``S0 > 0``: Daviet's single-unknown root (Daviet,
  Bertails-Descoubes and Boissieux 2011), ``s(alpha) = (A + alpha I)^-1 c`` with
  ``|s| = mu (S0 - g.s)``, by a safeguarded Newton iteration in the bracket
  ``[0, |c| (1 + mu |g|) / (mu S0) - e_min]``, started from the unit's previous
  root of the solve when that lies in the bracket; a Schur complement that
  vanishes (a rank-one block) takes the closed-form root;
* press slide, where ``S0 <= 0`` and take-off does not solve the problem (an
  anchor whose ring's load lets its friction press it into approach): only a
  friction that presses the normal carries load. ``F = a1 a2 (mu S - |s|)``,
  with ``a_i = e_i + alpha``, is concave in alpha, so its roots are at most two
  and Newton from ``alpha = 0`` rises to the smaller one, the slide of the larger
  load, without overshoot (within the evaluation cap). That slide is admissible
  whenever the other one is, so with the stick this finds a loaded solution
  wherever one exists. An unloaded unit takes it without resuming the root its
  take-off friction continues, the slide of the smaller load.

A resumed root's load answers the contact's approach (``S0``) with
``dS/dS0 = (d|s|/dalpha) / h'``: negative where ``h`` rises through zero (the
middle one of several slides, and every press slide of the smaller load) and
large where the friction's pressing nearly balances the normal stiffness. Where
that response is negative or above :data:`CONTACT_UNIT_SPLIT_GAIN` times the
normal row's own, a loaded unit that holds the slide (its friction within
:data:`CONTACT_UNIT_SPLIT_HOLD` of the root's) takes its normal with the
friction held, then the friction on that load's disk, if that normal is
loaded: the normal row's own response. A held solution stays put, as under the
exact solve. The step stabilizes sweeps whose exact solves the coupling of units
drives away from such a slide; a lone unit inside the hold nears a slide of gain
``G`` by ``1 - 1/G`` per visit, to within the friction routine's tolerance
amplified by ``G``, and leaves a slide of negative gain by ``1 + 1/|G|`` until
the exact solve returns it. The friction is finally clamped to the disk of the
final load, so the result is exactly feasible.

A ring unit solves its normals by an active set over their proximally
regularized block, from its loaded members, then each anchor by the point
routine on the friction Schur complement from its current friction (two
alternations for two anchors; a ring takes no split step and no press slide);
a basis that loses a member restarts the anchors from their current friction. A
member that depends on the basis and whose velocity agrees with the basis
members' to within ``1e-5`` of the ring's velocity scale holds its load: the
velocities do not fix the split among such members, which the proximal step
would move by their residuals over its weight. A ring therefore settles with
such a member up to ``1e-5`` of its velocity scale from the basis velocities,
and the split stays where it was: the fourth corner of a rigid planar face keeps
its load (none from a cold start), so the face rests on three corners wherever
they carry it, one of the solutions of the indeterminate problem. The
alternation latch relaxes a unit whose load change keeps reversing, which every
fixed point of the exact sweep survives: a unit whose exact solution equals its
impulses does not move for any relaxation factor.

The law is written once, as C++ text. :func:`get_unit_source` binds backend
intrinsics around it: the CUDA binding is the device prelude
:data:`CONTACT_UNIT` for native sweeps, and the CPU binding backs the
``wp.func_native`` functions of this module, which the Warp loop sweeps call.
"""

import warp as wp

from .device_preludes import DevicePrelude, substitute

CONTACT_UPDATES = ("sequential", "unit")
"""The contact updates a sweep can apply (``SolverFeatherPGS(contact_update=...)``)."""

CONTACT_UNIT_STATE_FIELDS = 3
"""Per-row unit state: slide root, last applied load change, latch factor."""


def check_contact_update(update: str) -> str:
    """Return ``update`` when it names a contact update.

    Raises:
        ValueError: If ``update`` is not one of :data:`CONTACT_UPDATES`.
    """
    if update not in CONTACT_UPDATES:
        raise ValueError(f"contact_update must be one of {CONTACT_UPDATES}, got {update!r}")
    return update


CONTACT_UNIT_RING_NORMALS = 8
"""Most normal rows of a ring unit; a larger ring solves its anchors as anchor units."""
CONTACT_UNIT_RING_ANCHORS = 2
"""Most anchors of a ring unit; a ring with more solves its anchors as anchor units."""
CONTACT_UNIT_NEWTON_CAP = 8
"""Newton evaluations of one slide root."""
CONTACT_UNIT_RING_PASSES = 2
"""Alternations of the two anchors of a ring unit."""
CONTACT_UNIT_PROXIMAL = 1.0e-4
"""Proximal weight of a ring's normal block: SPD in float32 for coplanar faces, the same fixed point."""
CONTACT_UNIT_SPLIT_GAIN = 8.0
"""Largest load response of a held slide to the contact's approach, relative to the normal row's own, that keeps the
exact solve; a larger or negative one takes the normal with the friction held, then the friction."""
CONTACT_UNIT_SPLIT_HOLD = 1.0e-3
"""Largest distance of a unit's friction from its resumed slide root, relative to the root's, at which the unit holds
that slide: only a held slide takes the split step, any other friction the exact solve."""

# The block a ring unit reads and writes, per world: its row-major Delassus block (12 x 12), the residuals with its
# own impulses removed, its impulses, the solution and the two anchors' slide roots.
RING_BLOCK = 0
RING_RESIDUAL = 144
RING_IMPULSE = 156
RING_SOLUTION = 168
RING_ALPHA = 180
RING_BLOCK_SIZE = 192

_UNIT = r"""
    // split: the root's load answers the contact's approach with the wrong sign or above ${gain} times the normal
    // row's own response, where the coupling of units can drive exact solves away from it.
    struct ContactUnitRoot { float s0, s1, alpha; int evaluations, split; };
    // mode: 1 stick, 2 slide, 3 no load to slide on, 4 held slide of such a gain (the split step).
    struct ContactUnitFriction { float friction0, friction1, alpha; int evaluations, mode; };
    struct ContactUnitSolution { float normal, friction0, friction1, alpha; int evaluations; };
    struct ContactUnitLatch { float weight, factor, change; };
    struct ContactUnit {
        // Eigenpairs of the SPD 2x2 [[a, c], [c, d]]: e1 >= e2 and the unit axis of e1.
        static ${inline} void eigen(float a, float c, float d, float& e1, float& e2, float& vx, float& vy) {
            const float tr = a + d;
            const float disc = ${sqrt}(${max}((a - d) * (a - d) + 4.0f * c * c, 0.0f));
            e1 = 0.5f * (tr + disc);
            e2 = 0.0f;
            if (e1 > 0.0f) e2 = ${max}((a * d - c * c) / e1, 0.0f);
            vx = c;
            vy = e1 - a;
            if (a >= d) {
                vx = e1 - d;
                vy = c;
            }
            const float nv = ${sqrt}(vx * vx + vy * vy);
            if (nv > 0.0f) {
                vx = vx / nv;
                vy = vy / nv;
            } else {
                vx = 1.0f;
                vy = 0.0f;
            }
        }

        // Daviet's slide root in the eigenbasis: s_i = c_i / (e_i + alpha), |s| = mu (S0 - g.s), by safeguarded
        // Newton on G(alpha) = 1 / (|s| + mu g.s) - 1 / (mu S0) inside the bracket; a vanishing Schur complement
        // takes the closed-form root.
        static ${noinline} ContactUnitRoot slide(float e1, float e2, float cx, float cy, float gx, float gy, float S0,
                                                 float mu, float alpha0, int cap) {
            const float target = 1.0f / (mu * S0);
            const float nc = ${sqrt}(cx * cx + cy * cy);
            const float ng = ${sqrt}(gx * gx + gy * gy);
            float lo = 0.0f;
            float hi = ${max}(nc * (1.0f + mu * ng) * target - e2, 0.0f) * 1.000001f + 1.0e-30f;
            ContactUnitRoot out{0.0f, 0.0f, 0.0f, 0, 0};
            float x = -1.0f;
            if (e1 == 0.0f) {
                const float ee = (e1 * cx * cx + e2 * cy * cy) / ${max}(nc * nc, 1.0e-30f);
                x = (nc + mu * (gx * cx + gy * cy)) * target - ee;
                if (x > lo && x < hi) {
                    out.s0 = cx * (1.0f / (e1 + x));
                    out.s1 = cy * (1.0f / (e2 + x));
                    out.alpha = x;
                    out.evaluations = 1;
                    return out;
                }
            }
            if (alpha0 > 0.0f && alpha0 < hi) {
                x = alpha0;
            } else {
                const float ee = (e1 * cx * cx + e2 * cy * cy) / ${max}(nc * nc, 1.0e-30f);
                x = (nc + mu * (gx * cx + gy * cy)) * target - ee;
                if (!(x > lo && x < hi)) x = 0.5f * (lo + hi);
            }
            float sx = 0.0f;
            float sy = 0.0f;
            for (int it = 0; it < cap; ++it) {
                const float ix = 1.0f / (e1 + x);
                float iy = 0.0f;
                if (e2 + x > 0.0f) iy = 1.0f / (e2 + x);
                sx = cx * ix;
                sy = cy * iy;
                out.evaluations += 1;
                const float ns = ${sqrt}(sx * sx + sy * sy);
                const float q = ns + mu * (gx * sx + gy * sy);
                float G = 0.0f;
                if (q > 0.0f) {
                    G = 1.0f / q - target;
                } else {
                    G = 1.0e30f;
                }
                if (${abs}(G) <= 2.0e-6f * target || hi - lo <= 1.0e-7f * hi) break;
                if (G < 0.0f) {
                    lo = x;
                } else {
                    hi = x;
                }
                float xn = -1.0f;
                if (q > 0.0f && ns > 0.0f) {
                    const float dq = -(sx * sx * ix + sy * sy * iy) / ns - mu * (gx * sx * ix + gy * sy * iy);
                    const float dG = -dq / (q * q);
                    if (dG != 0.0f) xn = x - G / dG;
                }
                if (xn > lo && xn < hi) {
                    x = xn;
                } else {
                    x = 0.5f * (lo + hi);
                }
            }
            out.s0 = sx;
            out.s1 = sy;
            out.alpha = x;
            return out;
        }

        // The slide root of the current friction: Newton on h(alpha) = |s| + mu g.s - mu S0, the friction's excess
        // over its disk, from the root that friction implies. Every root lies in (0, |c| (1 + mu |g|) / (mu |S0|) -
        // e2) for either sign of S0 (the friction presses the normal where S0 < 0); the iteration stops when it
        // leaves that interval, it does not bracket, so it continues the root the friction holds and may miss others.
        // Converged within 2e-6 of the terms of h, which stay well-scaled when the pressing friction carries most of
        // the load. alpha -1 when no root is met within cap evaluations. The root's load S = |s| / mu answers S0 with
        // dS/dS0 = (d|s|/dalpha) / h': negative where h rises (the middle one of several slides), large where the
        // friction's pressing nearly balances the normal stiffness.
        static ${noinline} ContactUnitRoot resume(float e1, float e2, float cx, float cy, float gx, float gy, float S0,
                                                  float mu, float start, int cap) {
            const float radius = mu * S0;
            const float nc = ${sqrt}(cx * cx + cy * cy);
            const float ng = ${sqrt}(gx * gx + gy * gy);
            const float hi = ${max}(nc * (1.0f + mu * ng) / ${abs}(radius) - e2, 0.0f) * 1.000001f + 1.0e-30f;
            ContactUnitRoot out{0.0f, 0.0f, -1.0f, 0, 0};
            float x = start;
            for (int it = 0; it < cap; ++it) {
                if (!(x > 0.0f && x < hi)) break;
                const float ix = 1.0f / (e1 + x);
                const float iy = 1.0f / (e2 + x);
                const float sx = cx * ix;
                const float sy = cy * iy;
                out.evaluations += 1;
                const float ns = ${sqrt}(sx * sx + sy * sy);
                if (!(ns > 0.0f)) break;
                const float gs = mu * (gx * sx + gy * sy);
                const float h = ns + gs - radius;
                const float dns = -(sx * sx * ix + sy * sy * iy) / ns;
                const float dh = dns - mu * (gx * sx * ix + gy * sy * iy);
                if (${abs}(h) <= 2.0e-6f * (ns + ${abs}(gs))) {
                    out.s0 = sx;
                    out.s1 = sy;
                    out.alpha = x;
                    const float gain = dns / dh;
                    if (dh > 0.0f || gain > ${gain}f) out.split = 1;
                    return out;
                }
                if (!(dh != 0.0f)) break;
                x = x - h / dh;
            }
            return out;
        }

        // The slide of a friction that presses its normal (S0 <= 0) with the larger load, the smallest root of h.
        // With a1 = e1 + alpha and a2 = e2 + alpha, F = a1 a2 (mu S - |s|) = mu (S0 a1 a2 - gx cx a2 - gy cy a1) -
        // |(cx a2, cy a1)|, a quadratic of leading coefficient mu S0 less the norm of an affine map, is concave for
        // S0 <= 0: it has at most two roots, and Newton from alpha = 0, where F < 0 below the smaller one, rises to it
        // without overshoot. Its load S = |s| / mu is the larger of the two (|s| falls with alpha) and answers S0 with
        // a positive gain, as h falls through it; the larger root's is negative. alpha -1 when F(0) >= 0 (none below
        // the larger root), when F passes its maximum below zero (no root) or after cap evaluations.
        static ${noinline} ContactUnitRoot press(float e1, float e2, float cx, float cy, float gx, float gy, float S0,
                                                 float mu, int cap) {
            ContactUnitRoot out{0.0f, 0.0f, -1.0f, 0, 0};
            float x = 0.0f;
            for (int it = 0; it < cap; ++it) {
                const float a1 = e1 + x;
                const float a2 = e2 + x;
                const float q1 = cx * a2;
                const float q2 = cy * a1;
                const float N = ${sqrt}(q1 * q1 + q2 * q2);
                const float G = mu * (gx * q1 + gy * q2);
                const float F = mu * S0 * a1 * a2 - G - N;
                out.evaluations += 1;
                if (${abs}(F) <= 2.0e-6f * (N + ${abs}(G))) {
                    if (x > 0.0f) {
                        out.s0 = cx / a1;
                        out.s1 = cy / a2;
                        out.alpha = x;
                    }
                    return out;
                }
                const float dF = mu * (S0 * (a1 + a2) - gx * cx - gy * cy) - (cx * q1 + cy * q2) / N;
                if ((it == 0 && F > 0.0f) || !(dF > 0.0f)) break;
                x = x - F / dF;
            }
            return out;
        }

        // Friction of one anchor on its Schur complement A = [[a, c12], [c12, d]] with residual c and load
        // S(f) = S0 + g.f of at least Smin, current friction (f1, f2) on its current load: a current friction on its
        // disk's boundary resumes its slide root first, so a problem with several solutions keeps the one the unit
        // holds; else stick; else slide where S0 > 0. Where S0 <= 0 and take-off does not solve the unit's problem
        // (takeoff_fails), only a friction that presses the normal carries load: the unit takes the press slide of the
        // larger load, and an unloaded one takes it without resuming the root its take-off friction continues, which
        // is the press slide of the smaller load, whose gain is negative. No friction without a coefficient.
        static ${inline} ContactUnitFriction friction(float a, float c12, float d, float c1, float c2, float S0,
                                                     float g1, float g2, float mu, float alpha0, int cap, float Smin,
                                                     float f1, float f2, float load, int takeoff_fails) {
            ContactUnitFriction out{0.0f, 0.0f, 0.0f, 1, 1};
            if (!(mu > 0.0f)) return out;
            float e1, e2, vx, vy;
            eigen(a, c12, d, e1, e2, vx, vy);
            const float cx = vx * c1 + vy * c2;
            const float cy = -vy * c1 + vx * c2;
            const float gx = vx * g1 + vy * g2;
            const float gy = -vy * g1 + vx * g2;
            const float px = -(vx * f1 + vy * f2);
            const float py = vy * f1 - vx * f2;
            const float np2 = px * px + py * py;
            const float Rc = 0.99999f * mu * load;
            const int pressed = takeoff_fails != 0 && S0 <= 0.0f;
            if (np2 > 0.0f && load > 0.0f && np2 >= Rc * Rc && S0 != 0.0f && !(pressed != 0 && !(load > Smin))) {
                // Its root: (A + alpha I) s = c along s, the velocity c - A s parallel to s.
                const float start = (px * (cx - e1 * px) + py * (cy - e2 * py)) / np2;
                const ContactUnitRoot root = resume(e1, e2, cx, cy, gx, gy, S0, mu, start, cap);
                out.evaluations += root.evaluations;
                if (root.alpha > 0.0f && S0 - (gx * root.s0 + gy * root.s1) >= Smin) {
                    out.friction0 = -(vx * root.s0 - vy * root.s1);
                    out.friction1 = -(vy * root.s0 + vx * root.s1);
                    out.alpha = root.alpha;
                    out.mode = 2;
                    // The split step is for a slide the unit holds: its friction within ${hold} of the root's.
                    const float hx = root.s0 - px;
                    const float hy = root.s1 - py;
                    if (root.split != 0 && hx * hx + hy * hy <= ${hold2}f * (root.s0 * root.s0 + root.s1 * root.s1)) {
                        out.mode = 4;
                    }
                    return out;
                }
            }
            if (e2 > 0.0f) {
                const float sx = cx / e1;
                const float sy = cy / e2;
                const float S = S0 - (gx * sx + gy * sy);
                if (S >= Smin && ${sqrt}(sx * sx + sy * sy) <= mu * S) {
                    out.friction0 = -(vx * sx - vy * sy);
                    out.friction1 = -(vy * sx + vx * sy);
                    return out;
                }
            } else if (cx == 0.0f && cy == 0.0f) {
                return out;
            } else if (e1 > 0.0f && cy == 0.0f) {
                // A singular Schur complement whose residual lies in its range sticks with its least-norm force.
                const float sx = cx / e1;
                const float S = S0 - gx * sx;
                if (S >= Smin && ${abs}(sx) <= mu * S) {
                    out.friction0 = -(vx * sx);
                    out.friction1 = -(vy * sx);
                    return out;
                }
            }
            if (S0 <= 0.0f) {
                // No load to slide on without pressing; a press slide needs mu |g| > 1.
                if (pressed != 0 && mu * ${sqrt}(gx * gx + gy * gy) > 1.0f) {
                    const ContactUnitRoot root = press(e1, e2, cx, cy, gx, gy, S0, mu, cap);
                    out.evaluations += root.evaluations;
                    if (root.alpha > 0.0f && S0 - (gx * root.s0 + gy * root.s1) >= Smin) {
                        out.friction0 = -(vx * root.s0 - vy * root.s1);
                        out.friction1 = -(vy * root.s0 + vx * root.s1);
                        out.alpha = root.alpha;
                        out.mode = 2;
                        return out;
                    }
                }
                out.mode = 3;
                return out;
            }
            const ContactUnitRoot root = slide(e1, e2, cx, cy, gx, gy, S0, mu, alpha0, cap);
            out.friction0 = -(vx * root.s0 - vy * root.s1);
            out.friction1 = -(vy * root.s0 + vx * root.s1);
            out.alpha = root.alpha;
            out.evaluations += root.evaluations;
            out.mode = 2;
            return out;
        }

        // The squared W-norm of (d0, d1, d2) over the 3x3 block [[wnn, wn1, wn2], [wn1, w11, w12], [wn2, w12, w22]].
        static ${inline} float wnorm(float wnn, float w11, float w12, float w22, float wn1, float wn2, float d0,
                                     float d1, float d2) {
            return wnn * d0 * d0 + w11 * d1 * d1 + w22 * d2 * d2 + 2.0f * (wn1 * d0 * d1 + wn2 * d0 * d2 + w12 * d1 * d2);
        }

        // One contact (or one patch anchor whose ring keeps the load L of its other normals) against its 3x3 block
        // [[wnn, wn1, wn2], [wn1, w11, w12], [wn2, w12, w22]] with residual b at zero own impulse and current
        // impulses (on, of1, of2). An unloaded unit takes off first where take-off solves its problem; a loaded one
        // solves its loaded branches first (its slide root, else stick, else slide) and, where take-off solves its
        // problem too, keeps whichever solution lies nearer its impulses in the W-norm. Where take-off does not solve
        // it and the friction must press the normal to carry load (S0 <= 0), the loaded branches include the press
        // slide of the larger load. The solve returns the current impulses whenever they solve the problem (a held
        // slide whose load answers S0 with a gain outside (0, 8] takes the split step, which keeps it as well). The
        // returned alpha is the slide root, also where its normal came out negative and the unit takes off, so that
        // the next visit continues a root the evaluation cap stopped short; -1 for take-off and stick.
        static ${inline} ContactUnitSolution solve(float wnn, float w11, float w12, float w22, float wn1, float wn2,
                                                  float bn, float b1, float b2, float mu, float L, float alpha0, int cap,
                                                  float on, float of1, float of2) {
            ContactUnitSolution out{0.0f, 0.0f, 0.0f, -1.0f, 0};
            float tfx = 0.0f;
            float tfy = 0.0f;
            int takeoff = 0;
            const float load = on + L;
            const int unloaded = !(on > 0.0f);
            // Take-off's friction, where take-off may solve the problem: a contact that separates at zero impulse, and
            // an unloaded anchor holding friction, whose ring's load lets that friction lift it.
            if (L > 0.0f && (bn >= 0.0f || (unloaded != 0 && (of1 != 0.0f || of2 != 0.0f)))) {
                const ContactUnitFriction tf = friction(w11, w12, w22, b1, b2, L, 0.0f, 0.0f, mu, -1.0f, cap, L, of1,
                                                        of2, load, 0);
                tfx = tf.friction0;
                tfy = tf.friction1;
                takeoff = 1;
            }
            const int lifts = bn + wn1 * tfx + wn2 * tfy >= 0.0f;
            if (unloaded != 0 && lifts != 0) {
                out.friction0 = tfx;
                out.friction1 = tfy;
                return out;
            }
            const float inv = 1.0f / wnn;
            const ContactUnitFriction r = friction(w11 - wn1 * wn1 * inv, w12 - wn1 * wn2 * inv, w22 - wn2 * wn2 * inv,
                                                   b1 - bn * inv * wn1, b2 - bn * inv * wn2, -bn * inv + L, -wn1 * inv,
                                                   -wn2 * inv, mu, alpha0, cap, L, of1, of2, load, 1 - lifts);
            out.evaluations = r.evaluations;
            int found = 0;
            const float xn = (-bn - wn1 * r.friction0 - wn2 * r.friction1) * inv;
            const float held = -bn - wn1 * of1 - wn2 * of2;
            if (r.mode == 4 && on > 0.0f && xn >= 0.0f && held > 0.0f) {
                // A held slide whose load answers the contact's approach with the wrong sign, or more than 8 times the
                // normal row alone, takes the normal with its current friction held, then the friction on that load's
                // disk: the normal row's own response, which stabilizes a sweep whose exact solves the coupling of
                // units drives away from such a slide. A solution stays put, as under the exact solve; a lone unit
                // nears the slide by 1 - 1/gain per visit and leaves a slide of negative gain. A root whose normal
                // impulse is negative, or a held friction that would unload the normal, takes the exact solve.
                const float xs = held * inv;
                const ContactUnitFriction q = friction(w11, w12, w22, b1 + wn1 * xs, b2 + wn2 * xs, xs + L, 0.0f, 0.0f,
                                                       mu, -1.0f, cap, xs + L, of1, of2, load, 0);
                out.evaluations += q.evaluations;
                const float R = mu * (xs + L);
                float g1 = q.friction0;
                float g2 = q.friction1;
                const float ng = ${sqrt}(g1 * g1 + g2 * g2);
                if (ng > R) {
                    float sc = 0.0f;
                    if (ng > 0.0f) sc = R / ng;
                    g1 *= sc;
                    g2 *= sc;
                }
                out.normal = xs;
                out.friction0 = g1;
                out.friction1 = g2;
                out.alpha = r.alpha;
                found = 1;
            } else if (r.mode != 3 && xn >= 0.0f) {
                const float R = mu * (xn + L);
                float f1 = r.friction0;
                float f2 = r.friction1;
                const float nf = ${sqrt}(f1 * f1 + f2 * f2);
                if (nf > R) {
                    float sc = 0.0f;
                    if (nf > 0.0f) sc = R / nf;
                    f1 *= sc;
                    f2 *= sc;
                }
                out.normal = xn;
                out.friction0 = f1;
                out.friction1 = f2;
                if (r.mode != 1) out.alpha = r.alpha;
                found = 1;
            }
            if (found != 0) {
                if (bn >= 0.0f && on > 0.0f && lifts != 0) {
                    // Take-off solves this problem too (friction pressing the normal): keep the nearer solution.
                    const float dl = wnorm(wnn, w11, w12, w22, wn1, wn2, out.normal - on, out.friction0 - of1,
                                           out.friction1 - of2);
                    const float dt = wnorm(wnn, w11, w12, w22, wn1, wn2, -on, tfx - of1, tfy - of2);
                    if (dt < dl) {
                        out.normal = 0.0f;
                        out.friction0 = tfx;
                        out.friction1 = tfy;
                        out.alpha = -1.0f;
                    }
                }
                return out;
            }
            if (L > 0.0f && takeoff == 0) {
                const ContactUnitFriction tf = friction(w11, w12, w22, b1, b2, L, 0.0f, 0.0f, mu, -1.0f, cap, L, of1,
                                                        of2, load, 0);
                tfx = tf.friction0;
                tfy = tf.friction1;
            }
            out.friction0 = tfx;
            out.friction1 = tfy;
            if (r.mode == 2) out.alpha = r.alpha;
            return out;
        }

        // The alternation latch of one unit. factor: its relaxation, 0 while unlatched; previous: its last applied
        // load change. A change that reverses the previous one by at least 80% of it, above 5% of the load,
        // compounds the factor by 1 / (1 + kappa), kappa the change ratio clamped to [0.2, 3], down to 0.1.
        static ${inline} ContactUnitLatch latch(float factor, float previous, float old_load, float new_load) {
            float weight = factor;
            if (weight <= 0.0f) weight = 1.0f;
            float change = weight * (new_load - old_load);
            if (previous * change < 0.0f && ${abs}(change) >= 0.8f * ${abs}(previous)
                && ${abs}(change) > 0.05f * ${max}(${max}(new_load, old_load), 1.0e-30f)) {
                const float kappa = ${min}(${max}(${abs}(change) / ${abs}(previous), 0.2f), 3.0f);
                factor = ${max}(weight / (1.0f + kappa), 0.1f);
                weight = factor;
                change = weight * (new_load - old_load);
            }
            return ContactUnitLatch{weight, factor, change};
        }

        static ${inline} void cholesky_solve(const float* L, int n, const float* rhs, float* out) {
            for (int i = 0; i < n; ++i) {
                float v = rhs[i];
                for (int j = 0; j < i; ++j) v -= L[i * 8 + j] * out[j];
                out[i] = v / L[i * 8 + i];
            }
            for (int ii = 0; ii < n; ++ii) {
                const int i = n - 1 - ii;
                float v = out[i];
                for (int j = i + 1; j < n; ++j) v -= L[j * 8 + i] * out[j];
                out[i] = v / L[i * 8 + i];
            }
        }

        // A ring unit: k normals (at most 8) and A anchors (at most 2), rows ordered normals, then each anchor's
        // pair. W: its nu x nu block (nu = k + 2A, row-major), b: residuals at zero own impulse, old: its impulses,
        // x: the solution, alpha: the anchors' slide roots of this solve (in/out). Returns the root evaluations.
        static ${noinline} int ring(const float* W, const float* b, const float* old, float* x, float* alpha, int k,
                                    int A, float mu0, float mu1, int passes, int cap, float prox) {
            const int nu = k + 2 * A;
            const int nF = 2 * A;
            float B[8];
            int act[8];
            int bas[8];
            float LC[64];
            float Y[8];
            float Z[8];
            float XM[32];
            float TMP[8];
            float TMP2[8];
            float ASC[16];
            float CS[4];
            float GS[4];
            float F[4];
            float XB[8];
            float EX[8];
            float RV[8];
            float ZR[8];
            int evaluations = 0;
            // Proximal term on the normals: W_NN + eps D and b_N - eps D old, the same fixed point.
            float eps = 0.0f;
            if (k > 1) eps = prox;
            for (int i = 0; i < k; ++i) B[i] = b[i] - eps * W[i * nu + i] * old[i];
            // Initial active set: the loaded normals, else the most approaching one.
            int loaded = 0;
            for (int i = 0; i < k; ++i) {
                act[i] = 0;
                if (old[i] > 0.0f) {
                    act[i] = 1;
                    loaded += 1;
                }
            }
            if (loaded == 0) {
                int best = -1;
                float bv = 0.0f;
                for (int i = 0; i < k; ++i) {
                    if (B[i] < bv) {
                        bv = B[i];
                        best = i;
                    }
                }
                if (best < 0) {
                    for (int a = 0; a < nu; ++a) x[a] = 0.0f;
                    return 0;
                }
                act[best] = 1;
            }
            for (int f = 0; f < nF; ++f) F[f] = old[k + f];
            // The ring's current load, on whose disks its anchors' current friction lies, and its normals' current
            // velocities.
            float held = 0.0f;
            for (int i = 0; i < k; ++i) {
                held += old[i];
                float v = b[i];
                for (int c = 0; c < nu; ++c) v += W[i * nu + c] * old[c];
                RV[i] = v;
            }
            float vscale = 1.0e-4f;
            for (int i = 0; i < k; ++i) vscale = ${max}(vscale, ${abs}(B[i]));
            for (int outer = 0; outer < 3 * 8 + 3; ++outer) {
                // Basis: the active normals by decreasing old load; a pivot dependent on the basis leaves the set.
                int na = 0;
                for (int sel = 0; sel < k; ++sel) {
                    int best = -1;
                    float bl = -1.0f;
                    for (int i = 0; i < k; ++i) {
                        if (act[i] == 1) {
                            int used = 0;
                            for (int j = 0; j < na; ++j) {
                                if (bas[j] == i) used = 1;
                            }
                            if (used == 0 && old[i] > bl) {
                                bl = old[i];
                                best = i;
                            }
                        }
                    }
                    if (best < 0) break;
                    float piv = W[best * nu + best] * (1.0f + eps);
                    for (int j = 0; j < na; ++j) {
                        const int bj = bas[j];
                        float v = W[best * nu + bj];
                        for (int t = 0; t < j; ++t) v -= LC[na * 8 + t] * LC[j * 8 + t];
                        v = v / LC[j * 8 + j];
                        LC[na * 8 + j] = v;
                        piv -= v * v;
                    }
                    // A normal that depends on the basis and whose current velocity agrees with the basis normals' to
                    // within 1e-5 of the ring's velocity scale holds its load: the velocities do not fix the split
                    // among such normals, which a proximal step would move by the rounding of their residuals over
                    // eps. Its weight stands in for the proximal term, so the fixed point is the same.
                    float lag = RV[best];
                    for (int j = 0; j < na; ++j) lag -= LC[na * 8 + j] * ZR[j];
                    float extra = 0.0f;
                    if (piv < 1.0e-3f * W[best * nu + best] && ${abs}(lag) <= 1.0e-5f * vscale) {
                        extra = 1.0e6f * W[best * nu + best];
                    }
                    piv += extra;
                    if (piv > 1.0e-7f * W[best * nu + best]) {
                        LC[na * 8 + na] = ${sqrt}(piv);
                        ZR[na] = lag / LC[na * 8 + na];
                        bas[na] = best;
                        EX[na] = extra;
                        na += 1;
                    } else {
                        act[best] = -1;
                    }
                }
                if (na == 0) {
                    for (int i = 0; i < k; ++i) x[i] = 0.0f;
                    for (int f = 0; f < nF; ++f) x[k + f] = 0.0f;
                    int worst = -1;
                    float wv = -1.0e-6f * vscale;
                    for (int i = 0; i < k; ++i) {
                        if (act[i] >= 0 && B[i] < wv) {
                            wv = B[i];
                            worst = i;
                        }
                    }
                    if (worst < 0) break;
                    act[worst] = 1;
                    continue;
                }
                // y = K b_B, z = K 1 and XM = K W_BF with K the inverse of the basis block.
                for (int j = 0; j < na; ++j) TMP[j] = B[bas[j]] - EX[j] * old[bas[j]];
                cholesky_solve(LC, na, TMP, Y);
                for (int j = 0; j < na; ++j) TMP[j] = 1.0f;
                cholesky_solve(LC, na, TMP, Z);
                for (int f = 0; f < nF; ++f) {
                    for (int j = 0; j < na; ++j) TMP[j] = W[bas[j] * nu + k + f];
                    cholesky_solve(LC, na, TMP, TMP2);
                    for (int j = 0; j < na; ++j) XM[j * 4 + f] = TMP2[j];
                }
                if (nF > 0) {
                    float S0 = 0.0f;
                    for (int j = 0; j < na; ++j) S0 -= Y[j];
                    for (int f = 0; f < nF; ++f) {
                        float cs = b[k + f];
                        float gs = 0.0f;
                        for (int j = 0; j < na; ++j) {
                            const float wfb = W[(k + f) * nu + bas[j]];
                            cs -= wfb * Y[j];
                            gs -= wfb * Z[j];
                        }
                        CS[f] = cs;
                        GS[f] = gs;
                        for (int h = 0; h < nF; ++h) {
                            float v = W[(k + f) * nu + k + h];
                            for (int j = 0; j < na; ++j) v -= W[(k + f) * nu + bas[j]] * XM[j * 4 + h];
                            ASC[f * 4 + h] = v;
                        }
                    }
                    int npass = 1;
                    if (A == 2) npass = passes;
                    for (int p = 0; p < npass; ++p) {
                        for (int jj = 0; jj < A; ++jj) {
                            const int f0 = 2 * jj;
                            const int o0 = 2 - f0;
                            float c1 = CS[f0];
                            float c2 = CS[f0 + 1];
                            float s0j = S0;
                            if (A == 2) {
                                c1 += ASC[f0 * 4 + o0] * F[o0] + ASC[f0 * 4 + o0 + 1] * F[o0 + 1];
                                c2 += ASC[(f0 + 1) * 4 + o0] * F[o0] + ASC[(f0 + 1) * 4 + o0 + 1] * F[o0 + 1];
                                s0j += GS[o0] * F[o0] + GS[o0 + 1] * F[o0 + 1];
                            }
                            float muj = mu0;
                            if (jj == 1) muj = mu1;
                            const ContactUnitFriction r = friction(ASC[f0 * 4 + f0], ASC[f0 * 4 + f0 + 1],
                                                                   ASC[(f0 + 1) * 4 + f0 + 1], c1, c2, s0j, GS[f0],
                                                                   GS[f0 + 1], muj, alpha[jj], cap, 0.0f, F[f0],
                                                                   F[f0 + 1], held, 0);
                            evaluations += r.evaluations;
                            if (r.mode == 2 || r.mode == 4) alpha[jj] = r.alpha;
                            F[f0] = r.friction0;
                            F[f0 + 1] = r.friction1;
                        }
                    }
                }
                // x_B = -(y + XM f); a negative normal leaves the basis and is not re-added in this visit.
                int neg = -1;
                float nv = 0.0f;
                float xmax = 0.0f;
                for (int j = 0; j < na; ++j) {
                    float v = -Y[j];
                    for (int f = 0; f < nF; ++f) v -= XM[j * 4 + f] * F[f];
                    XB[j] = v;
                    xmax = ${max}(xmax, ${abs}(v));
                }
                for (int j = 0; j < na; ++j) {
                    if (XB[j] < nv && XB[j] < -1.0e-6f * xmax) {
                        nv = XB[j];
                        neg = j;
                    }
                }
                for (int i = 0; i < k; ++i) x[i] = 0.0f;
                for (int j = 0; j < na; ++j) x[bas[j]] = ${max}(XB[j], 0.0f);
                for (int f = 0; f < nF; ++f) x[k + f] = F[f];
                if (neg >= 0) {
                    // A basis that loses a normal was wrong: the next one solves its anchors from their current
                    // friction, which a solution keeps.
                    act[bas[neg]] = -1;
                    for (int f = 0; f < nF; ++f) F[f] = old[k + f];
                    continue;
                }
                // An inactive normal must not approach.
                int worst = -1;
                float wv = -1.0e-6f * vscale;
                for (int i = 0; i < k; ++i) {
                    if (act[i] == 0) {
                        float v = B[i];
                        for (int j = 0; j < na; ++j) v += W[i * nu + bas[j]] * XB[j];
                        for (int f = 0; f < nF; ++f) v += W[i * nu + k + f] * F[f];
                        if (v < wv) {
                            wv = v;
                            worst = i;
                        }
                    }
                }
                if (worst >= 0) {
                    act[worst] = 1;
                    continue;
                }
                break;
            }
            // Exact feasibility: each anchor's friction inside the disk of the final load.
            if (nF > 0) {
                float S = 0.0f;
                for (int i = 0; i < k; ++i) S += x[i];
                for (int jj = 0; jj < A; ++jj) {
                    float muj = mu0;
                    if (jj == 1) muj = mu1;
                    const float R = muj * S;
                    const float fx = x[k + 2 * jj];
                    const float fy = x[k + 2 * jj + 1];
                    const float nf = ${sqrt}(fx * fx + fy * fy);
                    if (nf > R) {
                        float sc = 0.0f;
                        if (nf > 0.0f) sc = R / nf;
                        x[k + 2 * jj] = fx * sc;
                        x[k + 2 * jj + 1] = fy * sc;
                    }
                }
            }
            return evaluations;
        }
    };"""


def get_unit_source(*, cpu: bool = False) -> str:
    """Bind backend intrinsics around the one contact-unit law.

    Args:
        cpu: Bind Warp's host and device helpers, for ``wp.func_native`` snippets compiled by Warp on any device,
            instead of CUDA intrinsics with the slide root and the ring routine out of line.
    """
    if cpu:
        bindings = {
            "inline": "CUDA_CALLABLE",
            "noinline": "CUDA_CALLABLE",
            "sqrt": "wp::sqrt",
            "max": "wp::max",
            "min": "wp::min",
            "abs": "wp::abs",
        }
    else:
        bindings = {
            "inline": "__device__ __forceinline__",
            "noinline": "__device__ __noinline__",
            "sqrt": "sqrtf",
            "max": "fmaxf",
            "min": "fminf",
            "abs": "fabsf",
        }
    # Float literals for the C++ text (``8.0f``, ``1e-06f``), whatever the constants' Python types.
    return substitute(
        _UNIT,
        bindings,
        gain=repr(float(CONTACT_UNIT_SPLIT_GAIN)),
        hold=f"{float(CONTACT_UNIT_SPLIT_HOLD):g}",
        hold2=f"{float(CONTACT_UNIT_SPLIT_HOLD) ** 2:.6g}",
    )


CONTACT_UNIT = DevicePrelude("contact_unit", get_unit_source(), get_unit_source(cpu=True))
"""The contact-unit law, ``ContactUnit::solve``, ``latch`` and ``ring``, for native sweeps, with its CPU twin."""

_CPU_UNIT = get_unit_source(cpu=True)

ContactUnitPoint = wp.types.vector(length=5, dtype=float)
"""A point or anchor unit's solution: normal impulse, the two friction impulses, slide root (-1 unless sliding) and
root evaluations."""


@wp.func_native(
    _CPU_UNIT
    + r"""
    const ContactUnitSolution s = ContactUnit::solve(w_nn, w_t0, w_t01, w_t1, w_nt0, w_nt1, b_n, b_t0, b_t1, mu,
                                                     ring_load, alpha, cap, x_n, x_t0, x_t1);
    wp::vec_t<5, wp::float32> out;
    out.c[0] = s.normal;
    out.c[1] = s.friction0;
    out.c[2] = s.friction1;
    out.c[3] = s.alpha;
    out.c[4] = static_cast<float>(s.evaluations);
    return out;
"""
)
def contact_unit_solve(
    w_nn: float,
    w_t0: float,
    w_t01: float,
    w_t1: float,
    w_nt0: float,
    w_nt1: float,
    b_n: float,
    b_t0: float,
    b_t1: float,
    mu: float,
    ring_load: float,
    alpha: float,
    cap: int,
    x_n: float,
    x_t0: float,
    x_t1: float,
) -> ContactUnitPoint:
    """Solve one point contact, or one patch anchor with its ring's other load held, against its 3x3 block.

    The block is ``[[w_nn, w_nt0, w_nt1], [w_nt0, w_t0, w_t01], [w_nt1, w_t01, w_t1]]``; ``b`` is the residual
    of the unit's rows with its own impulses removed and ``x`` its current impulses. Take-off keeps an anchor's
    friction on the disk of radius ``mu * ring_load``; ``alpha`` warm-starts the slide root when it lies in the
    bracket. The current impulses are returned whenever they solve the unit's problem, also where that problem has
    several solutions: an unloaded unit takes off first where take-off solves its problem, a loaded one solves its
    loaded branches first and keeps the nearer (in the W-norm) of that solution and take-off, and a current friction
    on its disk's boundary first resumes its own slide root (a held slide whose gain asks for it takes the split
    step, which keeps it as well). An anchor whose take-off fails where its friction must press its normal to carry
    load takes the press slide of the larger load.

    Returns:
        The normal impulse, the two friction impulses, the slide root (also where its normal came out negative and
        the unit takes off, for the next visit to continue; -1 for take-off and stick) and the root evaluations.
    """
    ...


@wp.func_native(
    _CPU_UNIT
    + r"""
    const ContactUnitLatch l = ContactUnit::latch(factor, previous, old_load, new_load);
    return wp::vec3(l.weight, l.factor, l.change);
"""
)
def contact_unit_latch(factor: float, previous: float, old_load: float, new_load: float) -> wp.vec3:
    """Update a unit's alternation latch for a visit that moves its total normal load from ``old_load`` to ``new_load``.

    Args:
        factor: The unit's relaxation factor, 0 while unlatched.
        previous: The unit's last applied load change.
        old_load: The unit's total normal impulse before the visit.
        new_load: The total normal impulse of the exact local solution.

    Returns:
        The weight to apply, ``x = old + weight (x* - old)``, the new factor and the applied load change.
    """
    ...


@wp.func_native(
    _CPU_UNIT
    + f"""
    float* data = block.data + base;
    return ContactUnit::ring(data + {RING_BLOCK}, data + {RING_RESIDUAL}, data + {RING_IMPULSE}, data + {RING_SOLUTION},
                             data + {RING_ALPHA}, normals, anchors, mu0, mu1, passes, cap, prox);
"""
)
def contact_unit_ring(
    block: wp.array[float],
    base: int,
    normals: int,
    anchors: int,
    mu0: float,
    mu1: float,
    passes: int,
    cap: int,
    prox: float,
) -> int:
    """Solve one ring unit held in ``block`` at ``base``.

    The layout is :data:`RING_BLOCK` (its row-major Delassus block, at most 12 x 12, rows ordered normals, then
    each anchor's pair), :data:`RING_RESIDUAL` (residuals with its own impulses removed), :data:`RING_IMPULSE`
    (its impulses), :data:`RING_SOLUTION` (the exact solution, written) and :data:`RING_ALPHA` (the two anchors'
    slide roots of this solve, read and written).

    Returns:
        The slide-root evaluations of the visit.
    """
    ...
