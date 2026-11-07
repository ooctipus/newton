Add experimental `NEWTON_NARROW_PHASE_CONVEX_CELLS=1` support candidates for
validated immutable convex hulls on the coherent rejection path. Unsupported
queries retain original support scans; invalidate cells before geometry edits
and rebuild the collision pipeline and captured graphs to enable them again.
