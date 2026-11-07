Add an experimental default-off `NEWTON_HEIGHTFIELD_GEOMETRIC_CULL=1` mode that
rejects separated, recognized cuboid/heightfield triangle queries before the
existing compact append. It requires finite queries, cell rejection and stock
nonpredictive global contact reduction; unsupported cases retain the original
path. Current geometry and detection thresholds remain live.
