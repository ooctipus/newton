Add an experimental, default-off `NEWTON_HEIGHTFIELD_CELL_REJECT=1` collision mode
that rejects vertically separated heightfield cells before triangle contact
queries, preserving current contact search gaps and existing buffer capacities.
