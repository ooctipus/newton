Add experimental `NEWTON_HEIGHTFIELD_PAIR_CSR=1` grouping of raw heightfield
witnesses with active and nearest-separation priority and at most four contacts
per supported pair. Optional `NEWTON_HEIGHTFIELD_PAIR_CSR_SHELL=1` enables finite
shell witnesses after CSR admission. Both modes require the supported geometric
culling and stock-writer pipeline, retain unsupported-pair fallback, and expose
sticky raw-pool and membership capacity failures.
