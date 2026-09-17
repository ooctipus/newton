# One row-bank mapping correction — 2026-09-17

Default-off experiment, not an accepted gain. This preserves the original
row-owned packet prototype at `56608c664fb1f10161743b9b09865f29d5f9fb26`.
The accepted comparator remains the corrected-limit owner, not that prototype.

## Why this one correction

Initial packets lost whole physics: RTX12.509309075 →13.346259925 ms;
GB14.415478075 →15.731324800 ms. All original source/idle/capacity/budget
guards and independent audits pass. Strict node attribution separates48
physics/12 auxiliary roots with zero unproven nodes. The small fused owner
costs3.583258750/4.317253083 ms, while complete row production retires only
about1.09/1.18 ms. The local formation cost cannot be called GS-only.

Compiled PTX/SASS retains the dependent shared-J load inside the triangular
WJ loop. With row stride44, equal-node accesses use only8 of32 banks: up to
four distinct addresses per bank. This establishes an address-pattern issue,
not a measured stall fraction; partial masks and mixed templates matter.

This single correction uses the exact physical width43. All105 tile allocation,
bound and row-address sites change44→43; arithmetic order, Gram registers,
prefix energy gate, contact law, publication/fallback and final dead-tile du
reuse remain unchanged. No new launch, matrix or copy is added. No stride grid
or further mapping correction is funded.

A1 ms whole saving would require recovering1.837 ms RTX /2.316 ms GB from the
losing prototype, more than its entire added small-owner cost versus compact.
The bank correction is a short causal test, not a credible promise of that
milestone. Recovery versus a losing candidate is not itself accepted progress.

## Qualification

Regression first fails on the44-wide source. Five CPU controls then pass;
the generated source is exactly the frozen original after only the105 checked
address substitutions. Native source SHA256:
`fe7539cbf85a5e111773b4b50101425179ccd392e91cd831892aed2262eac985`.
Test SHA256:
`28027e1714679499cb673a4051f0eeb8441e9daf671776467e43886ac3a38724`.

Hidden-device AOT `/tmp/fpgs-register-packets-width43-offline-MvplCiWc/offline01`
passes exact source guards, both SM120/103:118/112 registers unchanged,
5632 shared bytes (128 fewer), zero stack/spills/local memory. Text is
82048/82304 bytes. Six existing native groups pass on each card; complete paired
timing uses the original drivers, with unchanged physical tolerances and budgets.
Full precommit also reordered one observer import; observation logic is unchanged.
No new performance result is claimed in this pre-timing checkpoint.
