# Early contactless Franka completion experiment

Base: `fb0f075935c4abcfcf8f3932ae615774551a2d95`. This is a default-off
scheduling experiment, not a measured improvement or a claim of 20% savings.
Isaac Lab, capacities, timestep, eight sweeps and current constraint law stay
unchanged. Enable only with the existing local row-packet owner.

## Boundary and cost hypothesis

After current prefix/raw allocation and the original free6 velocity-limit
allocation, admit only original SINGLE9 worlds with zero MF rows and zero
current raw endpoint incidence (including rejected contacts). Run the existing
native9 solve and original integration/FK/body-finalization math on a private
stream while all contact, MF, owner40 and other-articulation work remains late.
Original parallel current force, mass and predictor producers are retained.
No new serial world solver or FK representation is introduced.

The pinned prior 16K node window contains 0.943/0.725 ms of **all-articulation**
publication (RTX/GB), an upper ceiling rather than the selected primary cost.
A duration-preserving counterfactual gives median 184/207 us from the proposed
ready boundary to owner40 completion; original SINGLE9 plus all-art publication
takes 148/133 us. This leaves nominal 36/74 us before incidence, compaction,
events, masked/fixed-grid launch costs and bandwidth contention. Historical
512 snapshots admit 486/482 worlds; this is not current 16K coverage evidence.
The original local9 solve is already hidden by owner40: do not count it as
deleted work. A meaningful first target is >=10% whole RTX improvement with
no GB regression, not the unachieved fixed fourfold target.

## Ownership and validation

Private current count/owner views prevent later queue publication from racing
the early solve. Initialize velocity once before the fork; late impulse clearing
and late SINGLE solve exclude early worlds. Guard compact publication indices
before any source/cache access. All free6 and prescribed articulations stay
late. Early and late qdd, integration, FK and finalizer writes are disjoint;
join the persistent done event before whole-state consumers and return. Reject
in-place/overlapping states and unsupported cache/debug/solver modes.

Validate actual same-input native lambda/velocity/diagonal and original public
q/qd/body pose/twist/cache outputs; test stale prior ownership, raw incidence,
MF and prefix transitions, zero/all/mixed cohorts, partial grids, refresh/reuse,
notification/cold paths, and two captured graphs. Negative controls must expose
late lambda/velocity overwrites. CPU/source tests and offline compilation precede
root-owned actual GPU controls and early complete 512/16K timing. All routing,
publication, fallback, streams and joins are charged. Diagnose a first timing
loss once; no block/lane grid or silent serial-math rewrite.

## Source-ready checkpoint

`FEATHER_PGS_EARLY_FRANKA=1` additionally requires
`FEATHER_PGS_LOCAL_ROW_PACKETS=1` and the original admitted direct-cache law.
The 512 snapshot has 1,536 articulations and 10,752 coordinates. Exact added
device array storage is 67,596 bytes at 512 and 2,162,700 bytes (2.063 MiB) at
16K, including masks/private solve views, valid early/late lists, and a DOF-to-art
map. The model's existing joint-to-art map is reused. Public capacities do not
change; list storage is bounded by actual worlds/articulations, not contacts.

All nine source-adapted publication kernels compile on CPU. Offline NVRTC and
ptxas pass all 20 module/architecture combinations (sm120/sm103), with source
guard, in `/tmp/fpgs-early-franka-ready-Hz08S9do/offline03`. Finalizer forward
uses 40 registers, zero stack/spills; FK retains callable math with a 72-byte
stack and no spills. These resources are not a performance result. Four
portable ownership/source regressions, the completed-event lifecycle regression,
and the existing notification regressions pass. The complete source-ready suite
has 34 CPU passes and four CUDA-only skips (38 registered tests). Independent
CPU checks exercise the actual A-entry/no-sentinel constructor and all 15
publication/cache outputs on four saved inputs. The former completed-event
implementation fails the new lifecycle test when restored in memory; the
original unmasked impulse clearer fails the nonzero early-lambda control.
Full precommit passes. Actual native/graph controls and timing remain pending.

The done event is joined before whole-state publication and `active` is then
cleared, so a new capture does not inherit a prior graph's event. Untimed
admission metadata should inspect the persistent device cohort counts/masks,
not interpret the deliberately cleared host `active` flag as non-admission.
