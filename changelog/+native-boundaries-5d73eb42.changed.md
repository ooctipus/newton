Separate experimental MuJoCo population records from operations. Replace
`MuJoCoWorlds(...)` with `mujoco_worlds_prepare(...)` and owner methods with
the `mujoco_worlds_*` free functions in `newton.solvers`. Native work layouts,
defaults and contact invalidation now come from MuJoCo Warp; Newton owns
storage admission and graph binding. Consume `SolverMuJoCo.model_mapping`
instead of reconstructing private native indices in downstream tasks.
