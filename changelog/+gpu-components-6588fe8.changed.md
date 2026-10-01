Extract numeric instance directories, typed field storage, CUDA memory backing,
and graph execution into the independent `gpu-components` package. Replace
`newton.worlds` and generic Newton utility imports with passive records from
`gpu_components.*_data` and explicit operations from `gpu_components.directory`,
`fields`, `backing`, and `graph`; keep native physics composition in
`newton.solvers.MuJoCoWorlds`.
