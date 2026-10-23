Deprecate `SolverFeatherPGS.reset_diagnostic_logs()`, which only clears the private `pgs_debug` logs and has no caller in Newton; it now warns and will be removed in a future release.
