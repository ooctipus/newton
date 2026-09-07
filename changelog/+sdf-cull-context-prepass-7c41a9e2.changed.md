Prepared the mesh-SDF cull contexts in a prepass kernel (one thread per active pair) so the cull kernel loads each context with one read instead of resolving it on a single thread per context.
