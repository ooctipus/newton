Add an experimental, default-off `NEWTON_HEIGHTFIELD_CYLINDER_SPLIT=1` option
that separates admitted heightfield/regular-cylinder queries into MPR, queued
GJK and manifold/write passes while retaining the existing contact math and
fallbacks. This requires `NEWTON_HEIGHTFIELD_PRIMITIVE_QUERY=1` and allocates
44 bytes of scratch storage per triangle capacity.
