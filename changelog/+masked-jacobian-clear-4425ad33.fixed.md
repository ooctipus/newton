Fix `newton.eval_jacobian` with a `mask` raising `NotImplementedError` when a CPU graph capture, or a CUDA graph capture with `apic=True`, records a call into a non-contiguous `J`.
