Add experimental, default-off `NEWTON_REDUCER_EXPORT_CLEANUP=1` to retire consumed
reducer entries during stock buffered contact export and reset counters at the
next collision. Preserve contact selection, capacity checks and the existing
lifecycle for unsupported routes; standalone export remains nondestructive by
default.
