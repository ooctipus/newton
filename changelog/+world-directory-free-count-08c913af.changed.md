Rename experimental `WorldDirectoryData.ready_count` to `free_count` to
distinguish free admission slots from physically backed rows. Update directory
field access to `free_count`; mechanical `RowStorage.ready_count` is unchanged
and no compatibility alias is provided.
