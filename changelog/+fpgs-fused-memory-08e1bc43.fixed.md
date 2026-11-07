Account for streamed row chunks and all live response tiles when selecting FeatherPGS fused dense response kernels, avoiding unnecessary matrix clears on GPUs where the fused working set fits.
