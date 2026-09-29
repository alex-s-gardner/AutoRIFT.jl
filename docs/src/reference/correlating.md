```@meta
CurrentModule = AutoRIFT
```

# Correlating

`autorift` is the entry point: two images in, a grid of sub-pixel displacements out. The remaining
names here are for reusing the work `autorift` does internally — the plans, buffers and FFT
workspace — across many pairs of the same size.

```@docs
AutoRIFT
autorift
autorift!
AutoRIFT.init
reinit!
Cache
imagepair
autorift_with_grid
ondisk
halo
block_size_for
warm_plans!
```
