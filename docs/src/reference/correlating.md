```@meta
CurrentModule = AutoRIFT
```

# Correlating

`autorift` is the entry point: two images in, a grid of sub-pixel displacements out. The remaining
names here are for reusing the work `autorift` does internally — the plans, buffers and FFT
workspace — across many pairs of the same size.

FFTW plans are measured on first use and persisted, so a process that finds FFTW wisdom on disk starts
at full speed and one that does not pays plan measurement first. That is invisible on a workstation,
where the file outlives every process, and costly in a deployment that discards its filesystem between
jobs: see [`AutoRIFT.precompute_fftw_wisdom`](@ref) for populating it ahead of the first job and
[`AutoRIFT.fftw_wisdom_path`](@ref) for where it goes and how to redirect it.

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
AutoRIFT.precompute_fftw_wisdom
AutoRIFT.fftw_wisdom_path
```
