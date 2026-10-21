```@meta
CurrentModule = AutoRIFT
```

# Correlating

`autorift` takes two images and returns a grid of sub-pixel displacements. The other names here reuse
the plans, buffers and FFT workspace across many pairs of the same size.

FFTW plans are measured on first use and saved to a wisdom file. A process that finds the file starts
at full speed; one that does not measures its plans first. On a workstation the file persists between
processes. In a deployment that discards its filesystem between jobs, populate it ahead of the first
job with [`AutoRIFT.precompute_fftw_wisdom`](@ref); [`AutoRIFT.fftw_wisdom_path`](@ref) gives its
location and how to redirect it.

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
