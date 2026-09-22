```@meta
CurrentModule = AutoRIFT
```

# Preprocessing

An `ImagePair` holds the two images and their validity masks. `preprocess` applies a
[preprocessing method](methods.md#Preprocessing) to one; the filters below are the same operations
called directly, for inspecting what a choice does to an image.

```@docs
ImagePair
FiniteMask
resident
valid
preprocess
replace_nonfinite
highpass
highpass!
wallis
wallis_gapfill
laplacian
sobel
deramp
ramp_phase
decibel
bytescale
```
