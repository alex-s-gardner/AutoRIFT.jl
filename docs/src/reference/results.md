```@meta
CurrentModule = AutoRIFT
```

# Results

`autorift` on plain arrays returns a `MultichipResult`: the displacement layers, the correlation
strength, and the chip size that produced each point. The rest of this page covers the steps behind
it, for callers who run a single chip size or resample a field themselves.

```@docs
MultichipResult
nmeasured
DisplacementField
displacement_field
track
track!
correlate_multichip
chipsize_level
pass_geometry
PassGeometry
```

## Outlier rejection

```@docs
reject_outliers
outlier_filter
```

## Windowing and rescaling

```@docs
window
relax
rescale
```

## Resampling

```@docs
resample
resample!
Nearest
Area
Bilinear
Bicubic
```

## Mask operations

```@docs
dilate_within
small_components
```
