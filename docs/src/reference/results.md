```@meta
CurrentModule = AutoRIFT
```

# Results

`autorift` on plain arrays returns a `MultichipResult`: the displacement layers, the correlation
strength, and which chip size answered at each point. The rest of this page is the machinery behind
it, exposed for callers who drive a single scale or resample a field themselves.

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
