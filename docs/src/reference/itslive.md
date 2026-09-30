```@meta
CurrentModule = AutoRIFT
```

# ITS_LIVE packaging

`write` packages autoRIFT output into an ITS_LIVE product netCDF from one `ItsLiveInput`, computing
velocity, stable-shift correction, error estimates, and (for a radar pair) the conversion matrix in a
single pass. Requires `using NCDatasets`; call as `AutoRIFT.write` since the name is not exported, to
avoid shadowing `Base.write`.

```@docs
ItsLiveInput
write
```

## Building an `ItsLiveInput`

Each field of `ItsLiveInput` names an external input `write` does not compute — see `ItsLiveInput`'s
own docstring for which package or caller responsibility supplies it.

```@docs
GeogridCoefficients
ReferenceVelocity
ItsLiveGeoref
ImagePairInfo
SwathOffsetBias
```

### From `ImagePairGeometry`

Defined when `ImagePairGeometry` is loaded, converting its `PairGeometry`/`GeometryInputs` into the
types above.

```@docs
coefficients
reference_velocity
image_location
```

### From a `Rasters.Raster`

Defined when `Rasters`/`ArchGDAL` are loaded.

```@docs
cf_grid_mapping
```
