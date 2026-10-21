# Geospatial

Loading `Rasters` adds an `autorift` method that reads the pixel size and coordinate reference system
from the inputs and returns map-oriented feature motion rather than pixel offsets. Loading
`DimensionalData` alone adds the same dimension handling without a CRS.

The core `autorift` is documented on [Correlating](correlating.md); this page documents the methods
these packages add. Each section resolves its names in the extension module that defines them.

## Rasters

```@meta
CurrentModule = AutoRIFTRastersExt
```

```@docs
AutoRIFTRastersExt
AutoRIFT.autorift(::Rasters.AbstractRaster, ::Rasters.AbstractRaster)
```

## DimensionalData

```@meta
CurrentModule = AutoRIFTDimensionalDataExt
```

```@docs
AutoRIFTDimensionalDataExt
AutoRIFT.autorift(::AbstractDimArray, ::AbstractDimArray)
```
