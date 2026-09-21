# Geospatial

Load `Rasters` and `autorift` gains a raster method: it reads the pixel size and the coordinate
reference system from the inputs, and returns map-oriented feature motion rather than pixel offsets.
Load `DimensionalData` alone for the same dimension handling without a CRS.

Signatures are given here because the core's `autorift` is documented on
[Correlating](correlating.md); these are the methods those packages add. Each section resolves its
names in the extension module that defines them, where the packages it triggers on are in scope.

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
