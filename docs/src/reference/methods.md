```@meta
CurrentModule = AutoRIFT
```

# Methods

Five choices a pass makes, each a small type hierarchy. Pass an instance as the corresponding
`params` keyword; the abstract type is the trait a new method would subtype.

## Similarity

How a chip is compared against a window.

```@docs
SimilarityMeasure
NCC
ZNCC
Coherence
```

## Sub-pixel refinement

How the peak is located between samples.

```@docs
SubpixelMethod
PyramidRefine
NoRefine
```

## Preprocessing

What is done to each image before correlation.

```@docs
PreprocessMethod
NoPreprocess
Highpass
Wallis
WallisGapfill
Laplacian
Sobel
Destripe
Deramp
Decibel
```

## Rotation

Whether chips are correlated at several orientations.

```@docs
RotationMethod
RotationSearch
NoRotationSearch
AutoRIFT.angles
```

## Outlier rejection

Which measured vectors are discarded as inconsistent with their neighbors.

```@docs
OutlierMethod
GardnerFilter
NoOutlierFilter
```
