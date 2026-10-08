```@meta
CurrentModule = AutoRIFT
```

# Points and grids

A `PointSet` says where a pass measures and, per point, how large a chip to use, how far to search,
and what displacement to search around. Building one is how you vary those fields across the scene —
`autorift`'s scalar keywords cannot.

```@docs
PointSet
pointset
gridpoints
npoints
nsearchable
issearchable
scatter
rebuild
sanitize!
chip_bounds
search_bounds
```
