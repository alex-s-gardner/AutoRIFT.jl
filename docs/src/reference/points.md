```@meta
CurrentModule = AutoRIFT
```

# Points and grids

A `PointSet` gives the points where a pass measures and, for each point, the chip size, the search
radius and the displacement to search around. Build one to vary those fields across the scene;
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
