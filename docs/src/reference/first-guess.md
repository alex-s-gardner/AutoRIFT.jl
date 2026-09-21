```@meta
CurrentModule = AutoRIFT
```

# First guess

Sparse feature matching, run before the dense pass to supply it with a prior. Useful when motion
exceeds the search radius, or when the scene rotates.

The detectors live in extensions: `ORBGuess` needs `ImageFeatures`, `AKAZEGuess` needs
`AkazeFeatures`. `required_package` reports which.

```@docs
FirstGuess
ORBGuess
AKAZEGuess
first_guess
scene_rotation
consistent_matches
required_package
```
