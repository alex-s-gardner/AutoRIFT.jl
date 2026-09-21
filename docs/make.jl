# `checkdocs = :public` resolves public names through `Base.ispublic`, which exists from Julia 1.11.
# Below that it sees only the exported names, so the missing-docs check passes while covering a third
# of the API. Build on 1.11 or later.
VERSION >= v"1.11" ||
    error("the docs build needs Julia 1.11 or later: `checkdocs = :public` silently degrades to \
           `:exports` without `Base.ispublic`")

using Documenter
using DocumenterVitepress
using AutoRIFT

# The extensions' own triggers. Loading them is what makes the dimensional and geospatial `autorift`
# methods, and their docstrings, exist at all.
using ArchGDAL, DimensionalData, DiskArrays, Rasters
using CairoMakie, Dates, Statistics

# An extension whose trigger is missing yields a reference page with no entries and no error, so
# resolve every one of them up front and fail on the first absence.
function extension(name::Symbol)
    m = Base.get_extension(AutoRIFT, name)
    m === nothing && error("extension $name did not load: one of its trigger packages is missing \
                            from the docs environment, and its docstrings would be dropped silently")
    return m
end

# Docstrings attached to a method defined in an extension are owned by that extension's module, and
# `makedocs` renders only docstrings whose module it was given. A `@meta` block's `CurrentModule`
# resolves in `Main`, so binding the names here is also what lets a reference page switch into one.
const AutoRIFTDimensionalDataExt = extension(:AutoRIFTDimensionalDataExt)
const AutoRIFTRastersExt = extension(:AutoRIFTRastersExt)
const MODULES = [AutoRIFT, AutoRIFTDimensionalDataExt, AutoRIFTRastersExt]

# Every jldoctest in the package writes its calls as `AutoRIFT.name`, so bringing the module itself
# into scope is all any of them need.
DocMeta.setdocmeta!(AutoRIFT, :DocTestSetup, :(using AutoRIFT); recursive = true)

const REFERENCE = ["Overview" => "reference/index.md",
                   "Correlating" => "reference/correlating.md",
                   "Parameters" => "reference/parameters.md",
                   "Methods" => "reference/methods.md",
                   "Points and grids" => "reference/points.md",
                   "Preprocessing" => "reference/preprocessing.md",
                   "Results" => "reference/results.md",
                   "First guess" => "reference/first-guess.md",
                   "Geospatial" => "reference/geospatial.md"]

# No scheme: `MarkdownVitepress` prepends `https://` when it writes the edit-link pattern into
# `config.mts`, so a scheme here yields `https://https://github.com/...` and every edit link 404s.
const REPO = "github.com/alex-s-gardner/AutoRIFT.jl"

makedocs(;
         sitename = "AutoRIFT.jl",
         authors = "Alex S. Gardner and contributors",
         modules = MODULES,
         repo = Remotes.GitHub("alex-s-gardner", "AutoRIFT.jl"),
         format = DocumenterVitepress.MarkdownVitepress(; repo = REPO),
         # `:public` covers the `public` declaration as well as the exports; `:all` would demand an
         # entry for every internal docstring, including the extensions' private helpers.
         checkdocs = :public,
         pages = ["Home" => "index.md",
                  "Getting started" => "getting-started.md",
                  "Tutorials" => ["A guided walkthrough" => "tutorials/walkthrough.md",
                                  "Geospatial data" => "tutorials/geospatial.md"],
                  "How-to" => ["Choosing chip size, spacing, and radius" => "howto/chip-sizes.md",
                               "Judging a result" => "howto/quality.md",
                               "Filtering outliers" => "howto/outliers.md",
                               "Masking invalid pixels" => "howto/masks.md",
                               "Choosing a preprocessing filter" => "howto/preprocessing.md",
                               "Giving the search a first guess" => "howto/first-guess.md",
                               "Correlating many pairs" => "howto/batch.md",
                               "Plotting a result" => "howto/plotting.md",
                               "Scenes larger than memory" => "howto/large-scenes.md",
                               "Running on a GPU" => "howto/gpu.md"],
                  "Concepts" => ["How feature tracking works" =>
                                     "explanation/feature-tracking.md",
                                 "Conventions" => "explanation/conventions.md",
                                 "Multiple chip sizes" => "explanation/multi-chip-size.md",
                                 "Memory" => "explanation/memory.md",
                                 "Performance" => "explanation/performance.md"],
                  "API reference" => REFERENCE])

DocumenterVitepress.deploydocs(; repo = "github.com/alex-s-gardner/AutoRIFT.jl",
                               target = joinpath(@__DIR__, "build"),
                               branch = "gh-pages", devbranch = "main", push_preview = true)
