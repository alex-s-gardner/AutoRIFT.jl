# Stage 1 of the A/B: one chip-size level, one similarity measure, no pyramid and no outlier
# filter, on a window of the Jakobshavn pair.
#
# Writes the preprocessed images, the grid, and this side's displacements to a `.npz`-shaped
# JLD-free bundle that `stage1_python.py` reads, so both sides correlate *identical* Float32
# arrays at identical grid coordinates. Anything they disagree about is then the correlator
# rather than the input.
#
#   julia --project=tools/ab tools/ab/stage1_julia.jl [npix] [chip] [radius] [row] [col]
#
# `row`/`col` centre the sub-window on the cached scene. The default is the fast trunk rather
# than the scene centre: the centre is slow, low-contrast ice where the correlation surface has
# no dominant peak, so both implementations pick noise and their disagreement measures the
# imagery rather than either correlator.

using AutoRIFT, Serialization, Statistics
using AutoRIFT: ImagePair, displacement_field, track!, gridpoints, preprocess, imagepair, init
# Qualified: `BenchmarkTools` also exports `params`, and the user's startup file loads it into
# `Main` for interactive sessions.
using AutoRIFT: params

const NPIX = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
const CHIP = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32
const RADIUS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
# Centred on Jakobshavn's trunk, where ITS_LIVE reports 3,000-11,800 m/yr over crevassed ice.
const CROW = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1678
const CCOL = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2495
const OUT = joinpath(@__DIR__, "stage1")

const CACHE = get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests"))

# A raw byte dump per array plus one text manifest, rather than a container format: NPZ is not a
# dependency of this package and the point of the harness is that both sides read the same bytes.
function dump_array(dir, name, A)
    open(joinpath(dir, name * ".bin"), "w") do io
        write(io, A)
    end
    return (name, string(eltype(A)), size(A))
end

function main()
    mkpath(OUT)
    w = deserialize(joinpath(CACHE, "window.jls"))

    # A sub-window about `(CROW, CCOL)`, clamped so it stays inside the cached scene and the
    # comparison never measures either side's edge handling.
    n = NPIX
    r0 = clamp(CROW - n ÷ 2, 1, size(w.reference, 1) - n + 1)
    c0 = clamp(CCOL - n ÷ 2, 1, size(w.reference, 2) - n + 1)
    rows, cols = r0:(r0 + n - 1), c0:(c0 + n - 1)
    ref = Matrix{Float32}(w.reference[rows, cols])
    sec = Matrix{Float32}(w.secondary[rows, cols])

    # `chip_size_max = chip` pins the run to a single level, so stage 1 compares the correlator
    # alone: no pyramid, no coarse pass, no merge.
    p = params(; chip_size = (X = CHIP, Y = CHIP), chip_size_max = (X = CHIP, Y = CHIP),
               grid_spacing = (X = CHIP, Y = CHIP), search_radius = (X = RADIUS, Y = RADIUS),
               preprocess = :highpass, filter_width = 5, subpixel = :pyramid, upsampling = 16)

    # Preprocess through the package's own path, then hand the *filtered* arrays to Python. The
    # filter is compared separately; mixing it in here would confound a correlator difference
    # with a filter difference.
    pair = imagepair(init(ref, sec;
                          chip_size = (X = CHIP, Y = CHIP),
                          chip_size_max = (X = CHIP, Y = CHIP),
                          grid_spacing = (X = CHIP, Y = CHIP),
                          search_radius = (X = RADIUS, Y = RADIUS),
                          preprocess = :highpass, filter_width = 5,
                          subpixel = :pyramid, upsampling = 16))
    fref = Matrix{Float32}(pair.reference)
    fsec = Matrix{Float32}(pair.secondary)

    grid = gridpoints(size(fref), p.grid_spacing;
                      chip_size = p.chip_size_max, search_radius = p.search_radius)
    out = displacement_field(grid)
    track!(out, pair, grid, p)

    manifest = Tuple{String,String,Tuple}[]
    push!(manifest, dump_array(OUT, "filtered_reference", fref))
    push!(manifest, dump_array(OUT, "filtered_secondary", fsec))
    push!(manifest, dump_array(OUT, "raw_reference", ref))
    push!(manifest, dump_array(OUT, "raw_secondary", sec))
    # Grid coordinates as Float64 in this package's 1-based pixel-centre convention. The Python
    # side converts to its own 0-based convention explicitly, and that conversion is the thing
    # under test.
    push!(manifest, dump_array(OUT, "grid_x", Matrix{Float64}(grid.x)))
    push!(manifest, dump_array(OUT, "grid_y", Matrix{Float64}(grid.y)))
    push!(manifest, dump_array(OUT, "julia_dx", Matrix{Float32}(out.dx)))
    push!(manifest, dump_array(OUT, "julia_dy", Matrix{Float32}(out.dy)))
    push!(manifest, dump_array(OUT, "julia_correlation", Matrix{Float32}(out.correlation)))

    open(joinpath(OUT, "manifest.txt"), "w") do io
        println(io, "# name dtype shape")
        for (name, T, sz) in manifest
            println(io, name, " ", T, " ", join(sz, "x"))
        end
        println(io, "chip ", CHIP)
        println(io, "radius ", RADIUS)
        println(io, "npix ", n)
        println(io, "filter_width 5")
        println(io, "upsampling 16")
    end

    ok = count(!isnan, out.dx)
    println("grid           ", size(grid), " = ", length(grid.x), " points")
    println("julia measured ", ok, " (", round(100ok / length(out.dx); digits = 1), "%)")
    if ok > 0
        println("dx range       ", extrema(filter(!isnan, out.dx)))
        println("dy range       ", extrema(filter(!isnan, out.dy)))
    end
    # ITS_LIVE's speed over the same window, as a texture check rather than a comparison: a
    # window it reports as slow is one where neither correlator has a dominant peak to find.
    sp = sqrt.(Float64.(w.reference_vx[rows, cols]) .^ 2 .+
               Float64.(w.reference_vy[rows, cols]) .^ 2)
    good = filter(!isnan, sp)
    println("window         rows ", rows, " cols ", cols)
    isempty(good) || println("ITS_LIVE speed median ", round(median(good); digits = 1),
                             " max ", round(maximum(good); digits = 1), " m/yr")
    println("wrote          ", OUT)
end

main()
