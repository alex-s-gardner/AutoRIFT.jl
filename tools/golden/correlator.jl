# Phase 1: AutoRIFT.jl against the reference correlator, on production imagery.
#
#   julia --project=tools/golden -t 8 tools/golden/correlator.jl S2B_MSIL1C_20200612
#
# A diagnostic, not the gate. The gate is the product comparison, and when that disagrees this says
# whether the correlator or the packaging is responsible — which is worth having before the packaging
# exists, because it is the difference between one question and two.
#
# Both sides are handed the *same* arrays: the filtered pair, grid, priors, per-point search limits
# and chip bounds that `capture.py` took at the reference's own `runAutorift` boundary. So a
# preprocessing difference cannot appear here as a correlator difference, and the comparison is of
# the correlator alone. This is the `tools/ab` stage-2 discipline on production imagery instead of a
# hand-cut window.
#
# Four conventions have to be right, each documented and asserted in `tools/ab/README.md` and each
# capable of producing a plausible-looking wrong answer rather than an obvious failure:
#
#   **Index base.** The reference's grid is 0-based pixel indices; `PointSet` is 1-based. So every
#   coordinate gains 1.
#
#   **The half pixel.** `arImgDisp_s` adds `+0.5` to the grid internally, and `runAutorift` snaps an
#   even chip's grid to `round(x + 0.5) - 0.5` before that. AutoRIFT.jl adds the same `0.5` in
#   `_shift_points`, so it must *not* be applied here — doing it twice moves every search centre.
#
#   **The `Dy` sign.** `arImgDisp_s` converts its answer from matrix-Y to cartesian-Y before
#   returning — "Y from down being positive to up being positive"
#   (`autoRIFT.py:1142-1143`, and again at `:1308-1309` for the unsigned entry point). So the
#   reference's `Dy` is up-positive while AutoRIFT.jl's `dy` is row-positive, and comparing them
#   needs one negation. This is the correlator's own convention and applies whatever `optical_flag`
#   says; the separate `if optical_flag == 0: Dy = -Dy` in `testautoRIFT.py` is a *second* flip that
#   the driver applies to radar afterwards, and it acts on the already-cartesian value.
#
#   The sign is *measured* rather than asserted, the way `tools/ab/bench_figures.jl` does it: all four
#   combinations are scored and the comparison reports which won. A hardcoded flip is only correct
#   while the writer's convention holds and fails silently when it changes, whereas a near-perfect
#   *anti*-correlation is a flipped sign and nothing else.
#
#   **Array layout.** Handled by `xchg`, which puts the element type and both dimensions in the file.
#   These grids are square, where a wrong-convention read is a silent transpose.
#
# Both are compared where both measured. The reference reports a search-window corner for a
# degenerate chip where AutoRIFT.jl reports nothing (`REFERENCE.md`), so a coverage difference is
# expected and is counted rather than hidden — it is the mechanism that can later shift
# `stable_shift`.

include("manifest.jl")
include("reference.jl")
include("intermediate.jl")

using AutoRIFT
using AutoRIFT: PointSet, params
using Printf, Statistics

"""
    pointset_from_capture(k::Capture) -> PointSet{2}

The reference's search grid as a `PointSet`.

Coordinates become 1-based, and a point the reference skipped — search limit zero — keeps a zero
radius, which is how `PointSet` marks a point to skip.

**The grid needs `+1`, not `+0.5`, and this was measured rather than reasoned.** `runAutorift` sets
`xGrid = round(xGrid) + 0.5` before correlating (`autoRIFT.py:890-891`) and `capture.py` dumps the
arrays after that, so the values arriving here are half-integers in 0-based pixel coordinates. The
tempting move is to subtract that half pixel while adding the index base, on the grounds that
`_shift_points` adds AutoRIFT.jl's own `0.5` for the even-chip centroid — but scanning the offset says
otherwise: at `+1.0` exact agreement is **85.0%** with a median residual of 0, and at `+0.5` it is
49.2% with a median of 1/32. The two half pixels do not cancel; they are the same convention counted
once on each side.

Getting this wrong is invisible in a median over the whole scene, because a half-pixel offset produces
no residual under uniform motion and one proportional to the local velocity gradient. Before the fix,
exact agreement fell from 66.5% in the flattest gradient decile to 5.6% in the steepest, and the
difference map showed structure only along the fast-flow margins. Scan the offset before believing any
argument about which convention applies — including this one.
"""
function pointset_from_capture(k::Capture)
    xg = k.arrays["in_xGrid"]
    yg = k.arrays["in_yGrid"]
    srx = k.arrays["in_SearchLimitX"]
    sry = k.arrays["in_SearchLimitY"]
    csmin = k.arrays["in_ChipSizeMinX"]
    csmax = k.arrays["in_ChipSizeMaxX"]
    dx0 = k.arrays["in_Dx0"]
    dy0 = k.arrays["in_Dy0"]

    chip0 = Int(k.scalars["ChipSize0X"])
    scale_y = Float64(k.scalars["ScaleChipSizeY"])

    return PointSet(
        Float64.(xg) .+ 1, Float64.(yg) .+ 1,
        Int.(srx), Int.(sry),
        Float64.(dx0), Float64.(dy0),
        fill(chip0, size(xg)), fill(round(Int, chip0 * scale_y), size(xg)),
        Int.(csmin), Int.(csmax),
    )
end

"""
    subpixel_from_capture(k::Capture) -> Tuple{Vararg{PyramidRefine}}

The subpixel methods the reference used, one per chip-size level, finest first.

`OverSampleRatio` is a per-chip-size dictionary the driver assembles at run time
(`testautoRIFT.py:488-510`) rather than a scalar, and `autoRIFT.py:652` looks it up per level. So the
quantization step is a property of the level: 1/16 px at the base chip size and 1/32 or 1/64 above
it for optical input. A scalar is also handled, because the driver sets one when the chip bounds are
absent.

Entries are ordered by chip size and truncated to the levels this run will use, since a tuple longer
than the level list is a configuration error rather than something to ignore.
"""
function subpixel_from_capture(k::Capture)
    osr = k.scalars["OverSampleRatio"]
    chip0 = Int(k.scalars["ChipSize0X"])
    maxchip = Int(maximum(k.arrays["in_ChipSizeMaxX"]))
    # The levels this run will actually correlate: chip0 * 2^j up to the largest bound present.
    chips = [chip0 << j for j in 0:floor(Int, log2(maxchip / chip0))]

    osr isa Number && return (PyramidRefine(Int(osr)),)
    # JSON object keys arrive as symbols, and the driver keys them by chip size.
    return Tuple(PyramidRefine(Int(osr[Symbol(c)])) for c in chips)
end

"""
    kwargs_from_capture(k::Capture) -> NamedTuple

Correlator settings taken from what the reference used, rather than from what the driver is believed
to set.

`autorift(a, b, ::PointSet)` takes keywords rather than a `Params`, so these are passed through as
such. The per-point fields — coordinates, priors, search radii, chip bounds — travel in the
`PointSet` instead; only the scene-wide settings are here.

`preprocess = :none` because the captured pair is already filtered. Filtering again would compare two
different images.
"""
function kwargs_from_capture(k::Capture)
    chip0 = Int(k.scalars["ChipSize0X"])
    scale_y = Float64(k.scalars["ScaleChipSizeY"])
    spacing = Int(k.scalars["GridSpacingX"])
    maxchip = Int(maximum(k.arrays["in_ChipSizeMaxX"]))

    return (; chip_size = (X = chip0, Y = round(Int, chip0 * scale_y)),
            chip_size_max = (X = maxchip, Y = maxchip),
            grid_spacing = (X = spacing, Y = spacing),
            subpixel = subpixel_from_capture(k),
            preprocess = :none)
end

"""
    compare_correlator(c::GoldenCase; n = 100) -> NamedTuple

Run AutoRIFT.jl on the reference's own captured inputs and diff `dx`/`dy` against its `Dx`/`Dy`.

Returns the per-axis statistics plus the coverage split, since a point one side answered and the
other did not is a different finding from a point they both answered differently.
"""
function compare_correlator(c::GoldenCase; n::Integer = 100)
    k = read_capture(c; n)

    grid = pointset_from_capture(k)
    kw = kwargs_from_capture(k)

    a = k.arrays["in_I1"]
    b = k.arrays["in_I2"]

    # Argument order. `arImgDisp_s(a, b)` cuts its chip from `b` and its window from `a`; the
    # reference calls it as `arImgDisp_s(self.I2, self.I1)`, so `I1` supplies the chip. AutoRIFT.jl's
    # `autorift(reference, secondary)` cuts its chip from `secondary`, so `I1` binds to `secondary`.
    @info "correlating" scene=size(a) npoints=length(grid.x) subpixel=kw.subpixel
    t = @elapsed out = autorift(b, a, grid; kw...)

    rdx = k.arrays["out_Dx"]
    rdy = k.arrays["out_Dy"]

    # The reference's output grid is one point smaller in each direction than its input grid, so the
    # comparison is over the overlap, stated rather than assumed.
    ny = min(size(out.dx, 1), size(rdx, 1))
    nx = min(size(out.dx, 2), size(rdx, 2))
    @info "grid overlap" julia=size(out.dx) reference=size(rdx) compared=(ny, nx)

    stats = map((:dx, :dy)) do axis
        jul = getproperty(out, axis)[1:ny, 1:nx]
        ref = (axis === :dx ? rdx : rdy)[1:ny, 1:nx]
        # Score both signs and keep the better. `dy` needs the flip and `dx` does not, but measuring
        # says so rather than a comment asserting it — and a flip that stops being needed shows up
        # here as a changed `sign` instead of as a silent bias.
        pos = _axis_stats(String(axis), jul, ref, +1)
        neg = _axis_stats(String(axis), jul, ref, -1)
        pos.exact >= neg.exact ? pos : neg
    end

    return (; time = t, overlap = (ny, nx), dx = stats[1], dy = stats[2],
            julia_size = size(out.dx), reference_size = size(rdx), result = out, capture = k)
end

"""
    _axis_stats(name, jul, ref, sign) -> NamedTuple

Compare `jul` against `sign * ref` on one axis.

`NaN` is the correlator's no-measurement marker on both sides, so a point one side answered alone is
counted as coverage rather than folded into the value statistics: the reference reports a
search-window corner for a degenerate chip where AutoRIFT.jl reports nothing (`REFERENCE.md`), and
that difference has a different cause from a disagreement about a measured value.

`correlation` is over both-measured points and is the diagnostic for a sign error specifically — a
near-perfect negative correlation is a flipped sign and nothing else.
"""
function _axis_stats(name, jul, ref, sign::Int)
    both = 0; only_j = 0; only_r = 0; exact = 0
    d = Float64[]
    va = Float64[]
    vb = Float64[]
    for i in eachindex(jul, ref)
        mj = isnan(jul[i]); mr = isnan(ref[i])
        if mj && mr
            continue
        elseif mr
            only_j += 1
        elseif mj
            only_r += 1
        else
            both += 1
            a = Float64(jul[i]); b = sign * Float64(ref[i])
            push!(va, a); push!(vb, b)
            a == b ? (exact += 1) : push!(d, a - b)
        end
    end
    ad = abs.(d)
    return (; name, sign, both, only_julia = only_j, only_reference = only_r, exact,
            exact_fraction = both == 0 ? 1.0 : exact / both,
            max_abs = isempty(ad) ? 0.0 : maximum(ad),
            p99 = isempty(ad) ? 0.0 : quantile(ad, 0.99),
            median = isempty(ad) ? 0.0 : median(ad),
            bias = isempty(d) ? 0.0 : mean(d),
            correlation = length(va) < 2 ? NaN : cor(va, vb))
end

function report(r)
    @printf("\ncorrelated in %.1f s; grid julia %s, reference %s, compared %s\n",
            r.time, r.julia_size, r.reference_size, r.overlap)
    @printf("\n%-5s %5s %9s %8s %8s %8s %9s %9s %9s %8s\n",
            "axis", "sign", "both", "only jl", "only ref", "exact", "median", "p99", "max", "corr")
    for s in (r.dx, r.dy)
        @printf("%-5s %5s %9d %8d %8d %7.2f%% %9.4g %9.4g %9.4g %+8.5f\n",
                s.name, s.sign > 0 ? "+" : "-", s.both, s.only_julia, s.only_reference,
                100 * s.exact_fraction, s.median, s.p99, s.max_abs, s.correlation)
    end
    @printf("\nbias: dx %+.6g, dy %+.6g\n", r.dx.bias, r.dy.bias)
    return nothing
end

function main(args)
    isempty(args) && error("usage: correlator.jl <product-name-fragment> [--run N]")
    c = only(cases(args[1]))
    n = 100
    i = findfirst(==("--run"), args); i === nothing || (n = parse(Int, args[i + 1]))
    report(compare_correlator(c; n))
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
