# The stage ladder: AutoRIFT.jl against the reference, one intermediate at a time.
#
#   CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=0 julia --project=tools/golden \
#       tools/golden/intermediate.jl LC08_L1TP_009011 --force
#   julia --project=tools/golden -t 8 tools/golden/stages.jl LC08_L1TP_009011
#
# **Why one stage at a time.** `autorift()` builds about two dozen intermediates inside its
# chip-size loop (`autoRIFT.py:407-874`) and every one of them is a local, so the function boundaries
# are the only place the two implementations have ever been compared. A comparison there is a
# comparison of the whole composition: when it disagrees, any of the twenty-four steps could be
# responsible, and `tools/golden/README.md` records five successive hypotheses that each survived an
# endpoint check and then turned out to be wrong. What refutes a hypothesis about step 12 is the
# array step 12 produced.
#
# **Every stage is fed the reference's own input.** A stage reads the reference's dumped input, runs
# AutoRIFT.jl's counterpart on exactly those bytes, and is diffed against the reference's dumped
# output. Julia output is never chained into the next Julia stage — that would rebuild the composed
# comparison this exists to take apart, and a difference entering at step 3 would then show up at
# every step after it.
#
# **The gate depends on what the stage produces.** A mask or an integer grid is a *decision*: it is
# exact or it is a semantic difference, and there is no rounding to hide behind. A displacement at the
# base chip size is quantized to the level's upsampling step, so it is also exact. A displacement
# above the base level is not comparable that way at all, because both sides replace the measurement
# with a bicubic resize (`autoRIFT.py:856-866`) and neither field is quantized — there, bias and the
# within-one-step fraction are the statistics and `exact` is meaningless. `tools/golden/README.md`
# records the measurement that closed that question.
#
# Stages run in order and the ladder stops at the first red one, because a stage fed a correct input
# tells you about itself, while the stages after a red one are being asked a question whose premise
# has already failed.

include("manifest.jl")
include("reference.jl")
include("intermediate.jl")
include("correlator.jl")

using AutoRIFT
using AutoRIFT: PointSet, windowmax, windowmean, windowrange, sanitize!, rebuild,
                params, extent
using Printf, Statistics

# ---------------------------------------------------------------------------
# Comparing one stage
# ---------------------------------------------------------------------------

"""
    StageResult

One rung of the ladder: what was compared, how it was judged, and whether it passed.

`detail` carries the numbers behind the verdict so a red rung says *how* it differs rather than only
that it does — a stage off by one quantization step everywhere and a stage transposed both report
"not equal".
"""
struct StageResult
    name::String
    reference::String
    gate::String
    passed::Bool
    n::Int
    detail::String
end

"""
    exact_stage(name, ref_name, jl, ref; nanok = true) -> StageResult

Compare a decision — a mask, a grid, a search radius, a chip size — for exact equality.

`NaN` counts as equal to `NaN`, since it is the no-value marker on both sides rather than a value.
Nothing else is tolerated: these arrays are selected or copied from existing values, so a difference
of any size is a difference in which value was selected.

The failure detail names the *first* disagreeing position and the two values there. A count alone
cannot distinguish a one-column shift from a scattered difference, and the position is what a heatmap
would show.
"""
function exact_stage(name, ref_name, jl, ref)
    if size(jl) != size(ref)
        return StageResult(name, ref_name, "exact", false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    bad = 0
    first_bad = nothing
    for i in eachindex(IndexCartesian(), jl)
        a, b = jl[i], ref[i]
        same = (a isa AbstractFloat && isnan(a) && b isa AbstractFloat && isnan(b)) || a == b
        same && continue
        bad += 1
        first_bad === nothing && (first_bad = (Tuple(i), a, b))
    end
    n = length(ref)
    detail = if bad == 0
        "all $n equal"
    else
        pos, a, b = first_bad
        @sprintf("%d of %d differ (%.4f%%), first at %s: julia %s, reference %s",
                 bad, n, 100 * bad / n, pos, a, b)
    end
    return StageResult(name, ref_name, "exact", bad == 0, n, detail)
end

"""
    quantized_stage(name, ref_name, jl, ref, step) -> StageResult

Compare two displacement fields that are both quantized to `step`.

This is the base chip size, where a residual below one upsampling step is not a disagreement about
position but about which of two adjacent representable values a peak rounded to. `exact` is the
headline and coverage is reported beside it, because a point one side measured and the other did not
is a different finding from a point they both measured differently.

Passing is judged against the pre-existing L8/L9 benchmark rather than against 100%: below a real
correlation peak both implementations are choosing from noise, and no two implementations agree about
that. `tools/ab/README.md` records the benchmark at 77.4% exact and 97.3% within one step.
"""
function quantized_stage(name, ref_name, jl, ref, step; min_exact = 0.774, min_within = 0.973)
    if size(jl) != size(ref)
        return StageResult(name, ref_name, "quantized", false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    both = 0; only_j = 0; only_r = 0; exact = 0; within = 0
    d = Float64[]
    for i in eachindex(jl, ref)
        mj = isnan(jl[i]); mr = isnan(ref[i])
        mj && mr && continue
        if mr
            only_j += 1
        elseif mj
            only_r += 1
        else
            both += 1
            δ = Float64(jl[i]) - Float64(ref[i])
            push!(d, δ)
            δ == 0 && (exact += 1)
            abs(δ) <= step && (within += 1)
        end
    end
    fe = both == 0 ? 0.0 : exact / both
    fw = both == 0 ? 0.0 : within / both
    ad = abs.(d)
    detail = @sprintf("both %d, only jl %d, only ref %d; exact %.2f%%, within one step %.2f%%, median %.4g, p99 %.4g, bias %+.4g",
                      both, only_j, only_r, 100fe, 100fw,
                      isempty(ad) ? 0.0 : median(ad), isempty(ad) ? 0.0 : quantile(ad, 0.99),
                      isempty(d) ? 0.0 : mean(d))
    gate = string("exact>=", round(100min_exact; digits = 1), "% within>=",
                  round(100min_within; digits = 1), "%")
    return StageResult(name, ref_name, gate, fe >= min_exact && fw >= min_within, both, detail)
end

"""
    unquantized_stage(name, ref_name, jl, ref) -> StageResult

Compare two displacement fields that are **not** quantized, which is every level above the base.

Both sides overwrite their own measurements with a bicubic resize of a decimated field
(`autoRIFT.py:811,856-866`; `_undecimate_level`), so neither field's values sit on any quantization
grid and exact agreement is unreachable by construction rather than by defect. Measured: multiples of
1/16, 1/32, 1/64 and 1/128 account for 0.01–0.03% of *either* side's chip-48 values, against 99.4% at
the base level.

So the gate is bias and dispersion. A bias means the two disagree about position systematically, which
is a bug; a symmetric spread is the accumulated Float32 difference between two implementations of a
chain that decimates, median-filters, area-resizes, hole-fills and bicubic-resizes.
"""
function unquantized_stage(name, ref_name, jl, ref; max_bias = 0.01, min_within_tenth = 0.45)
    if size(jl) != size(ref)
        return StageResult(name, ref_name, "unquantized", false, 0,
                           "shape $(size(jl)) against reference $(size(ref))")
    end
    d = Float64[]
    for i in eachindex(jl, ref)
        (isnan(jl[i]) || isnan(ref[i])) && continue
        push!(d, Float64(jl[i]) - Float64(ref[i]))
    end
    isempty(d) && return StageResult(name, ref_name, "unquantized", false, 0,
                                    "no point measured on both sides")
    ad = abs.(d)
    bias = mean(d)
    tenth = count(<=(0.1), ad) / length(ad)
    detail = @sprintf("both %d; bias %+.5f, median %.4g, p95 %.4g, within 0.1 px %.1f%%",
                      length(d), bias, median(ad), quantile(ad, 0.95), 100tenth)
    gate = string("|bias|<=", max_bias, " within0.1>=", round(Int, 100min_within_tenth), "%")
    return StageResult(name, ref_name, gate,
                       abs(bias) <= max_bias && tenth >= min_within_tenth, length(d), detail)
end

# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

# The stage trace records one chip-size level per run, and which one it was is in the keys.
function traced_level(k::Capture)
    levels = Set{Int}()
    for name in keys(k.stages)
        m = match(r"_L(\d+)$", name)
        m === nothing || push!(levels, parse(Int, m.captures[1]))
    end
    isempty(levels) && error(
        "this capture has no stage trace. Re-take it with " *
        "`CAPTURE_STAGES=1 CAPTURE_STAGE_LEVEL=<i> intermediate.jl <case> --force`, " *
        "where `<i>` is the chip-size loop's own 0-based index.")
    length(levels) == 1 || error("stage trace spans levels $(sort!(collect(levels))); one per run")
    return only(levels)
end

"""
    stage(k, name, level) -> Matrix

One traced array, by its bare name. Errors rather than returning `nothing` when it is absent, because
a stage silently skipped is a rung the ladder reports as green.
"""
function stage(k::Capture, name::AbstractString, level::Integer)
    key = "$(name)_L$(level)"
    haskey(k.stages, key) && return k.stages[key]
    # A name in `REDUMP` is written with a revision index instead, and *which* revision a stage
    # consumes is the question — `SearchLimitX0` holds three distinct values under one name. So a
    # bare lookup that finds only revisions is an error naming them rather than a guess at one.
    revs = sort!([kk for kk in keys(k.stages) if occursin(Regex("^$(name)_rev\\d+_L$(level)\$"), kk)])
    isempty(revs) && error("no stage `$key` in this capture; traced: " *
                           join(sort!(collect(keys(k.stages))), ", "))
    error("`$name` is mutated in place, so it has revisions rather than one value: " *
          join(revs, ", ") * ". Ask for the revision the consuming stage reads.")
end

stage_rev(k::Capture, name::AbstractString, rev::Integer, level::Integer) =
    k.stages["$(name)_rev$(rev)_L$(level)"]

"""
    ladder(c::GoldenCase; n = 100) -> Vector{StageResult}

Walk the stages of one chip-size level in order, stopping at the first that disagrees.

The level is whichever one the capture traced. Level 0 is the base chip size, where the grid passes
through unresized and the interesting stages are the search-limit rewrite onward; a level above it
additionally exercises the grid resize, the mask dilation and the radius widening.
"""
function ladder(c::GoldenCase; n::Integer = 100, stop_on_red::Bool = true)
    k = read_capture(c; n)
    L = traced_level(k)
    chip0 = Int(k.scalars["ChipSize0X"])
    chip = chip0 << L
    spacing = Int(k.scalars["GridSpacingX"])
    minsearch = Int(k.scalars["minSearch"])
    @info "stage ladder" case=c.product level=L chip spacing minsearch

    out = StageResult[]
    push!(out, rung_grid(k, L, chip, chip0))
    push!(out, rung_search_rewrite(k, L, minsearch))
    push!(out, rung_priors(k, L, chip, chip0, spacing))
    append!(out, rungs_coarse_sampling(k, L, chip))

    if stop_on_red
        i = findfirst(r -> !r.passed, out)
        i === nothing || (out = out[1:i])
    end
    return out
end

# ---------------------------------------------------------------------------
# 3.1 — the level's grid
# ---------------------------------------------------------------------------
#
# At the base chip size `xGrid0` is `self.xGrid` copied (`autoRIFT.py:587-593`), so this rung asserts
# the capture is self-consistent: the traced grid must be the one `runAutorift` was handed. That is
# worth a rung of its own rather than an assumption, because a capture taken before `runAutorift`
# rewrote the grid holds an integer grid rather than `round(x) + 0.5`, and comparing against it moves
# every search centre half a pixel — a residual that is zero under uniform motion and grows with the
# velocity gradient, so it hides in every summary statistic.
#
# Above the base size the reference resizes with `INTER_AREA` and snaps an even chip's grid to
# `round(x + 0.5) - 0.5` (`autoRIFT.py:509-530`), where AutoRIFT.jl decimates by taking every
# `stride`-th point and shifting to the cell centre (`_decimate_level`, `_cell_centres`). Those two
# reach the same place by different routes, so the rung compares the positions rather than the method.
function rung_grid(k::Capture, L::Int, chip::Int, chip0::Int)
    xg0 = stage(k, "xGrid0", L)
    if L == 0
        return exact_stage("3.1 grid, base level", "xGrid0", k.arrays["in_xGrid"], xg0)
    end
    stride = chip ÷ chip0
    full = pointset_from_capture(k)
    sub = AutoRIFT._decimate_level(full, trues(size(full)), stride)
    sub === nothing && return StageResult("3.1 grid, level $L", "xGrid0", "exact", false, 0,
                                          "AutoRIFT.jl decimated to nothing at stride $stride")
    # The reference's grid is 0-based and AutoRIFT.jl's 1-based, so the comparison subtracts the
    # index base rather than comparing raw coordinates. `pointset_from_capture` added it.
    return exact_stage("3.1 grid, level $L", "xGrid0",
                       Float32.(sub.grid.x .- 1), xg0)
end

# ---------------------------------------------------------------------------
# 3.5 — the search-limit rewrite
# ---------------------------------------------------------------------------
#
# `autoRIFT.py:596-602` rewrites the limits at the top of every level, and it is the rewritten array
# the correlator receives:
#
#     idxZero = (SearchLimitX0 <= 0) | (SearchLimitY0 <= 0)
#     SearchLimitX0[idxZero] = 0; SearchLimitY0[idxZero] = 0
#     SearchLimitX0[~idxZero & (SearchLimitX0 < minSearch)] = minSearch
#
# Two rules, each of which changes the window searched: either axis zero zeroes both, and a nonzero
# radius below `minSearch` is raised to it. `sanitize!` is AutoRIFT.jl's form of the same rewrite.
#
# The rung compares `rev1` against `rev0` — the value after the rewrite against the value before —
# because that pair isolates the rewrite from whatever produced its input. Reading `rev0` as the
# correlator's radius is the error this rung exists to prevent: it costs 944,036 points on this case,
# by up to 23 pixels, and biases the comparison opposite to its cause, since the rewritten points are
# the small-radius ones.
function rung_search_rewrite(k::Capture, L::Int, minsearch::Int)
    before_x = stage_rev(k, "SearchLimitX0", 0, L)
    before_y = stage_rev(k, "SearchLimitY0", 0, L)
    after_x = stage_rev(k, "SearchLimitX0", 1, L)

    pts = rebuild(pointset_from_capture(k);
                  radius_x = Int.(before_x), radius_y = Int.(before_y))
    sanitize!(pts, minsearch)
    return exact_stage("3.5 search-limit rewrite", "SearchLimitX0_rev1",
                       Float32.(pts.radius_x), after_x)
end

# ---------------------------------------------------------------------------
# 3.3 / 3.4 — a decimated cell's radius and prior
# ---------------------------------------------------------------------------
#
# A decimated point stands for its whole cell, so its window must cover every fine point in it: both
# the widest radius any of them asked for and the spread of their priors, since two fine points with
# different priors search around different centres. The reference adds exactly those two terms over
# `1 / Scale` cells (`autoRIFT.py:540-586`):
#
#     SearchLimitX0 = colfilt(SearchLimitX, (1/Scale, 1/Scale), 0)   # max
#                   + colfilt(Dx0,          (1/Scale, 1/Scale), 4)   # range
#     Dx00          = colfilt(Dx0,          (1/Scale, 1/Scale), 2)   # mean
#
# At the base chip size there is no decimation and the three are copies, so this rung asserts that —
# a level that widened its own base radius would be searching a window the reference does not.
function rung_priors(k::Capture, L::Int, chip::Int, chip0::Int, spacing::Int)
    dx00 = stage(k, "Dx00", L)
    if L == 0
        return exact_stage("3.4 prior, base level", "Dx00", k.arrays["in_Dx0"], dx00)
    end
    stride = chip ÷ chip0
    mx = windowmean(k.arrays["in_Dx0"], stride)
    # The reference resizes the reduced field with `INTER_NEAREST` and rounds
    # (`autoRIFT.py:585-586`), so the comparison is on the reduced values at the decimated nodes.
    rows = 1:stride:size(mx, 1)
    cols = 1:stride:size(mx, 2)
    return exact_stage("3.4 prior, level $L", "Dx00",
                       Float32.(round.([mx[i, j] for i in rows, j in cols])), dx00)
end

# ---------------------------------------------------------------------------
# 3.6 — the coarse pass's sample lattice, its radii, and its priors
# ---------------------------------------------------------------------------
#
# The coarse pass samples every `sparseSearchSampleRate * ratio`-th point of the level's grid
# (`autoRIFT.py:603-614`), and each sampled point stands in for the cell around it — so its radius is a
# maximum over a window rather than its own value, and its prior is the cell's mean.
#
# **The window is one wider than the step when the step is even.** `filtWidth = stride + 1` for an even
# stride and `stride` for an odd one (`autoRIFT.py:618-626`), so the reduction is symmetric about the
# node it is sampled at. Reducing over the stride instead under-covers a coarse point's window, and
# always downward: at stride 8 it cost 1,809 of 85,556 coarse points on this case, by up to 152 pixels.
# A radius too small searches a narrower window than the reference did, so it rails out or misses the
# peak at exactly the points where the prior was doing work — which reads as a correlator disagreement.
#
# This rung is *not* crop-safe: which fine points the lattice lands on depends on the grid's extent, so
# a sub-window samples a different set and the comparison would be against the wrong nodes.
function rungs_coarse_sampling(k::Capture, L::Int, chip::Int)
    pts = pointset_from_capture(k)
    p = params(; kwargs_from_capture(k)...)
    lp = AutoRIFT._level_points(pts, p, extent(chip), trues(size(pts)))
    setup = AutoRIFT._coarse_points(lp, p, extent(chip))
    setup === nothing && return [StageResult("3.6 coarse sampling", "xGrid0C", "exact", false, 0,
                                            "AutoRIFT.jl found no coarse grid at this level")]

    out = StageResult[]
    # The lattice, by the positions it lands on. `+1`/`-1` is the index base and nothing else.
    push!(out, exact_stage("3.6a coarse lattice", "xGrid0C",
                           Float32.(setup.coarse.x .- 1), stage(k, "xGrid0C", L)))
    # The radii, which are the reduction the width rule above governs. The reference dumps the
    # *undecimated* `colfilt` output under this name, so the comparison decimates it on the lattice
    # just verified rather than assuming the two lattices coincide.
    for (axis, refname, jl) in (("x", "SearchLimitX0C", setup.coarse.radius_x),
                                ("y", "SearchLimitY0C", setup.coarse.radius_y))
        ref = stage(k, refname, L)[setup.rows, setup.cols]
        push!(out, exact_stage("3.6b coarse radius $axis", refname, Float32.(jl), ref))
    end
    # The priors, sampled at the node rather than reduced: the reference slices `Dx00` with the same
    # `rIdxC` it slices the grid with (`autoRIFT.py:643-644`), so no window is involved here.
    for (axis, refname, jl) in (("x", "Dx0C", setup.coarse.dx_prior),
                                ("y", "Dy0C", setup.coarse.dy_prior))
        push!(out, exact_stage("3.6c coarse prior $axis", refname, Float32.(jl), stage(k, refname, L)))
    end
    return out
end

# ---------------------------------------------------------------------------

function report(rs::Vector{StageResult})
    @printf("\n%-28s %-24s %-28s %-6s %10s\n", "stage", "reference", "gate", "state", "n")
    for r in rs
        @printf("%-28s %-24s %-28s %-6s %10d\n", r.name, r.reference, r.gate,
                r.passed ? "GREEN" : "RED", r.n)
        @printf("%s%s\n", " "^4, r.detail)
    end
    red = count(r -> !r.passed, rs)
    @printf("\n%d rung%s, %d green, %d red\n", length(rs), length(rs) == 1 ? "" : "s",
            length(rs) - red, red)
    return red == 0
end

function main(args)
    isempty(args) && error("usage: stages.jl <product-name-fragment> [--run N] [--all]")
    c = only(cases(args[1]))
    n = 100
    i = findfirst(==("--run"), args); i === nothing || (n = parse(Int, args[i + 1]))
    ok = report(ladder(c; n, stop_on_red = !("--all" in args)))
    ok || exit(1)
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
