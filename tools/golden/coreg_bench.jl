# What the Sentinel-1 coregistration costs, against what the reference pipeline spends on it.
#
#   julia --project=tools/golden -t 12,1 tools/golden/coreg_bench.jl S1C_IW_SLC__1SSV_20250416
#   julia --project=tools/golden -t 12,1 tools/golden/coreg_bench.jl S1C_IW_SLC__1SSV_20250416 --no-gate
#
# **Coregistration, not correlation, is where a radar pair's time goes, and nothing measured it.** Every
# other harness here measures the correlator: `mem_nisar.jl` its peak, `profile_nisar.jl` where its wall
# clock goes, `e2e_table.jl` it against the reference. The stage in front of it had a correctness record
# and no timing at all. From the containers' own logs the reference spends 2,984 s coregistering a NISAR L1
# pair against 812 s correlating it, and about thirty minutes on the Sentinel-1 pair whose 43 bursts
# `runAutorift` then takes 116 s over — so a figure here is worth several of one there.
#
# What is measured is `secondary_swath_amplitude`: the secondary acquisition put on the reference's grid,
# which is the whole of what COMPASS's per-burst `rdr2geo`/`geo2rdr`/`ResampSlc` and hyp3's `merge_swaths`
# produce. The reference's counterpart is `isce_offsets.py --bench` in the `arift-ref` environment.
#
# **The headline row runs the real function and the breakdown runs one burst.** Attributing the stages
# means timing them separately, and timing them separately means reproducing the loop — which would then
# be a different implementation from the one the headline measured. So the headline calls
# `secondary_swath_amplitude` unmodified and the breakdown re-runs its three stages on a single burst,
# labelled as such. A breakdown that does not roughly sum to the headline over the burst count is a sign
# the loop does something the three stages do not.
#
# **The gate is the statistic `radar.jl` already established**, mean ratio and correlation against
# `secondary.tif`: 0.9976 and 0.99957 for the deramped eight-tap sinc with a Hann window. It is streamed
# rather than loaded, because the mosaic is already resident and the reference's copy is another 1.45 GB.
#
# Needs the case's run directory: both `.SAFE` trees, `dem.tif`, both orbit files and `secondary.tif`. A
# burst job's run directory holds all of them, so nothing is fetched.

include("correlator.jl")   # and, through it, `manifest.jl`, `reference.jl` and `intermediate.jl`
include("radar.jl")
include(joinpath(dirname(@__DIR__), "ab", "memtrace.jl"))

using Printf, Statistics
using ArchGDAL
using AutoRIFT

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

# ---------------------------------------------------------------------------
# The pair, from the run directory
# ---------------------------------------------------------------------------

"""
    burst_products(c::GoldenCase, run) -> (reference, secondary, swaths)

The two `Sentinel1Product`s of a burst job and the subswaths it mosaics.

`s1_burst_pair` returns the pair's *geometry* and discards the products, since that is all the geogrid
needs. The resample needs the products themselves, so this is the same construction kept.
"""
function burst_products(c::GoldenCase, run::AbstractString)
    early, late = _burst_safes(run)
    sws = burst_swaths(c)
    pol = lowercase(s1_polarization(basename(early)))
    mk(safe) = Sentinel1Product(safe; orbit = s1_orbit(run, basename(safe)),
                                polarization = pol, swaths = sws)
    return mk(early), mk(late), sws
end

# ---------------------------------------------------------------------------
# Agreement against the reference's own resample
# ---------------------------------------------------------------------------

"""
    row_lag(got, path; band = 1, span = 60) -> (lag, correlation)

The row offset between `got` and the raster at `path`, from a correlation search over a column strip.

**A subswath and a mosaic differ by the stacking crop, so the offset is measured rather than assumed.**
`merge_swaths` slices each subswath from `bursts[0].first_valid_line` and writes it at that subswath's
azimuth offset, zero for the first — so `secondary_swath_amplitude`, which is the counterpart of
`sec_swath_iw1.tif`, sits that many rows below `secondary.tif`. On `S1C_IW_SLC__1SSV_20250416` it is 19.
A `ResampledMosaic` carries the crop and compares at a lag of 0.

Searching for it is what keeps [`amplitude_agreement`](@ref) a statement about pixel values: 19 rows drop
the correlation from 0.989 to 0.077, which would otherwise read as a resampling failure.
"""
function row_lag(got::AbstractMatrix, path::AbstractString; band::Integer = 1, span::Integer = 60)
    ds = ArchGDAL.read(path)
    b = ArchGDAL.getband(ds, band)
    nr, nc = size(got)
    # A strip inside the swath rather than at its edge, over rows the data occupies on both sides.
    cols = max(1, nc ÷ 2 - 511):min(nc, nc ÷ 2 + 512)
    ny = min(4096, nr ÷ 2)
    y0 = nr ÷ 4
    ref = permutedims(ArchGDAL.read(b, first(cols) - 1, y0, length(cols), ny))
    best, bestlag = -Inf, 0
    for lag in (-span):span
        rows = (y0 + 1 + lag):(y0 + ny + lag)
        (first(rows) < 1 || last(rows) > nr) && continue
        A = @view got[rows, cols]
        m = (A .!= 0) .& (ref .!= 0)
        count(m) < 1000 && continue
        r = cor(Float64.(A[m]), Float64.(ref[m]))
        (isnan(r) || r <= best) && continue
        best, bestlag = r, lag
    end
    return (bestlag, best)
end

"""
    amplitude_agreement(got, path; band = 1, block = 2048, lag = 0) -> (ratio, correlation, n)

`got` against the raster at `path`, over the pixels both report as non-zero.

`lag` shifts `got`'s rows before comparing, so the statistic describes the resampled *values* rather
than the merge's row origin; [`row_lag`](@ref) measures it.

The ratio of means and the Pearson correlation, which is what `radar.jl` chose the interpolation kernel
on: a kernel that is too sharp or unwindowed shifts the ratio away from 1 while leaving the correlation
high, so neither statistic alone decides.

Zero is the reference's own no-data marker here — `slc_to_vrt_file` writes `NoDataValue` 0 and the ramp
margins outside a burst's valid rectangle read as zero on both sides — so a pixel zero on either side
carries no comparison and is skipped rather than counted as agreement.

**The comparison is over `got`'s rows, which are fewer than the reference's.** `merge_swaths:437-439`
sets the mosaic's azimuth extent from the last burst's start plus the *merged* height rather than one
burst's, so the reference's raster is about 1.8x the acquisition and everything past the acquisition is
zero — measured on this pair, rows 9000 and beyond hold no non-zero sample at all. `got` is the
acquisition's own extent, so it is compared against the leading rows and the empty tail is not counted
as agreement. Both widths must still match, since a width difference is a real disagreement about the
range origin.

Streamed a row block at a time. Both arrays are Float32 at the mosaic's shape, and the point of the
measurement is the footprint.
"""
function amplitude_agreement(got::AbstractMatrix, path::AbstractString;
                             band::Integer = 1, block::Integer = 2048, lag::Integer = 0)
    ds = ArchGDAL.read(path)
    w, h = ArchGDAL.width(ds), ArchGDAL.height(ds)
    nr, nc = size(got)
    nc == w || throw(DimensionMismatch(
        "the resampled mosaic is $nc samples wide and `$(basename(path))` is $w; the two describe " *
        "different range origins, which is a disagreement rather than a window to compare over"))
    nr <= h || throw(DimensionMismatch(
        "the resampled mosaic has $nr lines against `$(basename(path))`'s $h; it cannot extend past " *
        "the grid the reference resampled onto"))

    b = ArchGDAL.getband(ds, band)
    sa = sb = saa = sbb = sab = 0.0
    n = 0
    for r0 in 1:block:nr
        rows = r0:min(r0 + block - 1, nr)
        # GDAL's window is 0-based `(xoff, yoff, xsize, ysize)` and row-major, so the read comes back
        # transposed relative to the mosaic's `(line, sample)` orientation.
        ref = permutedims(ArchGDAL.read(b, 0, first(rows) - 1, w, length(rows)))
        @inbounds for j in 1:nc, ii in eachindex(rows)
            g = first(rows) + ii - 1 + lag
            (g < 1 || g > nr) && continue
            x = Float64(got[g, j])
            y = Float64(ref[ii, j])
            (x == 0 || y == 0 || !isfinite(x) || !isfinite(y)) && continue
            n += 1
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y
        end
    end
    n == 0 && return (NaN, NaN, 0)
    ma, mb = sa / n, sb / n
    cov = sab / n - ma * mb
    va = saa / n - ma * ma
    vb = sbb / n - mb * mb
    return (ma / mb, cov / sqrt(va * vb), n)
end

# ---------------------------------------------------------------------------
# The three stages, on one burst
# ---------------------------------------------------------------------------

"""
    burst_breakdown(rp, sp, swath, dem, burst) -> NamedTuple

Seconds in the offset solve, the deramp and the interpolation for one burst.

The stages `secondary_swath_amplitude` composes, timed apart so a total can be attributed. Each is the
same call that function makes; the numbers are one burst's and scale by the burst count only to the
extent the bursts are the same size, which within a subswath they are.
"""
function burst_breakdown(rp::Sentinel1Product, sp::Sentinel1Product, swath::Integer, dem,
                         burst::Integer)
    a = annotation(rp, swath)
    sa = annotation(sp, swath)
    lpb, spb = a.lines_per_burst, a.samples_per_burst

    ss = open_slc(sp.path; orbit = sp.orbit_path, swath = swath, burst = burst,
                  polarization = sp.polarization)
    cs = RadarCoordinate(ss)
    cr = RadarCoordinate(open_slc(rp.path; orbit = rp.orbit_path, swath = swath, burst = burst,
                                  polarization = rp.polarization))

    t_off = @elapsed dl, ds = _offset_lattice(cr, cs, dem, lpb, spb)
    t_fit = @elapsed carrier = CarrierFit(TopsCarrier(ss, cs, lpb), lpb, spb)

    sraster = burst_raster(first(collect(bursts(sp, swath))).backend).raster
    rows = ((burst - 1) * lpb + 1):(burst * lpb)
    t_der = @elapsed deramped = deramped_burst(sraster, rows, carrier)
    _zero_outside_valid!(deramped, sa.first_valid_line[burst] - 1, sa.last_valid_line[burst] - 1,
                         sa.first_valid_sample[burst] - 1, sa.last_valid_sample[burst] - 1)

    fvl = a.first_valid_line[burst] - 1
    lvl = a.last_valid_line[burst] - 1
    t_res = @elapsed resample_burst(deramped, dl, ds, fvl:lvl, 0:(spb - 1); doppler = (lpb, spb))

    return (; offsets = t_off, carrier_fit = t_fit, deramp = t_der, resample = t_res,
            burst_shape = (lpb, spb), output_lines = lvl - fvl + 1)
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

"""
    check_windows(r, full) -> Int

Windows read from `r` against the same windows of the materialized `full`, returning how many differ.

**The seams are the point.** A window inside one burst exercises nothing the materialized path does not;
a window straddling a seam has to pick up two bursts, each with its own carrier and offset field, and
place them in the same order the merge does — consecutive placements overlap by a row, so which burst
writes a shared row is an ordering the two paths must agree on rather than a detail. A single row and a
single column are included because a degenerate window is where a band's own bounds collapse.
"""
seam_rows(r::ResampledSwath) = [last(p.mosaic_rows) for p in r.places[1:(end - 1)]]

# Every burst seam of every subswath, in mosaic rows, plus the subswath boundaries themselves. A seam is
# where two sources meet, and that is the only place the two paths can order their writes differently.
function seam_rows(m::ResampledMosaic)
    rows = Int[]
    for (k, q) in enumerate(m.places)
        for s in seam_rows(m.swaths[k])
            s in q.swath_rows || continue
            push!(rows, first(q.mosaic_rows) + (s - first(q.swath_rows)))
        end
        k < length(m.places) && push!(rows, last(q.mosaic_rows))
    end
    return rows
end

# Where the array actually carries data, so a window lands on something. A mosaic is about 1.8x the
# acquisition and the tail is empty, where any two implementations agree trivially.
covered_rows(r::ResampledSwath) = first(first(r.places).mosaic_rows):last(last(r.places).mosaic_rows)
covered_rows(m::ResampledMosaic) =
    minimum(q -> first(q.mosaic_rows), m.places):maximum(q -> last(q.mosaic_rows), m.places)
covered_cols(r::ResampledSwath) = first(first(r.places).mosaic_cols):last(last(r.places).mosaic_cols)
covered_cols(m::ResampledMosaic) =
    minimum(q -> first(q.mosaic_cols), m.places):maximum(q -> last(q.mosaic_cols), m.places)

function check_windows(r, full::AbstractMatrix)
    nr, nc = size(r)
    size(full) == (nr, nc) || throw(DimensionMismatch(
        "the lazy array is $(nr)x$(nc) and the materialized one $(size(full)); they describe " *
        "different grids"))
    dr, dc = covered_rows(r), covered_cols(r)
    clampr(a, b) = clamp(a, 1, nr):clamp(b, 1, nr)
    clampc(a, b) = clamp(a, 1, nc):clamp(b, 1, nc)
    mid(x) = (first(x) + last(x)) ÷ 2
    windows = Tuple{UnitRange{Int},UnitRange{Int},String}[
        (clampr(mid(dr) - 127, mid(dr) + 128), clampc(mid(dc) - 255, mid(dc) + 256), "mid-swath"),
        (clampr(1, 256), clampc(1, 512), "top-left corner"),
        (clampr(nr - 255, nr), clampc(nc - 511, nc), "bottom-right corner"),
        (clampr(mid(dr), mid(dr)), clampc(first(dc), last(dc)), "one row, full width"),
        (clampr(first(dr), first(dr) + 511), clampc(mid(dc), mid(dc)), "one column"),
        (clampr(first(dr) - 8, first(dr) + 8), clampc(mid(dc) - 63, mid(dc) + 64), "the data's first row"),
        (clampr(last(dr) - 8, last(dr) + 8), clampc(mid(dc) - 63, mid(dc) + 64), "the data's last row"),
    ]
    for (k, s) in enumerate(seam_rows(r))
        push!(windows, (clampr(s - 127, s + 128), clampc(mid(dc) - 255, mid(dc) + 256), "across seam $k"))
    end
    bad = 0
    for (rows, cols, what) in windows
        got = r[rows, cols]
        ref = @view full[rows, cols]
        same = got == ref
        same || (bad += 1)
        @printf("    %-22s %s\n", what, same ? "equal" : @sprintf("DIFFERS, max %.3g",
                                                                  maximum(abs.(got .- ref))))
    end
    return bad
end

# ---------------------------------------------------------------------------
# The correlator over a resampled secondary
# ---------------------------------------------------------------------------

"""
    correlation_grid(c, n, scene) -> (PointSet{2}, kw, reference_measured)

The reference's own search grid for this pair, and the settings it was searched with.

**Taken from the capture rather than chosen here**, because neither the configuration nor the grid's
extent is free. This pair is searched at chip 56 by 16 — `ChipSize0X` 56 with `ScaleChipSizeY` 0.2857 —
behind a Wallis filter 21 samples wide, with a per-point search radius from the geogrid that reaches 75
samples in x and 13 in y. A square chip at a default filter width measures nothing whatever on
SLC-resolution speckle, and two arms then agree on a grid of `NaN`s.

**And it is the whole grid, not a patch of it.** A sub-window measures nothing either: the coarse pass
restricts the finer levels through the outlier filter's neighbourhood, so a grid shorter than that
window searches every point at full radius and keeps none of them — `tile.jl:1260` warns about exactly
this. Measured with the reference's own imagery, grid and settings on a 192x384 patch where the
reference measured 1,640 of 73,728 points: **0**. So the correlated area is the production one, and the
patch escape hatch does not exist.

`preprocess` is the one setting overridden. `kwargs_from_capture` sets `:none` because a capture's
imagery is already filtered; a resampled mosaic is raw amplitude, so it takes the filter the reference
applied — which is also what has the blocked path filter each block from raw input.

`reference_measured` is how many points the reference resolved, which is the yield to read a Julia
count against: 41,935 of 2,826,240 here, on an Antarctic coastal pair whose grid is mostly ocean.
"""
function correlation_grid(c::GoldenCase, n::Integer, scene::Tuple{Int,Int})
    k = read_capture(c; n, mmap = CAPTURE_IMAGERY)
    size(k.arrays["in_I1"]) == scene || throw(DimensionMismatch(
        "the reference correlated a $(join(size(k.arrays["in_I1"]), "x")) image and the mosaic is " *
        "$(join(scene, "x")); the grid's coordinates do not describe this array"))

    kw = merge(kwargs_from_capture(k),
               (; preprocess = AutoRIFT.Wallis(; width = Int(k.scalars["WallisFilterWidth"]))))
    return (pointset_from_capture(k), kw, count(isfinite, k.arrays["out_Dx"]))
end

"""
    correlate_arm(reference, secondary, grid, block, buf, hz; kw...) -> NamedTuple

One blocked correlation, with its wall clock, CPU seconds and peak footprint.

`secondary` is the whole point: a materialized mosaic and a [`ResampledMosaic`](@ref) go through the
same call, so what differs between two arms is where the secondary's pixels come from and nothing
else.
"""
function correlate_arm(reference::AbstractMatrix, secondary::AbstractMatrix, grid, block::Integer,
                       buf, hz; kw...)
    GC.gc(true); GC.gc(true)
    fl = last(rusage!(buf))
    cpu0 = cpu_seconds!(buf, hz)
    out, trace, secs = with_trace(; interval = 0.01) do
        autorift(reference, secondary, grid; kw..., process_block_size = (block, block))
    end
    return (; out, seconds = secs, cpu = cpu_seconds!(buf, hz) - cpu0,
            peak = maximum(trace.footprint), floor_bytes = fl)
end

# `isequal`, so a point both arms left unmeasured counts as agreement rather than as `NaN != NaN`.
same_field(a, b) = size(a) == size(b) && mapreduce(isequal, &, a, b)

function main()
    isempty(ARGS) && error("usage: coreg_bench.jl <product-name-fragment> [--run N] " *
                           "[--no-gate] [--no-trace] [--no-check] [--no-correlate] " *
                           "[--blocks a,b,c] [--corr-blocks a,b]")
    c = only(cases(ARGS[1]))
    n = parse(Int, argvalue("--run", "200"))
    run = run_dir(c, n)
    isdir(run) || error("no run $n of $(c.product) under $(dirname(run))")

    rp, sp, sws = burst_products(c, run)
    dem = dem_sampler(joinpath(run, "dem.tif"))
    swath = only(sws)   # A burst job over one subswath; the multi-swath mosaic is a separate rung.
    a = annotation(rp, swath)

    @printf("%s\n", c.product)
    @printf("  run %d, IW%d, %d bursts of %d x %d, %d threads\n", n, swath, nbursts(a),
            a.lines_per_burst, a.samples_per_burst, Threads.nthreads())
    flush(stdout)

    # Compile before anything is timed: the first call through the resample pays the JIT for the whole
    # interpolation kernel, which on one burst is a large share of what is being measured.
    @printf("  warmup (burst 1): %.1f s compiling\n",
            @elapsed burst_breakdown(rp, sp, swath, dem, 1))
    flush(stdout)

    br = burst_breakdown(rp, sp, swath, dem, 1)
    @printf("\n  one burst, by stage (%d x %d in, %d lines out)\n", br.burst_shape...,
            br.output_lines)
    for k in (:offsets, :carrier_fit, :deramp, :resample)
        @printf("    %-12s %8.3f s\n", k, getproperty(br, k))
    end
    total1 = br.offsets + br.carrier_fit + br.deramp + br.resample
    @printf("    %-12s %8.3f s   x %d bursts = %.1f s\n", "sum", total1, nbursts(a),
            total1 * nbursts(a))
    flush(stdout)

    # The floor the peak is measured against: what the process holds with the rasters open and nothing
    # running, collected first so the figure is this run's requirement and not the warmup's garbage.
    GC.gc(true); GC.gc(true)
    buf = zeros(UInt64, 64)
    floor_bytes = last(rusage!(buf))
    hz = tick_rate()
    cpu0 = cpu_seconds!(buf, hz)

    traced = !("--no-trace" in ARGS)
    work() = secondary_swath_amplitude(rp, sp, swath, dem)
    out, nlines, ns, seconds, peak = if traced
        # Only when the peak moves, so a redirected run leaves a readable log rather than one line per
        # sample.
        shown = Ref(0.0)
        progress = function (t, _)
            isempty(t.footprint) && return
            pk = maximum(t.footprint) / 2^20
            pk > shown[] + 64 || return
            shown[] = pk
            @printf(stderr, "    resampling  peak %8.0f MiB\n", pk)
            flush(stderr)
        end
        r, trace, secs = with_trace(; interval = 0.01, progress) do
            work()
        end
        println(stderr)
        (r..., secs, maximum(trace.footprint))
    else
        secs = @elapsed r = work()
        (r..., secs, last(rusage!(buf)))
    end
    cpu = cpu_seconds!(buf, hz) - cpu0

    @printf("\n  whole subswath, %d bursts\n", nbursts(a))
    @printf("    wall              %8.1f s\n", seconds)
    @printf("    cpu               %8.1f s  (%.2f of %d threads)\n", cpu, cpu / seconds,
            Threads.nthreads())
    @printf("    mosaic            %8d x %-8d %.2f GiB Float32\n", nlines, ns,
            4 * nlines * ns / 2^30)
    @printf("    peak              %8.2f GiB  (%.2f above floor %.2f)\n", peak / 2^30,
            (Int(peak) - Int(floor_bytes)) / 2^30, floor_bytes / 2^30)
    flush(stdout)

    out = nothing   # the subswath array is not what the rest compares against; let it go
    GC.gc(true)

    # Everything below is on the **mosaic** grid, not the subswath's. That is the grid the geogrid's
    # `window_*` rasters index and therefore the one a correlator's point set is expressed in, so it is
    # the grid a secondary has to be readable on. A subswath sits `first_valid_line` rows above it.
    GC.gc(true)
    mfloor = last(rusage!(buf))
    cpu2 = cpu_seconds!(buf, hz)
    mos, msecs = let
        r, _, s = with_trace(; interval = 0.01) do
            secondary_mosaic(rp, sp, sws, dem)
        end
        (r, s)
    end
    mcpu = cpu_seconds!(buf, hz) - cpu2
    mnl, mns = size(mos)
    @printf("\n  the mosaic, materialized\n")
    @printf("    wall              %8.1f s   cpu %.1f\n", msecs, mcpu)
    @printf("    size              %8d x %-8d %.2f GiB Float32\n", mnl, mns,
            4 * mnl * mns / 2^30)
    flush(stdout)

    if !("--no-check" in ARGS)
        println("\n  a lazily read mosaic window against the materialized mosaic")
        lazym = ResampledMosaic(rp, sp, sws, dem)
        bad = check_windows(lazym, mos)
        bad == 0 || error("$bad windows read from a ResampledMosaic differ from the mosaic")

        # The path a blocked correlation actually takes. `AutoRIFT._read_window!`'s generic method indexes
        # with ranges rather than viewing, because a view of a lazy array defers to a scalar read per
        # pixel — measured there at 444 s against 0.6 s. A `ResampledMosaic` is not a `StridedMatrix`, so
        # it takes that method, and asserting it here is what says the two agree about the contract rather
        # than about this case.
        rows, cols = 6000:6511, 9000:9511
        dest = Matrix{Float32}(undef, length(rows), length(cols))
        AutoRIFT._read_window!(dest, lazym, rows, cols)
        ok = dest == @view mos[rows, cols]
        @printf("    %-22s %s\n", "AutoRIFT._read_window!", ok ? "equal" : "DIFFERS")
        ok || error("AutoRIFT._read_window! on a ResampledMosaic does not match the mosaic")
        flush(stdout)
    end

    # The blocked arm: the mosaic read a window at a time from a `ResampledMosaic`, which is what a
    # correlator driving the resample through `process_block_size` does. Nothing holds the mosaic, so the
    # footprint is the block's rather than the scene's — and `deramped` against the area read is what the
    # eight-tap halo costs in repeated work.
    want = sum(Float64, mos)
    for spec in split(argvalue("--blocks", "4096,2048,1024"), ',')
        bs = parse(Int, spec)
        bs <= 0 && continue
        m = ResampledMosaic(rp, sp, sws, dem)
        GC.gc(true); GC.gc(true)
        fl = last(rusage!(buf))
        cpu1 = cpu_seconds!(buf, hz)
        acc = Ref(0.0)
        nblocks = Ref(0)
        blocked() = for r0 in 1:bs:mnl, c0 in 1:bs:mns
            rows = r0:min(r0 + bs - 1, mnl)
            cols = c0:min(c0 + bs - 1, mns)
            w = m[rows, cols]
            nblocks[] += 1
            # A checksum rather than the block, so the arm holds one block at a time and the answer is
            # still compared against the materialized mosaic.
            acc[] += sum(Float64, w)
        end
        _, btrace, bsecs = with_trace(; interval = 0.01) do
            blocked()
        end
        bcpu = cpu_seconds!(buf, hz) - cpu1
        bpeak = maximum(btrace.footprint)
        # Against the area actually resampled — the placements' own extent — rather than the mosaic's,
        # which on a burst job is 1.8x the acquisition and mostly empty.
        covered = sum(q -> length(q.mosaic_rows) * length(q.mosaic_cols), m.places)
        amp = sum(r -> r.deramped[], m.swaths) / covered
        verdict = acc[] ≈ want ? "matches" : @sprintf("DIFFERS by %.6g", acc[] - want)
        @printf("  block %5d: %6.1f s  cpu %6.1f  peak %6.2f GiB (%.2f above)  %5d blocks  deramp amp %.3fx  checksum %s\n",
                bs, bsecs, bcpu, bpeak / 2^30, (Int(bpeak) - Int(fl)) / 2^30, nblocks[], amp, verdict)
        flush(stdout)
    end

    if !("--no-gate" in ARGS)
        sec = joinpath(run, "secondary.tif")
        if isfile(sec)
            lag, lagcorr = row_lag(mos, sec)
            ratio, corr, npx = amplitude_agreement(mos, sec; lag)
            @printf("\n  the mosaic against %s\n", basename(sec))
            @printf("    row lag           %+8d   (%.5f on the search strip)\n", lag, lagcorr)
            @printf("    comparable px     %8d\n", npx)
            @printf("    mean ratio        %8.4f   (0.9976 is the recorded kernel)\n", ratio)
            @printf("    correlation       %8.5f   (0.99957 is the recorded kernel)\n", corr)
            lag == 0 || @printf("    NOTE a non-zero lag means the mosaic origin disagrees with the reference's\n")
        else
            @printf("\n  no secondary.tif in the run directory; agreement not measured\n")
        end
    end

    # **The claim this stage settles: a lazy secondary is an input `autorift` takes, and taking it
    # changes nothing about the answer.** Everything above compares a window read from a
    # `ResampledMosaic` against the same window of the materialized mosaic; this hands both to the
    # correlator and compares what comes out, which is the only statement that covers the reads the
    # driver actually issues — its own halo, its own block order, one read per pass per block.
    #
    # Both arms are blocked at the same size, so the secondary's storage is the only difference
    # between them. The materialized arm holds the mosaic as well as its blocks, which is the
    # footprint the lazy one exists to avoid; the lazy arm pays for it in repeated resampling.
    if !("--no-correlate" in ARGS)
        grid, kw, refmeasured = correlation_grid(c, n, size(mos))
        ref = radar_mosaic(rp, sws)
        size(ref) == size(mos) || error("the reference mosaic is $(size(ref)) and the secondary " *
                                        "$(size(mos)); they are not the same grid")
        @printf("\n  autorift over the resampled secondary\n")
        @printf("    grid            %d x %d = %d points, %d searchable, %d resolved by the reference\n",
                size(grid)..., length(grid), AutoRIFT.nsearchable(grid), refmeasured)
        @printf("    settings        chip %dx%d..%dx%d, %s, %s\n", kw.chip_size.X, kw.chip_size.Y,
                kw.chip_size_max.X, kw.chip_size_max.Y, kw.preprocess,
                kw.threaded ? "$(Threads.nthreads()) threads" : "one thread")
        flush(stdout)

        blocks = [parse(Int, s) for s in split(argvalue("--corr-blocks", "2048,1024"), ',')]
        base = correlate_arm(ref, mos, grid, first(blocks), buf, hz; kw...)
        measured = count(isfinite, base.out.dx)
        @printf("    a materialized secondary, block %d\n", first(blocks))
        @printf("      wall %6.1f s  cpu %6.1f  peak %5.2f GiB (%.2f above)  %d points measured\n",
                base.seconds, base.cpu, base.peak / 2^30,
                (Int(base.peak) - Int(base.floor_bytes)) / 2^30, measured)
        measured == 0 && error("no point was measured on either arm, so an agreement between them " *
                               "says nothing")
        flush(stdout)

        for bs in blocks
            arm = correlate_arm(ref, ResampledMosaic(rp, sp, sws, dem), grid, bs, buf, hz; kw...)
            agree = [k for k in (:dx, :dy, :correlation, :chip_size)
                     if !same_field(getproperty(arm.out, k), getproperty(base.out, k))]
            @printf("    a lazy secondary, block %d\n", bs)
            @printf("      wall %6.1f s  cpu %6.1f  peak %5.2f GiB (%.2f above)  %s\n",
                    arm.seconds, arm.cpu, arm.peak / 2^30,
                    (Int(arm.peak) - Int(arm.floor_bytes)) / 2^30,
                    isempty(agree) ? "dx, dy, correlation and chip_size identical" :
                    "DIFFERS in " * join(agree, ", "))
            flush(stdout)
            isempty(agree) || error("a lazily resampled secondary gives a different answer than " *
                                    "the materialized mosaic at block $bs: " * join(agree, ", "))
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
