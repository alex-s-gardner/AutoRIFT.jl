# Where the raw-SLC-to-`dx`/`dy` chain spends its time on a NISAR case, attributed over the whole
# `correlate` stage rather than over captured, mmap'd, pre-filtered bytes.
#
#   julia --project=tools/golden -t 10,1 tools/golden/profile_e2e_nisar.jl NISAR_L1_PR_RSLC
#   julia --project=tools/golden -t 10,1 tools/golden/profile_e2e_nisar.jl NISAR_L1_PR_RSLC --rows 1:582
#   julia --project=tools/golden -t 10,1 tools/golden/profile_e2e_nisar.jl NISAR_L1_PR_RSLC --no-profile
#
# **Why this exists alongside `profile_nisar.jl`.** That script attributes `correlator.jl`'s comparison
# — AutoRIFT.jl run on the reference's own captured, byte-quantized, mmap'd pair. Nothing there touches
# HDF5 decode, the lazy per-block resample (`ResampledRSLC`), or `TileCache`, which is what makes
# `e2e_run.jl`'s `correlate` stage on NISAR L1 cost 4,438.6 s against the 397-467 s the captured-bytes
# comparison measures on the same case — a ~10x gap with no attribution to say where inside it the time
# actually goes. This runs the real from-granule imagery through the profiler instead.
#
# **`--rows a:b` crops the *grid*, not the block size.** `AutoRIFT.block_size_for` is computed from the
# full, uncropped grid before any crop is applied, and that block size is then forced on the cropped
# run. Computing it from the cropped grid instead recomputes a smaller halo — the cropped grid's own
# extreme search radii are a subset of the full grid's — and silently measures a different, smaller-block
# configuration than the one the real run uses, which is a bug this harness had once already. A crop is
# for making the measurement affordable, not for changing what is being measured.
#
# **Runtime and attribution come from separate runs**, `profile_nisar.jl`'s discipline: sampling every
# thread every few milliseconds perturbs the wall clock it explains, so the clean row is unprofiled and
# the profiled row is a second pass at the same configuration, warmed first so neither pays compilation.

include("e2e_run.jl")
include("profile_attribution.jl")

using AutoRIFT: block_size_for
using Printf

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

function parse_rows(s::Union{Nothing,AbstractString})
    isnothing(s) && return nothing
    parts = split(s, ':')
    length(parts) == 2 || throw(ArgumentError("`--rows` wants `a:b`, got \"$s\""))
    return parse(Int, parts[1]):parse(Int, parts[2])
end

"""
    warmup_e2e(i1, i2, grid, p, block) -> Float64

Correlate a small dense window of `grid` before anything is timed, so the first recorded run is not
also the first compilation. Returns the seconds it took.

The window is found by scanning a coarse lattice of candidates, the way `profile_nisar.jl`'s `warmup`
does: on a NISAR geogrid the footprint is a rotated swath inside its bounding box, so a corner crop can
easily contain nothing to search, and any window with a few hundred searchable points compiles the same
code.
"""
function warmup_e2e(i1, i2, grid::AutoRIFT.PointSet{2}, p::AutoRIFT.Params, block::Tuple{Int,Int},
                    filter_cache_tile::Union{Nothing,Int} = nothing)
    side = 64
    n, i, j = _densest_window(grid, side)
    nr, nc = size(grid)
    n == 0 && error("no searchable window found for warmup; the grid appears to be entirely fill")
    patch = grid[i:min(nr, i + side - 1), j:min(nc, j + side - 1)]
    t0 = time()
    AutoRIFT.autorift(i2, i1, patch, p, block, 0, filter_cache_tile)
    seconds = time() - t0
    @printf("  warmup: %d x %d grid window at (%d, %d), %d searchable points, %.1f s compiling\n",
            size(patch)..., i, j, n, seconds)
    flush(stdout)
    return seconds
end

"""
    report_e2e(clean_seconds, scan, nthreads)

Print the same occupancy, thread-histogram and stage-attribution tables `profile_nisar.jl`'s `report`
does, for the one from-granule configuration this harness measures rather than a block-size sweep.
"""
function report_e2e(clean_seconds::Real, scan, nthreads::Integer)
    isnothing(scan) && return nothing
    println()
    report_profile_scan(scan.result, clean_seconds, nthreads, scan.cpu / clean_seconds;
                        delay = scan.delay, fill = scan.fill, top = 20, serial_top = 12)
    return nothing
end

function main()
    isempty(ARGS) && error("usage: profile_e2e_nisar.jl <product-fragment> [--rows a:b] " *
                          "[--no-profile] [--filter-cache-tile N]")
    c = only(cases(ARGS[1]))
    reason = unsupported(c)
    isnothing(reason) || error("$(c.product): $reason")
    rows = parse_rows(argvalue("--rows", nothing))
    profile = !("--no-profile" in ARGS)
    fct_arg = argvalue("--filter-cache-tile", nothing)
    filter_cache_tile = isnothing(fct_arg) ? nothing : parse(Int, fct_arg)
    nthreads = Threads.nthreads(:default)

    t0 = time(); s = setup(c); @printf("geometry: %.1f s\n", time() - t0)

    t0 = time()
    grid_full = AutoRIFT.pointset(s.geometry; pixel_size = ImagePairGeometry.xsize(s.pair.coordinate))
    m = correlator_filter(c)
    p = AutoRIFT.params(s.geometry; threaded = nthreads > 1, preprocess = isnothing(m) ? :none : m)
    @printf("grid: %.1f s   full grid %s, searchable %d\n", time() - t0, size(grid_full),
            AutoRIFT.nsearchable(grid_full))

    t0 = time()
    (i1, i2) = e2e_imagery(s)
    scene = size(i1)
    caches = TileCache[]
    # Kept even when `filter_cache_tile` is set — see the matching note in `e2e_run.jl` on why removing
    # it made things worse rather than better.
    (i1, i2) = map((i1, i2)) do img
        AutoRIFT.ondisk(img) || return img
        tc = TileCache(img; tile = 1024, dir = joinpath(CACHE, "scratch"))
        push!(caches, tc)
        return tc
    end
    @printf("imagery: %.1f s   scene %s\n", time() - t0, scene)

    # Forced from the full grid, before any crop — see the header note on why.
    b = block_size_for(grid_full, p, scene)
    @printf("block size (full-grid halo): %d x %d\n", b.X, b.Y)

    grid = isnothing(rows) ? grid_full : grid_full[rows, :]
    @printf("run grid: %s, searchable %d%s\n", size(grid), AutoRIFT.nsearchable(grid),
            isnothing(rows) ? "" : " (rows $rows of $(size(grid_full, 1)))")
    @printf("filter tile cache: %s\n", isnothing(filter_cache_tile) ? "off" : "$filter_cache_tile px")

    warmup_e2e(i1, i2, grid, p, (b.X, b.Y), filter_cache_tile)

    buf = zeros(UInt64, 64)
    hz = tick_rate()
    run1() = AutoRIFT.autorift(i2, i1, grid, p, (b.X, b.Y), 0, filter_cache_tile)

    GC.gc(true); GC.gc(true)
    clean_cpu0 = cpu_seconds!(buf)
    clean_t0 = time()
    out = run1()
    clean_seconds = time() - clean_t0
    clean_cpu = cpu_seconds!(buf) - clean_cpu0
    measured = count(!isnan, out.dx)
    @printf("\ncorrelate (clean): %.1f s   cpu %.1f s (occupancy %.2f of %d)   measured %d\n",
            clean_seconds, clean_cpu, clean_cpu / clean_seconds, nthreads, measured)

    scan = nothing
    if profile
        nwords, delay = plan_profile(clean_seconds, nthreads)
        Profile.clear()
        Profile.init(; n = nwords, delay)
        cpu0 = cpu_seconds!(buf)
        _, trace, prof_seconds = with_trace(; interval = 0.05) do
            Profile.@profile run1()
        end
        cpu = cpu_seconds!(buf) - cpu0
        data = Profile.fetch(include_meta = true)
        fill = Profile.len_data() / Profile.maxlen_data()
        scan = (; fill, delay, cpu, seconds = prof_seconds,
                result = scan_profile(data, trace.tick_hz; delay))
        @printf("correlate (profiled): %.1f s   cpu %.1f s\n", prof_seconds, cpu)
    end

    for tc in caches
        @printf("tile cache: %s\n", cache_report(tc))
        close(tc)
    end

    report_e2e(clean_seconds, scan, nthreads)
    return nothing
end

main()
