# The full-scene comparison table: the reference, the library eager and lazy, and the trimmed binary
# at each block size. Prints the table as Markdown, records it as JSON, and renders it to PDF.
#
# One fresh process per row, timed by `/usr/bin/time -l`, so runtime is whole-process wall clock and
# peak RSS is what a memory limit would see. A process per row is required rather than tidy:
# `ru_maxrss` is a high-water mark, so two configurations measured in one process both report the
# larger. One rep per row — each is minutes, and the spread across repeats is under 1%.
#
# Requires the real-data cache (`tools/realdata/README.md`), the reference env (see this directory's
# README), the scene planes written by `bench_scene.jl`, and a built binary (`app/README.md`).
#
#   julia --project=tools/ab tools/ab/bench_table.jl              # measure, then render
#   julia --project=tools/ab tools/ab/bench_table.jl --replot     # re-render from results.json
#
# `--replot` exists because rendering is the part worth iterating on and measuring is the part that
# costs 12 minutes.

using Base64, JSON3, Printf

include(joinpath(@__DIR__, "bench_figures.jl"))

const HERE = @__DIR__
const ROOT = dirname(dirname(HERE))
const BIN = joinpath(ROOT, "app", "build", "bin", "autorift")
const PLOTS = joinpath(HERE, "plots")
const RESULTS = joinpath(HERE, "bench_table.json")
# Scene planes and per-row logs. Defaults to a fresh temp dir; set this to reuse planes across runs.
const WORK = get(ENV, "AUTORIFT_BENCH_DIR", mktempdir())

# The configuration every row shares: chips 16/64 at spacing 8 is the ITS_LIVE optical setting, and the
# one the accuracy comparison in this directory's README was made at.
const CHIP, CHIP_MAX, SPACING, RADIUS, UPSAMPLING = 16, 64, 8, 20, 16

# `gridpoints` insets by the widest chip's half-extent plus the search radius plus one; `block_layout`
# snaps a block outward to a whole number of grid points, so a block of `bs` pixels spans `bs ÷ spacing`.
gridlen(n) = length((cld(CHIP_MAX, 2) + RADIUS + 2):SPACING:(n - cld(CHIP_MAX, 2) - RADIUS - 1))

# One row of the table, run in a fresh process. Written to a file rather than kept as a function
# because each row must be its own process, for the peak-RSS reason above.
#
# `eager` and `lazy` differ in exactly one thing — whether the pixels are on disk or in memory when
# correlation starts — since both go through `autorift(::AbstractRaster, ...)`. `raw` is the
# plain-array path on bare `Float32` planes, which is what the binary reads and what every other
# benchmark in this repository measures.
const ROW_SCRIPT = """
    using AutoRIFT, Printf
    mode, block = ARGS[1], parse(Int, ARGS[2])
    work = ARGS[3]
    kw = (; chip_size = $CHIP, chip_size_max = $CHIP_MAX, grid_spacing = $SPACING,
          search_radius = $RADIUS, preprocess = :highpass, filter_width = 5,
          upsampling = $UPSAMPLING, threaded = Threads.nthreads() > 1,
          process_block_size = block == 0 ? nothing : (block, block))
    if mode == "raw"
        nr, nc = Tuple(parse.(Int, split(read(joinpath(work, "dims.txt"), String))))
        ref = Matrix{Float32}(undef, nr, nc); read!(joinpath(work, "ref.bin"), ref)
        sec = Matrix{Float32}(undef, nr, nc); read!(joinpath(work, "sec.bin"), sec)
        out = autorift(ref, sec; kw...)
        measured = count(!isnan, out.dx)
    else
        using Rasters, ArchGDAL, Extents
        cache = joinpath(get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests")), "landsat")
        a = Raster(joinpath(cache, "LC09_L1TP_009011_20230618_20230618_02_T1_B8.TIF"); lazy = true)
        b = Raster(joinpath(cache, "LC08_L1TP_009011_20230626_20230710_02_T1_B8.TIF"); lazy = true)
        # `view` and not `getindex`: indexing a lazy `Raster` by an `Extent` materializes it, which
        # would make every mode eager.
        ex = Extents.intersection(Extents.extent(a), Extents.extent(b))
        va, vb = view(a, ex), view(b, ex)
        out = mode == "lazy" ? autorift(va, vb; kw...) : autorift(read(va), read(vb); kw...)
        measured = count(!isnan, parent(out.vx))
    end
    @printf("MEASURED %d\\n", measured)
"""

# What produced these numbers. Recorded with the results rather than in prose, because a timing without
# its machine, OS and versions is not comparable to anything — including a later run of this script.
function provenance()
    cpu = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
    ram = parse(Int, strip(read(`sysctl -n hw.memsize`, String))) / 2^30
    osname = strip(read(`sw_vers -productName`, String))
    osver = strip(read(`sw_vers -productVersion`, String))
    # The reference's own version string, from the env that ran it.
    pyver = try
        strip(read(`micromamba run -n arift-ref python -c "import autoRIFT,sys;print(autoRIFT.__version__)"`,
                   String))
    catch
        "unknown"
    end
    # `hw.ncpu`, not `Sys.CPU_THREADS`: on an Apple M2 Max the latter reports the 8 performance cores
    # only, while the thread counts in this table are what `-t` was given, which can use all 12.
    cores = parse(Int, strip(read(`sysctl -n hw.ncpu`, String)))
    return (; cpu, cores, ram_gib = round(ram; digits = 0),
            os = "$osname $osver", julia = string(VERSION),
            autorift_jl = autorift_version(), autorift_py = pyver,
            commit = readchomp(`git -C $ROOT rev-parse --short HEAD`))
end

# The package's declared version and the commit it was measured at. Both, since a `0.1.0` in
# `Project.toml` moves far less often than the code it labels.
function autorift_version()
    for line in eachline(joinpath(ROOT, "Project.toml"))
        m = match(r"^version\s*=\s*\"(.*)\"", line)
        m === nothing || return m.captures[1]
    end
    return "unknown"
end

# Runtime and peak RSS of one child process. `/usr/bin/time -l` reports peak in bytes on macOS, and is
# the only way to get it for a process this one did not itself build.
function timed(cmd::Cmd, tag::AbstractString)
    log = joinpath(WORK, "$tag.log")
    open(log, "w") do io
        run(pipeline(`/usr/bin/time -l $cmd`; stdout = io, stderr = io))
    end
    seconds = peak = cpu = nothing
    for line in eachline(log)
        # `/usr/bin/time -l` opens with "<real> real <user> user <sys> sys". Summing user and system
        # over elapsed gives mean cores used — which is measured rather than assumed, and for
        # autoRIFT.py is far below the core count.
        m = match(r"^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", line)
        if m !== nothing
            seconds = parse(Float64, m.captures[1])
            cpu = parse(Float64, m.captures[2]) + parse(Float64, m.captures[3])
        end
        occursin("maximum resident set size", line) &&
            (peak = parse(Float64, first(split(strip(line)))) / 2^20)
    end
    seconds === nothing && error("no timing for $tag; see $log")
    cores = something(cpu, 0.0) / seconds
    @printf("  %-32s %7.1f s  %6.0f MiB  %5.2f cores\n", tag, seconds, something(peak, 0.0), cores)
    flush(stdout)
    return seconds, something(peak, 0.0), cores
end

function measure()
    rowfile = joinpath(WORK, "row.jl")
    write(rowfile, ROW_SCRIPT)
    nr, nc = Tuple(parse.(Int, split(read(joinpath(WORK, "dims.txt"), String))))
    gy, gx = gridlen(nr), gridlen(nc)
    nblocks(bs) = bs == 0 ? 1 : cld(gy, bs ÷ SPACING) * cld(gx, bs ÷ SPACING)
    @printf("overlap %d x %d = %.1f Mpixel; grid %d x %d = %.2f M points; load %.2f\n\n",
            nr, nc, nr * nc / 1e6, gy, gx, gy * gx / 1e6, first(Sys.loadavg()))

    rows = Any[]
    add!(label, threads, lazy, blocks, r) =
        push!(rows, (; label, threads, lazy, blocks, seconds = r[1], peak = r[2], cores = r[3]))
    julia(th, mode, bs, tag) = timed(
        `$(Base.julia_cmd()) --startup-file=no --project=$HERE -t $th $rowfile $mode $bs $WORK`, tag)

    # The reference gets no thread count to set: its correlation loop is serial, so what it uses is
    # whatever OpenCV's filter calls take. Reported as the measured mean rather than a requested count.
    r = timed(`micromamba run -n arift-ref python $(joinpath(HERE, "bench_scene.py"))`, "python")
    add!("python autoRIFT v2.1.2", @sprintf("%.1f†", r[3]), "no", 1, r)

    for th in (12, 1)
        add!("AutoRIFT.jl, eager, no blocks", string(th), "no", 1, julia(th, "raw", 0, "eager_$th"))
    end

    add!("AutoRIFT.jl, lazy, block 2048", "12", "yes", nblocks(2048), julia(12, "lazy", 2048, "lazy"))

    # `[mmap] = 1` is the binary's lazy path: its contract is raw planes, so there is no GDAL in a
    # trimmed build and demand-paged pages are what stand in for a lazy read.
    for bs in (0, 2048, 1024, 512, 256)
        label = bs == 0 ? "juliac binary, lazy, no blocks" : "juliac binary, lazy, block $bs"
        cmd = `$BIN $(joinpath(WORK, "ref.bin")) $(joinpath(WORK, "sec.bin")) $nr $nc
               $(joinpath(WORK, "bin_$bs.bin")) $CHIP $RADIUS $SPACING $bs $UPSAMPLING 1`
        add!(label, "1*", "yes", nblocks(bs), timed(cmd, "bin_$bs"))
    end

    result = (; scene = (; rows = nr, cols = nc), grid = (; rows = gy, cols = gx),
              chip = CHIP, chip_max = CHIP_MAX, spacing = SPACING, radius = RADIUS,
              upsampling = UPSAMPLING, provenance = provenance(), table = rows)
    open(RESULTS, "w") do io
        JSON3.pretty(io, result)
    end
    return result
end

# A PNG inlined as a data URI. Chrome resolves `file://` subresources inconsistently in headless
# `--print-to-pdf`; embedding the bytes removes the question.
data_uri(path) = "data:image/png;base64," * base64encode(read(path))

# The report as a PDF, through headless Chrome — the one HTML-to-PDF engine present on a stock macOS
# with no TeX installation.
#
# Three pages: the performance table, then the accuracy comparison against the Python reference as
# maps and as distributions. Performance and accuracy belong in one artifact because neither answers
# the question alone — a faster implementation that disagrees is not faster at the same job.
function render(r)
    mkpath(PLOTS)
    figs = accuracy_figures()
    s = figs.stats
    # Both the requested thread count and the cores actually used: a run given 12 threads that achieves
    # 3.8 has told you something the setting alone hides, and for autoRIFT.py there is no setting at all.
    body = join(("""<tr><td class="l">$(t.label)</td><td>$(t.threads)</td>""" *
                 """<td>$(@sprintf("%.2f", t.cores))</td><td>$(t.lazy)</td>""" *
                 """<td>$(t.blocks)</td><td>$(@sprintf("%.1f s", t.seconds))</td>""" *
                 """<td>$(@sprintf("%.0f MiB", t.peak))</td></tr>""" for t in r.table), "\n")
    html = """
    <!DOCTYPE html><html><head><meta charset="utf-8"><style>
      @page { size: letter landscape; margin: 0.8in; }
      body { font: 11pt/1.45 "Helvetica Neue", Helvetica, sans-serif; color: #111; }
      h1 { font-size: 15pt; margin: 0 0 3pt; }
      p.sub { color: #555; font-size: 9.5pt; margin: 0 0 16pt; }
      table { border-collapse: collapse; width: 100%; font-size: 10pt; }
      th { text-align: right; border-bottom: 1.5px solid #111; padding: 5pt 9pt; font-weight: 600; }
      td { text-align: right; padding: 5pt 9pt; border-bottom: 0.5px solid #ddd; }
      th:first-child, td.l { text-align: left; }
      tbody tr:last-child td { border-bottom: 1.5px solid #111; }
      p.note { color: #555; font-size: 9pt; margin-top: 14pt; }
      .page { page-break-before: always; }
      /* The figures are authored at the aspect of the space left on a landscape letter page after the
         heading — 9.4in by 5.6in — so scaling to full width fills the page without overflowing it.
         `page-break-inside: avoid` is deliberately absent: it makes Chrome shrink the whole block to
         fit rather than honoring the size, which lands the figure in the top half of the page. */
      img.fig { display: block; width: 100%; height: auto; margin-top: 4pt; }
      table.stats { width: auto; font-size: 9.5pt; margin: 0 0 2pt; }
      table.stats td { border: none; padding: 0 20pt 0 0; white-space: nowrap; }
    </style></head><body>
      <h1>AutoRIFT: runtime and peak memory, full Landsat 8/9 scene</h1>
      <p class="sub">$(r.scene.rows) &times; $(r.scene.cols) pixels &middot;
        grid $(r.grid.rows) &times; $(r.grid.cols) =
        $(@sprintf("%.2f", r.grid.rows * r.grid.cols / 1e6)) M points &middot;
        chip $(r.chip)&ndash;$(r.chip_max) px, spacing $(r.spacing),
        search radius $(r.radius), upsampling $(r.upsampling)<br>
        $(r.provenance.cpu), $(r.provenance.cores) cores,
        $(@sprintf("%.0f", r.provenance.ram_gib)) GiB &middot; $(r.provenance.os) &middot;
        Julia $(r.provenance.julia) &middot;
        AutoRIFT.jl $(r.provenance.autorift_jl) ($(r.provenance.commit)) &middot;
        autoRIFT $(r.provenance.autorift_py)</p>
      <table><thead><tr><th>configuration</th><th>threads</th><th>cores used</th><th>lazy</th>
        <th>blocks</th><th>runtime</th><th>peak RSS</th></tr></thead>
      <tbody>
    $body
      </tbody></table>
      <p class="note">* Multi-threading is not currently supported in the standalone binary.<br>
        &dagger; Mean cores used, measured as CPU time over elapsed time. autoRIFT.py's correlation loop
        is serial &mdash; it runs at one core for 85% of the run, reaching ~11 cores only during the
        OpenCV filter calls &mdash; so this is what it achieves on a 12-core machine, not a setting.</p>

      <div class="page">
        <h1>Agreement with autoRIFT.py: displacement</h1>
        <p class="sub">Identical input arrays and parameters on both sides, so the difference is the
          pipeline alone &middot; $(s.npix)&thinsp;&times;&thinsp;$(s.npix) px over Jakobshavn,
          $(s.shared) points measured by both &middot; chip $(s.chip)&ndash;$(s.chip_max) px, spacing
          $(s.spacing), radius $(s.radius), upsampling $(s.upsampling)</p>
        <table class="stats"><tr>
          <td><b>signed median bias</b> $(@sprintf("%+.4f", s.bias_dx)) px (dx),
              $(@sprintf("%+.4f", s.bias_dy)) px (dy)</td>
          <td><b>median difference</b> $(@sprintf("%.3f", s.median)) px</td>
          <td><b>correlation</b> $(@sprintf("%.4f", s.cor_dx)) (dx),
              $(@sprintf("%.4f", s.cor_dy)) (dy)</td>
          <td><b>identical to the last bit</b> at $(@sprintf("%.1f%%", 100 * s.exact)) of points;
              the tail reaches $(@sprintf("%.1f", s.steps_p99)) steps of 1/$(s.upsampling)&thinsp;px
              at p99</td>
        </tr></table>
        <img class="fig" src="$(data_uri(figs.heatmaps))">
      </div>

      <div class="page">
        <h1>Agreement with autoRIFT.py: chip size and coverage</h1>
        <p class="sub">Which level answered each point, and where the two disagree &middot; the
          histogram is binned in whole steps of 1/$(s.upsampling)&thinsp;px because the sub-pixel search
          quantizes the differences &mdash; in pixels it is one unresolvable spike at zero</p>
        <img class="fig" src="$(data_uri(figs.histograms))">
      </div>

      <div class="page">
        <h1>AutoRIFT.jl's remaining outputs</h1>
        <p class="sub">The reference reports none of these, so no comparison page can show them
          &middot; both read <b>zero</b> where the peak lay against the search boundary, so any positive
          threshold rejects those points and <code>correlation .== 0</code> finds where
          <code>search_radius</code> is too small &middot; peak SNR reads high on a tall peak over
          low-contrast ground, which is why it is the weaker gate</p>
        <table class="stats"><tr>
          <td><b>gate on correlation, not peak SNR</b></td>
          <td>by decile, exact agreement rises monotonically with correlation, 31% &rarr; 98%</td>
          <td>peak SNR sits near 80% for eight deciles, then <i>falls</i> to 45% in its highest</td>
        </tr></table>
        <img class="fig" src="$(data_uri(figs.outputs))">
      </div>
    </body></html>
    """
    src = joinpath(WORK, "table.html")
    write(src, html)
    pdf = joinpath(PLOTS, "bench_table.pdf")
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    run(pipeline(`$chrome --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=$pdf $src`;
                 stdout = devnull, stderr = devnull))
    return pdf
end

function main()
    r = "--replot" in ARGS ? JSON3.read(read(RESULTS, String)) : measure()
    println()
    println("| configuration | threads | cores used | lazy | blocks | runtime | peak RSS |")
    println("|---|---:|---:|:---:|---:|---:|---:|")
    for t in r.table
        @printf("| %s | %s | %.2f | %s | %d | %.1f s | %.0f MiB |\n",
                t.label, t.threads, t.cores, t.lazy, t.blocks, t.seconds, t.peak)
    end
    @printf("\nwrote %s\n      %s\n", RESULTS, render(r))
end

main()
