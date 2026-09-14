# Does a smaller block cut peak memory, and what does it cost in time?
#
# The question this answers is production sizing: the pipeline runs tens of millions of image pairs,
# and a scheduler packing those onto instances is bounded by resident memory per worker rather than by
# the speed of any one pair. `process_block_size` is the one knob a caller has over that, and the
# figure that decides whether it is usable is peak memory against runtime at each size.
#
#   julia --project=tools/ab tools/ab/mem_blocks.jl                    # sweep, then render
#   julia --project=tools/ab tools/ab/mem_blocks.jl --replot           # re-render from the traces
#   julia --project=tools/ab tools/ab/mem_blocks.jl --blocks 0,512     # a subset of the sweep
#   julia --project=tools/ab tools/ab/mem_blocks.jl --threads 1        # serial
#
# One child process per block size (`mem_child.jl`), for the reason `benchmark/memory.jl` records:
# peak memory is a high-water mark, so two configurations measured in one process both report the
# larger. What is new here is that each child *traces* its resident memory rather than reading the
# mark once, and reads the profiler's stacks at the sample where the trace peaked — so the output says
# where the peak was, not only how large it was.
#
# Needs the scene planes `bench_scene.jl` writes:
#
#   julia --project=tools/ab tools/ab/bench_scene.jl ~/data/autorift/bench

using CairoMakie, Printf, Serialization, Statistics

include(joinpath(@__DIR__, "memtrace.jl"))

const HERE = @__DIR__
const ROOT = dirname(dirname(HERE))
const PLOTS = joinpath(HERE, "plots")
# Traces live beside the scene planes rather than in the repository: one is a few hundred KiB and the
# sweep is re-run whenever the correlator changes.
const WORK = get(ENV, "AUTORIFT_BENCH_DIR", expanduser("~/data/autorift/bench"))
const TRACES = joinpath(WORK, "traces")

# The sweep. 0 is one block over the whole scene — the baseline every other row is read against.
#
# 8192 is deliberately included even though it partitions this scene into only 9 blocks, which is
# fewer than the machine has threads: every block is then in flight at once, so the run holds nine
# 8331² working sets and peak *exceeds* the untiled run. That is the shape of the failure a block size
# chosen by block count rather than by block area produces, and it belongs on the curve rather than
# being quietly omitted.
const BLOCKS = [0, 8192, 4096, 2048, 1024, 512, 256, 128]

argvalue(flag, default) = (i = findfirst(==(flag), ARGS);
                           isnothing(i) ? default : ARGS[i + 1])

# ---------------------------------------------------------------------------
# Running the sweep
# ---------------------------------------------------------------------------

# One child, with its progress line passed through to this terminal.
#
# `stderr` is inherited rather than captured, which is what lets the child's rewritten progress line
# reach the user live; `stdout` is read as data. The child prints `READY` before correlating and `DONE`
# after, so a failure before the run is distinguishable from one during it.
function run_child(block::Int, nthreads::Int)
    out = joinpath(TRACES, "trace_$(block)_t$(nthreads).jls")
    script = joinpath(HERE, "mem_child.jl")
    # `-t N,1`: the second figure is the interactive thread the sampler runs on, without which it
    # would queue behind the correlation's own tasks and leave gaps in the trace where the peak is.
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$HERE -t $nthreads,1
           $script $WORK $block $nthreads $out`
    @printf("%s\n", block == 0 ? "untiled (one block over the scene)" : "block $(block) px")
    flush(stdout)
    run(pipeline(cmd; stdout = stdout, stderr = stderr))
    return deserialize(out)
end

function sweep(blocks::Vector{Int}, nthreads::Int)
    mkpath(TRACES)
    isfile(joinpath(WORK, "dims.txt")) ||
        error("no scene planes in $WORK; run `bench_scene.jl $WORK` first")
    return [run_child(b, nthreads) for b in blocks]
end

# Traces already on disk, for `--replot`.
function load_traces(blocks::Vector{Int}, nthreads::Int)
    rs = Any[]
    for b in blocks
        f = joinpath(TRACES, "trace_$(b)_t$(nthreads).jls")
        isfile(f) || continue
        push!(rs, deserialize(f))
    end
    isempty(rs) && error("no traces in $TRACES; run without `--replot` first")
    return rs
end

# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

label_of(m) = m.block == 0 ? "untiled" : "$(m.block) px"

# `phys_footprint` and not `resident_size`, as the headline figure, and the difference is the whole
# reason both are traced. Every configuration here reads its imagery through a memory mapping, so
# `resident_size` counts the clean file-backed pages the read populated — pages the kernel evicts
# under pressure rather than OOM-killing for. Charging a lazily-read run for its own page cache would
# credit the untiled run, which touches each page once, against a blocked run whose halo touches some
# of them several times.
function print_table(rs)
    base = first(r.meta for r in rs if r.meta.block == 0)
    @printf("\n%-10s %7s %10s %10s %9s %9s %8s %9s %8s\n",
            "block", "blocks", "peak MiB", "resident", "live MiB", "seconds", "vs base", "read amp",
            "measured")
    for r in rs
        m = r.meta
        @printf("%-10s %7d %10.0f %10.0f %9.0f %9.1f %8.2fx %8.2fx %9d\n",
                label_of(m), m.nblocks, m.peak_footprint / 2^20, m.peak_resident / 2^20,
                m.peak_live / 2^20, m.seconds, m.peak_footprint / base.peak_footprint,
                m.read_pixels / (2 * prod(m.scene)), m.measured)
    end
    # Where each configuration's peak sat, which is the question a single figure cannot answer. A peak
    # attributed to the collector rather than to a stage says the memory is being fought over.
    println("\nwhat was running at each peak (share of running samples in the peak window):")
    for r in rs
        m = r.meta
        total = max(1, sum(last, m.stacks; init = 0))
        @printf("  %-10s peak at %5.1f%% of the run%s\n", label_of(m),
                100 * m.peak_index / max(1, length(r.trace.tick)),
                m.profile_overflow ? "  [profile buffer filled: attribution is partial]" : "")
        for (lbl, n) in first(m.stacks, 3)
            @printf("      %5.1f%%  %s\n", 100 * n / total, lbl)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

const FIG_SIZE = (1560, 1000)

# Seconds from the start of each trace, on the profiler's clock so the axis and the peak attribution
# are the same quantity.
seconds_axis(t) = (t.tick .- first(t.tick)) ./ t.tick_hz

# The trade curve, the traces, and where each peak sits — three panels, because no one of them answers
# the sizing question alone. The curve says which size to pick; the traces say whether a peak is a
# plateau the whole run holds or a spike one stage sets; the attribution says what is holding it.
function render(rs; nthreads::Int)
    mkpath(PLOTS)
    sorted = sort(rs; by = r -> (r.meta.block == 0 ? typemax(Int) : r.meta.block))
    base = first(r.meta for r in rs if r.meta.block == 0)
    m1 = first(sorted).meta

    fig = Figure(; size = FIG_SIZE, fontsize = 15)
    Label(fig[0, 1:2],
          "Peak memory against block size — Landsat 8/9 scene, $(m1.scene[1])×$(m1.scene[2]) px, " *
          "$(nthreads) thread$(nthreads == 1 ? "" : "s")";
          fontsize = 20, font = :bold, halign = :left)
    Label(fig[1, 1:2],
          "chip $(m1.chip)–$(m1.chip_max) px, spacing $(m1.spacing), search radius $(m1.radius), " *
          "upsampling $(m1.upsampling) · grid $(m1.grid[1])×$(m1.grid[2]) · " *
          "memory-mapped input, physical footprint sampled every 4 ms";
          fontsize = 13, color = :gray35, halign = :left)

    # Panel 1: the trade. Peak on the left axis and runtime on the right, against block size, because
    # the decision is a trade between them and reading it off two separate panels invites picking a
    # size that is good on one axis only.
    ax1 = Axis(fig[2, 1]; xlabel = "block size (pixels)", ylabel = "peak memory (MiB)",
               xscale = log2, title = "The trade", titlealign = :left)
    ax1r = Axis(fig[2, 1]; ylabel = "runtime (s)", yaxisposition = :right, xscale = log2,
                yticklabelcolor = :firebrick, ylabelcolor = :firebrick)
    hidespines!(ax1r)
    hidexdecorations!(ax1r)
    linkxaxes!(ax1, ax1r)

    tiled = [r.meta for r in sorted if r.meta.block != 0]
    bx = [Float64(m.block) for m in tiled]
    scatterlines!(ax1, bx, [m.peak_footprint / 2^20 for m in tiled];
                  color = :steelblue, markersize = 11, linewidth = 2.5, label = "peak, blocked")
    scatterlines!(ax1r, bx, [m.seconds for m in tiled];
                  color = :firebrick, markersize = 11, linewidth = 2.5, linestyle = :dash)
    hlines!(ax1, [base.peak_footprint / 2^20]; color = :gray30, linestyle = :dot, linewidth = 2,
            label = "peak, untiled")
    hlines!(ax1r, [base.seconds]; color = (:firebrick, 0.45), linestyle = :dot, linewidth = 2)
    ax1.xticks = (bx, string.(Int.(bx)))
    axislegend(ax1; position = :lt, framevisible = false, labelsize = 12)
    ylims!(ax1, 0, nothing)
    ylims!(ax1r, 0, nothing)

    # Panel 2: peak against the working-set arithmetic, which is the mechanism. `BlockBuffers` holds
    # nine block-sized arrays per task, so a run's buffer footprint is the block's *area* times the
    # task count — and if that is what sets the peak, the points lie on a line through it. Plotted
    # against the measurement rather than asserted, since the read amplification and the pooled
    # correlation workspaces are also in the total.
    ax2 = Axis(fig[2, 2]; xlabel = "block working set: max read window × tasks (MiB)",
               ylabel = "peak memory (MiB)", xscale = log2, yscale = log2,
               title = "Peak tracks block area, not block count", titlealign = :left)
    # Nine arrays per block buffer set: two `Float32` planes, three `Float32` filter outputs, four
    # `Bool` masks. Matches `AutoRIFT.BlockBuffers`.
    bytes_per_px = 2 * 4 + 3 * 4 + 4 * 1
    ws = [prod(m.maxread) * bytes_per_px * min(m.nblocks, nthreads) / 2^20 for m in tiled]
    scatterlines!(ax2, ws, [m.peak_footprint / 2^20 for m in tiled];
                  color = :steelblue, markersize = 11, linewidth = 2.5)
    for (m, w) in zip(tiled, ws)
        text!(ax2, w, m.peak_footprint / 2^20; text = "  $(m.block)", fontsize = 11,
              align = (:left, :top), color = :gray25)
    end
    hlines!(ax2, [base.peak_footprint / 2^20]; color = :gray30, linestyle = :dot, linewidth = 2)

    # Panel 3: the traces themselves, which is what makes a peak interpretable. A plateau the run
    # holds throughout and a spike one stage sets are the same number and different problems.
    ax3 = Axis(fig[3, 1:2]; xlabel = "seconds", ylabel = "physical footprint (MiB)",
               title = "The trace, and where each run peaked", titlealign = :left)
    colors = cgrad(:viridis, max(2, length(sorted)); categorical = true)
    for (k, r) in pairs(sorted)
        t, m = r.trace, r.meta
        x = seconds_axis(t)
        y = t.footprint ./ 2^20
        lines!(ax3, x, y; color = colors[k], linewidth = 2, label = label_of(m))
        scatter!(ax3, [x[m.peak_index]], [y[m.peak_index]];
                 color = colors[k], marker = :star5, markersize = 17, strokewidth = 0.5)
    end
    axislegend(ax3; position = :rb, framevisible = false, labelsize = 12, nbanks = 2)
    ylims!(ax3, 0, nothing)

    # What held the peak, as text beside the traces: the stack is the finding, and a caption is the
    # only honest place for it — it is a label, not a quantity to be plotted.
    lines_out = String[]
    for r in sorted
        m = r.meta
        total = max(1, sum(last, m.stacks; init = 0))
        top = isempty(m.stacks) ? "(no samples in the peak window)" : first(first(m.stacks))
        share = isempty(m.stacks) ? 0.0 : 100 * last(first(m.stacks)) / total
        push!(lines_out, @sprintf("%-9s %6.0f MiB   %5.1f%%  %s", label_of(m),
                                  m.peak_footprint / 2^20, share, first(top, 96)))
    end
    Label(fig[4, 1:2], "At the peak:\n" * join(lines_out, "\n");
          fontsize = 11.5, font = :regular, halign = :left, justification = :left,
          color = :gray20)

    rowsize!(fig.layout, 2, Relative(0.36))
    rowsize!(fig.layout, 3, Relative(0.34))
    path = joinpath(PLOTS, "mem_blocks_t$(nthreads).png")
    save(path, fig; px_per_unit = 2)
    return path
end

function main()
    nthreads = parse(Int, argvalue("--threads", string(min(12, Sys.CPU_THREADS))))
    blocks = parse.(Int, split(argvalue("--blocks", join(BLOCKS, ',')), ','))
    rs = "--replot" in ARGS ? load_traces(blocks, nthreads) : sweep(blocks, nthreads)
    print_table(rs)
    @printf("\nwrote %s\n", render(rs; nthreads))
    return nothing
end

main()
