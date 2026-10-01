# The FFT plan cache and its persisted wisdom.
#
# Wisdom is a pure optimization: it must make a cold process faster and must never be able to
# make one fail. Both halves are tested, and the second matters more — a correlator that throws
# because a scratch directory is read-only would be worse than one that plans every time.

using AutoRIFT: fft_plan, ifft_plan, warm_plans!, clear_plans!, fftw_wisdom_path,
                load_fftw_wisdom!, save_fftw_wisdom!, next_fft_size,
                precompute_fftw_wisdom, _bucket_rungs, _ladder_sizes, chip_sizes,
                FFTW_WISDOM_DIR_VAR, RFFT_PLANS, IRFFT_PLANS, reset_fftw_wisdom_path!
using Scratch: with_scratch_directory

@testset "plan cache" begin
    clear_plans!()
    # The same size returns the identical plan object, which is the point: planning is
    # milliseconds and executing is microseconds, so a cache miss per grid point would dominate.
    p1 = fft_plan(96, 96)
    p2 = fft_plan(96, 96)
    @test p1 === p2
    @test fft_plan(128, 128) !== p1        # different size, different plan

    i1 = ifft_plan(96, 96)
    @test ifft_plan(96, 96) === i1

    # `warm_plans!` is what a pass calls before spawning tasks, so every size it names must be
    # resident afterwards — that is what keeps the planner lock off the parallel path.
    clear_plans!()
    sizes = [(96, 96), (128, 128)]
    warm_plans!(sizes)
    for (ny, nx) in sizes
        # A second call must not plan again; identity is how that is observable.
        @test fft_plan(ny, nx) === fft_plan(ny, nx)
        @test ifft_plan(ny, nx) === ifft_plan(ny, nx)
    end
end

@testset "no plan survives precompilation" begin
    # A regression test for a segfault, not a style preference.
    #
    # An FFTW plan is a handle to a C structure. Caching one in a `const Dict` means the
    # precompile image serialises it, and on reload the handle points nowhere — executing it
    # crashes the process inside `fftwf_execute_dft_r2c`. The precompile workload in
    # `src/AutoRIFT.jl` plans several sizes, so without `clear_plans!()` in `__init__` the package
    # segfaults on its first correlation.
    #
    # This asserts the state a freshly-loaded process must be in. It passes trivially in a session
    # that has already correlated something, so it runs in a subprocess that has only just loaded
    # the package — which is the only place the property is observable.
    script = """
        using AutoRIFT
        n = length(AutoRIFT.RFFT_PLANS) + length(AutoRIFT.IRFFT_PLANS)
        n == 0 || error("\$n plan(s) cached at load time; a deserialised FFTW plan segfaults")
        # And prove it by executing one: this is the call that crashed.
        a = [Float32((i * 7 + j * 13) % 251) / 251 for i in 1:150, j in 1:150]
        autorift(a, circshift(a, (2, 3)); chip_size = 32, search_radius = 6, chip_size_max = 32)
        print("ok")
    """
    out = read(`$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) -e $script`, String)
    @test out == "ok"
end

@testset "wisdom path is machine-specific" begin
    path = fftw_wisdom_path()
    # `nothing` is a legitimate answer — a sandbox with no writable depot — so the test accepts
    # it rather than asserting a path exists.
    if !isnothing(path)
        name = basename(path)
        # Keyed by CPU and FFTW version, because wisdom is portable across neither: it encodes
        # cache sizes and SIMD widths, and FFTW's serialisation format is its own business.
        # Reading another machine's file would produce plans tuned for the wrong hardware.
        @test occursin("fftw", name)
        @test endswith(name, ".wisdom")
        # The CPU model is interpolated into the filename, so it must have survived being made
        # filesystem-safe: no spaces, no parentheses, no slashes.
        @test !occursin(" ", name)
        @test !occursin("/", basename(path))
        @test isdirpath(dirname(path)) || isdir(dirname(path))
    end
end

@testset "the scratch directory is resolved per call, not cached" begin
    # A regression test for a bug that made the *other* wisdom tests pass for the wrong reason.
    #
    # `fftw_wisdom_path` caches the expensive part of its answer — the CPU-derived filename — but must
    # re-resolve the directory on every call, because `Scratch.with_scratch_directory` redirects it
    # dynamically and the two testsets below rely on that redirect to avoid touching the real depot.
    # When the whole path was cached, the redirect became a silent no-op: those testsets kept
    # passing while writing into the developer's own scratch space, and the "read-only depot" case
    # was no longer testing a read-only depot at all.
    outside = fftw_wisdom_path()
    if !isnothing(outside)
        mktempdir() do dir
            with_scratch_directory(dir) do
                inside = fftw_wisdom_path()
                @test !isnothing(inside)
                # The load-bearing assertion: the redirect is honoured even though `fftw_wisdom_path`
                # was already called once in this process.
                @test startswith(inside, dir)
                @test inside != outside
                # And the cached half is genuinely reused rather than re-derived differently.
                @test basename(inside) == basename(outside)
            end
        end
        # Leaving the block restores the real location.
        @test fftw_wisdom_path() == outside
    end
end

@testset "a hopeless export is not retried forever" begin
    # `save_fftw_wisdom!` clears `WISDOM_DIRTY` on every exit path, not only the successful one. None of
    # the ways it fails is worth retrying — an unwritable depot does not become writable between two
    # chip-size levels of one image pair — and leaving the flag set turned one dead export into a
    # permanent one: `warm_plans!` calls this per level per pair, so every later call re-derived the
    # path and retried the doomed write, measured at 116 us and 6.6 KiB a time.
    mktempdir() do dir
        ro = joinpath(dir, "readonly")
        mkpath(ro)
        chmod(ro, 0o500)
        try
            with_scratch_directory(ro) do
                clear_plans!()
                fft_plan(next_fft_size(149), next_fft_size(149))
                @test AutoRIFT.WISDOM_DIRTY[]          # a new size was planned
                save_fftw_wisdom!()
                # The export could not have succeeded here, and the flag is clear regardless.
                @test !AutoRIFT.WISDOM_DIRTY[]
            end
        finally
            chmod(ro, 0o700)
        end
    end
end

@testset "wisdom round-trips" begin
    # The actual claim: exporting and re-importing wisdom makes a re-plan cheap. Done in an
    # isolated scratch directory so the test cannot disturb the real one, and so a machine with
    # no writable depot still runs the rest of the file.
    mktempdir() do dir
        with_scratch_directory(dir) do
            path = fftw_wisdom_path()
            if !isnothing(path)
                # Plan something unusual, so this test is not measuring a size another testset
                # already planned.
                sz = next_fft_size(151)
                clear_plans!()
                fft_plan(sz, sz)
                save_fftw_wisdom!()
                @test isfile(path)
                @test filesize(path) > 0

                # No temporary left behind: the export writes to a unique name and renames, so a
                # crashed process cannot leave a half-written file that later runs fail to read.
                @test isempty(filter(f -> endswith(f, ".tmp"), readdir(dirname(path))))

                # Importing it is idempotent and does not throw.
                @test load_fftw_wisdom!() === nothing
            end
        end
    end
end

@testset "wisdom failures degrade rather than throw" begin
    # The half that matters most. Every filesystem touch is guarded, because production may run
    # in a read-only container, on a full disk, or with no depot at all — and in every one of
    # those cases the correct behaviour is to plan from scratch, not to fail a correlation.
    mktempdir() do dir
        ro = joinpath(dir, "readonly")
        mkpath(ro)
        chmod(ro, 0o500)              # readable and traversable, not writable
        try
            with_scratch_directory(ro) do
                # Neither of these may throw, whatever the filesystem says.
                @test load_fftw_wisdom!() === nothing
                @test save_fftw_wisdom!() === nothing
                # And correlation still works, which is the property all of this protects.
                clear_plans!()
                @test fft_plan(96, 96) !== nothing
            end
        finally
            chmod(ro, 0o700)          # so the temp dir can be removed
        end
    end

    # A corrupt file is not an error either: the next export overwrites it, and until then the
    # planner simply measures.
    mktempdir() do dir
        with_scratch_directory(dir) do
            path = fftw_wisdom_path()
            if !isnothing(path)
                mkpath(dirname(path))
                write(path, "this is not FFTW wisdom")
                AutoRIFT.WISDOM_LOADED[] = false      # allow a re-import for the test
                @test load_fftw_wisdom!() === nothing
                clear_plans!()
                @test fft_plan(96, 96) !== nothing
            end
        end
    end
end

@testset "AUTORIFT_FFTW_WISDOM_DIR overrides where wisdom lives" begin
    # The deployment knob: a process whose filesystem is discarded between jobs needs the file to
    # land somewhere that outlives it, without depending on where the depot happens to be.
    @test !haskey(ENV, FFTW_WISDOM_DIR_VAR)       # the default path must not depend on the harness
    default = fftw_wisdom_path()
    mktempdir() do dir
        withenv(FFTW_WISDOM_DIR_VAR => dir) do
            p = fftw_wisdom_path()
            @test dirname(p) == dir
            @test p != default
            # The file name still carries CPU and FFTW version, so one directory may hold several
            # machines' files and each process reads only its own.
            @test basename(p) == basename(default)
        end
    end
    @test fftw_wisdom_path() == default           # restored once the override is gone
end

@testset "the bucket ladder is data-independent" begin
    # The property `precompute_fftw_wisdom` rests on: the rungs are fixed, so a pair's time
    # separation changes which are reached and never what they are.
    r16 = _bucket_rungs(16)
    r64 = _bucket_rungs(64)
    @test issorted(r16) && issorted(r64)
    @test r16 ⊆ r64                      # widening the bound only adds rungs
    @test all(1 .<= r64 .<= 64)
    @test 16 in r64 && 32 in r64 && 64 in r64        # the octaves themselves are rungs
    # Four per octave: the powers of two and three evenly spaced steps between consecutive ones.
    @test count(r -> 16 <= r <= 32, r64) == 5        # 16, 20, 24, 28, 32
end

@testset "precompute_fftw_wisdom plans the reachable set" begin
    mktempdir() do dir
        withenv(FFTW_WISDOM_DIR_VAR => dir) do
            # Deliberately tiny: one chip level and a 4 px bound, so the set is a handful of sizes.
            p = AutoRIFT.params(; chip_size = (X = 8, Y = 8), chip_size_max = (X = 8, Y = 8),
                                 grid_spacing = (X = 4, Y = 4))
            rungs = _bucket_rungs(4)
            expected = _ladder_sizes(only(chip_sizes(p)), rungs)

            clear_plans!()
            n = precompute_fftw_wisdom(p; max_search_radius = 4)
            @test n == length(expected)
            # Deterministic rather than timed: every size it claims to have planned is in the cache.
            @test all(sz -> haskey(RFFT_PLANS, sz), expected)
            @test all(sz -> haskey(IRFFT_PLANS, sz), expected)
            # And it persisted them, which is what makes the next process warm.
            @test isfile(fftw_wisdom_path())
            @test filesize(fftw_wisdom_path()) > 0

            @test_throws "at least 1" precompute_fftw_wisdom(p; max_search_radius = 0)
        end
    end
end

@testset "measuring FFTW wisdom on a cold machine is warned about" begin
    # The warning exists for measurement honesty: plan measurement lands inside whatever the caller is
    # timing, so a benchmark or profile from a run with no FFTW wisdom describes planning rather than
    # correlation. It also surfaces FFTW wisdom that is not being found — a container that discards its
    # filesystem, or `AUTORIFT_FFTW_WISDOM_DIR` pointing somewhere unwritable.
    mktempdir() do dir
        withenv(FFTW_WISDOM_DIR_VAR => dir) do
            reset_fftw_wisdom_path!()          # as a fresh process starts: nothing imported, nothing said
            clear_plans!()
            @test_logs (:warn, r"No FFTW wisdom") match_mode = :any warm_plans!(((64, 64),))

            # Once per process. The condition cannot change mid-run, and a correlation reaches this
            # path thousands of times.
            clear_plans!()
            @test_logs min_level = Base.CoreLogging.Warn warm_plans!(((72, 72),))

            # And silent whenever FFTW wisdom *was* imported, because measuring a size the file does
            # not hold is normal: a pass's widest bucket is the grid's own maximum rather than a rung,
            # so no precomputed file can cover it.
            AutoRIFT.WISDOM_IMPORTED[] = true
            AutoRIFT.WISDOM_WARNED[] = false
            clear_plans!()
            @test_logs min_level = Base.CoreLogging.Warn warm_plans!(((80, 80),))
        end
    end
    reset_fftw_wisdom_path!()
end
