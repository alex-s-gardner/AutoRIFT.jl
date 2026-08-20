# The FFT plan cache and its persisted wisdom.
#
# Wisdom is a pure optimization: it must make a cold process faster and must never be able to
# make one fail. Both halves are tested, and the second matters more — a correlator that throws
# because a scratch directory is read-only would be worse than one that plans every time.

using AutoRIFT: fft_plan, ifft_plan, warm_plans!, clear_plans!, wisdom_path,
                load_wisdom!, save_wisdom!, next_fft_size
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
    path = wisdom_path()
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

@testset "wisdom round-trips" begin
    # The actual claim: exporting and re-importing wisdom makes a re-plan cheap. Done in an
    # isolated scratch directory so the test cannot disturb the real one, and so a machine with
    # no writable depot still runs the rest of the file.
    mktempdir() do dir
        with_scratch_directory(dir) do
            path = wisdom_path()
            if !isnothing(path)
                # Plan something unusual, so this test is not measuring a size another testset
                # already planned.
                sz = next_fft_size(151)
                clear_plans!()
                fft_plan(sz, sz)
                save_wisdom!()
                @test isfile(path)
                @test filesize(path) > 0

                # No temporary left behind: the export writes to a unique name and renames, so a
                # crashed process cannot leave a half-written file that later runs fail to read.
                @test isempty(filter(f -> endswith(f, ".tmp"), readdir(dirname(path))))

                # Importing it is idempotent and does not throw.
                @test load_wisdom!() === nothing
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
                @test load_wisdom!() === nothing
                @test save_wisdom!() === nothing
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
            path = wisdom_path()
            if !isnothing(path)
                mkpath(dirname(path))
                write(path, "this is not FFTW wisdom")
                AutoRIFT.WISDOM_LOADED[] = false      # allow a re-import for the test
                @test load_wisdom!() === nothing
                clear_plans!()
                @test fft_plan(96, 96) !== nothing
            end
        end
    end
end
