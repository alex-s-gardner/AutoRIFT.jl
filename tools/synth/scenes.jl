# Synthetic scenes with analytically known displacement, for scoring every implementation against
# truth rather than against each other.
#
# Each case writes a bundle -- two images, the true displacement at every pixel, and the grid -- in the
# same layout `tools/ab` uses, so the Python readers there work unchanged.
#
#   julia --project=tools/synth tools/synth/scenes.jl [case ...]
#
# With no arguments every case is generated. Named cases regenerate a subset.

using Printf, Random, Serialization, Statistics
using FFTW, Interpolations

const OUT = joinpath(@__DIR__, "scenes")
const CACHE = get(ENV, "AUTORIFT_TESTDATA", expanduser("~/data/autorift/tests"))

# Scene side, and the correlation parameters every case shares unless it overrides them.
const NPIX = 1024
const RADIUS = 20
const SPACING = 8
const UPSAMPLING = 16

# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

"""
    white_texture(n; seed, cutoff = 0.5) -> Matrix{Float32}

Random texture band-limited to `cutoff` of Nyquist, normalized to unit standard deviation.

Band-limited rather than spectrally white, because a sub-pixel shift of a white field is not a
well-posed operation: energy at Nyquist cannot be translated by any interpolator, local or exact. The
cost is measurable and is not small — recovering a known 0.729 px shift gives 0.0000 px error at 0.5
Nyquist against 0.0263 px at full band, and that 0.026 px floor would otherwise be attributed to the
estimators under test. An exact Fourier phase-ramp shift shows the same floor, so it is the texture and
not the warp.

Half Nyquist also matches real imagery better than white noise does: an optical sensor's point-spread
function and its atmosphere are both low-pass.
"""
function white_texture(n::Integer; seed::Integer, cutoff::Real = 0.5)
    F = FFTW.fftshift(FFTW.fft(randn(Random.MersenneTwister(seed), n, n)))
    mid = n ÷ 2 + 1
    lim = cutoff * n / 2
    for j in 1:n, i in 1:n
        hypot(i - mid, j - mid) > lim && (F[i, j] = 0)
    end
    a = real(FFTW.ifft(FFTW.ifftshift(F)))
    s = std(a)
    return Float32.(s > 0 ? a ./ s : a)
end

"""
    landsat_texture(n; seed) -> Matrix{Float32}

An `n`-by-`n` patch of the cached Landsat reference scene, high-pass filtered as the correlator would
see it.

Real texture is anisotropic — crevasse fields are directional — and whether that changes the comparison
cannot be established from isotropic synthetic texture. The patch is taken from the fast trunk, where
the real disagreement lives.

No low-pass is applied: this imagery is already band-limited, carrying 95.8% of its power within half
Nyquist and 90.8% within a quarter, which is why [`white_texture`](@ref)'s cutoff of half Nyquist is the
comparable synthetic choice rather than an arbitrary one.
"""
function landsat_texture(n::Integer; seed::Integer)
    path = joinpath(CACHE, "window.jls")
    isfile(path) || error("real texture needs the cache at $path; see tools/realdata/README.md")
    w = deserialize(path)
    # A fixed patch rather than a random one: the case must be reproducible, and this window is the
    # trunk the real anomaly was found in.
    rng = Random.MersenneTwister(seed)
    r0 = 1678 - n ÷ 2 + rand(rng, -64:64)
    c0 = 2495 - n ÷ 2 + rand(rng, -64:64)
    r0 = clamp(r0, 1, size(w.reference, 1) - n + 1)
    c0 = clamp(c0, 1, size(w.reference, 2) - n + 1)
    patch = Matrix{Float32}(w.reference[r0:(r0 + n - 1), c0:(c0 + n - 1)])
    # Standardized so the noise SNRs below mean the same thing here as on white texture.
    m, s = mean(patch), std(patch)
    return s > 0 ? (patch .- m) ./ s : patch .- m
end

# ---------------------------------------------------------------------------
# Displacement fields
# ---------------------------------------------------------------------------
#
# Each returns `(dx, dy)` in pixels at every pixel of an `n`-by-`n` scene, with `dx` along columns and
# `dy` down rows. These are the ground truth: no implementation is consulted.

# A uniform shift. The case every arm must recover exactly, since there is no deformation for a rigid
# chip to misrepresent.
translation(n, sx, sy) = (fill(Float32(sx), n, n), fill(Float32(sy), n, n))

"""
    shear(n; amp, width, ratio) -> (dx, dy)

A shear margin: `dx` ramping from 0 to `amp` across `width` rows through a `tanh`, with `dy` a fixed
`ratio` of it.

The geometry of a glacier trunk edge — fast ice beside slow, with a steep cross-flow gradient — and the
one condition the anomalous patch in the real scene has that the rest of the grid does not. The maximum
gradient is `amp / (2 * width)` px/px, which is what a chip of side `c` turns into `c * amp / (2 * width)`
px of differential motion inside its own footprint.
"""
function shear(n; amp = 10.0, width = 40.0, ratio = -0.6)
    dx = Matrix{Float32}(undef, n, n)
    for j in 1:n, i in 1:n
        dx[i, j] = amp * 0.5 * (1 + tanh((i - n / 2) / width))
    end
    return dx, Float32(ratio) .* dx
end

"""
    rigid_rotation(n; degrees) -> (dx, dy)

Rotation about the scene centre. Displacement grows with radius, so the local gradient is uniform —
which separates "the chip sees deformation" from "the displacement is large", two things a shear field
confounds.
"""
function rigid_rotation(n; degrees = 0.5)
    θ = deg2rad(degrees)
    c, s = cos(θ), sin(θ)
    dx = Matrix{Float32}(undef, n, n)
    dy = Matrix{Float32}(undef, n, n)
    mid = (n + 1) / 2
    for j in 1:n, i in 1:n
        x, y = j - mid, i - mid
        dx[i, j] = (c * x - s * y) - x
        dy[i, j] = (s * x + c * y) - y
    end
    return dx, dy
end

"""
    divergence(n; rate) -> (dx, dy)

Uniform radial stretch: every chip sees its texture dilated rather than sheared or rotated. A rigid
translation model cannot represent dilation either, and this is the mode that isolates it.
"""
function divergence(n; rate = 0.004)
    dx = Matrix{Float32}(undef, n, n)
    dy = Matrix{Float32}(undef, n, n)
    mid = (n + 1) / 2
    for j in 1:n, i in 1:n
        dx[i, j] = rate * (j - mid)
        dy[i, j] = rate * (i - mid)
    end
    return dx, dy
end

# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------

"""
    gradient_magnitude(dx, dy) -> Matrix{Float32}

The local deformation rate: the Frobenius norm of the displacement gradient, in px per px.

This is the variable the disagreement between implementations tracks, so it is what results are
stratified by. It is the right quantity because a chip of side `c` centred on a point sees roughly
`c * gradient` px of differential motion across its own footprint — the thing a single rigid
translation cannot represent — whereas the displacement *magnitude* is something a rigid chip handles
exactly.

All four partial derivatives, not just `∂dx/∂row`: shear, rotation and divergence load different
components, and a measure blind to any of them would rank those modes wrongly. Central differences
inside, one-sided at the frame.
"""
function gradient_magnitude(dx::AbstractMatrix, dy::AbstractMatrix)
    axes(dx) == axes(dy) ||
        throw(DimensionMismatch("components must match: $(axes(dx)) vs $(axes(dy))"))
    out = similar(dx, Float32)
    rows, cols = axes(dx)
    for j in cols, i in rows
        ip, im = min(i + 1, last(rows)), max(i - 1, first(rows))
        jp, jm = min(j + 1, last(cols)), max(j - 1, first(cols))
        # The denominator is the actual index span, so a one-sided difference at the frame is scaled
        # like a derivative rather than reading half as steep as it is.
        xr = (dx[ip, j] - dx[im, j]) / max(ip - im, 1)
        xc = (dx[i, jp] - dx[i, jm]) / max(jp - jm, 1)
        yr = (dy[ip, j] - dy[im, j]) / max(ip - im, 1)
        yc = (dy[i, jp] - dy[i, jm]) / max(jp - jm, 1)
        out[i, j] = sqrt(xr^2 + xc^2 + yr^2 + yc^2)
    end
    return out
end

"""
    warp(img, dx, dy) -> Matrix{Float32}

`img` displaced by `(dx, dy)`: the result at `(i, j)` samples `img` at `(i - dy, j - dx)`, so a feature
at `(i, j)` in `img` appears at `(i + dy, j + dx)` in the result.

Cubic B-spline interpolation, which is not the limiting error here: against an exact Fourier phase-ramp
shift — the analytically correct answer for a band-limited field — the recovered displacement is
identical to 4 decimal places at band limits of 0.5, 0.8 and 1.0 Nyquist. What remains is the
upsampling quantization step, which is a property of the estimators under test rather than of the warp.

`Cubic(Line(OnGrid()))` and not `Cubic(Reflect(OnGrid()))`: the reflecting boundary condition fails to
interpolate its own nodes on broadband data, by up to 1.85 in units of the texture's standard deviation.
That would corrupt the truth field silently, since a warp is only ever compared against the correlator's
answer and never against the input at the nodes.
"""
function warp(img::AbstractMatrix, dx::AbstractMatrix, dy::AbstractMatrix)
    axes(img) == axes(dx) == axes(dy) ||
        throw(DimensionMismatch("image and displacement must match: " *
                                "$(axes(img)) vs $(axes(dx)) vs $(axes(dy))"))
    itp = interpolate(Float64.(img), BSpline(Cubic(Line(OnGrid()))))
    ext = extrapolate(itp, Reflect())
    out = similar(img, Float32)
    for j in axes(img, 2), i in axes(img, 1)
        out[i, j] = ext(i - dy[i, j], j - dx[i, j])
    end
    return out
end

# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

"""
    add_noise(img, snr; seed) -> Matrix{Float32}

Additive Gaussian noise at the given signal-to-noise *amplitude* ratio, measured against `img`'s own
standard deviation so the level means the same thing on white and on Landsat texture.
"""
function add_noise(img::AbstractMatrix, snr::Real; seed::Integer)
    isinf(snr) && return Matrix{Float32}(img)
    σ = std(img) / snr
    return Float32.(img .+ σ .* randn(Random.MersenneTwister(seed), Float32, size(img)))
end

"""
    decorrelate(img, fraction; seed) -> Matrix{Float32}

Replace `fraction` of the scene's variance with texture unrelated to `img`, in patches rather than per
pixel.

A crevasse field does not decorrelate pixel by pixel: crevasses open and close, so whole neighbourhoods
change their appearance between acquisitions while the surrounding ice does not. Per-pixel replacement
would instead behave like white noise, which `add_noise` already covers.
"""
function decorrelate(img::AbstractMatrix, fraction::Real; seed::Integer, patch::Integer = 24)
    fraction <= 0 && return Matrix{Float32}(img)
    rng = Random.MersenneTwister(seed)
    out = Matrix{Float32}(img)
    fresh = randn(rng, Float32, size(img))
    nr, nc = size(img)
    for j0 in 1:patch:nc, i0 in 1:patch:nr
        rand(rng) < fraction || continue
        rows, cols = i0:min(i0 + patch - 1, nr), j0:min(j0 + patch - 1, nc)
        out[rows, cols] .= fresh[rows, cols]
    end
    return out
end

"""
    to_uint8(img) -> Matrix{Float32}

The reference's `uniform_data_type` quantization for `DataType = 0` (`autoRIFT.py:359-385`): rescale so
±3σ spans 0–255, round, clip.

Returned as `Float32` holding integer values rather than as `UInt8`, so both sides correlate byte-
identical numbers and the quantization is the only difference from the float case. This is the path
ITS_LIVE production runs.
"""
function to_uint8(img::AbstractMatrix)
    m = mean(img)
    # The reference's own unbiased correction, so the scaling matches to the last bit.
    n = length(img)
    s = std(img; corrected = false) * sqrt(n / (n - 1.0))
    scaled = (Float64.(img) .- (m - 3s)) ./ (6s) .* 256
    return Float32.(round.(clamp.(scaled, 0, 255)))
end

# ---------------------------------------------------------------------------
# The case grid
# ---------------------------------------------------------------------------

"""
    Case

One scene to be correlated by every arm. `field` names the deformation, `chip`/`radius`/`spacing` the
correlation parameters, and the remaining fields the degradations applied.

`gradient_amp` scales the shear field; it is the variable the disagreement tracks, so it is recorded
per case rather than derived at scoring time.
"""
Base.@kwdef struct Case
    name::String
    field::Symbol
    texture::Symbol = :white
    chip::Int = 16
    radius::Int = RADIUS
    spacing::Int = SPACING
    npix::Int = NPIX
    gradient_amp::Float64 = 10.0
    shear_width::Float64 = 40.0
    shift::Tuple{Float64,Float64} = (0.0, 0.0)
    degrees::Float64 = 0.5
    rate::Float64 = 0.004
    snr::Float64 = Inf
    noise_secondary_only::Bool = false
    decorrelation::Float64 = 0.0
    quantize::Bool = false
    seed::Int = 20260830
end

"""
    cases() -> Vector{Case}

The designed grid: a base case, a one-factor-at-a-time sweep around it, and a full crossing of
gradient amplitude with chip size.

Not a full factorial. Every factor crossed with every other is 432 scenes, most of which would
re-measure what one factor already establishes. The two factors crossed fully are gradient and chip
size, because the effect under study is differential motion *within a chip footprint* — the product of
the two — so their interaction is the mechanism rather than an extra dimension.
"""
function cases()
    cs = Case[]

    # Deformation mode. Translation is the gate: no deformation, so every arm must be exact.
    push!(cs, Case(; name = "translate_integer", field = :translation, shift = (3.0, -2.0)))
    push!(cs, Case(; name = "translate_subpixel", field = :translation, shift = (2.375, -1.625)))
    push!(cs, Case(; name = "rotation_0p5deg", field = :rotation, degrees = 0.5))
    push!(cs, Case(; name = "rotation_2deg", field = :rotation, degrees = 2.0))
    push!(cs, Case(; name = "divergence", field = :divergence, rate = 0.004))

    # Gradient amplitude crossed with chip size: the interaction that is the mechanism.
    for amp in (2.5, 10.0, 20.0), chip in (16, 32, 64)
        push!(cs, Case(; name = @sprintf("shear_a%g_c%d", amp, chip), field = :shear,
                       gradient_amp = amp, chip))
    end

    # Noise, one factor at a time off the base case.
    for snr in (10.0, 4.0, 2.0)
        push!(cs, Case(; name = @sprintf("shear_snr%g", snr), field = :shear, snr))
    end
    # Noise on the secondary only: the asymmetric case a real pair presents, since the two
    # acquisitions differ in illumination and atmosphere.
    push!(cs, Case(; name = "shear_snr4_secondary", field = :shear, snr = 4.0,
                   noise_secondary_only = true))

    # Decorrelation: texture that changes between acquisitions rather than moving.
    for f in (0.1, 0.25, 0.5)
        push!(cs, Case(; name = @sprintf("shear_decorr%g", f), field = :shear, decorrelation = f))
    end

    # Quantization, on the shear case and on a clean translation. Both, because the question is
    # whether uint8 changes the disagreement, and a translation case says whether it costs accuracy
    # at all.
    push!(cs, Case(; name = "shear_uint8", field = :shear, quantize = true))
    push!(cs, Case(; name = "translate_subpixel_uint8", field = :translation,
                   shift = (2.375, -1.625), quantize = true))

    # Real texture, warped by the same analytic fields.
    push!(cs, Case(; name = "shear_landsat", field = :shear, texture = :landsat))
    push!(cs, Case(; name = "translate_subpixel_landsat", field = :translation,
                   texture = :landsat, shift = (2.375, -1.625)))
    push!(cs, Case(; name = "shear_landsat_uint8", field = :shear, texture = :landsat,
                   quantize = true))

    return cs
end

# ---------------------------------------------------------------------------
# Building and writing a case
# ---------------------------------------------------------------------------

"""
    truth(c) -> (dx, dy)

The analytic displacement field for a case, in pixels.
"""
function truth(c::Case)
    n = c.npix
    c.field === :translation && return translation(n, c.shift[1], c.shift[2])
    c.field === :shear && return shear(n; amp = c.gradient_amp, width = c.shear_width)
    c.field === :rotation && return rigid_rotation(n; degrees = c.degrees)
    c.field === :divergence && return divergence(n; rate = c.rate)
    throw(ArgumentError("unknown deformation field $(c.field)"))
end

"""
    build(c) -> (reference, secondary, dx_true, dy_true)

The image pair for a case, with the degradations applied in acquisition order: warp, then
decorrelation, then noise, then quantization.

That order is the physical one and it matters. Noise added before warping would be interpolated along
with the signal and so become correlated between the two images, which is not what sensor noise does;
quantizing before adding noise would let the noise carry sub-quantum information the sensor could not
have recorded.
"""
function build(c::Case)
    n = c.npix
    tex = c.texture === :white ? white_texture(n; seed = c.seed) :
          c.texture === :landsat ? landsat_texture(n; seed = c.seed) :
          throw(ArgumentError("unknown texture source $(c.texture)"))
    dxt, dyt = truth(c)

    ref = Matrix{Float32}(tex)
    sec = warp(tex, dxt, dyt)

    if c.decorrelation > 0
        sec = decorrelate(sec, c.decorrelation; seed = c.seed + 1)
    end
    if isfinite(c.snr)
        c.noise_secondary_only || (ref = add_noise(ref, c.snr; seed = c.seed + 2))
        sec = add_noise(sec, c.snr; seed = c.seed + 3)
    end
    if c.quantize
        ref, sec = to_uint8(ref), to_uint8(sec)
    end
    return ref, sec, dxt, dyt
end

"""
    grid_axes(c) -> (rows, cols)

The pixel indices, 1-based, the case is correlated at.

The margin keeps every chip and its whole search window inside the scene: half the chip, plus the
search radius, plus one. Derived here rather than taken from `_build_grid` so the Python arms can
reproduce it from the manifest alone.
"""
function grid_axes(c::Case)
    margin = cld(c.chip, 2) + c.radius + 1
    ax = (margin + 1):(c.spacing):(c.npix - margin)
    return ax, ax
end

dump_array(dir, name, A) = begin
    open(joinpath(dir, name * ".bin"), "w") do io
        write(io, A)
    end
    (name, string(eltype(A)), size(A))
end

"""
    write_case(c) -> String

Generate a case and write its bundle. Returns the directory.

The bundle holds the two images, the truth sampled at the grid points, the grid itself, and a manifest
naming every array's dtype and shape — the same convention `tools/ab/stage2_julia.jl` writes and
`tools/ab/stage2_python.py` reads, so the Python arms need no reader of their own.
"""
function write_case(c::Case)
    dir = joinpath(OUT, c.name)
    mkpath(dir)
    ref, sec, dxt, dyt = build(c)
    rows, cols = grid_axes(c)

    # Truth at the grid points, and the grid in 1-based pixel centres. The Python arms subtract one.
    gx = Float64[Float64(j) for _ in rows, j in cols]
    gy = Float64[Float64(i) for i in rows, _ in cols]
    tdx = Float32[dxt[i, j] for i in rows, j in cols]
    tdy = Float32[dyt[i, j] for i in rows, j in cols]
    # Computed on the full-resolution field and then sampled, not from the decimated grid: a central
    # difference over the 8 px grid spacing would understate the gradient of a 40 px-wide shear margin.
    gmag = gradient_magnitude(dxt, dyt)
    tgrad = Float32[gmag[i, j] for i in rows, j in cols]

    manifest = Tuple{String,String,Tuple}[]
    push!(manifest, dump_array(dir, "reference", ref))
    push!(manifest, dump_array(dir, "secondary", sec))
    push!(manifest, dump_array(dir, "grid_x", gx))
    push!(manifest, dump_array(dir, "grid_y", gy))
    push!(manifest, dump_array(dir, "true_dx", tdx))
    push!(manifest, dump_array(dir, "true_dy", tdy))
    push!(manifest, dump_array(dir, "true_gradient", tgrad))

    open(joinpath(dir, "manifest.txt"), "w") do io
        println(io, "# name dtype shape")
        for (name, T, sz) in manifest
            println(io, name, " ", T, " ", join(sz, "x"))
        end
        println(io, "chip ", c.chip)
        println(io, "radius ", c.radius)
        println(io, "grid_spacing ", c.spacing)
        println(io, "npix ", c.npix)
        println(io, "upsampling ", UPSAMPLING)
        println(io, "quantize ", c.quantize ? 1 : 0)
        println(io, "is_shear ", c.field === :shear ? 1 : 0)
    end
    # Every knob as text, so a results table can be stratified by any factor without re-deriving it
    # from the case name. Text and not a serialized struct: the Python arms read this too, and a
    # `Serialization` blob would need the `Case` definition in scope to load even in Julia.
    open(joinpath(dir, "case.txt"), "w") do io
        for f in fieldnames(Case)
            v = getfield(c, f)
            println(io, f, " ", v isa Tuple ? join(v, ",") : v)
        end
    end
    return dir
end

function main()
    mkpath(OUT)
    all = cases()
    wanted = isempty(ARGS) ? all : filter(c -> c.name in ARGS, all)
    if isempty(wanted)
        names = join((c.name for c in all), "\n  ")
        error("no case matched $(ARGS). Known cases:\n  $names")
    end
    for c in wanted
        t = @elapsed dir = write_case(c)
        # The maximum gradient determines whether a chip spans differential motion, and it is the
        # independent variable of the whole comparison, so it is reported per case. Taken over both
        # axes and both components: a field that varies only along columns, as `divergence` does, has
        # a zero row-difference and would otherwise print as uniform motion.
        g = maximum(gradient_magnitude(truth(c)...))
        @printf("%-28s %s %4d px chip  max deformation %.4f px/px  %.1f s\n",
                c.name, c.texture === :white ? "white  " : "landsat", c.chip, g, t)
    end
    @printf("\n%d case(s) in %s\n", length(wanted), OUT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
