# How well each quality measure predicts disagreement with the reference.
#
# `correlation` and `peak_ratio` both come back from every point and both look like quality numbers, so
# the question a user actually has is which one to threshold. This answers it against an *independent*
# label -- "this point disagrees with `autoRIFT.py` by more than 0.25 px" -- rather than against either
# measure's own notion of a good peak.
#
# Reads the stage-2 dump, so it measures the same run the rest of `tools/ab/` reports on and needs no
# imagery of its own. Run `stage2_julia.jl` and `stage2_python.py` first.
#
#   julia --project=tools/ab tools/ab/peak_ratio_skill.jl

using Printf, Statistics

const DIR = joinpath(@__DIR__, "stage2")

# The manifest carries each array's dtype and shape, so nothing here restates them.
function read_dumps(dir)
    shapes = Dict{String,Tuple{DataType,Tuple{Int,Int}}}()
    for line in eachline(joinpath(dir, "manifest.txt"))
        (isempty(line) || startswith(line, '#')) && continue
        f = split(line)
        length(f) == 3 || continue
        T = f[2] == "Float32" ? Float32 : f[2] == "Float64" ? Float64 :
            f[2] == "Int32" ? Int32 : continue
        dims = parse.(Int, split(f[3], 'x'))
        length(dims) == 2 || continue
        shapes[f[1]] = (T, (dims[1], dims[2]))
    end
    rd(name, T, sz) = reshape(reinterpret(T, read(joinpath(dir, name * ".bin"))), sz)
    out = Dict{String,Matrix}()
    for (name, (T, sz)) in shapes
        out[name] = Matrix(rd(name, T, sz))
    end
    # Python's grid is derived independently and can differ by a row or column; the reference dumps
    # carry no manifest entry, so their shape is read from the file it does write.
    ps = Tuple(parse.(Int, split(strip(read(joinpath(dir, "python_shape.txt"), String)))))
    for name in ("python_dx", "python_dy")
        out[name] = Matrix(rd(name, Float32, ps))
    end
    return out
end

# Area under the ROC curve for "does a low `score` predict `label`", by the rank identity: AUC is the
# probability that a random labelled point scores below a random unlabelled one. Ties count a half,
# which is what makes an uninformative measure come out at exactly 0.5 rather than near it.
#
# Computed from ranks rather than by sweeping thresholds: a threshold sweep has to choose its
# thresholds, and the choice shows up in the answer for a measure whose values cluster.
function auc_low_is_bad(score::AbstractVector{Float64}, label::AbstractVector{Bool})
    ord = sortperm(score)
    s = score[ord]
    l = label[ord]
    npos = count(l)
    nneg = length(l) - npos
    (npos == 0 || nneg == 0) && return NaN
    # Mid-ranks within each run of equal scores, so a tie contributes 0.5 rather than 0 or 1.
    ranksum = 0.0
    i = 1
    while i <= length(s)
        j = i
        while j < length(s) && s[j + 1] == s[i]
            j += 1
        end
        mid = (i + j) / 2
        for k in i:j
            l[k] && (ranksum += mid)
        end
        i = j + 1
    end
    # `ranksum` is the rank sum of the positives, lowest-first. A perfect low-is-bad measure puts every
    # positive at the bottom, giving the minimum possible sum, hence the complement.
    return 1.0 - (ranksum - npos * (npos + 1) / 2) / (npos * nneg)
end

# Rate of `label` within each decile of `score`, lowest decile first.
function decile_rates(score::AbstractVector{Float64}, label::AbstractVector{Bool})
    ord = sortperm(score)
    n = length(ord)
    return [begin
                lo = div((d - 1) * n, 10) + 1
                hi = div(d * n, 10)
                mean(label[ord[lo:hi]])
            end for d in 1:10]
end

# Spearman rank correlation: Pearson on ranks, with ties averaged.
function spearman(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    rank(v) = begin
        ord = sortperm(v)
        r = zeros(Float64, length(v))
        i = 1
        while i <= length(ord)
            j = i
            while j < length(ord) && v[ord[j + 1]] == v[ord[i]]
                j += 1
            end
            for k in i:j
                r[ord[k]] = (i + j) / 2
            end
            i = j + 1
        end
        r
    end
    return cor(rank(x), rank(y))
end

function main()
    d = read_dumps(DIR)
    js = size(d["julia_dx"])
    ps = size(d["python_dx"])
    nr, nc = min(js[1], ps[1]), min(js[2], ps[2])
    crop(A) = A[1:nr, 1:nc]

    jdx, jdy = crop(d["julia_dx"]), crop(d["julia_dy"])
    corr = crop(d["julia_correlation"])
    ppr = crop(d["julia_peak_ratio"])
    interp = crop(d["julia_interpolated"])

    # The reference reports a search-window corner where it failed rather than a no-measurement, and
    # `autorift` reports `NaN`. Only points both sides measured can be labelled, and only directly
    # measured ones carry a `correlation` -- a filled point has no surface of its own.
    pdx, pdy = crop(d["python_dx"]), crop(d["python_dy"])
    ok = [!isnan(jdx[i]) && !isnan(jdy[i]) && !isnan(pdx[i]) && !isnan(pdy[i]) &&
          interp[i] == 0 && !isnan(corr[i]) && !isnan(ppr[i])
          for i in eachindex(jdx)]
    idx = findall(ok)

    # `hypot` rather than a per-axis test: the label is "the displacement disagrees", and a
    # disagreement split across both axes is one disagreement.
    err = [hypot(Float64(jdx[i] - pdx[i]), Float64(jdy[i] - pdy[i])) for i in idx]
    label = err .> 0.25

    c = Float64.(corr[idx])
    r = Float64.(ppr[idx])

    @printf("points compared:      %d\n", length(idx))
    @printf("labelled (>0.25 px):  %d (%.1f%%)\n", count(label), 100 * mean(label))
    @printf("peak_ratio: finite %d, Inf %d\n", count(isfinite, r), count(isinf, r))
    @printf("railed (ratio == 0):  %d\n", count(iszero, r))

    # `Inf` is an honest value -- an unrivalled peak -- and it ranks above every finite one, which the
    # rank-based AUC handles directly. No filtering, since dropping the cleanest points would flatter
    # the measure.
    for (name, s) in (("correlation", c), ("peak_ratio", r))
        @printf("\n%s\n", name)
        @printf("  AUC                %.3f\n", auc_low_is_bad(s, label))
        rates = decile_rates(s, label)
        @printf("  decile rates (worst-scoring first): %s\n",
                join((@sprintf("%.1f%%", 100 * v) for v in rates), " "))
        @printf("  worst decile %.1f%%   best decile %.1f%%\n", 100 * rates[1], 100 * rates[10])
        fin = [v for v in s if isfinite(v)]
        @printf("  median %.3f   10th %.3f   90th %.3f  (finite only)\n",
                median(fin), quantile(fin, 0.1), quantile(fin, 0.9))
    end

    # Whether the two measures say the same thing. Finite points only: a rank correlation over `Inf`
    # ties would report the size of that tie group rather than the relationship.
    both = [k for k in eachindex(c) if isfinite(c[k]) && isfinite(r[k])]
    @printf("\nSpearman(correlation, peak_ratio) = %.3f  over %d finite points\n",
            spearman(c[both], r[both]), length(both))

    # Within a band of fixed `correlation`, does the ratio still rank correctly? A measure that only
    # works by proxying for peak height would collapse here.
    println("\npeak_ratio AUC within bands of fixed correlation:")
    for (lo, hi) in ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
        sel = [k for k in eachindex(c) if lo <= c[k] < hi]
        n = length(sel)
        if n >= 200 && 0 < count(label[sel]) < n
            @printf("  %.1f-%.1f  n=%6d  rate %.1f%%  AUC %.3f\n",
                    lo, hi, n, 100 * mean(label[sel]), auc_low_is_bad(r[sel], label[sel]))
        else
            @printf("  %.1f-%.1f  n=%6d  (too few, or all one class)\n", lo, hi, n)
        end
    end
end

main()
