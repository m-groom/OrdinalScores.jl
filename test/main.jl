using Test
using CategoricalArrays
using CategoricalDistributions: UnivariateFinite
using StatisticalMeasuresBase
using Distributions: pdf, support
using StatisticalMeasures
using OrdinalScores

# -- Helpers ---------------------------------------------------------------

# Construct a UnivariateFinite over the exact pool of y (ordered categorical)
# probs is a Dict(level=>prob) or Pair... form; unspecified levels get prob 0.
function uf_from_probs(y_pool::CategoricalVector{T}, probs::AbstractDict) where {T}
    labs = levels(y_pool)  # pool labels (CategoricalValue-compatible)
    ps   = [get(probs, ℓ, 0.0) for ℓ in labs]
    UnivariateFinite(labs, ps; pool=y_pool)  # ensures pool compatibility/order
end

# Scalar (per-observation) RPS re-implementation used for "expected" values in tests
# This must match the mathematical definition; default normalised by (K-1).
function rps_scalar(p::UnivariateFinite, y::CategoricalValue; normalise::Bool=true)
    labs = levels(y)
    K = length(labs)
    j = findfirst(isequal(y), labs)::Int
    s = 0.0
    cum = 0.0
    @inbounds for k in 1:(K-1)
        cum += pdf(p, labs[k])          # F_k
        d = cum - (k < j ? 0.0 : 1.0)   # F_k - O_k (O_k = 1 for k ≥ j)
        s += d*d
    end
    normalise ? s/(K-1) : s
end

# Convenience to aggregate with SMBase semantics
function expected_aggregate(ms::AbstractVector; weights=nothing, mode=Mean())
    StatisticalMeasuresBase.aggregate(ms; weights=weights, mode=mode)
end

# Composite weights (observation weights ⨉ class-weights) per SMBase
composite(y, w, cw) = StatisticalMeasuresBase.CompositeWeights(y, w, cw)
composite(y, cw)    = StatisticalMeasuresBase.CompositeWeights(y, cw)

# -- Basic functionality ---------------------------------------------------

@testset "RPS: basic single/multiple observation behavior" begin
    # Ordered 3-class pool
    ypool = categorical(["low","med","high"]; ordered=true)

    # One observation
    y1 = categorical(["med"]; ordered=true, levels=levels(ypool))[1]
    p1 = uf_from_probs(ypool, Dict("low"=>0.2, "med"=>0.5, "high"=>0.3))
    @test RPS()( [p1], [y1] ) ≈ rps_scalar(p1, y1)

    # Multiple observations
    y = categorical(["low","high","med","low"]; ordered=true, levels=levels(ypool))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("low"=>0.7, "med"=>0.2, "high"=>0.1)),
        uf_from_probs(ypool, Dict("low"=>0.1, "med"=>0.3, "high"=>0.6)),
        uf_from_probs(ypool, Dict("low"=>0.2, "med"=>0.5, "high"=>0.3)),
        uf_from_probs(ypool, Dict("low"=>1.0, "med"=>0.0, "high"=>0.0)),
    ]
    ms = [rps_scalar(ŷ[i], y[i]) for i in eachindex(y)]
    @test measurements(RPS(), ŷ, y) ≈ ms
    @test RPS()(ŷ, y) ≈ expected_aggregate(ms)
end

@testset "RPS: normalization flag" begin
    ypool = categorical(["a","b","c","d"]; ordered=true)
    y = categorical(["a","d","c","b"]; ordered=true, levels=levels(ypool))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("a"=>0.4,"b"=>0.3,"c"=>0.2,"d"=>0.1)),
        uf_from_probs(ypool, Dict("a"=>0.1,"b"=>0.2,"c"=>0.3,"d"=>0.4)),
        uf_from_probs(ypool, Dict("a"=>0.2,"b"=>0.2,"c"=>0.3,"d"=>0.3)),
        uf_from_probs(ypool, Dict("a"=>0.8,"b"=>0.1,"c"=>0.05,"d"=>0.05)),
    ]
    ms_norm   = [rps_scalar(ŷ[i], y[i]; normalise=true)  for i in eachindex(y)]
    ms_unnorm = [rps_scalar(ŷ[i], y[i]; normalise=false) for i in eachindex(y)]
    @test all(ms_unnorm .≈ (length(levels(ypool))-1) .* ms_norm)
    @test RPS(normalise=false)(ŷ, y) ≈ expected_aggregate(ms_unnorm)
end

@testset "RPS: degenerate perfect-forecast gives zero" begin
    ypool = categorical(["low","med","high"]; ordered=true)
    y = categorical(["low","med","high","high"]; ordered=true, levels=levels(ypool))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("low"=>1.0)),
        uf_from_probs(ypool, Dict("med"=>1.0)),
        uf_from_probs(ypool, Dict("high"=>1.0)),
        uf_from_probs(ypool, Dict("high"=>1.0)),
    ]
    @test RPS()(ŷ, y) == 0.0
end

# -- Weights and class-weights --------------------------------------------

@testset "RPS: observation weights only" begin
    ypool = categorical(["low","med","high"]; ordered=true)
    y = categorical(["low","high","med","low"]; ordered=true, levels=levels(ypool))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("low"=>0.7,"med"=>0.2,"high"=>0.1)),
        uf_from_probs(ypool, Dict("low"=>0.1,"med"=>0.3,"high"=>0.6)),
        uf_from_probs(ypool, Dict("low"=>0.2,"med"=>0.5,"high"=>0.3)),
        uf_from_probs(ypool, Dict("low"=>1.0)),
    ]
    w = [1.0, 2.0, 0.5, 0.0] # include a zero-weight example
    ms = [rps_scalar(ŷ[i], y[i]) for i in eachindex(y)]

    # Expected aggregate per StatisticalMeasuresBase Mean() convention:
    expected = expected_aggregate(ms; weights=w)
    @test RPS()(ŷ, y, w) ≈ expected

    # Rescaling invariance
    expected = expected_aggregate(ms; weights=w, mode=IMean())
    m = measurements(RPS(), ŷ, y)
    @test StatisticalMeasuresBase.aggregate(m; weights=10 .* w, mode=IMean()) ≈ expected # scaled weights
end

@testset "RPS: class weights only" begin
    ypool = categorical(["low","med","high"]; ordered=true)
    y = categorical(["low","high","med","low","med"]; ordered=true, levels=levels(ypool))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("low"=>0.6,"med"=>0.3,"high"=>0.1)),
        uf_from_probs(ypool, Dict("low"=>0.1,"med"=>0.3,"high"=>0.6)),
        uf_from_probs(ypool, Dict("low"=>0.2,"med"=>0.5,"high"=>0.3)),
        uf_from_probs(ypool, Dict("low"=>1.0)),
        uf_from_probs(ypool, Dict("low"=>0.4,"med"=>0.4,"high"=>0.2)),
    ]
    cw = Dict("low"=>1.0, "med"=>2.0, "high"=>3.0) # keys must match pool exactly
    ms = [rps_scalar(ŷ[i], y[i]) for i in eachindex(y)]
    expected = expected_aggregate(ms; weights=composite(y, cw))
    @test RPS()(ŷ, y, cw) ≈ expected
end

@testset "RPS: observation + class weights (multiplicative)" begin
    ypool = categorical(["low","med","high"]; ordered=true)
    y = categorical(["low","high","med","low"]; ordered=true, levels=levels(ypool))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("low"=>0.7,"med"=>0.2,"high"=>0.1)),
        uf_from_probs(ypool, Dict("low"=>0.1,"med"=>0.3,"high"=>0.6)),
        uf_from_probs(ypool, Dict("low"=>0.2,"med"=>0.5,"high"=>0.3)),
        uf_from_probs(ypool, Dict("low"=>1.0)),
    ]
    w  = [1.0, 2.0, 0.5, 1.0]
    cw = Dict("low"=>1.0, "med"=>0.5, "high"=>2.0)
    ms = [rps_scalar(ŷ[i], y[i]) for i in eachindex(y)]
    expected = expected_aggregate(ms; weights=composite(y, w, cw))
    @test RPS()(ŷ, y, w, cw) ≈ expected

    # 3-arg vs 4-arg API: passing `nothing` for weights should match 3-arg class_weights call
    @test RPS()(ŷ, y, nothing, cw) ≈ RPS()(ŷ, y, cw)
end

# -- Binary (K=2) relationship to BrierLoss --------------------------------

@testset "K=2: RPS == 0.5 * BrierLoss() (binary case)" begin
    # Binary ordered pool. MLJ convention: second level is the "positive" class.
    ypool = categorical(["no","yes"]; ordered=true)
    y = categorical(["no","yes","yes","no","no"]; ordered=true, levels=levels(ypool))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("no"=>0.8, "yes"=>0.2)),
        uf_from_probs(ypool, Dict("no"=>0.3, "yes"=>0.7)),
        uf_from_probs(ypool, Dict("no"=>0.4, "yes"=>0.6)),
        uf_from_probs(ypool, Dict("no"=>0.2, "yes"=>0.8)),
        uf_from_probs(ypool, Dict("no"=>0.9, "yes"=>0.1)),
    ]
    # RPS aggregate:
    rps_val = RPS()(ŷ, y)
    # StatisticalMeasures' BrierLoss uses the multiclass formula even for K=2
    bl_val  = BrierLoss()(ŷ, y)
    @test rps_val ≈ 0.5 * bl_val atol=1e-12
end

# -- Argument checks / fussy behavior --------------------------------------

@testset "Pool and order checks; unordered targets rejected" begin
    # Same labels but different order in pools → should error
    ypool = categorical(["low","med","high"]; ordered=true)

    y = categorical(["low","med"]; ordered=true, levels=reverse(levels(ypool)))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("low"=>0.7,"med"=>0.2,"high"=>0.1)),
        uf_from_probs(ypool, Dict("low"=>0.1,"med"=>0.3,"high"=>0.6)), # mismatched pool order
    ]
    @test_throws ArgumentError RPS()(ŷ, y)  # check_pools enforces pool equality with order

    # Unordered target should be rejected by your measure (either via method signature or extra_check)
    y_unordered = categorical(["low","high"]; ordered=false)
    p = uf_from_probs(y_unordered, Dict("low"=>0.6,"high"=>0.4))
    # If the implementation restricts to OrderedFactor at the method level, this will
    # be a MethodError; otherwise ensure an ArgumentError is thrown by an explicit check.
    @test_throws Exception RPS()([p], [y_unordered[1]])
end

@testset "Class-weights dictionary must match pool (no missing keys)" begin
    ypool = categorical(["a","b","c"]; ordered=true)
    y = categorical(["a","b","c"]; ordered=true, levels=levels(ypool))
    ŷ = UnivariateFinite[
        uf_from_probs(ypool, Dict("a"=>0.2,"b"=>0.3,"c"=>0.5)),
        uf_from_probs(ypool, Dict("a"=>0.3,"b"=>0.4,"c"=>0.3)),
        uf_from_probs(ypool, Dict("a"=>0.9,"b"=>0.05,"c"=>0.05)),
    ]
    cw = Dict("a"=>1.0, "b"=>1.0, "c"=>1.0)
    cw_missing = Dict("a"=>1.0, "b"=>1.0)      # missing "c"
    cw_extra   = Dict("a"=>1.0, "b"=>1.0, "c"=>1.0, "d"=>2.0) # extra key

    @test_throws ArgumentError RPS()(ŷ, y, cw_missing)
    @test RPS()(ŷ, y, cw) ≈ RPS()(ŷ, y, cw_extra) # Extra key is ignored
end
