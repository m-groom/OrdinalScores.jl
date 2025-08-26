# Shared helper to accumulate squared CDF differences up to K-1
@inline function _rps(Fp, target_at, K::Int, normalise::Bool)
    s = zero(eltype(Fp))
    @inbounds for k in 1:(K - 1)
        d = Fp[k] - target_at(k)
        s += d*d
    end
    return normalise && K > 1 ? s/(K-1) : s
end

# ---- atomic measure + multimeasure constructor ----
struct RPSOnScalars
    normalise::Bool
end

@inline function (m::RPSOnScalars)(p::UnivariateFinite, y::CategoricalValue)
    # Require an ordered target and matching pool/order with the prediction
    if !CategoricalArrays.isordered(y)
        throw(ArgumentError("target must be an ordered categorical value"))
    end
    lvls = levels(y)
    if support(p) != lvls
        throw(ArgumentError("prediction and target pools (with order) must match"))
    end
    K = length(lvls)
    @inbounds begin
        Fp = cumsum(pdf.(Ref(p), lvls))
        j = levelcode(y)
        onev = one(eltype(Fp))
        zerov = zero(eltype(Fp))
        target_at = (k -> (k < j ? zerov : onev))
        return _rps(Fp, target_at, K, m.normalise)
    end
end

@inline function (m::RPSOnScalars)(p::UnivariateFinite, y::UnivariateFinite)
    lvls = support(y)
    K = length(lvls)
    @inbounds begin
        Fp = cumsum(pdf.(Ref(p), lvls))
        Fy = cumsum(pdf.(Ref(y), lvls))
        target_at = (k -> Fy[k])
        return _rps(Fp, target_at, K, m.normalise)
    end
end

StatisticalMeasuresBase.orientation(::RPSOnScalars) = Loss()
StatisticalMeasuresBase.external_aggregation_mode(::RPSOnScalars) = Mean()
StatisticalMeasuresBase.observation_scitype(::RPSOnScalars) = Union{Missing,OrderedFactor}
StatisticalMeasuresBase.human_name(::RPSOnScalars) = "ranked probability score"

# Minimal constructor using standard wrappers: robust_measure → fussy_measure
function RankedProbabilityScore(; normalise::Bool=true)
    StatisticalMeasuresBase.fussy_measure(StatisticalMeasuresBase.robust_measure(StatisticalMeasuresBase.multimeasure(
        RPSOnScalars(normalise)
    )))
end

# Create aliases
const rps = RankedProbabilityScore
const ranked_probability_score = RankedProbabilityScore
const RPS = RankedProbabilityScore
