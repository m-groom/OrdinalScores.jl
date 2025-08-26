# --- Atomic scorer (one observation) ---
# Normalised by (K-1) by default (toggle with the constructor parameter)
@inline function _rps(p::UnivariateFinite{S}, y::CategoricalValue{S}, normalise::Bool) where {S<:OrderedFactor}
    lvls = levels(y)                # ordered levels from the pool
    K = length(lvls)
    @inbounds begin
        # cumulative predicted probs in pool order
        F = cumsum(pdf.(Ref(p), lvls))
        # index of observed class
        j = findfirst(isequal(y), lvls)
        # cumulative observed one-hot: 1 up to j, then 0
        s = zero(eltype(F))
        for k in 1:(K-1)
            d = F[k] - (k <= j ? one(s) : zero(s))
            s += d*d
        end
        return normalise ? s/(K-1) : s
    end
end

# ---- public measure constructor + traits (via @combination) ----
# This builds a multi-measure that broadcasts _rps_atomic over observations,
# adds argument checks, supports `nothing` weights, etc.
@combination(
    RankedProbabilityScore(; normalise=true) =
        multimeasure( (ŷ, y) -> _rps(ŷ, y, normalise) ),
    kind_of_proxy       = LearnAPI.Distribution(),          # ŷ are distributions
    observation_scitype = Union{OrderedFactor, Missing},    # ordered classes only
    orientation         = Loss(),
    external_aggregation_mode = Mean(),
    human_name = "ranked probability score",
)

# Create aliases
const rps = RankedProbabilityScore
const ranked_probability_score = RankedProbabilityScore
const RPS = RankedProbabilityScore
