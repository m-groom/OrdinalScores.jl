module OrdinalScores

using StatisticalMeasuresBase
using CategoricalArrays
using CategoricalDistributions: UnivariateFinite
using LearnAPI
using ScientificTypesBase: OrderedFactor

# Include measures
include("main.jl")
export RankedProbabilityScore, rps, ranked_probability_score, RPS

end # module OrdinalScores
