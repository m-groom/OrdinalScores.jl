# OrdinalScores.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Julia](https://img.shields.io/badge/julia-1.9%20%7C%201.10%20%7C%201.11-9558B2.svg)](https://julialang.org)

**OrdinalScores.jl** provides proper scoring rules for evaluating probabilistic predictions on ordered categorical data, with full integration into the [MLJ](https://github.com/JuliaAI/MLJ.jl) and [StatisticalMeasures.jl](https://github.com/JuliaAI/StatisticalMeasures.jl) ecosystems. Currently, it implements the Ranked Probability Score (RPS).

## Overview

The Ranked Probability Score (RPS) is a proper scoring rule for assessing the quality of probabilistic predictions when the target variable has a natural ordering. Unlike standard classification metrics that ignore the ordinal structure, RPS penalises predictions proportionally to the distance between predicted and actual categories.

### Key Features

- **Proper scoring rule**: RPS is strictly proper, encouraging honest probabilistic predictions
- **Ordinal-aware**: Accounts for the natural ordering of categories (e.g., "low" < "medium" < "high")
- **Flexible inputs**: Supports both deterministic (`CategoricalValue`) and probabilistic (`UnivariateFinite`) predictions
- **Weight handling**: Fully supports observation weights and class weights via StatisticalMeasuresBase
- **MLJ integration**: Works seamlessly with MLJ's model evaluation and tuning workflows

### Mathematical Background

RPS is computed as the sum of squared differences between cumulative distribution functions:

```
RPS = (1/(K-1)) × Σᵢ (Fᵢ - Gᵢ)²
```

where `F` is the predicted CDF, `G` is the true CDF (indicator function), and `K` is the number of ordered categories. For binary classification (K=2), RPS equals 0.5 × Brier score.

## Installation

```julia
using Pkg
Pkg.add("OrdinalScores")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/OrdinalScores.jl")
```

## Quick Start

```julia
using MLJBase
using StatisticalMeasures
using CategoricalArrays
using OrdinalScores

# Create ordinal data (ordered categories)
y_true = categorical(
    ["low", "medium", "high", "medium", "low"],
    ordered=true
)

# Probabilistic predictions (UnivariateFinite distributions)
y_pred = [
    UnivariateFinite(levels(y_true) , [0.7, 0.2, 0.1]; pool=y_true, ordered=true),
    UnivariateFinite(levels(y_true) , [0.1, 0.6, 0.3]; pool=y_true, ordered=true),
    UnivariateFinite(levels(y_true) , [0.1, 0.2, 0.7]; pool=y_true, ordered=true),
    UnivariateFinite(levels(y_true) , [0.2, 0.5, 0.3]; pool=y_true, ordered=true),
    UnivariateFinite(levels(y_true) , [0.8, 0.15, 0.05]; pool=y_true, ordered=true),
]

# Evaluate predictions using RPS
score = RPS()(y_pred, y_true)
println("Ranked Probability Score: ", score)
```
### Use with MLJ model evaluation
```julia
using MLJ
X, y = make_blobs(100, 3; centers=3)
y = coerce(y, OrderedFactor)        
levels!(y, sort(levels(y)))   # ensure 1 < 2 < 3 ordering is explicit
@load DecisionTreeClassifier pkg=DecisionTree
Tree = @load DecisionTreeClassifier pkg=DecisionTree
model = Tree()
mach = machine(model, X, y)
evaluate!(mach, resampling=CV(nfolds=5), measure=RPS())
```

## Licence

This software is distributed under the MIT Licence.
