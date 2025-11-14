module FFFSampler

# Structure is very inspired by https://github.com/torfjelde/AutomaticMALA.jl

import AbstractMCMC
import Random
import LogDensityProblems
using LinearAlgebra
using PDMats
import Statistics

# Utility definitions of (un)whiten for UniformScaling
function PDMats.whiten!(r::AbstractVecOrMat, a::UniformScaling, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    ldiv!(r, sqrt(a.λ), x)
end
function PDMats.unwhiten!(r::AbstractVecOrMat, a::UniformScaling, x::AbstractVecOrMat)
    PDMats.@check_argdims axes(r) == axes(x)
    mul!(r, x, sqrt(a.λ))
end
PDMats.whiten(a::UniformScaling, x::AbstractVecOrMat) = x / sqrt(a.λ)
PDMats.unwhiten(a::UniformScaling, x::AbstractVecOrMat) = sqrt(a.λ) * x
PDMats.invquad(a::UniformScaling, x::AbstractVector) = dot(x,x) / a.λ

# Utility max
#struct Plus end
#const ⁺ = Plus()
#Base.:*(x::Number, ::Plus) = max(x, zero(x))
pos(x) = max(x, zero(x))
finiteorzero(x) = isfinite(x) ? x : zero(x)

# Compatibility with staticarrays
abstract type StaticLogDensityProblem end
_randn(rng, model) = randn(rng, LogDensityProblems.dimension(model))

# Balancing functions
abstract type BalancingFunction end
struct MHBalancing <: BalancingFunction end
struct BarkerBalancing <: BalancingFunction end
struct SqrtBalancing <: BalancingFunction end
balancing_g(::Type{MHBalancing}, t) = min(t,one(t))
balancing_g(::Type{BarkerBalancing}, t) = 2*t/(t+1)  # normalize to g(1) = 1
balancing_g(::Type{SqrtBalancing}, t) = sqrt(t)

abstract type RebalancedSampler{B<:BalancingFunction} <: AbstractMCMC.AbstractSampler end
balancing_g(::RebalancedSampler{B}, t) where {B<:BalancingFunction} = balancing_g(B, t)

@enum Action FORWARD=1 FLIP FRESH MIRROR
Base.to_index(a::Action) = Int(a)
Base.getindex(t::Tuple, a::Action) = Base.getindex(t, Int(a))
Base.:(==)(a::Int, b::Action) = a == Int(b)
Base.:(==)(a::Action, b::Int) = Int(a) == b

include("sampler.jl")
include("observable.jl")
include("discrete.jl")
include("adapt.jl")
include("mirror.jl")
include("bouncy.jl")
include("rhmc.jl")
include("rgw.jl")

export FFF, trajectory
export Discretizer, DiscreteFFF, discretize
export Mirrorer, FFFMirror
export BouncyJump

## Extensions

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" include(
            "../ext/FFFSamplerTuringExt.jl"
        )
        @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" include(
            "../ext/FFFSamplerStaticArraysExt.jl"
        )
    end
end

end
