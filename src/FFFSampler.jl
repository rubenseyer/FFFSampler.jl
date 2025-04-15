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

include("sampler.jl")
include("observable.jl")
include("discrete.jl")

export FFF, DiscreteFFF, trajectory

## Extensions

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" include(
            "../ext/FFFSamplerTuringExt.jl"
        )
    end
end

end
