module FFFSamplerStaticArraysExt

if isdefined(Base, :get_extension)
    using FFFSampler: FFFSampler
    using LogDensityProblems: LogDensityProblems
    using StaticArrays: StaticArrays
    using PDMats: PDMats
    using Random: Random
else
    using ..FFFSampler: FFFSampler
    using ..LogDensityProblems: LogDensityProblems
    using ..StaticArrays: StaticArrays
    using ..PDMats: PDMats
    using ..Random: Random
end


struct StaticLogDensityProblem{T} <: FFFSampler.StaticLogDensityProblem
    inner_problem::T
end
FFFSampler.StaticLogDensityProblem(inner_problem) = StaticLogDensityProblem(inner_problem)
LogDensityProblems.dimension(ℓ::StaticLogDensityProblem) = LogDensityProblems.dimension(ℓ.inner_problem)
LogDensityProblems.capabilities(::StaticLogDensityProblem{T}) where {T} = LogDensityProblems.capabilities(T)
LogDensityProblems.logdensity(ℓ::StaticLogDensityProblem, x) = LogDensityProblems.logdensity(ℓ.inner_problem, x)
LogDensityProblems.logdensity_and_gradient(ℓ::StaticLogDensityProblem, x) = LogDensityProblems.logdensity_and_gradient(ℓ.inner_problem, x)
LogDensityProblems.logdensity_gradient_and_hessian(ℓ::StaticLogDensityProblem, x) = LogDensityProblems.logdensity_gradient_and_hessian(ℓ.inner_problem, x)

FFFSampler._randn(rng, model::FFFSampler.StaticLogDensityProblem) = StaticArrays.randn(rng, StaticArrays.SVector{LogDensityProblems.dimension(model)})

function FFFSampler.refresh(rng::Random.AbstractRNG, model::FFFSampler.StaticLogDensityProblem, sampler, p=nothing)
    prop = FFFSampler._randn(rng, model)
    prop = PDMats.whiten(sampler.Minv, prop)  # we store the inverse, so it's whiten and not unwhiten
    if p === nothing
        return prop
    else
        ρ = sampler.ρfresh
        return ρ * p  +  √(1 - ρ^2) * prop
    end
end

end