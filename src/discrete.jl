struct DiscreteFFF{T}  <: AbstractMCMC.AbstractSampler
    inner_sampler::FFF{T}
    "discretization step size"
    Δ::T
end
DiscreteFFF(args...; Δ=1.0) = DiscreteFFF(FFF(args...), Δ)
DiscreteFFF(sampler::FFF) = DiscreteFFF(sampler, 1.0)

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::DiscreteFFF;
    initial_params=nothing,
    kwargs...
)
    return AbstractMCMC.step(rng, model_wrapper, sampler.inner_sampler; initial_params)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::DiscreteFFF,
    state::FFFSampler.FFFTransition;
    kwargs...
)
    t = Random.randexp(rng)/sum(state.Λ)
    while t < sampler.Δ
        state, _ = AbstractMCMC.step(rng, model_wrapper, sampler.inner_sampler, state)
        t += Random.randexp(rng)/sum(state.Λ)
    end
    
    return state, state
end

function discretize(samples::AbstractVector{<:FFFSampler.AbstractFFFTransition{T1}}, N) where {T1}
    ts = cumsum(1.0/sum(t.Λ) for t in samples)
    Δ = ts[end]/N
    xs = Vector{T1}(undef, N)
    xs[1] = samples[1].current.q
    t = 0.0
    j = 1
    for i in 2:N
        t += Δ
        while t > ts[j]
            j += 1
        end
        xs[i] = samples[j].current.q
    end
    return xs
end