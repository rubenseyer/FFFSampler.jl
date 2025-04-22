struct DiscreteFFF{F<:FFF, T}  <: AbstractMCMC.AbstractSampler
    inner_sampler::F
    "discretization step size"
    Δ::T
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::DiscreteFFF;
    initial_params=nothing,
    kwargs...
)
    sample, _ = AbstractMCMC.step(rng, model_wrapper, sampler.inner_sampler; initial_params)
    τ = 1.0/sum(sample.Λ)
    return sample, (τ, sample)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::DiscreteFFF,
    state::Tuple{Float64,<:AbstractFFFTransition};
    kwargs...
)
    τ, sample = state
    τ -= sampler.Δ
    while τ < 0
        sample, _ = AbstractMCMC.step(rng, model_wrapper, sampler.inner_sampler, sample)
        τ += 1.0/sum(sample.Λ)
    end
    
    return sample, (τ, sample)
end

function discretize(samples::AbstractVector{<:AbstractFFFTransition{T1}}, N) where {T1}
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