struct Discretizer{S, T}  <: AbstractMCMC.AbstractSampler
    inner_sampler::S
    "discretization step size"
    Δ::T
end
const DiscreteFFF = Discretizer

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::Discretizer;
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
    sampler::Discretizer,
    state;
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

discretize(samples, N) = _discretize(trajectory(samples)..., N)
discretize(xs, ws, N) = _discretize(cumsum(ws), xs, N)
function _discretize(ts, zs::AbstractVector{T}, N) where {T}
    Δ = ts[end]/N
    xs = Vector{T}(undef, N)
    xs[1] = zs[1]
    t = 0.0
    j = 1
    for i in 2:N
        t += Δ
        while t > ts[j]
            j += 1
        end
        xs[i] = zs[j]
    end
    return xs
end

# Here we expand to a discrete sample with the same information (stochastically).
# In principle this means adding virtual rejections to the trajectory.
# This works well if we have bounded rates, since that upper bounds the expanded sample size.
# However, with unbounded rates the sample size can increase uncontrollably.
expand(samples; rng=Random.default_rng()) = expand(trajectory(samples)...; rng)
function expand(xs, ws; rng=Random.default_rng())
    ys = empty(xs)
    min_weight = minimum(ws)
    for (x,w) in zip(xs, ws)
        push!(ys, x)
        while rand(rng) > min_weight/w  # Geometric(p)
            push!(ys, x)
        end
    end
    return ys
end