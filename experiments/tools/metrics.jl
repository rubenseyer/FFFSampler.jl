using Statistics, FFFSampler, AbstractMCMC

_inftozero(x) = isfinite(x) ? x : zero(x)

# Kolmogorov-Smirnov
# Adapted from HypothesisTests.jl
function distance_ks(xs, F)
    n = length(xs)
    Fs = F.(sort(xs))
    Dp = maximum((1:n) / n - Fs)
    Dn = -minimum((0:n-1) / n - Fs)
    return max(Dn, Dp)
end
function distance_ks(xs, weights, F)
    indices = sortperm(xs)
    ws = [0.; cumsum(view(weights, indices))]; ws ./= ws[end]
    Fs = F.(view(xs, indices))
    Dp = maximum(ws[2:end] - Fs)
    Dn = -minimum(ws[1:end-1] - Fs)
    return max(Dn, Dp)
end

# Cramér-von Mises
# FIXME no weight support yet
function distance_cvm(xs, F; statistic=false)
    n = length(xs)
    Fs = F.(sort(xs))
    T = 1/(12n) + sum(((1:2:2n) ./ (2n) .- Fs).^2)
    return statistic ? T : T/n  # the statistic is scaled by n
end

# Anderson-Darling
function distance_ad(xs, F; statistic=false)
    n = length(xs)
    Fs = F.(sort(xs))
    A2 = -sum((1:2:2n)./n .* _inftozero.(log.(Fs) .+ reverse(log.(1 .- Fs)))) - n
    return statistic ? A2 : A2/n  # the statistic is scaled by n
end

function distance_ad(xs, weights, F; statistic=false)
    n = length(xs)
    indices = sortperm(xs)
    cum_weights = [0.; cumsum(view(weights, indices))]; cum_weights ./= cum_weights[end]
    Fs = F.(view(xs, indices))

    w2 = -1 + sum(
        @views ((cum_weights[2:end] .- 1).^2 .- (cum_weights[1:end-1] .- 1).^2) .* _inftozero.(log.(1 .- Fs))
    ) + sum(
        @views (cum_weights[1:end-1].^2 .- cum_weights[2:end].^2) .* _inftozero.(log.(Fs))
    )
    return statistic ? n*w2 : w2  # the statistic is scaled by n
end

################################################################################

# extraction helpers
function _n_steps(samples::AbstractVector{<:FFFSampler.AbstractFFFTransition})
    cumsum([t.n_steps for t in samples])
end
function _states(f, samples::AbstractVector{<:FFFSampler.AbstractFFFTransition})
    [f(t.current.q) for t in samples]
end
function _weights(samples::AbstractVector{<:FFFSampler.AbstractFFFTransition})
    [1/sum(t.Λ) for t in samples]
end

function budget_cut(samples, budget)
    ns = _n_steps(samples)
    i = findlast(ns .≤ budget)
    return i
end

################################################################################
import AdvancedHMC

## Patch AdvancedHMC
function FFFSampler.trajectory(raw::AbstractVector{<:AdvancedHMC.Transition})
    return eachindex(raw), [u.z.θ for u in raw]
end

Statistics.mean(samples::AbstractVector{<:AdvancedHMC.Transition}) = Statistics.mean(identity, samples)
function Statistics.mean(f, samples::AbstractVector{<:AdvancedHMC.Transition})
    S = zero(f(samples[begin].z.θ))
    for sample in samples
        S += f(sample.z.θ)
    end
    S / length(samples)
end
function FFFSampler.cummean(f, samples::AbstractVector{<:AdvancedHMC.Transition})
    s = zero(f(samples[begin].z.θ))
    S = [s][1:0]
    for i in eachindex(samples)
        s += f(samples[i].z.θ)
        push!(S, s/i)
    end
    S
end
function Statistics.std(samples::AbstractVector{<:AdvancedHMC.Transition}; mean=nothing)
    sqrt.(Statistics.var(samples; mean))
end
function Statistics.var(samples::AbstractVector{<:AdvancedHMC.Transition}; mean=nothing)
    μ = isnothing(mean) ? Statistics.mean(samples) : mean
    Statistics.mean(Base.Fix2(.^,2), samples) - μ.^2
end

# extraction helpers
function _n_steps(samples::AbstractVector{<:AdvancedHMC.Transition})
    cumsum([t.stat.n_steps for t in samples])
end
function _states(f, samples::AbstractVector{<:AdvancedHMC.Transition})
    [f(t.z.θ) for t in samples]
end
function _weights(samples::AbstractVector{<:AdvancedHMC.Transition})
    [1 for t in samples]
end

