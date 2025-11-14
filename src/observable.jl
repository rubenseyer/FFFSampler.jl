Statistics.mean(samples::AbstractVector{<:AbstractFFFTransition}) = Statistics.mean(identity, samples)
function Statistics.mean(f, samples::AbstractVector{<:AbstractFFFTransition})
    S = zero(f(samples[begin].current.q))
    W = zero(sum(samples[begin].Λ))
    for sample in samples
        τ = 1/sum(sample.Λ)
        S += f(sample.current.q) * τ
        W += τ
    end
    S / W
end

function cummean(f, samples::AbstractVector{<:AbstractFFFTransition})
    S = zero(f(samples[begin].current.q))
    W = zero(sum(samples[begin].Λ))
    Ms = [S/one(W)][1:0]
    for sample in samples
        τ = 1/sum(sample.Λ)
        S += f(sample.current.q) * τ
        W += τ
        push!(Ms, S/W)
    end
    Ms
end

function Statistics.std(samples::AbstractVector{<:AbstractFFFTransition}; mean=nothing)
    sqrt.(Statistics.var(samples; mean))
end

function Statistics.var(samples::AbstractVector{<:AbstractFFFTransition}; mean=nothing)
    μ = isnothing(mean) ? Statistics.mean(samples) : mean
    Statistics.mean(Base.Fix2(.^,2), samples) - μ.^2
end

function trajectory(samples::AbstractVector{<:AbstractFFFTransition})
    ts = Float64[]
    xs = typeof(samples[begin].current.q)[]
    τ = 0.0
    for sample in samples
        τ += 1/sum(sample.Λ)
        push!(ts, τ) 
        push!(xs, sample.current.q) 
    end
    ts, xs
end

function acceptance_rate(samples::AbstractVector{<:AbstractFFFTransition})
    return Statistics.mean(t.action == FORWARD for t in Iterators.drop(samples,1))
end

function flip_rate(samples::AbstractVector{<:AbstractFFFTransition})
    return Statistics.mean(t.action == FLIP for t in Iterators.drop(samples,1))
end