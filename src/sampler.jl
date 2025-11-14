struct FFF{B<:BalancingFunction,T,MS} <: RebalancedSampler{B}
    "leapfrog step size"
    ϵ::T
    "leapfrog step count"
    L::Int
    "fresh event rate"
    λfresh::T
    "fresh partial refreshment correlation"
    ρfresh::T
    "inverse mass matrix"
    Minv::MS
end

FFF{B}(ϵ::T, L, λfresh::T, ρfresh::T=zero(ϵ), Minv::MS=I) where {B,T,MS} = FFF{B,T,MS}(ϵ, L, λfresh, ρfresh, Minv)
FFF(ϵ=0.1, L=1, λfresh=0.1, ρfresh=0.0, Minv=I) = FFF{MHBalancing}(float(ϵ), L, float(λfresh), float(ρfresh), Minv)

struct FFFState{T1,T2,T3}
    "position"
    q::T1
    "momentum"
    p::T1
    "marginal log probability of position"
    ℓ_q::T2
    "marginal gradient log probability of position"
    ∇ℓ_q::T3
end

abstract type AbstractFFFTransition{T1,T2,T3} end
struct FFFTransition{T1,T2,T3} <: AbstractFFFTransition{T1,T2,T3}
    "current state"
    current::FFFState{T1,T2,T3}
    "proposed forward state"
    proposed::FFFState{T1,T2,T3}
    "last state (backward with flipped p)"
    previous::FFFState{T1,T2,T3}
    #"current balance"
    #λ::NTuple{4,T2}  # TODO reconsider?
    "current rates"
    Λ::NTuple{3,T2}
    "number of leapfrog steps"
    n_steps::Int
    "action coming here"
    action::Action
end

function refresh(rng::Random.AbstractRNG, model, sampler, p=nothing)
    prop = _randn(rng, model)
    whiten!(prop, sampler.Minv, prop)  # we store the inverse, so it's whiten and not unwhiten
    if p === nothing
        return prop
    else
        ρ = sampler.ρfresh
        return ρ * p  +  √(1 - ρ^2) * prop
    end
end

function leapfrog(model, sampler, q, p, ℓ_q, ∇ℓ_q)
    for _ in 1:sampler.L
        p_half = p + (sampler.ϵ / 2) .* ∇ℓ_q
        q = q + sampler.ϵ .* sampler.Minv * p_half
        ℓ_q, ∇ℓ_q = LogDensityProblems.logdensity_and_gradient(model, q)
        p = p_half + (sampler.ϵ / 2) .* ∇ℓ_q
    end
    return q, p, ℓ_q, ∇ℓ_q
end
H(sampler, ℓ_q, p) = -ℓ_q + dot(p, sampler.Minv, p)/2

## Step functions

# Initializer
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::RebalancedSampler;
    initial_params=nothing,
    kwargs...
)
    model = model_wrapper.logdensity
    q = if initial_params === nothing
        # Here we are just starting gaussian...
        _randn(rng, model)
    else
        initial_params
    end
    p = refresh(rng, model, sampler)
    ℓ_q, ∇ℓ_q = LogDensityProblems.logdensity_and_gradient(model, q)
    current_state = FFFState(q, p, ℓ_q, ∇ℓ_q)
    transition = propose(model, sampler, current_state; action=FRESH)
    return transition, transition
end

# Execute transition and propose next
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::FFF,
    transition::FFFTransition;
    kwargs...
)
    model = model_wrapper.logdensity
    # For the skeleton chain the actual times are not important.
    # We just need to draw who wins! So it could be done by a categorical method too.
    τ🐸 = Random.randexp(rng)/transition.Λ[FORWARD]
    τflip = Random.randexp(rng)/transition.Λ[FLIP]
    τfresh = Random.randexp(rng)/transition.Λ[FRESH]

    if τ🐸 < τflip && τ🐸 < τfresh
        transition_new = propose(model, sampler, transition.proposed, transition.current)
    elseif τflip < τfresh
        transition_new = flip(transition)
    else
        p_new = refresh(rng, model, sampler, transition.current.p)
        next = FFFState(transition.current.q, p_new, transition.current.ℓ_q, transition.current.∇ℓ_q)
        transition_new = propose(model, sampler, next; action=FRESH)
    end
    
    return transition_new, transition_new
end

# Actual proposal of next transition
function propose(model, sampler::FFF, current::FFFState, previous::Union{FFFState,Nothing}=nothing; action::Action=FORWARD)
    q, p, ℓ_q, ∇ℓ_q = current.q, current.p, current.ℓ_q, current.∇ℓ_q

    # Do leapfrog forwards and backwards
    q_fw, p_fw, ℓ_q_fw, ∇ℓ_q_fw = leapfrog(model, sampler, q, p, ℓ_q, ∇ℓ_q)
    if previous === nothing
        q_bw, p_bw, ℓ_q_bw, ∇ℓ_q_bw = leapfrog(model, sampler, q, -p, ℓ_q, ∇ℓ_q)
    else
        # wrong direction of p, but does not matter for H_bw
        q_bw, p_bw, ℓ_q_bw, ∇ℓ_q_bw = previous.q, previous.p, previous.ℓ_q, previous.∇ℓ_q

        #lf = leapfrog(model, sampler, q, -p, ℓ_q, ∇ℓ_q)
        #@assert lf[1] ≈ q_bw
        #@assert lf[2] ≈ -p_bw
        #@assert lf[3] ≈ ℓ_q_bw
        #@assert lf[4] ≈ ∇ℓ_q_bw
    end

    H_cur = H(sampler, ℓ_q, p)
    H_fw = H(sampler, ℓ_q_fw, p_fw)
    H_bw = H(sampler, ℓ_q_bw, p_bw)

    # Compute the rates
    λ🐸 = finiteorzero(balancing_g(sampler, exp(H_cur - H_fw)))
    anti_λ🐸 = finiteorzero(balancing_g(sampler, exp(H_cur - H_bw)))
    λflip = max(anti_λ🐸 - λ🐸, zero(λ🐸))

    Λ = (λ🐸, λflip, sampler.λfresh)
    proposed = FFFState(q_fw, p_fw, ℓ_q_fw, ∇ℓ_q_fw)
    previous_new = previous === nothing ? FFFState(q_bw, -p_bw, ℓ_q_bw, ∇ℓ_q_bw) : previous
    transition = FFFTransition(current, proposed, previous_new, Λ, (1 + (previous === nothing)) * sampler.L, action)
    return transition
end

function flip(transition::FFFTransition)
    current_new = FFFState(transition.current.q, -transition.current.p, transition.current.ℓ_q, transition.current.∇ℓ_q)
    proposed_new = FFFState(transition.previous.q, -transition.previous.p, transition.previous.ℓ_q, transition.previous.∇ℓ_q)
    previous_new = FFFState(transition.proposed.q, -transition.proposed.p, transition.proposed.ℓ_q, transition.proposed.∇ℓ_q)
    Λ_new = (transition.Λ[FORWARD] + transition.Λ[FLIP], zero(transition.Λ[FLIP]), transition.Λ[FRESH])
    return FFFTransition(current_new, proposed_new, previous_new, Λ_new, 0, FLIP)
end
