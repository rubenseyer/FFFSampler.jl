## Randomized Guided Random Walk
# Reversible comparison to BJS
struct RGW{T} <: RebalancedSampler{MHBalancing}
    "step size"
    ϵ::T
    "fresh event rate"
    λfresh::T
end

## Step functions
# TODO. This is identical to RHMC.

# Initializer
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::RGW;
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
    p = _randn(rng, model)
    ℓ_q, ∇ℓ_q = LogDensityProblems.logdensity_and_gradient(model, q)
    current_state = FFFState(q, p, ℓ_q, ∇ℓ_q)
    transition = propose(model, sampler, current_state; action=FRESH)
    return transition, transition
end

# Execute transition and propose next
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::RGW,
    transition::BouncyJumpTransition;
    kwargs...
)
    model = model_wrapper.logdensity
    # For the skeleton chain the actual times are not important.
    # We just need to draw who wins! So it could be done by a categorical method too.
    τjump = Random.randexp(rng)/transition.Λ[FORWARD]
    τflip = Random.randexp(rng)/transition.Λ[FLIP]
    τfresh = Random.randexp(rng)/transition.Λ[FRESH]

    if τjump < τflip && τjump < τfresh
        transition_new = propose(model, sampler, transition.proposed, transition.current)
    elseif τflip < τfresh
        prev = FFFState(transition.current.q, -transition.current.p, transition.current.ℓ_q, transition.current.∇ℓ_q)
        transition_new = propose(model, sampler, prev, transition.current; action=FLIP)
    else
        p_new = _randn(rng, model)
        next = FFFState(transition.current.q, p_new, transition.current.ℓ_q, transition.current.∇ℓ_q)
        transition_new = propose(model, sampler, next, transition.current; action=FRESH)
    end
    
    return transition_new, transition_new
end

# Actual proposal of next transition
function propose(model, sampler::RGW, current::FFFState, previous::Union{FFFState,Nothing}=current; action::Action=FORWARD)
    q, p, ℓ_q, ∇ℓ_q = current.q, current.p, current.ℓ_q, current.∇ℓ_q

    # Walk one step forwards and backwards
    q_fw = q + sampler.ϵ .* p
    ℓ_q_fw, ∇ℓ_q_fw = LogDensityProblems.logdensity_and_gradient(model, q_fw)

    # p is unchanged anyway
    H_cur = -ℓ_q
    H_fw = -ℓ_q_fw

    # Compute the rates
    λjump = finiteorzero(balancing_g(MHBalancing, exp(H_cur - H_fw)))
    λflip = 1 - λjump

    Λ = (λjump, λflip, sampler.λfresh, zero(λjump))
    proposed = FFFState(q_fw, p, ℓ_q_fw, ∇ℓ_q_fw)
    transition = BouncyJumpTransition(current, proposed, previous, Λ, 1, action)
    return transition
end

################################################################################
## reBalanced Guided Random Walk
# Nonreversible comparison to BJS (without bounces)
struct BGW{B<:BalancingFunction, T} <: RebalancedSampler{B}
    "step size"
    ϵ::T
    "fresh event rate"
    λfresh::T
end
BGW{B}(ϵ::T, λfresh::T) where {B,T} = BGW{B,T}(ϵ, λfresh)
BGW(ϵ, λfresh) = BGW{MHBalancing}(float(ϵ), float(λfresh))

# Execute transition and propose next
# TODO. Basically identical to BouncyJump (with zero mirror rate)
# Initializer
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::BGW;
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
    p = _randn(rng, model)
    ℓ_q, ∇ℓ_q = LogDensityProblems.logdensity_and_gradient(model, q)
    current_state = FFFState(q, p, ℓ_q, ∇ℓ_q)
    transition = propose(model, sampler, current_state; action=FRESH)
    return transition, transition
end

# Execute transition and propose next
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::BGW,
    transition::BouncyJumpTransition;
    kwargs...
)
    model = model_wrapper.logdensity
    # For the skeleton chain the actual times are not important.
    # We just need to draw who wins! So it could be done by a categorical method too.
    τjump = Random.randexp(rng)/transition.Λ[FORWARD]
    τflip = Random.randexp(rng)/transition.Λ[FLIP]
    τfresh = Random.randexp(rng)/transition.Λ[FRESH]

    if τjump < τflip && τjump < τfresh
        transition_new = propose(model, sampler, transition.proposed, transition.current)
    elseif τflip < τfresh
        current_new = FFFState(transition.current.q, -transition.current.p, transition.current.ℓ_q, transition.current.∇ℓ_q)
        proposed_new = FFFState(transition.previous.q, -transition.previous.p, transition.previous.ℓ_q, transition.previous.∇ℓ_q)
        previous_new = FFFState(transition.proposed.q, -transition.proposed.p, transition.proposed.ℓ_q, transition.proposed.∇ℓ_q)
        transition_new = propose(model, sampler, current_new, previous_new, proposed_new; action=FLIP)
    else
        p_new = _randn(rng, model)
        next = FFFState(transition.current.q, p_new, transition.current.ℓ_q, transition.current.∇ℓ_q)
        transition_new = propose(model, sampler, next; action=FRESH)
    end
    
    return transition_new, transition_new
end

# Actual proposal of next transition
function propose(model, sampler::BGW, current::FFFState, previous::Union{FFFState,Nothing}=nothing, forward::Union{FFFState,Nothing}=nothing; action=FORWARD)
    q, p, ℓ_q, ∇ℓ_q = current.q, current.p, current.ℓ_q, current.∇ℓ_q

    # Walk one step forwards and backwards
    # TODO actually never uses the gradient.
    if forward === nothing
        q_fw = q + sampler.ϵ .* p
        ℓ_q_fw, ∇ℓ_q_fw = LogDensityProblems.logdensity_and_gradient(model, q_fw)
    else
        q_fw, ℓ_q_fw, ∇ℓ_q_fw = forward.q, forward.ℓ_q, forward.∇ℓ_q
    end
    if previous === nothing
        q_bw = q - sampler.ϵ .* p
        ℓ_q_bw, ∇ℓ_q_bw = LogDensityProblems.logdensity_and_gradient(model, q_bw)
    else
        q_bw, ℓ_q_bw, ∇ℓ_q_bw = previous.q, previous.ℓ_q, previous.∇ℓ_q
    end

    # p is unchanged anyway
    H_cur = -ℓ_q
    H_fw = -ℓ_q_fw
    H_bw = -ℓ_q_bw

    # Compute the rates
    λjump = finiteorzero(balancing_g(sampler, exp(H_cur - H_fw)))
    anti_λjump = finiteorzero(balancing_g(sampler, exp(H_cur - H_bw)))
    λflip = max(anti_λjump - λjump, zero(λjump))

    Λ = (λjump, λflip, sampler.λfresh, zero(λjump))
    proposed = forward === nothing ? FFFState(q_fw, p, ℓ_q_fw, ∇ℓ_q_fw) : forward
    previous_new = previous === nothing ? FFFState(q_bw, -p, ℓ_q_bw, ∇ℓ_q_bw) : previous
    transition = BouncyJumpTransition(current, proposed, previous_new, Λ, ((forward === nothing) + (previous === nothing)), action)
    return transition
end