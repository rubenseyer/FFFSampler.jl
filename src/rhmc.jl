struct RHMC{T,MS} <: RebalancedSampler{MHBalancing}
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

RHMC(ϵ=0.1, L=1, λfresh=0.1, ρfresh=0.0, Minv=I) = RHMC{Float64,Matrix{Float64}}(float(ϵ), L, float(λfresh), float(ρfresh), Minv)

# Execute transition and propose next
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::RHMC,
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
        prev = FFFState(transition.current.q, -transition.current.p, transition.current.ℓ_q, transition.current.∇ℓ_q)
        transition_new = propose(model, sampler, prev, transition.current; action=FLIP)
    else
        p_new = refresh(rng, model, sampler, transition.current.p)
        next = FFFState(transition.current.q, p_new, transition.current.ℓ_q, transition.current.∇ℓ_q)
        transition_new = propose(model, sampler, next, transition.current; action=FRESH)
    end
    
    return transition_new, transition_new
end

# Actual proposal of next transition
function propose(model, sampler::RHMC, current::FFFState, previous::Union{FFFState,Nothing}=current; action::Action=FORWARD)
    q, p, ℓ_q, ∇ℓ_q = current.q, current.p, current.ℓ_q, current.∇ℓ_q

    # Do leapfrog forwards
    q_fw, p_fw, ℓ_q_fw, ∇ℓ_q_fw = leapfrog(model, sampler, q, p, ℓ_q, ∇ℓ_q)
    # FIXME just fill the previous slot in the struct for now

    H_cur = H(sampler, ℓ_q, p)
    H_fw = H(sampler, ℓ_q_fw, p_fw)

    # Compute the rates
    λ🐸 = finiteorzero(balancing_g(MHBalancing, exp(H_cur - H_fw)))
    λflip = 1 - λ🐸

    Λ = (λ🐸, λflip, sampler.λfresh)
    proposed = FFFState(q_fw, p_fw, ℓ_q_fw, ∇ℓ_q_fw)
    transition = FFFTransition(current, proposed, previous, Λ, sampler.L, action)
    return transition
end
