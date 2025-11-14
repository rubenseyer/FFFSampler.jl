struct BouncyJump{B<:BalancingFunction,T} <: RebalancedSampler{B}
    "step size"
    ϵ::T
    "fresh event rate"
    λfresh::T
    #"randomized reflection orthogonal correlation"
    #ρrefl::T
end

BouncyJump{B}(ϵ::T, λfresh::T) where {B,T} = BouncyJump{B,T}(ϵ, λfresh)
#BouncyJump{B}(ϵ::T, λfresh::T, ρrefl=1.0) where {B,T} = BouncyJump{B,T}(ϵ, λfresh, ρrefl)
BouncyJump(ϵ, λfresh) = BouncyJump{MHBalancing}(float(ϵ), float(λfresh))

struct BouncyJumpTransition{T1,T2,T3} <: AbstractFFFTransition{T1,T2,T3}
    "current state"
    current::FFFState{T1,T2,T3}
    "proposed forward state"
    proposed::FFFState{T1,T2,T3}
    "last state (backward with flipped p)"
    previous::FFFState{T1,T2,T3}
    "current rates"
    Λ::NTuple{4,T2}
    "number of gradient evaluations used"
    n_steps::Int
    "action coming here"
    action::Action
end

reflection_desire(ϵ,p,∇ℓ_q) = -ϵ*dot(p,∇ℓ_q)

#=
function oscn(rng, v, ∇U, ρ=1.0; normalize=false)
    # -∇U i.e. gradlog is also OK to input, the signs cancel.
    # Decompose v
    vₚ = (dot(v,∇U)/dot(∇U,∇U))*∇U
    if ρ == 1
        return v - 2vₚ
    else
        # Sample and project
        v⊥ = √(1.0 - ρ^2) * randn(rng, length(v))
        v⊥ -= (dot(v⊥, ∇U)/dot(∇U,∇U))*∇U
        v⊥ += ρ*(v - vₚ)
        if normalize  # unit sphere
            v⊥ *= sqrt(1 - dot(vₚ,vₚ))/norm(v⊥)
        end
        return -vₚ + v⊥
    end
end
=#
_reflect(p, ∇U) = p - 2(dot(p,∇U)/dot(∇U,∇U))*∇U  # -∇U i.e. gradlog is also OK to input, the signs cancel.

## Step functions
# Initializer
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::BouncyJump;
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
    sampler::BouncyJump,
    transition::BouncyJumpTransition;
    kwargs...
)
    model = model_wrapper.logdensity
    # For the skeleton chain the actual times are not important.
    # We just need to draw who wins! So it could be done by a categorical method too.
    τ, i = findmin(Random.randexp(rng)/λ for λ in transition.Λ)

    if i == FORWARD
        transition_new = propose(model, sampler, transition.proposed, transition.current)
    elseif i == MIRROR
        # apply reflection
        #p_new = oscn(rng, transition.current.p, transition.current.∇ℓ_q, sampler.ρrefl)
        p_new = _reflect(transition.current.p, transition.current.∇ℓ_q)
        next = FFFState(transition.current.q, p_new, transition.current.ℓ_q, transition.current.∇ℓ_q)
        transition_new = propose(model, sampler, next; action=MIRROR)
    elseif i == FLIP
        current_new = FFFState(transition.current.q, -transition.current.p, transition.current.ℓ_q, transition.current.∇ℓ_q)
        proposed_new = FFFState(transition.previous.q, -transition.previous.p, transition.previous.ℓ_q, transition.previous.∇ℓ_q)
        previous_new = FFFState(transition.proposed.q, -transition.proposed.p, transition.proposed.ℓ_q, transition.proposed.∇ℓ_q)
        transition_new = propose(model, sampler, current_new, previous_new, proposed_new; action=FLIP)
    else # i == FRESH
        p_new = _randn(rng, model)
        next = FFFState(transition.current.q, p_new, transition.current.ℓ_q, transition.current.∇ℓ_q)
        transition_new = propose(model, sampler, next; action=FRESH)
    end
    
    return transition_new, transition_new
end

# Actual proposal of next transition
function propose(model, sampler::BouncyJump, current::FFFState, previous::Union{FFFState,Nothing}=nothing, forward::Union{FFFState,Nothing}=nothing; action=FORWARD)
    q, p, ℓ_q, ∇ℓ_q = current.q, current.p, current.ℓ_q, current.∇ℓ_q

    # Walk one step forwards and backwards
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

    # Might as well pretend we have a Gaussian reference measure for p.
    H_cur = -ℓ_q
    H_fw = -ℓ_q_fw
    H_bw = -ℓ_q_bw
    refl_d = reflection_desire(sampler.ϵ, p, ∇ℓ_q)

    # Compute the rates
    λjump = finiteorzero(balancing_g(sampler, exp(H_cur - H_fw)))
    anti_λjump = finiteorzero(balancing_g(sampler, exp(H_cur - H_bw)))
    λreflect = pos(refl_d)  # H is preserved exactly. Assuming g(1) = 1
    #anti_λreflect = pos(-refl_d) # H is preserved exactly. Assuming g(1) = 1
    # pos(-refl_d) - pos(refl_d) = -refl_d
    λflip = pos(anti_λjump - λjump - refl_d)
    

    Λ = (λjump, λflip, sampler.λfresh, λreflect)
    proposed = forward === nothing ? FFFState(q_fw, p, ℓ_q_fw, ∇ℓ_q_fw) : forward
    previous_new = previous === nothing ? FFFState(q_bw, -p, ℓ_q_bw, ∇ℓ_q_bw) : previous
    transition = BouncyJumpTransition(current, proposed, previous_new, Λ, ((forward === nothing) + (previous === nothing)), action)
    return transition
end
