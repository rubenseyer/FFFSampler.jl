struct Mirrorer{S} <: AbstractMCMC.AbstractSampler
    inner_sampler::S
end
FFFMirror(args...) = Mirrorer(FFF(args...))

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::Mirrorer;
    initial_params=nothing,
    kwargs...
)
    transition, state = AbstractMCMC.step(rng, model_wrapper, sampler.inner_sampler; initial_params)
    @assert isa(transition, AbstractFFFTransition)
    return transition, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::Mirrorer,
    transition_old;
    kwargs...
)
    transition, state = AbstractMCMC.step(rng, model_wrapper, sampler.inner_sampler, transition_old)
    transition.action != FLIP && return transition, state

    p_refl = reflect(transition.current.∇ℓ_q, transition_old.current.p)
    next_refl = FFFState(transition.current.q, p_refl, transition.current.ℓ_q, transition.current.∇ℓ_q)
    transition_refl = propose(model_wrapper.logdensity, sampler.inner_sampler, next_refl; action=MIRROR)
    prob_refl = min(transition_old.Λ[FLIP], transition_refl.Λ[FLIP]) / transition_old.Λ[FLIP]
    # Note that we could theoretically MIRROR and then FLIP or MIRROR back to back.
    # This works here as well, since we could transform if we elect to FLIP out of the chosen transition.
    #@show prob_refl
    chosen_transition = rand(rng) ≤ prob_refl ? transition_refl : transition
    chosen_transition = FFFTransition(chosen_transition.current, chosen_transition.proposed, chosen_transition.previous,
                                      chosen_transition.Λ, transition.n_steps + transition_refl.n_steps, chosen_transition.action)
    return chosen_transition, chosen_transition
end

normsq(x) = dot(x, x)
function reflect(∇ℓ_q, p; L=I)
    iszero(∇ℓ_q) && return p  # no-op if we are at the mode, edge case
    # FIXME should L match mass matrix?
    return p - (2*dot(∇ℓ_q, p)/normsq(L\∇ℓ_q))*(L'\(L\∇ℓ_q))
end
# R(n) = I - 2*n*n'/dot(n,n)