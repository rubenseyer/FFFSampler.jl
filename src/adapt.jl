
"""
    adapt_refresh(problem, fff, N=5000; ρ=1.0, initial_params=nothing, verbose=false)

Find "optimal" refreshment rate. If ρ = 1.0, try to have 1 refreshment between each flip respective
u-turn in the NUTS sense on average. If ρ = 0.0, try to have 1 refreshment in the longest sequence
of accepted moves so far. Intermediate choices interpolate.
"""
function adapt_refresh(problem, fff, N=5000; ρ=1.0, initial_params=nothing, verbose=false)
    λfresh = fff.λfresh
    maxl = 1
    transition, _ = AbstractMCMC.step(Random.TaskLocalRNG(), AbstractMCMC._model(problem), fff; initial_params)
    p0 = transition.current.p; i0 = 0
    for i in 1:N
        maxl = max(maxl, i - i0)
        fff = FFF(fff.ϵ, fff.L, λfresh*ρ +(1-ρ)/maxl)
        transition_old = transition
        transition, _ = AbstractMCMC.step(Random.TaskLocalRNG(), AbstractMCMC._model(problem), fff, transition; initial_params)
        acc = transition.Λ[1]/sum(transition.Λ)
        verbose && print(" λ ", λfresh, " maxl ", maxl, " acc $acc \t")

        if transition.current.p == -transition_old.current.p
            p0 = transition.current.p; i0 = i
            verbose && println("flip $i")
            λfresh *= 1.0 + 0.01*ρ
        elseif (transition.current.q == transition_old.current.q) && (transition.current.p != transition_old.current.p)
            p0 = transition.current.p; i0 = i
            verbose && println("fresh $i")
            λfresh /= 1.01
        elseif dot(p0, transition.current.p) < 0 
            p0 = transition.current.p; i0 = i
            verbose && println("u-turn $i")
            λfresh *= 1.0 + 0.01*ρ
        else 
            @assert transition.current == transition_old.proposed
            verbose && println("frog")
        end
    end
    fff
end