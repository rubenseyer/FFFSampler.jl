module FFFSamplerTuringExt

if isdefined(Base, :get_extension)
    using FFFSampler: FFFSampler
    using AbstractMCMC: AbstractMCMC
    using Turing: Turing
else
    using ..FFFSampler: FFFSampler
    using ..AbstractMCMC: AbstractMCMC
    using ..Turing: Turing
end

Turing.Inference.getparams(::Turing.DynamicPPL.Model, state::FFFSampler.FFFTransition) = state.current.q
Turing.Inference.getstats(state::FFFSampler.FFFTransition) = (;
    Λ = sum(state.Λ), n_steps = state.n_steps
)

end