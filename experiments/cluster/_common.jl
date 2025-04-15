using LogDensityProblems
using FFFSampler, AdvancedHMC, AbstractMCMC
using LinearAlgebra, Statistics, StatsBase
using Distributions, PDMats
using DelimitedFiles
import Dates, Random

include("../tools/metrics.jl")

LogDensityProblems.dimension(D::T) where {T<:MvNormal} = length(D)
LogDensityProblems.capabilities(::Type{<:MvNormal}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(D::T, x) where {T<:MvNormal} = Distributions.logpdf(D, x)
LogDensityProblems.logdensity_and_gradient(D::T, x) where {T<:MvNormal} = (Distributions.logpdf(D, x), Distributions.gradlogpdf(D, x))
marginal(D::T, k) where {T<:MvNormal} = Normal(D.μ[k], √D.Σ.diag[k])

logrange(start, stop; length) = exp10.(range(log10(start), log10(stop); length=length))
step_param(sampler::FFF) = sampler.L
step_param(sampler::HMC) = sampler.n_leapfrog
step_param(sampler::FFFMirror) = sampler.inner_sampler.L

function grid_experiment(sampler_init, problem, initial_state, rcdfs, grid; T=500_000, replicates=32, f=identity, metrics=(KS=distance_ks,AD=distance_ad), progress=isinteractive())
    d = LogDensityProblems.dimension(problem)
    params = collect(Iterators.product(grid...))
    buffer = zeros(length(metrics), d, replicates)

    filename = get(ENV,"SLURM_JOB_NAME","frogmc") * "-" * String(Symbol(sampler_init)) * "-" * Dates.format(Dates.now(),"yyyymmddHHMM") * ".csv"
    fd = open(filename, "w")
    writedlm(fd, [String.(keys(grid))... "d" ["marginal_$m" for m in keys(metrics)]...], ',')
    flush(fd)

    for i in eachindex(params)
        sampler = sampler_init(params[i]...)
        Threads.@threads for rep in 1:replicates
            local traj
            try
                traj = AbstractMCMC.sample(problem, sampler, Int(ceil(1.1T / step_param(sampler))); initial_params=initial_state, progress);
            catch e
                isa(e, InterruptException) && rethrow(e)
                @warn "Sampler failed" sampler params[i] rep e
                buffer[:, :, rep] .= NaN
            else
                resize!(traj, budget_cut(traj, T))
                xs = _states(f, traj)
                ws = _weights(traj)
                for k in 1:d
                    for (m,metric) in enumerate(metrics)
                        buffer[m, k, rep] = metric(getindex.(xs,k), ws, rcdfs[k])
                    end
                end
            end
        end

        for k in 1:d, rep in 1:replicates
            writedlm(fd, [params[i]... k buffer[:,k,rep]...], ',')
        end
        flush(fd)
    end
end