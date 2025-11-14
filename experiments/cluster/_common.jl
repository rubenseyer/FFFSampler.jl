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

if !isdefined(Base, :logrange)
    logrange(start, stop; length) = exp10.(range(log10(start), log10(stop); length=length))
end
step_param(sampler::FFF) = sampler.L
step_param(sampler::HMC) = sampler.n_leapfrog
step_param(sampler::AdvancedHMC.HMCSampler) = AdvancedHMC.nsteps(sampler.κ.τ)
step_param(sampler::Mirrorer) = step_param(sampler.inner_sampler)
step_param(sampler::BouncyJump) = 1
step_param(sampler::FFFSampler.RHMC) = sampler.L
step_param(sampler::FFFSampler.RGW) = 1
step_param(sampler::FFFSampler.BGW) = 1

default_metrics = (
    KS = distance_ks,
    #AD = distance_ad,
    W2 = distance_W,
    dESS = (x,w,ref) -> discretized_ess(x,w),
)

function grid_experiment(sampler_init, problem, initial_state, reference_marginals, grid;
        T=500_000, replicates=1, f=identity, metrics=default_metrics,
        progress=isinteractive(), name=String(Symbol(sampler_init)))
    d = length(f(initial_state))
    params = collect(Iterators.product(grid...))
    buffer = zeros(length(metrics), d, replicates)
    meta_buffer = zeros(4, replicates) # accrate, fliprate, ngrads, walltime

    # TODO: change format?
    filename = get(ENV,"SLURM_JOB_NAME","frogmc") * "-" * name * ".csv"
    fd = open(filename, "w")
    writedlm(fd, [String.(keys(grid))... "d" ["marginal_$m" for m in keys(metrics)]... "accrate" "fliprate" "ngrads" "walltime"], ',')
    flush(fd)

    # precompile?
    #AbstractMCMC.sample(problem, sampler_init(params[begin]...), ceil(Int, 0.1T); initial_params=initial_state, progress=false);

    start_time = time()
    for i in eachindex(params)
        sampler = sampler_init(params[i]...)
        for rep in 1:replicates
            local traj, walltime
            try
                walltime = @elapsed (
                    traj = AbstractMCMC.sample(problem, sampler, ceil(Int, 1.1T / step_param(sampler)); initial_params=initial_state, progress);
                )
            catch e
                isa(e, InterruptException) && rethrow(e)
                @warn "Sampler failed" sampler params[i] rep e
                buffer[:, :, rep] .= NaN
            else
                #sz, cost = budget_cut(traj, T)
                #resize!(traj, sz)
                sz, ngrads = length(traj), _n_steps(traj)[end]
                ws = _weights(traj)
                xs = _states(f, traj)
                for k in 1:d
                    marginal_xs = getindex.(xs, k)
                    for (m,metric) in enumerate(metrics)
                        buffer[m, k, rep] = metric(marginal_xs, ws, reference_marginals[k])
                    end
                end
                meta_buffer[1, rep] = FFFSampler.acceptance_rate(traj)
                meta_buffer[2, rep] = FFFSampler.flip_rate(traj)
                meta_buffer[3, rep] = ngrads
                meta_buffer[4, rep] = walltime
            end
        end

        writedlm(fd, ((params[i]..., k, buffer[:,k,rep]..., meta_buffer[:,rep]...) for k in 1:d, rep in 1:replicates), ',')
        flush(fd)

        # Timing report
        if i == 2
            elapsed = time() - start_time
            est_total = elapsed * length(params)/2
            println("# One iteration took $(round(elapsed/120; digits=2)) minutes. Estimated total time: $(round(est_total/3600; digits=2)) hours")
            flush(stdout)
        end
    end
end

function evaluate_experiment(params, sampler_init, problem, initial_state, reference_marginals;
        T=1_000_000, replicates=Threads.nthreads(), f=identity, metric=distance_W, reduction_dim=maximum, reduction_rep=mean)
    d = LogDensityProblems.dimension(problem)
    sampler = sampler_init(params...)
    results = zeros(replicates)
    Threads.@threads for rep in 1:replicates
        local traj
        try
            traj = AbstractMCMC.sample(problem, sampler, Int(ceil(1.1T / step_param(sampler))); initial_params=initial_state, progress=false);
        catch e
            isa(e, InterruptException) && rethrow(e)
            @warn "Sampler failed" sampler params rep e
            results[rep] = NaN
        else
            sz, cost = budget_cut(traj, T)
            resize!(traj, sz)
            xs = _states(f, traj)
            ws = _weights(traj)
            results[rep] = reduction_dim(metric(getindex.(xs,k), ws, reference_marginals[k]) for k in 1:d)
        end
    end
    return reduction_rep(results)
end

###

# Hack to work around typing restriction in HMCSampler (want to infer dimension from target problem)
struct MetricWrap{T} <: AdvancedHMC.AbstractMetric
    s::Symbol
end
MetricWrap(s::Symbol) = MetricWrap{AdvancedHMC.DEFAULT_FLOAT_TYPE}(s)
AdvancedHMC.make_metric(i::MetricWrap, ::Type{T}, d::Int) where {T} = AdvancedHMC.make_metric(i.s, T, d)
Base.eltype(::MetricWrap{T}) where {T} = T

function HMCρ(epsilon, L, ρ)
    integrator = Leapfrog(epsilon)
    kernel = HMCKernel(AdvancedHMC.PartialMomentumRefreshment(ρ), Trajectory{EndPointTS}(integrator, FixedNSteps(L)))
    return HMCSampler(kernel, MetricWrap(:unit), NoAdaptation())
end;