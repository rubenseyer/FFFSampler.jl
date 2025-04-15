#!/bin/bash
#
#SBATCH -J naiss2024-22-1100-fff-posteriordb
#SBATCH -t 52:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rubense@chalmers.se

#=
echo "# This is $SLURM_JOB_NAME on version $(git -C /proj/pdmps/repos/frogflipfresh/ describe --always --dirty)."
echo -e "\n########## The script follows ##########"
cat $0
echo -e "########## This concludes the script ##########\n"
module add julia/1.10.2-bdist
export JULIA_DEPOT_PATH="/proj/pdmps/julia:$JULIA_DEPOT_PATH"
export JULIA_PROJECT=/proj/pdmps/repos/frogflipfresh
exec julia -t 32 /proj/pdmps/repos/frogflipfresh/experiments/cluster/posteriordb.jl
=#
# WARNING: includes are relative to file, so you must have the absolute
# path as argument to julia

include("_common.jl")
using PosteriorDB, StanLogDensityProblems

post = PosteriorDB.posterior(PosteriorDB.database(), "one_comp_mm_elim_abs-one_comp_mm_elim_abs")

# Compile stan model
problem = StanProblem(post, ".", force=true, make_args=Threads.nthreads() > 1 ? ["STAN_THREADS=true"] : String[])
d = LogDensityProblems.dimension(problem)
initial_state = [0.0,0.0,0.0,-2.0]

## Consider correlations between parameters to show the tricky banana mixing?
##f(x) = [x[1]*x[3]; x[2]*x[3]; x[3]*x[4]; x[4]]

# Use gold standard posterior draws to approximate the reference cdf
begin
    gold_samples = PosteriorDB.load(PosteriorDB.reference_posterior(post))
    parameter_names = collect(keys(gold_samples[begin]))
    
    gold_matrix = reduce(hcat, reduce(vcat, gold_samples[i][pname] for i in eachindex(gold_samples)) for pname in parameter_names)
    for i in 1:size(gold_matrix,1)
        gold_matrix[i,:] .= StanLogDensityProblems.BridgeStan.param_unconstrain(problem.model, gold_matrix[i,:])
    end

    means = vec(mean(gold_matrix; dims=1))
    vars = vec(var(gold_matrix; dims=1))
    rcdfs = vec(mapslices(ecdf, gold_matrix; dims=1))
    #rcdfs = vec(mapslices(ecdf, mapslices(f, gold_matrix; dims=2); dims=1))
end

# mass matrix
#Minv = Diagonal(vars)
#HMCM(ϵ, L) = HMC(L, Leapfrog(ϵ), DiagEuclideanMetric(vars))
#FFFM(ϵ, L, λ, ρ=0.0) = FFF(ϵ, L, λ, ρ, Minv)

#= # exploration
using PairPlots
pairplot(mapslices(f, gold_matrix; dims=2))
#raw_nuts = AbstractMCMC.sample(problem, NUTS(0.8), 50_000; initial_params=initial_state)
metric = DiagEuclideanMetric(vars)
raw_hmc = AbstractMCMC.sample(problem, HMC(9, Leapfrog(0.6), metric), 2000; initial_params=initial_state)

include("../tools/viewer2.jl")
plot_trajectory(raw_hmc)
=#

Random.seed!(20250328)
grid_experiment(HMC, problem, initial_state, rcdfs, (;
    epsilon = range(0.02, 0.4; length=11),
    L = [1 3 7 15 31],
); T=150_000)

Random.seed!(20250328)
grid_experiment(FFF, problem, initial_state, rcdfs, (;
    epsilon = range(0.02, 0.4; length=11),
    L = [1 3 7 15 31],
    lambda = logrange(0.0158, 1.0; length=11),
    #ρ = [0.0, 0.5],
); T=150_000)
