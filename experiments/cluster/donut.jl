#!/bin/bash
#
#SBATCH -J naiss2024-22-1100-fff-donut
#SBATCH -t 6:00:00
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
exec julia -t 32 /proj/pdmps/repos/frogflipfresh/experiments/cluster/donut.jl
=#
# WARNING: includes are relative to file, so you must have the absolute
# path as argument to julia

include("_common.jl")
using LogDensityProblemsAD, ForwardDiff

struct Donut{T}
    R::T
    σ2::T
    p::T
end
Donut(R::T, σ2::T) where T = Donut{T}(R, σ2, 2)
LogDensityProblems.dimension(D::Donut) = 2
LogDensityProblems.capabilities(::Donut) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(D::Donut, x)
    return -(norm(x,D.p) - D.R)^2 / 2D.σ2
end

D = Donut(2.6, 0.0165)
problem = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), D)
initial_state = zeros(LogDensityProblems.dimension(D))
initial_state[1] = D.R

# sample from the donut, in 2D we can sample angle and radius
@assert D.p == 2
M = 5_000_000
reference_samples = Matrix{Float64}(undef, M, 2)
for i in 1:M
    r = rand(truncated(Normal(D.R, sqrt(D.σ2)), 0, nothing))
    θ = rand(Uniform(-π, π))
    reference_samples[i, 1] = r * cos(θ)
    reference_samples[i, 2] = r * sin(θ)
end
rcdfs = [ecdf(reference_samples[:,k]) for k in 1:size(reference_samples,2)]


Random.seed!(20250211)
grid_experiment(HMC, problem, initial_state, rcdfs, (;
    epsilon = range(0.01, 0.5; length=21),
    L = [1 2 3 7 15 31 47 63 127],
))

Random.seed!(20250211)
grid_experiment(FFF, problem, initial_state, rcdfs, (;
    epsilon = range(0.01, 0.5; length=21),
    L = [1 2 3 7 15 31 47 63 127],
    lambda = logrange(0.001, 1.0; length=11),
))
