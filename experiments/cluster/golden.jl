#!/bin/bash
#
#SBATCH -J naiss2024-22-1100-fff-golden
#SBATCH -t 72:00:00
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
exec julia -t 32 /proj/pdmps/repos/frogflipfresh/experiments/cluster/golden.jl
=#
# WARNING: includes are relative to file, so you must have the absolute
# path as argument to julia

include("_common.jl")

const goldens = [1.0, 0.8566748838545029, 0.733891856627126, 0.6287067210378086, 0.53859725722361, 100.0]

Σ = Diagonal(goldens .^ 2)
d = length(goldens)
problem = MvNormal(zeros(d),Σ)
initial_state = zeros(d)
rcdfs = [Base.Fix1(cdf, marginal(problem, k)) for k in 1:d]

Random.seed!(20250211)
grid_experiment(HMC, problem, initial_state, rcdfs, (;
    epsilon = range(0.1, 1.1; length=81),
    L = [1 2 4 8 16 32 64],
))

Random.seed!(20250211)
grid_experiment(FFF, problem, initial_state, rcdfs, (;
    epsilon = range(0.1, 1.1; length=81),
    L = [1 2 4 8 16 32 64],
    lambda = logrange(0.001, 1.0; length=21),
    #ρ = [0.0, 0.5],
))
