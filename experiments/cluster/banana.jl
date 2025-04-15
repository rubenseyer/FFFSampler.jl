#!/bin/bash
#
#SBATCH -J naiss2024-22-1100-fff-banana
#SBATCH -t 12:00:00
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
exec julia -t 32 /proj/pdmps/repos/frogflipfresh/experiments/cluster/banana.jl
=#
# WARNING: includes are relative to file, so you must have the absolute
# path as argument to julia
begin 
include("_common.jl")

struct HybridRosenbrock{Tμ,Ta,Tb} <: ContinuousMultivariateDistribution
    μ::Tμ
    a::Ta
    b::Tb
end
Base.length(D::HybridRosenbrock) = Base.length(D.b) + 1

LogDensityProblems.dimension(D::HybridRosenbrock) = length(D)
LogDensityProblems.capabilities(::Type{<:HybridRosenbrock}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(D::HybridRosenbrock, x::AbstractArray{<:Real, M}) where M
    n2, n1 = size(D.b)
    X = reshape(@view(x[2:end]), n1, n2)
    y = -D.a*(x[1] - D.μ)^2 
    for j in 1:n2 
        y -= D.b[j,1]*(X[j,1] - x[1]^2)^2
        for i in 2:n1
            y -= D.b[j,i]*(X[j,i] - X[j,i-1]^2)^2
        end
    end
    C = 0.5log(D.a) + 0.5sum(log.(D.b)) - length(x)/2*log(π)
    y + C
end

function LogDensityProblems.logdensity_and_gradient(D::HybridRosenbrock, x::AbstractArray{<:Real, M}) where M
    n2, n1 = size(D.b)
    X = reshape(@view(x[2:end]), n1, n2)

    y = -D.a*(x[1] - D.μ)^2 
    ∇y = similar(x)
    ∇y[1] = -2D.a*(x[1] - D.μ)

    ∇Y = reshape(@view(∇y[2:end]), n1, n2)
    for j in 1:n2
        y -= D.b[j,1]*(X[j,1] - x[1]^2)^2
        ∇Y[j,1] = -2D.b[j,1]*(X[j,1] - x[1]^2)
        ∇y[1] += 4D.b[j,1]*(X[j,1] - x[1]^2)*x[1]
        for i in 2:n1
            y -= D.b[j,i]*(X[j,i] - X[j,i-1]^2)^2
            ∇Y[j,i] = -2D.b[j,i]*(X[j,i] - X[j,i-1]^2)
            ∇Y[j,i-1] += 4D.b[j,i]*(X[j,i] - X[j,i-1]^2)*X[j,i-1]
        end
    end
    C = 0.5log(D.a) + 0.5sum(log.(D.b)) - length(x)/2*log(π)
    y + C, ∇y
end

function Distributions._rand!(rng::Random.AbstractRNG, D::HybridRosenbrock, x::AbstractVector{<:Real})
    n2, n1 = size(D.b)
    X = reshape(@view(x[2:end]), n2, n1)
    x[1] = D.μ + randn(rng)/√(2*D.a)
    for j in 1:n2 
        X[j,1] = x[1]^2 + randn(rng)/√(2D.b[j,1])
        for i in 2:n1
            X[j,i] = X[j,i-1]^2 + randn(rng)/√(2D.b[j,i])
        end
    end
    x
end
marginal1(D::HybridRosenbrock) = Normal(D.μ, 1/√(2*D.a))

problem = HybridRosenbrock(1., 1/10, 100/10*ones(1,1))
d = LogDensityProblems.dimension(problem)
initial_state = [4.678, 4.678^2]
reference_samples = rand(problem, 5_000_000)
rcdfs = [Base.Fix1(cdf, marginal1(problem)); map(k -> ecdf(view(reference_samples, k, :)), 2:d)]
end

Random.seed!(20250310)
grid_experiment(HMC, problem, initial_state, rcdfs, (;
    epsilon = range(0.01, 0.06; length=21),
    L = [1 2 5 10 20 50 100 200 500 1000 2000],
))

Random.seed!(20250310)
grid_experiment(FFF, problem, initial_state, rcdfs, (;
    epsilon = range(0.01, 0.06; length=21),
    L = [1 2 5 10 20 50 100 200 500 1000 2000],
    lambda = logrange(0.001, 0.5; length=11),
))

Random.seed!(20250310)
grid_experiment(FFFMirror, problem, initial_state, rcdfs, (;
    epsilon = range(0.01, 0.06; length=21),
    L = [1 2 5 10 20 50 100 200 500 1000 2000],
    lambda = logrange(0.001, 0.5; length=11),
))
