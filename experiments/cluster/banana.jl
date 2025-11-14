#!/bin/bash
#SBATCH -A C3SE2025-1-15 -p vera
#SBATCH -J fff-robustness-banana
#SBATCH -t 7-00:00:00
#SBATCH -n32
#SBATCH --array=0-7
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rubense@chalmers.se

#=
echo "# This is $SLURM_JOB_NAME on version $(git -C $REPOS/frogflipfresh/ describe --always --dirty)."
if [ -n "$(git -C $REPOS/frogflipfresh/ status --porcelain)" ]; then 
  echo -e "\n########## The script follows ##########"
  cat $0
  echo -e "########## This concludes the script ##########\n"
fi

cd $TMPDIR
function cleanup() {
    OUTPUT_NAME=(*.csv)
    OUTPUT_NAME=${OUTPUT_NAME%-*.*}
    awk '(NR == 1) || (FNR > 1)' *.csv | pigz > "$SLURM_SUBMIT_DIR/$OUTPUT_NAME-$SLURM_ARRAY_JOB_ID.csv.gz"
}
trap cleanup EXIT

module load Julia/1.10.2-linux-x86_64
# ensure your environment sets the JULIA_DEPOT_PATH correctly
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JULIA_PROJECT=$REPOS/frogflipfresh/experiments
CMD="srun --exact -n1 -c1 --mem=7G julia $REPOS/frogflipfresh/experiments/cluster/banana.jl $SLURM_ARRAY_TASK_ID"
parallel -N1 -u -j $SLURM_NTASKS "$CMD {1}" ::: $(seq $SLURM_NTASKS)
exit 0
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

problem = HybridRosenbrock(1., 1/10, 100/10*ones(1,1))
d = LogDensityProblems.dimension(problem)
initial_state = [4.678, 4.678^2]
reference_samples = rand(problem, 5_000_000)
reference_marginals = (
    WithReferenceSample(Normal(problem.μ, 1/√(2*problem.a)), reference_samples[1,:]),
    map(k -> ecdf(view(reference_samples, k, :)), 2:d)...,
    #map(k -> ecdf(abs2.(view(reference_samples, k, :))), 1:d)...
)
end

# first and second moment
#f(x) = [x; abs2.(x)]
f = identity

T = 3_500_000

hmc_settings = (;
    epsilon = range(0.01, 0.05; length=31),
    L = [1 2 3 4 5 10 20 50 100 200 500 1000 2000]
)
fff_settings = (;
    epsilon = range(0.01, 0.05; length=31),
    L = [1 2 3 4 5 10 20 50 100 200 500 1000 2000],
    lambda = logrange(0.001, 0.5; length=21),
)
bjs_settings = (;
    epsilon = logrange(0.001, 0.05; length=31),
    lambda = logrange(0.001, 0.5; length=21),
)

samplers = [
    "HMC" => (HMC, hmc_settings),
    "RHMC" => (FFFSampler.RHMC, fff_settings),
    "FFFmh" => (FFF{FFFSampler.MHBalancing}, fff_settings),
    "FFFsqrt" => (FFF{FFFSampler.SqrtBalancing}, fff_settings),
    "RGW" => (FFFSampler.RGW, bjs_settings),
    "BGW" => (FFFSampler.BGW{FFFSampler.MHBalancing}, bjs_settings),
    "BJSmh" => (BouncyJump{FFFSampler.MHBalancing}, bjs_settings),
    "BJSsqrt" => (BouncyJump{FFFSampler.SqrtBalancing}, bjs_settings),
]

@assert !isempty(ARGS) && 0 ≤ (ix = parse(Int, ARGS[1])) < length(samplers)
sampler_name, (sampler, settings) = samplers[ix + 1]
replicate_offset = length(ARGS) == 2 ? parse(Int, ARGS[2]) : 0
sampler_name *= "-$replicate_offset"
println("Running $sampler_name (index $(ix))")

Random.seed!(20250310)
grid_experiment(sampler, problem, initial_state, reference_marginals, settings; T, f, name=sampler_name)