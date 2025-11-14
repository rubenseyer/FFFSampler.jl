#!/bin/bash
#SBATCH -A C3SE2025-1-15 -p vera
#SBATCH -J fff-robustness-german
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
CMD="srun --exact -n1 -c1 --mem=7G julia $REPOS/frogflipfresh/experiments/cluster/german.jl $SLURM_ARRAY_TASK_ID"
parallel -N1 -u -j $SLURM_NTASKS "$CMD {1}" ::: $(seq $SLURM_NTASKS)
exit 0
=#
# WARNING: includes are relative to file, so you must have the absolute
# path as argument to julia
###begin
include("_common.jl")
using LinearAlgebra, Distributions
using LogDensityProblemsAD, LogDensityProblems, ForwardDiff
using DelimitedFiles: readdlm
using LogExpFunctions: logistic, log1pexp

data = readdlm(joinpath(dirname(@__FILE__),"german.data-numeric"))
X = mapslices(zscore, @view(data[:,1:end-1]); dims=1)  # standardize predictors
X = hcat(ones(size(X,1)), X)  # add intercept
y = Int.(data[:,end] .- 1)  # make response 0/1

struct BayesianLogisticRegression{T,S}
    X::Matrix{T}
    y::Vector{Int}
    Γ::S
end
LogDensityProblems.dimension(D::BayesianLogisticRegression) = size(D.X,2)
LogDensityProblems.capabilities(::Type{<:BayesianLogisticRegression}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity(D::BayesianLogisticRegression, β::AbstractArray{<:Real, M}) where M
    #u = D.X * β
    #return sum(D.y .* u .- log1pexp.(u)) - 0.5 * dot(β, D.Γ, β)
    logp = -0.5 * dot(β, D.Γ, β)
    for i in eachindex(D.y)
        u = dot(@view(D.X[i, :]), β)
        logp += D.y[i]*u - log1pexp(u)
    end
    return logp
end
function LogDensityProblems.logdensity_and_gradient(D::BayesianLogisticRegression, β::AbstractArray{<:Real, M}) where M
    #u = D.X * β
    #z = D.Γ * β
    #logp = sum(D.y .* u .- log1pexp.(u)) - 0.5 * dot(β, z)
    #∇logp = vec(sum(D.X .* (D.y .- logistic.(u)); dims=1)) .- z
    #return logp, ∇logp
    ∇logp = -D.Γ * β
    logp = 0.5*dot(β, ∇logp)
    for i in eachindex(D.y)
        @views u = dot(D.X[i, :], β)
        logp += D.y[i]*u - log1pexp(u)
        w = D.y[i] - logistic(u)
        for j in eachindex(∇logp)
            ∇logp[j] += D.X[i, j] * w
        end
    end
    return logp, ∇logp
end

# first and second moment
f(x) = [x; abs2.(x)]

problem = BayesianLogisticRegression(X,y,0.01I)
reference_samples, _ = readdlm(joinpath(dirname(@__FILE__),"german.reference"); header=true)
reference_samples = [reference_samples abs2.(reference_samples)]
reference_marginals = map(ecdf, eachcol(reference_samples))
initial_state = zeros(LogDensityProblems.dimension(problem))
###end

T = 250_000

hmc_settings = (;
    epsilon = range(0.01, 0.11; length=21),
    L = [1 2 3 4 5 10]
)
fff_settings = (;
    epsilon = range(0.01, 0.11; length=21),
    L = [1 2 3 4 5 10],
    lambda = logrange(0.01, 1.0; length=11),
)
bjs_settings = (;
    epsilon = logrange(0.001, 0.1; length=31),
    lambda = logrange(0.001, 1.0; length=21),
)

samplers = [
    "HMC" => (HMC, hmc_settings),
    "RHMC" => (FFFSampler.RHMC, fff_settings),
    "FFFmh" => (FFF{FFFSampler.MHBalancing}, fff_settings),
    #"FFFbarker" => (FFF{FFFSampler.BarkerBalancing}, fff_settings),
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

Random.seed!(20250924 + replicate_offset)
grid_experiment(sampler, problem, initial_state, reference_marginals, settings; T, f, name=sampler_name)
