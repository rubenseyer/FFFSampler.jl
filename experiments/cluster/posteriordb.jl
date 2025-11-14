#!/bin/bash
#SBATCH -A C3SE2025-1-15 -p vera
#SBATCH -J fff-robustness-posteriordb
#SBATCH -t 7-00:00:00
#SBATCH -n32
#SBATCH --array=0-3
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

# Download BridgeStan source to TMPDIR because of inode quota...
wget --progress=bar:force:noscroll -O $TMPDIR/bridgestan.tar.gz https://github.com/roualdes/bridgestan/releases/download/v2.7.0/bridgestan-2.7.0.tar.gz
mkdir $TMPDIR/bridgestan
tar -xzf $TMPDIR/bridgestan.tar.gz -C $TMPDIR/bridgestan --strip-components=1
export BRIDGESTAN=$TMPDIR/bridgestan

module load Julia/1.10.2-linux-x86_64
# ensure your environment sets the JULIA_DEPOT_PATH correctly
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JULIA_PROJECT=$REPOS/frogflipfresh/experiments

# Compile the stan models so that multiple jobs can share the artifacts
julia << EOF
    using PosteriorDB, StanLogDensityProblems
    post = PosteriorDB.posterior(PosteriorDB.database(), "one_comp_mm_elim_abs-one_comp_mm_elim_abs")
    problem = StanProblem(post, ".", force=true, make_args=["STAN_THREADS=true"])
EOF

# Run the problem
CMD="srun --exact -n1 -c1 --mem=7G julia $REPOS/frogflipfresh/experiments/cluster/posteriordb.jl $SLURM_ARRAY_TASK_ID"
parallel -N1 -u -j $SLURM_NTASKS "$CMD {1}" ::: $(seq $SLURM_NTASKS)
exit 0
=#
# WARNING: includes are relative to file, so you must have the absolute
# path as argument to julia

include("_common.jl")
using PosteriorDB, StanLogDensityProblems

post = PosteriorDB.posterior(PosteriorDB.database(), "one_comp_mm_elim_abs-one_comp_mm_elim_abs")

# Compile stan model
problem = StanProblem(post, ".", make_args=["STAN_THREADS=true"])
d = LogDensityProblems.dimension(problem)
initial_state = [0.0,0.0,0.0,-2.0]

# first and second moment
f(x) = [x; abs2.(x)]

# Use PosteriorDB gold standard draws as approximate reference
begin
    gold_samples = PosteriorDB.load(PosteriorDB.reference_posterior(post))
    parameter_names = collect(keys(gold_samples[begin]))
    
    gold_matrix = reduce(hcat, reduce(vcat, gold_samples[i][pname] for i in eachindex(gold_samples)) for pname in parameter_names)
    for i in 1:size(gold_matrix,1)
        gold_matrix[i,:] .= StanLogDensityProblems.BridgeStan.param_unconstrain(problem.model, gold_matrix[i,:])
    end

    #means = vec(mean(gold_matrix; dims=1))
    #vars = vec(var(gold_matrix; dims=1))
    #reference_marginals = vec(mapslices(ecdf, gold_matrix; dims=1))
    reference_marginals = vec(mapslices(ecdf, mapslices(f, gold_matrix; dims=2); dims=1))
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

T = 250_000

hmc_settings = (;
    epsilon = range(0.02, 0.2; length=21),
    L = [1 3 7 15 31],
)
fff_settings = (;
    epsilon = range(0.02, 0.2; length=21),
    L = [1 3 7 15 31],
    lambda = logrange(0.0158, 1.0; length=11),
)

samplers = [
    "HMC" => (HMC, hmc_settings),
    "RHMC" => (FFFSampler.RHMC, fff_settings),
    "FFFmh" => (FFF{FFFSampler.MHBalancing}, fff_settings),
    "FFFsqrt" => (FFF{FFFSampler.SqrtBalancing}, fff_settings),
    #"BJSmh" => (BouncyJump{FFFSampler.MHBalancing}, bjs_settings),
    #"BJSsqrt" => (BouncyJump{FFFSampler.SqrtBalancing}, bjs_settings),
]

@assert !isempty(ARGS) && 0 ≤ (ix = parse(Int, ARGS[1])) < length(samplers)
sampler_name, (sampler, settings) = samplers[ix + 1]
replicate_offset = length(ARGS) == 2 ? parse(Int, ARGS[2]) : 0
sampler_name *= "-$replicate_offset"
println("Running $sampler_name (index $(ix))")

Random.seed!(20250328)
grid_experiment(sampler, problem, initial_state, reference_marginals, settings; T, f, name=sampler_name)
