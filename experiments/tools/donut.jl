using LogDensityProblems, LogDensityProblemsAD, ForwardDiff
using FFFSampler, AdvancedHMC, AbstractMCMC
using LinearAlgebra, Statistics, Distributions, Random
using CairoMakie

include("./metrics.jl");

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

function trace_plot!(f,D,title,trajectory; limits=(-1.25D.R,1.25D.R), color=nothing, markerscale=5.0)
    ax = Axis(f; title, titlesize=12, aspect=DataAspect(), xticklabelsvisible=false, yticklabelsvisible=false)

    grid = range(limits...,length=501)
    contourf!(ax, grid, grid, (x,y) -> exp(LogDensityProblems.logdensity(D,[x;y])), levels=25, colormap=(Reverse(:grays), .3))

    ws = _weights(trajectory)
    ws ./= sum(ws)/length(ws)
    #ts = [0; cumsum(ws)[1:end-1]]  # expected time instead to correspond to estimate
    xs = _states(Point2f, trajectory)

    #lines!(ax, xs; linewidth=0.2)
    scatter!(ax, xs; alpha=0.5, markersize=markerscale .* sqrt.(ws), color)

    return ax
end

#Random.seed!(1234); raw_nuts = AbstractMCMC.sample(problem, NUTS(0.65), 100_000; initial_params=initial_state)
#nuts_L = Int(median([t.stat.n_steps for t in raw_nuts[50000:end]]))
#nuts_ϵ = mean([t.stat.step_size for t in raw_nuts[50000:end]])

ϵ = 0.1855
L_HMC = 1
L_RHMC = 1
L_FFF = 1
λ = 0.01
budget = 250

Random.seed!(1234); raw_hmc = AbstractMCMC.sample(problem, HMC(ϵ,L_HMC), budget; initial_params=initial_state)
Random.seed!(1234); raw_rhmc = AbstractMCMC.sample(problem, FFFSampler.RHMC(ϵ,L_RHMC,λ), budget; initial_params=initial_state)
Random.seed!(1234); raw_fff = AbstractMCMC.sample(problem, FFF{FFFSampler.MHBalancing}(ϵ,L_FFF,λ), budget; initial_params=initial_state)
Random.seed!(1234); raw_bjs = AbstractMCMC.sample(problem, BouncyJump{FFFSampler.SqrtBalancing}(ϵ,0.1*λ), budget; initial_params=initial_state)

# budget cuts
resize!(raw_hmc, first(budget_cut(raw_hmc, budget)));
resize!(raw_rhmc, first(budget_cut(raw_rhmc, budget)));
resize!(raw_fff, first(budget_cut(raw_fff, budget)));
resize!(raw_bjs, first(budget_cut(raw_bjs, budget)));

palette = Dict(
    "HMC" => colorant"#882255",
    "RHMC" => colorant"#cc6677",
    "FFFmh" => colorant"#117733",
    "FFFsqrt" => colorant"#44aa99",
    "RGW" => colorant"#999933",
    "BGW" => colorant"#ddcc77",
    "BJSmh" => colorant"#88ccee",
    "BJSsqrt" => colorant"#332288",
)
begin
    rmse(traj) = round(norm(mean(traj)); digits=2)
    acc(traj) = round(FFFSampler.acceptance_rate(traj); digits=2)
    fig = Figure(size=(725,192), figure_padding=1.0);
    #  ϵ=$ϵ L=$L\nRMSE=$(rmse(raw_hmc)) α=$(acc(raw_hmc))
    trace_plot!(fig[1,1],D,"HMC",raw_hmc; color=palette["HMC"])
    trace_plot!(fig[1,2],D,"RHMC",raw_rhmc; color=palette["RHMC"])
    trace_plot!(fig[1,3],D,"FFF (Sec. 3.1)",raw_fff; color=palette["FFFmh"])
    trace_plot!(fig[1,4],D,"BJS (Sec. 3.2)",raw_bjs; color=palette["BJSsqrt"])
    fig
end

save("out/donut.pdf", fig, scalefactor=3)