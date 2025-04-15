using LogDensityProblems, LogDensityProblemsAD, ForwardDiff
using FFFSampler, AdvancedHMC, AbstractMCMC
using LinearAlgebra, Statistics, Distributions, Random

include("../tools/metrics.jl");

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

function trace_plot!(f,D,title,trajectory)
    ax = Axis(f; title, aspect=1, titlesize=12)

    grid = range(-1.25D.R,1.25D.R,length=250)
    contourf!(ax, grid, grid, (x,y) -> exp(LogDensityProblems.logdensity(D,[x;y])), colormap=(Reverse(:grays), 0.3))

    ws = _weights(trajectory)
    ws ./= sum(ws)/length(ws)
    #ts = [0; cumsum(ws)[1:end-1]]  # expected time instead to correspond to estimate
    xs = _states(Point2f, trajectory)

    #lines!(ax, xs; linewidth=0.2)
    scatter!(ax, xs; alpha=0.5, markersize=7.5 .* sqrt.(ws))

    return ax
end

Random.seed!(1234)
ϵ = 0.18
L = 7
λ = 0.1
raw_hmc = AbstractMCMC.sample(problem, HMC(ϵ,L), 50; initial_params=initial_state)
raw_mala = AbstractMCMC.sample(problem, HMC(ϵ,1), 50L; initial_params=initial_state)
raw_fff = AbstractMCMC.sample(problem, FFF(ϵ,1,λ), 50L; initial_params=initial_state)
raw_fff_long = AbstractMCMC.sample(problem, FFF(ϵ,L,λ), 50; initial_params=initial_state)

using GLMakie
begin
    fig = Figure(size=(900,225));
    trace_plot!(fig[1,1],D,"FFF ϵ=$ϵ L=1 λ=$λ",raw_fff)
    trace_plot!(fig[1,2],D,"FFF ϵ=$ϵ L=$L λ=$λ",raw_fff_long)
    trace_plot!(fig[1,3],D,"HMC ϵ=$ϵ L=1",raw_mala)
    trace_plot!(fig[1,4],D,"HMC ϵ=$ϵ L=$L",raw_hmc)
    save("donut_trajectories.png", fig, scalefactor=3)
    fig
end