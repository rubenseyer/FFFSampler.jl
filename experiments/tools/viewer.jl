using GLMakie
using GLMakie.Colors

const palette = parse.(Ref(Colors.Colorant), ("#004d40", "#ffc107", "#1e8835", "#d81b60"))

function stuck(raw)
    vs, fs = rle([r.stat.is_accept for r in raw])
    vs[1] ? median(fs[2:2:end]) :  median(fs[1:2:end])
end

function plot_trajectory(traj::AbstractVector{<:FFFSampler.AbstractFFFTransition}; size=(800,800))
    d = min(length(traj[1].current.q), 5)
    ws = 1.0*[1/sum(t.Λ) for t in traj]
    ts = [0; cumsum(ws)[1:end-1]]  # expected time instead to correspond to estimate
    xs = [t.current.q for t in traj]

    # figure out what each transition was
    ttypes = map(t -> palette[t.action], traj)
    #ws2 = ws .* [t.action != FFFSampler.FORWARD ? 4.0 : 1.0 for t in ttypes]
    ttypes_doubled = collect(Iterators.flatten(zip(ttypes,ttypes)))[begin+1:end] # for stairs

    fig = Figure(size=size)
    for i in 1:d, j in 1:d
        if i == j
            ax, _ = stairs(fig[i,j], ts, getindex.(xs, i); step=:post, color=ttypes_doubled)
            ax.ylabel = "x$i"
        elseif i < j
            xi, xj = getindex.(xs, i), getindex.(xs, j)
            ax, _ = lines(fig[i,j], xi, xj; color=:black, linewidth=0.1)
            ax.xlabel = "x$i"
            ax.ylabel = "x$j"
            scatter!(ax, xi, xj; markersize=sqrt.(ws), color=ttypes, alpha=0.2)
        elseif i > j
            xi, xj = getindex.(xs, i), getindex.(xs, j)
            #ax, _ = scatter(fig[i,j], xj, xi; markersize=ws, color=:black, alpha=0.1)
            cs = RGBA.(0.0,0.0,0.0, ws/2median(ws))
            ax, _ = scatter(fig[i,j], xj, xi;  axis=(;aspect = DataAspect()), markersize=2.0, color=cs)

            ax.xlabel = "x$j"
            ax.ylabel = "x$i"
        end
    end
    return fig
end

function plot_trajectory(traj::AbstractVector{<:AdvancedHMC.Transition}; size=(800,800))
    d = min(length(traj[1].z.θ), 5)
    ts = 0:(length(traj)-1)
    xs = [t.z.θ for t in traj]
    st = stuck(traj)

    # figure out what each transition was
    ttypes = map(t -> t.stat.is_accept ? palette[FFFSampler.FORWARD] : palette[FFFSampler.FLIP], traj)
    ttypes_doubled = collect(Iterators.flatten(zip(ttypes,ttypes)))[begin+1:end] # for stairs

    fig = Figure(size=size)
    for i in 1:d, j in 1:d
        if i == j
            ax, _ = stairs(fig[i,j], ts, getindex.(xs, i); step=:post, color=ttypes_doubled)
            ax.ylabel = "x$i"
        elseif i < j
            xi, xj = getindex.(xs, i), getindex.(xs, j)
            ax, _ = lines(fig[i,j], xi, xj; color=:black, linewidth=0.1)
            scatter!(fig[i,j], xi, xj; color=ttypes, alpha=0.2, markersize=1.0)
            ax.xlabel = "x$i"
            ax.ylabel = "x$j"
        elseif i > j
            xi, xj = getindex.(xs, i), getindex.(xs, j)
            ax, _ = scatter(fig[i,j], xj, xi; markersize=2.0, axis=(;aspect = DataAspect()), color=:black)
            ax.xlabel = "x$j"
            ax.ylabel = "x$i"    
        end
    end
    return fig
end