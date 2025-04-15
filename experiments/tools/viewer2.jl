using GLMakie
using GLMakie.Colors

function stuck(raw)
    vs, fs = rle([r.stat.is_accept for r in raw])
    vs[1] ? median(fs[2:2:end]) :  median(fs[1:2:end])
end

function plot_trajectory(traj::AbstractVector{<:FFFSampler.AbstractFFFTransition}; size=(800,800), ds=1:min(length(traj[1].current.q), 5))
    ws = [1.0/sum(t.Λ) for t in traj]
    ts = [0; cumsum(ws)[1:end-1]]  # expected time instead to correspond to estimate
    xs = [t.current.q for t in traj]

    # figure out what each transition was
    ttypes = map(1:(length(traj)-1)) do i # IterTools.partition(xs, 2, 1)
        [:green, :orange, :darkblue, :red][Int(traj[i].action)]
    end
    ttypes = [ttypes; :black]
    ws2 = [ttypes[i] in (:darkblue, :orange, :red) ? 105.0 : ws[i] for i in eachindex(ws)]
    ttypes_doubled = collect(Iterators.flatten(zip(ttypes,ttypes)))[begin:end-1] # for stairs

    fig = Figure(size=size)
    for i in ds, j in ds
        if i == j
            ax, _ = stairs(fig[i,j], ts, getindex.(xs, ds[i]); step=:post, color=ttypes_doubled)
            scatter!(ax, ts, getindex.(xs, ds[i]), color=ttypes, markersize=5*(ttypes .!= :green))
            ax.ylabel = "x$(ds[i])"
        elseif i < j
            xi, xj = getindex.(xs, ds[i]), getindex.(xs, ds[j])
            ax, _ = lines(fig[i,j], xi, xj; color=:black, linewidth=0.1)
            ax.xlabel = "x$i"
            ax.ylabel = "x$j"
            scatter!(ax, xi, xj; markersize=ws2, color=ttypes, alpha=0.5)
        elseif i > j
            xi, xj = getindex.(xs, ds[i]), getindex.(xs, ds[j])
            #ax, _ = scatter(fig[i,j], xj, xi; markersize=ws, color=:black, alpha=0.1)
            cs = RGBA.(0.0,0.0,0.0, ws/2median(ws))
            ax, _ = scatter(fig[i,j], xj, xi;  axis=(;aspect = DataAspect()), markersize=2.0, color=cs)

            ax.xlabel = "x$(ds[j])"
            ax.ylabel = "x$(ds[i])"
        end
    end
    return fig
end

function plot_trajectory(traj::AbstractVector{<:AdvancedHMC.Transition}; size=(800,800), ds=1:min(length(traj[1].z.θ), 5))
    ts = 0:(length(traj)-1)
    xs = [t.z.θ for t in traj]
    st = try stuck(traj) catch _; 1.0 end

    # figure out what each transition was
    ttypes = map(1:(length(traj)-1)) do i # IterTools.partition(xs, 2, 1)
        t0, t1 = traj[i], traj[i+1]
        if t0.z.θ != t1.z.θ
            :green
        else
            :yellow
        end
    end
    ttypes = [ttypes; :black]
    ttypes_doubled = collect(Iterators.flatten(zip(ttypes,ttypes)))[begin:end-1] # for stairs

    fig = Figure(size=size)
    for i in ds, j in ds
        if i == j
            ax, _ = stairs(fig[i,j], ts, getindex.(xs, i); step=:post, color=ttypes_doubled)
            ax.ylabel = "x$i"
        elseif i < j
            xi, xj = getindex.(xs, i), getindex.(xs, j)
            ax, _ = lines(fig[i,j], xi, xj; color=:black, linewidth=0.1)
            scatter!(fig[i,j], xi, xj; color=:blue, alpha=0.1, markersize=2.0)
            ax.xlabel = "x$i"
            ax.ylabel = "x$j"
        elseif i > j
            xi, xj = getindex.(xs, i), getindex.(xs, j)
            ax, _ = scatter(fig[i,j], xj, xi; markersize=2.0, axis=(;aspect = DataAspect()), color=:black, alpha=1/2st)
            ax.xlabel = "x$j"
            ax.ylabel = "x$i"    
        end
    end
    return fig
end

function plot_trajectory2(traj::AbstractVector{<:AdvancedHMC.Transition}; size=(800,800), trunc=5)
    d = min(length(traj[1].z.θ), trunc)
    ts = 0:(length(traj)-1)
    xs = [t.z.θ for t in traj]
    st = try stuck(traj) catch _; 1.0 end

    # figure out what each transition was
    ttypes = map(1:(length(traj)-1)) do i # IterTools.partition(xs, 2, 1)
        t0, t1 = traj[i], traj[i+1]
        if t0.z.θ != t1.z.θ
            :green
        else
            :yellow
        end
    end
    ttypes = [ttypes; :black]
    ttypes_doubled = collect(Iterators.flatten(zip(ttypes,ttypes)))[begin:end-1] # for stairs
    d2 = ceil(Int, sqrt(d))
    fig = Figure(size=size)
    for k in 1:d
        ij = CartesianIndices((d2,d2))[k]
        ax, _ = stairs(fig[ij[1],ij[2]], ts, getindex.(xs, k); step=:post, color=ttypes_doubled)
        ax.ylabel = "x$k"
    end
    return fig
end