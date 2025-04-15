using Statistics, FFFSampler, AbstractMCMC
using GLMakie

function analysis_estimate(M, f, problem, setups, initial_params; resolution=250)
    results = map(setups) do (name, args)
        trajectory = AbstractMCMC.sample(problem, args...; initial_params)
        indices = 1:(length(trajectory) ÷ resolution):length(trajectory)
        n = _n_steps(trajectory)[indices]
        values = [FFFSampler.cummean(f, trajectory)[indices]]

        for m in 2:M
            #GC.gc()  # maybe?
            trajectory = AbstractMCMC.sample(problem, args...; initial_params)
            push!(values, FFFSampler.cummean(f, trajectory)[indices])
        end

        (; name, n, values)
    end

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="leapfrog steps", ylabel="estimate")
    for (; name, n, values) in results
        values_mean = [mean(v[i] for v in values) for i in eachindex(values[1])]
        values_min = [minimum(v[i] for v in values) for i in eachindex(values[1])]
        values_max = [maximum(v[i] for v in values) for i in eachindex(values[1])]
        band!(ax, n, values_min, values_max, alpha=0.33, label=name)
        lines!(ax, n, values_mean, label=name)
    end
    axislegend(ax, merge=true)
    return fig
end

function analysis_distance(M, costs, f, problem, setups, initial_params, cdf; distance=distance_ks)
    results = map(setups) do (name, args)
        trajectory = AbstractMCMC.sample(problem, args...; initial_params)
        ns = _n_steps(trajectory)
        indices = searchsortedlast.(Ref(ns), costs)
        xs = _states(f, trajectory)
        ws = _weights(trajectory)
        values = [[N == 0 ? NaN : distance(view(xs, 1:N), view(ws, 1:N), cdf) for N in indices]]

        for m in 2:M
            #GC.gc()  # maybe?
            trajectory = AbstractMCMC.sample(problem, args...; initial_params)
            ns = _n_steps(trajectory)
            indices = searchsortedlast.(Ref(ns), costs)
            xs = _states(f, trajectory)
            ws = _weights(trajectory)
            push!(values, [N == 0 ? NaN : distance(view(xs, 1:N), view(ws, 1:N), cdf) for N in indices])
        end

        (; name, values)
    end

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="gradient evaluations", ylabel="distance metric")
    for (; name, values) in results
        values_mean = [mean(v[i] for v in values) for i in eachindex(values[1])]
        values_std = [std(v[i] for v in values) for i in eachindex(values[1])]
        #values_min = [minimum(v[i] for v in values) for i in eachindex(values[1])]
        #values_max = [maximum(v[i] for v in values) for i in eachindex(values[1])]
        #band!(ax, costs, values_min, values_max, alpha=0.33, label=name)
        band!(ax, costs, values_mean .- 2/√M .* values_std, values_mean .+ 2/√M .* values_std, alpha=0.33, label=name)
        lines!(ax, costs, values_mean, label=name)
    end
    axislegend(ax, merge=true)
    return fig
end