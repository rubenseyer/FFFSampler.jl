using CSV, DataFrames
using Statistics, StatsBase
#using GLMakie
using CairoMakie
const palette = Dict(
    "HMC" => colorant"#882255",
    "RHMC" => colorant"#cc6677",
    "FFFmh" => colorant"#117733",
    "FFFsqrt" => colorant"#44aa99",
    "RGW" => colorant"#999933",
    "BGW" => colorant"#ddcc77",
    "BJSmh" => colorant"#88ccee",
    "BJSsqrt" => colorant"#332288",
)
const linestyles = Dict(
    "HMC" => :solid,
    "RHMC" => :dash,
    "FFFmh" => :dot,
    "FFFsqrt" => :dashdot,
    "RGW" => :solid,
    "BGW" => :dash,
    "BJSmh" => (:dot,:dense),
    "BJSsqrt" => :dashdotdot,
)
const pseudolog2 = Makie.ReversibleScale(
    x -> sign(x) * log2(abs(x) + 1),
    x -> sign(x) * (2^(abs(x)) - 1);
    limits = (0.0f0, 3.0f0),
    name = :pseudolog2
)


mean_se(x) = let m = mean(x); (m, sem(x; mean=m)) end
# Let's commit type crimes. Need the lexicographic tuple ordering to be poisoned by NaN
Base.min(a::Tuple{Float64,Float64}, b::Tuple{Float64,Float64}) = let cond = all(isnan, a) || (a[1] < b[1]) || (a[1] == b[1] && a[2] < b[2]); ifelse(cond, a, b) end
Base.max(a::Tuple{Float64,Float64}, b::Tuple{Float64,Float64}) = let cond = all(isnan, a) || (a[1] > b[1]) || (a[1] == b[1] && a[2] > b[2]); ifelse(cond, a, b) end

median_nansafe(xs) = median(x for x in xs if !isnan(x))
function quantile_nansafe(xs, p; kwargs...)
    if all(isnan, xs)
        return p ≥ 0.5 ? -Inf : +Inf  # If they all suck, return extremes
    end
    return quantile((x for x in xs if !isnan(x)), p; kwargs...)
end

function load_performance(modelname, samplers; cutoff_W2=Inf, cutoff_KS=Inf)
    ls = filter(s -> contains(s, "csv") && contains(s, "-$(modelname)-"), readdir("."))
    datafiles = map(sampler -> first(sort(filter(s -> split(s, '-')[end-1] == sampler, ls), rev=true)), samplers)
    return map(datafiles) do filename
        df = CSV.read(filename, DataFrame)

        # Change ESS to be per grads
        transform!(df,
            [:marginal_dESS, :ngrads] => ((ess, ngrads) -> ess ./ (1e-3 .* ngrads)) => "marginal_dESS/kgrad",
            [:marginal_dESS, :walltime] => ((ess, walltime) -> ess ./ walltime) => "marginal_dESS/s"
        )

        #df[df.rhat .> 1.01, r"marginal_"] .= NaN
        if isfinite(cutoff_W2)
            df[df.marginal_W2 .> cutoff_W2, r"marginal_"] .= NaN
        end
        if isfinite(cutoff_KS)
            df[df.marginal_KS .> cutoff_KS, r"marginal_"] .= NaN
        end

        groupcols = filter(in(propertynames(df)), [:epsilon, :L, :lambda, :ρ])
        df = groupby(df, [groupcols; :d])
        df = combine(df, valuecols(df) .=> mean_se)
        select!(df, Not([:d]))
        df = groupby(df, groupcols)
        df = combine(df, valuecols(df) .=> [minimum maximum])
        df = groupby(df, groupcols)
        df = combine(df, valuecols(df) .=> (c -> reshape(reinterpret(Float64, c), (2,:))') .=> (name -> let n = replace(name, "_se" => ""); [n, n * "_se"] end))

        return df
    end
end

function performance_slice(df, filtercol, slicecols = [:L, :lambda, :ρ]; p=0.0)
    # over the range of values in the slicecols,
    # find the lowest median value of filtercol.
    # This avoids outliers

    slicecols = filter(in(propertynames(df)), slicecols)
    if endswith(filtercol, "maximum")
        fdf = combine(groupby(df, slicecols), filtercol => (x -> quantile_nansafe(x, p)) => :target)
        best_ix = findmin(fdf.target)[2]  # best is the smallest despite max
    elseif endswith(filtercol, "minimum")
        fdf = combine(groupby(df, slicecols), filtercol => (x -> quantile_nansafe(x, 1-p)) => :target)
        best_ix = findmax(fdf.target)[2]  # best is the largest despite min
    else
        error("unsupported reduction")
    end

    best_vals = fdf[best_ix, slicecols]
    
    # slice to only those rows that have the best value in the slicecols
    df2 = df
    for col in slicecols
        if col in propertynames(df)
            df2 = filter(row -> row[col] == best_vals[col], df2)
        end
    end
    return df2
end

function strip_hanging_nan(df, ycol)
    # assumes sorted in xcol
    mask = fill(false, nrow(df))
    seen_valid = false
    seen_following_invalid = false
    for (i, row) in enumerate(eachrow(df))
        if seen_valid && seen_following_invalid
            break
        elseif !isnan(row[ycol])
            mask[i] = true
            seen_valid = true
        elseif seen_valid && !seen_following_invalid
            seen_following_invalid = true
        end
    end
    return df[mask,:]
end

function merge_plots_legend(axes)
    plots_in_fig = AbstractPlot[]
    labels_in_fig = AbstractString[]
    for ax in axes
        pl, lb = Makie.get_labeled_plots(ax, merge=false, unique=false)
        append!(plots_in_fig, pl)
        append!(labels_in_fig, lb)
    end

    ulabels = Base.unique(labels_in_fig)
    mergedplots = [[lp for (i, lp) in enumerate(plots_in_fig) if labels_in_fig[i] == ul]
            for ul in ulabels]
    return mergedplots, ulabels
end

function plot_robustness_1d(samplerss, datas, stats_and_reductions = [("marginal_W2", maximum), ("marginal_dESS/kgrad", minimum)]; filtercol="marginal_W2_mean_maximum")
    fig = Figure(size=(650,235), figure_padding=0.1);
    axes = Makie.Axis[]

    for (i, (stat, reduction)) in enumerate(stats_and_reductions)
        outcol = "$(stat)_mean_$(reduction)"
        ylabel = uppercasefirst(stat == "fliprate" ? "flip proportion" : "$reduction $(replace(stat[10:end], "_"=>" "))")
        ax = Axis(fig[1,i]; xlabel="ϵ", ylabel) #, yscale=Makie.log10, xscale=Makie.log10)
        samplers = samplerss isa Vector{Vector{String}} ? samplerss[i] : samplerss
        data = datas isa Vector{Vector{DataFrame}} ? datas[i] : datas
        for (sampler, df_orig) in zip(samplers, data)
            df = performance_slice(df_orig, filtercol, [:L, :lambda, :ρ]; p=0.0)
            df = strip_hanging_nan(df, outcol)
            label = sampler
            :L in propertynames(df) && (label *= " L=$(round(Int,df[begin,:L]))")
            :lambda in propertynames(df) && (label *= " λ=$(round(df[begin,:lambda]; digits=3))")
            #scatter!(ax, df.epsilon, df[:,outcol]; label, markersize=4)
            errorbars!(ax, df.epsilon, df[:,outcol], 2 .* df[:,outcol * "_se"]; whiskerwidth=5, alpha=0.8, color=palette[sampler])
            lines!(ax, df.epsilon, df[:,outcol]; label, color=palette[sampler], linestyle=linestyles[sampler])
        end
        push!(axes, ax)
    end
    mergedplots, ulabels = merge_plots_legend(axes)
    fig[0,begin:end] = Legend(fig, mergedplots, ulabels, framevisible=false, merge=true, orientation=:horizontal, nbanks=length(ulabels) > 4 ? 2 : 1)

    return fig
end

function plot_robustness_2d(samplers, data, stat="marginal_W2", reduction=maximum, filtercol=nothing; slicecols=[:lambda, :ρ], slicep=0.0, yaxis=:L, kwargs...)
    fig = Figure(size=(725,192), figure_padding=0);
    outcol = "$(stat)_mean_$(reduction)"

    data = map(df -> performance_slice(df, !isnothing(filtercol) ? filtercol : outcol, slicecols; p=slicep), data)
    crange = round.(extrema(filter(isfinite, hcat([df[:,outcol] for df in data]...))); digits=2)
    reverse = (reduction === minimum)  # flip if we want to minimize so the same color is good

    for (ix, (sampler, df)) in enumerate(zip(samplers, data))
        title = sampler
        titlecols = intersect(slicecols, propertynames(df))
        :L in titlecols && (title *= " L=$(round(df[begin,:L]; digits=0))")
        :lambda in titlecols && (title *= " λ=$(round(df[begin,:lambda]; digits=3))")
        :ρ in titlecols && (title *= " ρ=$(round(df[begin,:ρ]; digits=2))")

        ylabel = (ix == 1 ? (yaxis == :lambda ? "λ" : String(yaxis)) : "")
        ax = Axis(fig[1,ix]; xlabel="ϵ", ylabel, aspect=1, title, titlesize=12,
                xticks=WilkinsonTicks(5), xminorticks=IntervalsBetween(5), xminorticksvisible=true,
                yminorticks=IntervalsBetween(5), yminorticksvisible=true, kwargs...)
        heatmap!(ax, df.epsilon, df[:,yaxis], df[:,outcol]; colorrange=crange, colormap=reverse ? Reverse(:magma) : :magma, nan_color=:white)
        #scatter!(ax, best_HMC.epsilon, best_HMC.L; color=:white, marker=:diamond)
    end

    Colorbar(fig[1,length(samplers)+1]; limits=crange, colormap=reverse ? Reverse(:magma) : :magma,
            #ticks=range(crange[1], crange[2]; length=5),
            label=uppercasefirst("$reduction $(replace(stat[10:end], "_"=>" "))"))
    #colgap!(fig.layout, 1, 50)
    return fig
end

# Output the plots for the paper
begin
    modelname = "german"
    samplers = ["HMC", "RHMC", "FFFmh", "FFFsqrt"]
    data = load_performance(modelname, samplers; cutoff_KS=0.75)
    save("out/robustness1d-$modelname.pdf",
        plot_robustness_1d(samplers, data, [("marginal_dESS/kgrad", minimum), ("fliprate", maximum)]; filtercol="marginal_dESS/kgrad_mean_minimum"), scalefactor=3)
    save("out/robustness2d-$modelname-ess.pdf",
        plot_robustness_2d(samplers, data, "marginal_dESS/kgrad", minimum; yscale=pseudolog2, yticks=[2,5,10]), scalefactor=3)
    save("out/robustness2d-$modelname-W2.pdf",
        plot_robustness_2d(samplers, data, "marginal_W2", maximum, "marginal_dESS/kgrad_mean_minimum"; yscale=pseudolog2, yticks=[2,5,10]), scalefactor=3)
    save("out/robustness2d-$modelname-ess-lambda.pdf",
        plot_robustness_2d(samplers[begin+1:end], data[begin+1:end], "marginal_dESS/kgrad", minimum; slicecols=[:L], yaxis=:lambda), scalefactor=3)

    samplers = ["RGW", "BGW", "BJSmh", "BJSsqrt"]
    data = load_performance(modelname, samplers; cutoff_KS=0.75)
    save("out/robustness2dbjs-$modelname-ess.pdf",
        plot_robustness_2d(samplers, data, "marginal_dESS/s", minimum; slicecols=[], yaxis=:lambda, yscale=log10, xscale=log10, xticks=LogTicks(WilkinsonTicks(3))), scalefactor=3)
    save("out/robustness2dbjs-$modelname-W2.pdf",
        plot_robustness_2d(samplers, data, "marginal_W2", maximum, "marginal_dESS/s_mean_minimum"; slicecols=[], yaxis=:lambda, yscale=log10, xscale=log10, xticks=LogTicks(WilkinsonTicks(3))), scalefactor=3)
    
    samplers = ["RGW", "BGW", "BJSmh", "BJSsqrt", "FFFmh", "FFFsqrt"]
    data = load_performance(modelname, samplers; cutoff_KS=0.75)
    save("out/robustness1dbjs-$modelname.pdf",
        plot_robustness_1d([samplers[1:4], samplers[3:end]], [data[1:4], data[3:end]], [("marginal_dESS/s", minimum), ("marginal_dESS/kgrad", minimum)]; filtercol="marginal_dESS/kgrad_mean_minimum"), scalefactor=3)
end

begin
    modelname = "banana"
    samplers = ["HMC", "RHMC", "FFFmh", "FFFsqrt"]
    data = load_performance(modelname, samplers; cutoff_KS=0.75, cutoff_W2=20.)

    save("out/robustness2d-$modelname-ess.pdf",
        plot_robustness_2d(samplers, data, "marginal_dESS/kgrad", minimum, "marginal_W2_mean_maximum"; yscale=Makie.log10, slicep=0.5), scalefactor=3)
    save("out/robustness2d-$modelname-W2.pdf",
        plot_robustness_2d(samplers, data, "marginal_W2", maximum, "marginal_W2_mean_maximum"; yscale=Makie.log10, slicep=0.5), scalefactor=3)
    save("out/robustness2d-$modelname-ess-lambda.pdf",
        plot_robustness_2d(samplers[begin+1:end], data[begin+1:end], "marginal_dESS/kgrad", minimum, "marginal_W2_mean_maximum"; slicecols=[:L], slicep=0.5, yaxis=:lambda, yscale=log10), scalefactor=3)
end

begin
    modelname = "posteriordb"
    samplers = ["HMC", "RHMC", "FFFmh", "FFFsqrt"]
    data = load_performance(modelname, samplers; cutoff_KS=0.75)

    save("out/robustness2d-$modelname-ess.pdf",
        plot_robustness_2d(samplers, data, "marginal_dESS/kgrad", minimum, "marginal_dESS/kgrad_mean_minimum"; yscale=pseudolog2), scalefactor=3)
    save("out/robustness2d-$modelname-W2.pdf",
        plot_robustness_2d(samplers, data, "marginal_W2", maximum, "marginal_dESS/kgrad_mean_minimum"; yscale=pseudolog2), scalefactor=3)
    save("out/robustness2d-$modelname-ess-lambda.pdf",
        plot_robustness_2d(samplers[begin+1:end], data[begin+1:end], "marginal_dESS/kgrad", minimum, "marginal_dESS/kgrad_mean_minimum"; slicecols=[:L], yaxis=:lambda, yscale=log10), scalefactor=3)
end