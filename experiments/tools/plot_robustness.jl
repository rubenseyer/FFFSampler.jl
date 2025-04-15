using DataFrames, DelimitedFiles
using Statistics, StatsBase
using GLMakie

# Settings
#filename_HMC = "naiss2024-22-1100-fff-golden-HMC-202503061528.csv"
#filename_FFF = "naiss2024-22-1100-fff-golden-FFF-202503061718.csv"
#filename_HMC = "naiss2024-22-1100-fff-donut-HMC-202504021314.csv"
#filename_FFF = "naiss2024-22-1100-fff-donut-FFF-202504021334.csv"
#filename_HMC = "naiss2024-22-1100-fff-banana-HMC-202504041205.csv"
#filename_FFF = "naiss2024-22-1100-fff-banana-FFF-202504041225.csv"
filename_HMC = "naiss2024-22-1100-fff-posteriordb-HMC-202504071817.csv"
filename_FFF = "naiss2024-22-1100-fff-posteriordb-FFF-202504072052.csv"

ρ = 0.0
stat = "KS"

# Process data: max marginal KS distance
begin
    # HMC
    data, header = readdlm(filename_HMC, ',', header=true)
    df_HMC = begin
        df = DataFrame(data, Symbol.(vec(header)))
        df = groupby(df, [:epsilon, :L, :d])
        df = combine(df, valuecols(df) .=> mean)
        df = groupby(df, [:epsilon, :L])
        df = combine(df, valuecols(df) .=> maximum)
        #display(eachrow(df)[findmin(df[:,"marginal_$(stat)_mean_maximum"])[2]])
        df
    end

    # FFF
    data, header = readdlm(filename_FFF, ',', header=true)
    df_FFF = begin
        df = DataFrame(data, Symbol.(vec(header)))

        if :ρ in propertynames(df) 
            df = filter(row -> row[:ρ] == ρ, df)
            groupcols = [:epsilon, :L, :lambda, :ρ]
        else
            groupcols = [:epsilon, :L, :lambda]
        end

        df = groupby(df, [groupcols; :d])
        df = combine(df, valuecols(df) .=> mean)
        df = groupby(df, groupcols)
        df = combine(df, valuecols(df) .=> maximum)
        #display(eachrow(df)[findmin(df[:,"marginal_$(stat)_mean_maximum"])[2]])
        df
    end

    best_HMC = df_HMC[findmin(isnan(x) ? Inf : x for x in df_HMC[:,"marginal_$(stat)_mean_maximum"])[2], :]
    best_FFF = df_FFF[findmin(isnan(x) ? Inf : x for x in df_FFF[:,"marginal_$(stat)_mean_maximum"])[2], :]

    lambdas = (minimum(df_FFF.lambda), best_FFF.lambda, maximum(df_FFF.lambda))
end

# Render figure
begin
    fig = Figure(size=(900,225));
    crange = extrema(filter(isfinite, [df_HMC[:,"marginal_$(stat)_mean_maximum"]; df_FFF[:,"marginal_$(stat)_mean_maximum"]]))

    ax = Axis(fig[1,1]; xlabel="ϵ", ylabel="L", title="HMC", yscale=Makie.pseudolog10, aspect=1, titlesize=12)
    heatmap!(ax, df_HMC.epsilon, df_HMC.L, df_HMC[:,"marginal_$(stat)_mean_maximum"]; colorrange=crange, colormap=:magma)
    scatter!(ax, best_HMC.epsilon, best_HMC.L; color=:white, marker=:diamond)

    for j in 1:3
        df2 = df_FFF[df_FFF.lambda .== lambdas[j],:]
        ax = Axis(fig[1,1+j]; xlabel="ϵ", title="FFF λ=$(round(lambdas[j],digits=3))", yscale=Makie.pseudolog10, aspect=1, titlesize=12)
        heatmap!(ax, df2.epsilon, df2.L, df2[:,"marginal_$(stat)_mean_maximum"]; colorrange=crange, colormap=:magma)
        j == 2 && scatter!(ax, best_FFF.epsilon, best_FFF.L; color=:white, marker=:diamond)
    end

    Colorbar(fig[1,5]; limits=crange, colormap=:magma, label="max marginal $stat dist.")
    colgap!(fig.layout, 1, 50)
    fig
end

display(best_HMC)
display(best_FFF)
save("robustness-$(split(filename_FFF, '-')[5]).png", fig, scalefactor=3)