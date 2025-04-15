using DataFrames, DelimitedFiles
using Statistics, StatsBase
using GLMakie
using REPL.TerminalMenus: request, RadioMenu

# SETTINGS
ρ = 0.0

begin
    files = filter(s -> endswith(s, "csv") && contains(s, "FFF"), readdir("."))
    filename = files[request(RadioMenu(files))]

    data, header = readdlm(filename, ',', header=true)
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
   
    display(eachrow(df)[findmin(isnan(x) ? Inf : x for x in df.marginal_KS_mean_maximum)[2]])
end

#=
# Plot over L
begin 
    fig = Figure(size=(800, 600))

    df2 = groupby(df, [:L])
    indices = CartesianIndices((3,3))
    @assert length(df2) ≤ length(indices)
    crange = extrema(df.marginal_KS_mean_maximum)

    for (l, ((L,), rows)) in enumerate(pairs(df2))
        j, i = Tuple(indices[l])
        ax = Axis(fig[i,j]; xlabel="ϵ", ylabel="λ", title="L=$L", yscale=log10)
        heatmap!(ax, rows.epsilon, rows.lambda, rows.marginal_KS_mean_maximum; colorrange=crange)
    end
    Colorbar(fig[:, last(indices)[1]+1]; limits=crange, label="max marginal KS")
    Label(fig[0,:], "$(split(filename,'-')[5]) FFF ρ=$ρ", fontsize=18, font=:bold)

    show(fig)
end
=#

# Plot over λ
begin
    fig = Figure(size=(800, 600))

    df2 = groupby(df, [:lambda])
    indices = CartesianIndices((7,3))
    @assert length(df2) ≤ length(indices)
    crange = extrema(filter(!isnan, df.marginal_KS_mean_maximum))

    for (l, ((λ,), rows)) in enumerate(pairs(df2))
        j, i = Tuple(indices[l])
        ax = Axis(fig[i,j]; xlabel="ϵ", ylabel="L", title="λ=$(round(λ,digits=3))", yscale=log10)
        heatmap!(ax, rows.epsilon, rows.L, rows.marginal_KS_mean_maximum; colorrange=crange)
    end
    Colorbar(fig[:, last(indices)[1]+1]; limits=crange, label="max marginal KS")
    Label(fig[0,:], "$(split(filename,'-')[5]) FFF ρ=$ρ", fontsize=18, font=:bold)

    fig
end