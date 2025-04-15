using DataFrames, DelimitedFiles
using Statistics, StatsBase
using GLMakie
using REPL.TerminalMenus: request, RadioMenu

# SETTINGS
begin
    files = filter(s -> endswith(s, "csv") && contains(s, "HMC"), readdir("."))
    filename = files[request(RadioMenu(files))]

    data, header = readdlm(filename, ',', header=true)
    df = DataFrame(data, Symbol.(vec(header)))
    df = groupby(df, [:epsilon, :L, :d])
    df = combine(df, valuecols(df) .=> mean)
    df = groupby(df, [:epsilon, :L])
    df = combine(df, valuecols(df) .=> maximum)
   
    display(eachrow(df)[findmin(isnan(x) ? Inf : x for x in df.marginal_KS_mean_maximum)[2]])
end
begin
    fig = Figure(size=(800, 600))
    crange = extrema(filter(!isnan, df.marginal_KS_mean_maximum))
    ax = Axis(fig[1,1]; xlabel="ϵ", ylabel="L", yscale=log10)
    heatmap!(ax, df.epsilon, df.L, df.marginal_KS_mean_maximum; colorrange=crange)
    Colorbar(fig[1,2]; limits=crange, label="max marginal KS")
    
    namesegs = split(filename,'-')
    model = length(namesegs) ≥ 5 ? namesegs[5] : ""
    Label(fig[0,:], "$model HMC", fontsize=18, font=:bold)
    fig
end