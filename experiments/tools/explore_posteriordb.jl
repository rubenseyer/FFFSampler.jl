using PosteriorDB, StanLogDensityProblems
using GLMakie, MCMCChains, PairPlots

pdb = PosteriorDB.database()
PosteriorDB.posterior_names(pdb)

candidates = filter(x -> !isnothing(PosteriorDB.reference_posterior(PosteriorDB.posterior(pdb, x))), PosteriorDB.posterior_names(pdb))

function pdb_to_chain(pdb, pname; compile_model=false)
    post = PosteriorDB.posterior(pdb, pname)
    ref = PosteriorDB.reference_posterior(post)
    gold = PosteriorDB.load(ref)
    model = compile_model ? StanProblem(post, ".", force=true, make_args=Threads.nthreads() > 1 ? ["STAN_THREADS=true"] : String[]).model : nothing

    parameter_names = collect(keys(gold[begin]))
    N = length(gold[begin][parameter_names[begin]])
    P = length(parameter_names)
    M = length(gold)
    values = Array{Float64,3}(undef, N, P, M)
    for i in 1:M
        for j in 1:P
            values[:, j, i] .= gold[i][parameter_names[j]]
        end
        if model !== nothing
            for r in 1:N
                values[r, :, i] .= StanLogDensityProblems.BridgeStan.param_unconstrain(model, values[r, :, i])
            end
        end
    end
    #stds = sqrt.(var(values; dims=(1,3)))
    #values ./= stds
    return Chains(values, Symbol.(parameter_names))
end

# bball_drive_event_1-hmm_drive_1
# garch-garch11
# hudson_lynx_hare-lotka_volterra #??
# kilpisjarvi_mod-kilpisjarvi
# one_comp_mm_elim_abs-one_comp_mm_elim_abs
# gp_pois_regr-gp_pois_regr  # might have some annyoing ones
# mcycle_gp-accel_gp  # big and annoying
begin
    name = "one_comp_mm_elim_abs-one_comp_mm_elim_abs"
    fig = pairplot(pdb_to_chain(pdb, name; compile_model=true))
end
#save("pp.png", fig)

# one_comp_mm_elim_abs
ocm_chn = pdb_to_chain(pdb, "one_comp_mm_elim_abs-one_comp_mm_elim_abs"; compile_model=true)
begin
    fig = pairplot(ocm_chn[:,1:3,:], labels = Dict(:k_a => L"k_a", :K_m => L"K_m", :V_m => L"V_m"))
    grid = fig[1,1]
    delete!.(contents(grid.layout)[[1,2,3,6]])
    for ax in contents(grid.layout)
        ax.xlabelsize = 18
        ax.ylabelsize = 18
    end
    tight_yticklabel_spacing!(contents(grid.layout)[1])
    GLMakie.trim!(grid.layout)
    resize!(fig.scene, (400,240))
    save("pkpd_pairs.png", fig, scalefactor=3)
end