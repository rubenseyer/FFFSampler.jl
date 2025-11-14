module CodaESS

using LinearAlgebra
using Statistics
using StatsBase

function spectrum0_ar(X; pmax=nothing, atol=1e-12)
    X = Matrix(X)
    n, p = size(X)
    pmax === nothing && (pmax = max(0, min(floor(Int,10*log10(n)), n-1)))

    spec  = fill(0.0, p)
    order = fill(0,   p)

    Z = hcat(ones(Int, n), 1:n)     # [1  z]
    P = pinv(Z)                     # 2×n

    for j in 1:p
        # OLS fit and residuals
        β = P * X[:, j]                  # 2×1
        r = X[:, j] .- (Z * β)           # detrended residuals

        # if all residuals practically identical → variance ≈ 0
        if std(r) ≤ atol
            spec[j]  = 0.0
            order[j] = 0
            continue
        end

        # Select AR order by AIC using Yule–Walker / Levinson–Durbin
        a_best, σ2_best, p_best = select_ar_yw(r; pmax=pmax)

        # spectrum at frequency 0: σ² / (1 - ∑ a_k)^2
        s = 1.0 - sum(a_best)
        spec[j]  = σ2_best / (s*s)
        order[j] = p_best
    end

    return spec, order
end

function effective_size(X::AbstractMatrix; pmax=nothing)
    n, p = size(X)
    spec, _ = spectrum0_ar(X; pmax)
    ess = similar(spec)
    for j in 1:p
        vj = var(view(X, :, j))                 # sample variance (corrected=true)
        ess[j] = spec[j] == 0.0 ? 0.0 : n * vj / spec[j]
    end
    return ess
end

effective_size(X::AbstractVector; pmax=nothing) = effective_size(reshape(X, :, 1); pmax)[1]

# ---------------------- internals ----------------------

# Levinson–Durbin to solve Yule–Walker and return AR coeffs and innovation variance σ²
function levinson_durbin(γ::AbstractVector{<:Real}, p::Int)
    # γ[1] = γ0, γ[2] = γ1, ...
    a = zeros(Float64, p)   # a_1..a_p
    E = γ[1]                # prediction error variance at order 0
    if p == 0
        return a, E
    end
    for m in 1:p
        # reflection coefficient κ_m
        # κ_m = (γ_m - sum_{i=1}^{m-1} a_i^{(m-1)} γ_{m-i}) / E_{m-1}
        num = γ[m+1]
        @inbounds for i in 1:m-1
            num -= a[i] * γ[m - i + 1]
        end
        κ = num / E

        # update coefficients
        a_new = copy(a)
        a_new[m] = κ
        @inbounds for i in 1:m-1
            a_new[i] = a[i] - κ * a[m - i]
        end
        a = a_new

        # update prediction error
        E *= (1 - κ^2)
        E = max(E, eps())   # guard against tiny negatives from roundoff
    end
    return a, E
end

# Choose AR order 0..pmax by AIC: AIC(p) = n * log(σ²_p) + 2p
function select_ar_yw(x::AbstractVector{<:Real}; pmax::Int)
    n = length(x)
    pmax = max(0, min(pmax, n-1))
    best_p   = 0
    best_a   = Float64[]
    best_σ2  = var(x) # p=0 → innovation variance equals variance
    best_aic = n * log(best_σ2 + eps()) + 0.0

    if pmax == 0
        return best_a, best_σ2, best_p
    end

    γ = autocov(x, 0:pmax)
    for p in 1:pmax
        a_p, σ2_p = levinson_durbin(γ, p)
        aic_p = n * log(σ2_p + eps()) + 2p
        if aic_p < best_aic
            best_aic = aic_p
            best_p   = p
            best_a   = a_p
            best_σ2  = σ2_p
        end
    end
    return best_a, best_σ2, best_p
end

export effective_size

end # module
