[![arXiv](https://img.shields.io/badge/article-arXiv%3A2504.12190-B31B1B)](https://arxiv.org/abs/2504.12190)

# Rebalancing Markov jump processes for non-reversible continuous-time sampling

This package implements the samplers for use in the Julia ecosystem.
Under `experiments/` we provide the code used to produce the results in our paper.

## Demo


Let's take the evil banana target as a Turing.jl model.
```julia
using Turing

@model function banana(μ, a, b)
    X ~ Normal(μ, 1/√(2a))
    Y ~ Normal(X^2, 1/√(2b))
    return (X, Y)
end
model = banana(1., 1/10, 100/10)
```
<img src="https://github.com/user-attachments/assets/20111807-96f3-4ec0-a093-0a37612750ff" width="400">


Then it's just plug and play:
```julia 
using FFFSampler

ϵ, L, λ  = 0.035, 20, 0.04 # from paper
fff = FFF(ϵ, L, λ)

chain_fff = @time sample(model, externalsampler(Discretize(fff,0.1)), 200_000)
chain_nuts = @time sample(model, NUTS(), 200_000)
```
<img src="https://github.com/user-attachments/assets/b894085c-faef-4803-828c-58a6462fa742" width="400">

So what does FFF do? 

* It moves along level curves of the Hamiltonians like HMC, but it visits more than one point on each Hamiltonian, keeping going in the same direction
* Internally it uses a continuous notion of time and spends a random time in each point, removing the need for rejections (here we call a wrapper `Discretize` to hide this from Turing; see https://github.com/TuringLang/MCMCChains.jl/issues/253)

Try it, we are curious about your experience!

Directly accessing weighted trajectories is also easy with some glue code to reinterpret the model:
```julia
# bypass MCMCChains
using LogDensityProblemsAD, LogDensityProblems, ForwardDiff
problem = LogDensityProblemsAD.ADgradient(Val(:ForwardDiff), DynamicPPL.LogDensityFunction(model));
trace_fff = AbstractMCMC.sample(problem, fff, 200_000)
```

## Citation
```
@misc{jansson_rebalancing_2025,
    title={Rebalancing Markov jump processes for non-reversible continuous-time sampling},
    author={Erik Jansson and Moritz Schauer and Ruben Seyer and Akash Sharma},
    year={2025},
    eprint={2504.12190},
    archivePrefix={arXiv}
}
```

