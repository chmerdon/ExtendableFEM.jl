#=

# 202 : Poisson-Problem (Mixed)
([source code](@__SOURCE_URL__))

This example computes the solution ``u`` and its stress ``\mathbf{\sigma} := - \mu \nabla u``
of the two-dimensional Poisson problem in the mixed form
```math
\begin{aligned}
\mathbf{\sigma} + \mu \nabla u &= 0\\
\mathrm{div} \mathbf{\sigma} & = f \quad \text{in } \Omega
\end{aligned}
```
with right-hand side ``f(x,y) \equiv xy`` and homogeneous Dirichlet boundary conditions
on the unit square domain ``\Omega`` on a given grid.

The computed solution looks like this:

![](example202.png)
=#

module Example202_MixedPoissonProblem

using ExtendableFEM
using ExtendableGrids
using Test #hide

## define unknowns
σ = Unknown("σ"; name = "pseudostress")
u = Unknown("u"; name = "potential")

## bilinearform kernel for mixed Poisson problem
function blf!(result, u_ops, qpinfo)
    σ, divσ, u = view(u_ops, 1:2), view(u_ops, 3), view(u_ops, 4)
    μ = qpinfo.params[1]
    result[1] = σ[1] / μ
    result[2] = σ[2] / μ
    result[3] = -u[1]
    result[4] = divσ[1]
    return nothing
end
## right-hand side data
function f!(fval, qpinfo)
    fval[1] = qpinfo.x[1] * qpinfo.x[2]
    return nothing
end
## boundary data
function boundarydata!(result, qpinfo)
    result[1] = 0
    return nothing
end

function main(; nrefs = 5, μ = 0.25, order = 0, Plotter = nothing, kwargs...)

    ## problem description
    PD = ProblemDescription()
    assign_unknown!(PD, u)
    assign_unknown!(PD, σ)
    assign_operator!(PD, BilinearOperator(blf!, [id(σ), div(σ), id(u)]; params = [μ], kwargs...))
    assign_operator!(PD, LinearOperator(boundarydata!, [normalflux(σ)]; entities = ON_BFACES, regions = 1:4, kwargs...))
    assign_operator!(PD, LinearOperator(f!, [id(u)]; kwargs...))
    assign_operator!(PD, FixDofs(u; dofs = [1], vals = [0]))

    ## discretize
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefs)
    FES = Dict(
        u => FESpace{order == 0 ? L2P0{1} : H1Pk{1, 2, order}}(xgrid; broken = true),
        σ => FESpace{HDIVRTk{2, order}}(xgrid)
    )

    ## solve
    sol = ExtendableFEM.solve(PD, FES; kwargs...)

    ## plot
    plt = plot([id(u), id(σ)], sol; Plotter = Plotter)

    return sol, plt
end

generateplots = ExtendableFEM.default_generateplots(Example202_MixedPoissonProblem, "example202.png") #hide
function runtests() #hide
    sol, plt = main(; μ = 0.25, order = 0, nrefs = 2) #hide
    @test maximum(view(sol[1])) ≈ 0.08463539106946043 #hide
    return nothing #hide
end #hide
end # module
