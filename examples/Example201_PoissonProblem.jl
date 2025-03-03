#=

# 201 : Poisson-Problem
([source code](@__SOURCE_URL__))

This example computes the solution ``u`` of the two-dimensional Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with right-hand side ``f(x,y) \equiv xy`` and homogeneous Dirichlet boundary conditions
on the unit square domain ``\Omega`` on a given grid.

The computed solution for the default parameters looks like this:

![](example201.png)

=#

module Example201_PoissonProblem

using ExtendableFEM
using ExtendableGrids
using GridVisualize
using Metis
using Test #hide

## define variables
u = Unknown("u"; name = "potential")

## define data functions
function f!(fval, qpinfo)
    fval[1] = qpinfo.x[1] * qpinfo.x[2]
    return nothing
end

function main(; μ = 1.0, nrefs = 4, order = 2, Plotter = nothing, parallel = false, npart = parallel ? 8 : 1, kwargs...)
    ## problem description
    PD = ProblemDescription()
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; parallel = parallel, factor = μ, kwargs...))
    assign_operator!(PD, LinearOperator(f!, [id(u)]; parallel = parallel, kwargs...))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4))

    ## discretize
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefs)
    if npart > 1
        xgrid = partition(xgrid, PlainMetisPartitioning(npart = npart))
    end
    FES = FESpace{H1Pk{1, 2, order}}(xgrid)

    ## solve
    sol = solve(PD, FES; kwargs...)

    ## plot
    plt = plot([id(u), grad(u)], sol; Plotter = Plotter)

    return sol, plt
end

generateplots = ExtendableFEM.default_generateplots(Example201_PoissonProblem, "example201.png") #hide
function runtests() #hide
    sol, plt = main(; μ = 1.0, nrefs = 2, parallel = false, order = 2, npart = 2) #hide
    @test sum(sol.entries) ≈ 1.1140313632246377 #hide
    sol_parallel, plt = main(; μ = 1.0, nrefs = 2, order = 2, parallel = true, npart = 2) #hide
    @assert sum((sol_parallel.entries .- sol.entries) .^ 2) ≈ 0.0 #hide
    return nothing #hide
end #hide
end # module
