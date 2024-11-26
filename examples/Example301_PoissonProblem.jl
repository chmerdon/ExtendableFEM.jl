#=

# 301 : Poisson-Problem
([source code](@__SOURCE_URL__))

This example computes the solution ``u`` of the two-dimensional Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with right-hand side ``f(x,y) \equiv xy`` and homogeneous Dirichlet boundary conditions
on the unit cube domain ``\Omega`` on a given grid. The computed solution for the default
parameters looks like this:

![](example301.png)

=#

module Example301_PoissonProblem

using ExtendableFEM
using ExtendableGrids
using Test #hide

function f!(fval, qpinfo)
    return fval[1] = qpinfo.x[1] * qpinfo.x[2] * qpinfo.x[3]
end

function main(; μ = 1.0, nrefs = 3, Plotter = nothing, kwargs...)

    ## problem description
    PD = ProblemDescription()
    u = Unknown("u"; name = "potential")
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, kwargs...))
    assign_operator!(PD, LinearOperator(f!, [id(u)]; kwargs...))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4))

    ## discretize
    xgrid = uniform_refine(grid_unitcube(Tetrahedron3D), nrefs)
    FES = FESpace{H1P2{1, 3}}(xgrid)

    ## solve
    sol = solve(PD, FES; kwargs...)

    ## plot
    plt = plot([id(u)], sol; Plotter = Plotter)

    return sol, plt
end

generateplots = ExtendableFEM.default_generateplots(Example301_PoissonProblem, "example301.png") #hide
function runtests() #hide
    sol, plt = main() #hide
    return @test sum(sol.entries) ≈ 21.874305144549524 #hide
end #hide
end # module
