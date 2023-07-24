#= 

# 201 : Poisson-Problem
([source code](SOURCE_URL))

This example computes the solution ``u`` of the two-dimensional Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with right-hand side ``f(x,y) \equiv xy`` and homogeneous Dirichlet boundary conditions
on the unit square domain ``\Omega`` on a given grid.

=#

module Example201_PoissonProblem

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using GridVisualize

function f!(fval, qpinfo)
    fval[1] = qpinfo.x[1] * qpinfo.x[2]
end

function main(; μ = 1.0, nrefs = 4, order = 2, Plotter = nothing, kwargs...)

    ## problem description
    PD = ProblemDescription()
    u = Unknown("u"; name = "u")
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, kwargs...))
    assign_operator!(PD, LinearOperator(f!, [id(u)]; kwargs...))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4))

    ## discretize
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefs)
    FES = FESpace{H1Pk{1, 2, order}}(xgrid)

    ## solve
    sol = solve(PD, [FES]; kwargs...)
       
    ## plot
    p=GridVisualizer(; Plotter = Plotter, layout = (1,1), clear = true, resolution = (600,600))
    scalarplot!(p[1,1], xgrid, nodevalues_view(sol[u])[1], levels = 7, title = "u_h")
end

end # module