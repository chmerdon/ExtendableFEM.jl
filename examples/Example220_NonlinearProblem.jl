module Example220_NonlinearProblem

using ExtendableFEM
using ExtendableFEMBase
using GridVisualize
using ExtendableGrids
using ExtendableSparse
using LinearSolve
using Krylov
using Symbolics

const f = x -> x[1]+x[2]
const μ = 0.25

## kernel of nonlinear form
function kernel!(result, u_ops, qpinfo)
    u = view(u_ops, 1)
    ∇u = view(u_ops,2:3)
    result[1] = - f(qpinfo.x)
    result[2] = μ*∇u[1] + 0.5 * u[1]^4
    result[3] = μ*∇u[2] + 2 * u[1]^2
    return nothing
end

function boundarydata!(result, qpinfo)
    x = qpinfo.x
    result[1] = x[1]
end

function main(; nrefs = 5, Plotter = nothing, kwargs...)

    ## problem description
    PD = ProblemDescription()
    u = Unknown("u"; name = "potential")
    assign_unknown!(PD, u)
    assign_operator!(PD, NonlinearOperator(kernel!, [id(u),grad(u)]; kwargs...))
    assign_operator!(PD, InterpolateBoundaryData(u, boundarydata!; regions = 1:3, kwargs...))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = [4], kwargs...))

    ## grid
    xgrid = uniform_refine(grid_unitsquare_mixedgeometries(), nrefs)

    ## solve
    FES = FESpace{H1Q2{1,2}}(xgrid)
    sol = ExtendableFEM.solve(PD, [FES]; kwargs...)

    @info sol
    scalarplot(split_grid_into(xgrid, Triangle2D), nodevalues(sol[1])[:]; Plotter = Plotter)
end

end # module