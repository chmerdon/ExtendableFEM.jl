module Example201_PoissonProblem

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using GridVisualize

function main(; μ = 1.0, nrefs = 4, order = 3, Plotter = nothing, kwargs...)

    ## problem description
    PD = ProblemDescription()
    u = Unknown("u"; name = "u")
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, kwargs...))
    assign_operator!(PD, LinearOperator([id(u)]; kwargs...))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4))

    ## discretize
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefs)
    FES = FESpace{H1Pk{1, 2, order}}(xgrid)

    ## solve
    sol = ExtendableFEM.solve!(PD, [FES]; kwargs...)
       
    ## plot
    p=GridVisualizer(; Plotter = Plotter, layout = (1,1), clear = true, resolution = (600,600))
    scalarplot!(p[1,1], xgrid, nodevalues_view(sol[u])[1], levels = 7, title = "u_h")
end

end # module