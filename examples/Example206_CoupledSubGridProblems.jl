#=

# 206 : CoupledSubGridProblems
([source code](@__SOURCE_URL__))

This example demonstrates how to solve a coupled problem where two
variables only live on a sub-domain and are coupled through an interface condition.
Consider the unit square domain cut in half through on of its diagonals.
On each subdomain a solutiong ``u_j`` of the two-dimensional Poisson problem
```math
\begin{aligned}
-\Delta u & = f \quad \text{in } \Omega
\end{aligned}
```
with inhomogeneous boundary conditions on the former boundaries of the full square is searched.
Along the common boundary between the two subdomains a new interface region is assigned (appended to BFaceNodes)
and an interface condition is assembled that couples the two solutions ``u_1`` and ``u_2``
to each other.
In this toy example, this interface conditions penalizes the jump between the two solutions on each side
of the diagonal. Oberserve, that if the penalization factor ``\tau`` is large, the two solutions are
almost equal along the interface.

The computed solution(s) looks like this:

![](example206.svg)

Each column of the plot shows the solution, the subgrid it lives on. The last row shows the full grid.

=#

module Example206_CoupledSubGridProblems

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using ExtendableSparse
using GridVisualize
using UnicodePlots
using Test #


function boundary_conditions!(result, qpinfo)
    result[1] = 1 - qpinfo.x[1] - qpinfo.x[2] # used for both subsolutions
end

function interface_condition!(result, u, qpinfo)
    result[1] = -(u[2] - u[1])
    result[2] = -result[1]
end


function main(; μ = [1.0,1.0], f = [10,-10], τ = 1, nref = 4, order = 2, Plotter = nothing, kwargs...)

	## Finite element type
	FEType = H1Pk{1, 2, order}

	## generate mesh
	xgrid = grid_unitsquare(Triangle2D)

    ## define regions
    xgrid[CellRegions] = Int32[1,2,2,1]

    ## add an interface between region 1 and 2
    ## (one can use the BFace storages for that)
    xgrid[BFaceNodes] = Int32[xgrid[BFaceNodes] [2 5; 5 4]]
    append!(xgrid[BFaceRegions], [5,5])
    xgrid[BFaceGeometries] = VectorOfConstants{ElementGeometries, Int}(Edge1D, 6)

    ## refine
    xgrid = uniform_refine(xgrid, nref)

    ## define an FESpace just on region 1 and one just on region 2
    FES1 = FESpace{FEType}(xgrid; regions = [1])
    FES2 = FESpace{FEType}(xgrid; regions = [2])

    ## define variables
    u1 = Unknown("u1"; name = "potential in region 1")
    u2 = Unknown("u2"; name = "potential in region 2")

    ## problem description
	PD = ProblemDescription()
	assign_unknown!(PD, u1)
	assign_unknown!(PD, u2)
	assign_operator!(PD, BilinearOperator([grad(u1)]; regions = [1], factor = μ[1], kwargs...))
	assign_operator!(PD, BilinearOperator([grad(u2)]; regions = [2], factor = μ[2], kwargs...))
    assign_operator!(PD, LinearOperator([id(u1)]; regions = [1], factor = f[1]))
    assign_operator!(PD, LinearOperator([id(u2)]; regions = [2], factor = f[2]))
	assign_operator!(PD, BilinearOperator(interface_condition!, [id(u1), id(u2)]; regions = [5], factor = τ, entities = ON_FACES, kwargs...))
	assign_operator!(PD, InterpolateBoundaryData(u1, boundary_conditions!; regions = 1:4))
	assign_operator!(PD, InterpolateBoundaryData(u2, boundary_conditions!; regions = 1:4))

    sol = solve(PD, [FES1, FES2])

    plt = plot([id(u1), id(u2), dofgrid(u1), dofgrid(u2), grid(u1)], sol; Plotter = Plotter)

	return sol, plt
end

generateplots = default_generateplots(Example206_CoupledSubGridProblems, "example206.svg") #hide


function jump_l2norm!(result, u, qpinfo)
    result[1] = (u[1] - u[2])^2
end
function runtests() #hide
    ## test if jump at interface vanishes for large penalty
	sol, plt = main(; τ = 1e9, nrefs = 2, order = 2) #hide
    jump_integrator = ItemIntegrator(jump_l2norm!, [id(1), id(2)]; entities = ON_BFACES, regions = [5], resultdim = 1, quadorder = 4)
    jump_error = sqrt(sum(evaluate(jump_integrator, sol)))
    @info "||[u_1 - u_2]|| = $(jump_error)"
	@test jump_error < 1e-8 #hide
end #hide
end #module