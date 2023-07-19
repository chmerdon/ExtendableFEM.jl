module Example250_NavierStokesProblem

using ExtendableFEM
using ExtendableFEMLowLevel
using GridVisualize
using ExtendableGrids

function kernel_nonlinear!(result, u_ops, qpinfo)
    u, ∇u, p = view(u_ops, 1:2), view(u_ops,3:6), view(u_ops, 7)
    μ = qpinfo.params[1]
    result[1] = dot(u, view(∇u,1:2))
    result[2] = dot(u, view(∇u,3:4))
    result[3] = μ*∇u[1] - p[1]
    result[4] = μ*∇u[2]
    result[5] = μ*∇u[3]
    result[6] = μ*∇u[4] - p[1]
    result[7] = -(∇u[1] + ∇u[4])
    return nothing
end

function boundarydata!(result, qpinfo)
    result[1] = 1
    result[2] = 0
end

function initialgrid_cone()
    xgrid=ExtendableGrid{Float64,Int32}()
    xgrid[Coordinates] = Array{Float64,2}([-1 0; 0 -2; 1 0]')
    xgrid[CellNodes]=Array{Int32,2}([1 2 3]')
    xgrid[CellGeometries]=VectorOfConstants{ElementGeometries,Int}(Triangle2D,1)
    xgrid[CellRegions]=ones(Int32,1)
    xgrid[BFaceRegions]=Array{Int32,1}([1,2,3])
    xgrid[BFaceNodes]=Array{Int32,2}([1 2; 2 3; 3 1]')
    xgrid[BFaceGeometries]=VectorOfConstants{ElementGeometries,Int}(Edge1D,3)
    xgrid[CoordinateSystem]=Cartesian2D
    return xgrid
end

function main(; μ_final = 0.0005, nrefs = 6, Plotter = nothing, kwargs...)

    ## prepare parameter field
	extra_params = Array{Float64,1}([max(μ_final, 0.01)])

    ## problem description
    PD = ProblemDescription()
    u = Unknown("u"; name = "velocity")
    p = Unknown("p"; name = "pressure")
    assign_unknown!(PD, u)
    assign_unknown!(PD, p)
    assign_operator!(PD, ExtendableFEM.NonlinearOperator(kernel_nonlinear!, [apply(u, ReconstructionIdentity{HDIVRT0{2}}),grad(u),id(p)]; params = extra_params, kwargs...))#; jacobian = kernel_jacobian!)) 
    assign_operator!(PD, InterpolateBoundaryData(u, boundarydata!; regions = 3))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = [1,2]))

    ## grid
    xgrid = uniform_refine(initialgrid_cone(), nrefs)

    ## prepare FESpace
    FES = [FESpace{H1BR{2}}(xgrid), FESpace{L2P0{1}}(xgrid)]

    ## prepare plots
    p=GridVisualizer(; Plotter = Plotter, layout = (1,1), clear = true, resolution = (1200,1200))
    

    ## solve by μ embedding
	step = 0
    sol = nothing
    SC = nothing
	while (true)
		step += 1
		@info "Step $step : solving for μ=$(extra_params[1])"
        sol, SC = ExtendableFEM.solve!(PD, FES, SC; return_config = true, target_residual = 1e-10, maxiterations = 20, kwargs...)
        scalarplot!(p[1,1], xgrid, nodevalues(sol[1]; abs = true)[1,:]; title = "μ = $(extra_params[1])", Plotter = Plotter)
		vectorplot!(p[1,1], xgrid, evaluate(PointEvaluator(sol[1], Identity)), spacing = 0.05, clear = false)
        
        if extra_params[1] <= μ_final
			break
        else
            extra_params[1] = max(μ_final, extra_params[1]/2)
		end
	end

    @info sol
    scalarplot!(p[1,1], xgrid, nodevalues(sol[1]; abs = true)[1,:]; title = "μ = $(extra_params[1])", Plotter = Plotter)
    vectorplot!(p[1,1], xgrid, evaluate(PointEvaluator(sol[1], Identity)), spacing = 0.05, clear = false)
    
    writeVTK("Example250_output.vtu", xgrid; velocity = nodevalues(sol[1]), pressure = nodevalues(sol[2]), cellregions = xgrid[CellRegions])
end

end # module