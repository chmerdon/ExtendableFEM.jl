module Example260_AxisymmetricNavierStokesProblem

using ExtendableFEM
using ExtendableFEMBase
using GridVisualize
using ExtendableGrids
using ExtendableSparse
using SimplexGridFactory
using Triangulate


function kernel_convection!(result, input, qpinfo)
    u, ∇u = view(input, 1:2), view(input, 3:6)
    r = qpinfo.x[1]
    result[1] = r*(∇u[1]*u[1] + ∇u[2]*u[2])
    result[2] = r*(∇u[3]*u[1] + ∇u[4]*u[2])
    return nothing
end

function kernel_stokes_axisymmetric!(result, u_ops, qpinfo)
    u, ∇u, p = view(u_ops,1:2), view(u_ops,3:6), view(u_ops, 7)
    r = qpinfo.x[1]
    μ = qpinfo.params[1]
    ## add Laplacian
    result[1] = μ/r * u[1] - p[1]
    result[2] = 0
    result[3] = μ*r * ∇u[1] - r*p[1]
    result[4] = μ*r * ∇u[2]
    result[5] = μ*r * ∇u[3]
    result[6] = μ*r * ∇u[4] - r*p[1]
    result[7] = -(r*(∇u[1]+∇u[4]) + u[1])
    return nothing
end

function u!(result, qpinfo)
    x = qpinfo.x
    result[1] = x[1]
    result[2] = -2*x[2]
end

function kernel_l2div(result, u_ops, qpinfo)
    u, divu = view(u_ops,1:2), view(u_ops,3)
    result[1] = (qpinfo.x[1]*divu[1] + u[1])^2
end

function kernel_normalflux(result, un, qpinfo)
    result[1] = qpinfo.x[1]*un[1]
end

function kernel_flux(result, un, qpinfo)
    result[1] = qpinfo.x[1]*un[1]
    result[2] = qpinfo.x[1]*un[2]
end


function main(; μ = 0.1, nrefs = 4, nonlinear = false, uniform = false, Plotter = nothing, kwargs...)

    ## problem description
    PD = ProblemDescription()
    u = Unknown("u"; name = "velocity")
    p = Unknown("p"; name = "pressure")
    assign_unknown!(PD, u)
    assign_unknown!(PD, p)
    assign_operator!(PD, BilinearOperator(kernel_stokes_axisymmetric!, [id(u),grad(u),id(p)]; bonus_quadorder = 1, params = [μ], kwargs...))#; jacobian = kernel_jacobian!)) 
    if nonlinear
        assign_operator!(PD, NonlinearOperator(kernel_convection!, [id(u)], [id(u),grad(u)]; bonus_quadorder = 2, kwargs...))#; jacobian = kernel_jacobian!)) 
    end
    assign_operator!(PD, InterpolateBoundaryData(u, u!; regions = 1:2))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = [4], mask = (1,0,1)))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = [1], mask = (0,1,1)))

    ## grid
    if uniform
        xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefs)
    else
        xgrid = simplexgrid(Triangulate;
        points=[0 0 ; 1 0 ; 1 1 ; 0 1]',
        bfaces=[1 2 ; 2 3 ; 3 4 ; 4 1 ]',
        bfaceregions=[1, 2, 3, 4],
        regionpoints=[0.5 0.5;]',
        regionnumbers=[1],
        regionvolumes=[4.0^(-nrefs-1)])
    end

    ## solve
    #FES = [FESpace{H1P2{2,2}}(xgrid), FESpace{H1P1{1}}(xgrid)]
    FES = [FESpace{H1BR{2}}(xgrid), FESpace{L2P0{1}}(xgrid)]
    sol = ExtendableFEM.solve!(PD, FES; kwargs...)

    ## compute divergence in cylindrical coordinates by volume integrals
    DivIntegrator = ItemIntegrator(kernel_l2div, [id(u), div(u)]; quadorder = 8, resultdim = 1)
    @info "||div(u)|| = $(sqrt(sum(evaluate(DivIntegrator, sol))))"

    ## compute normafluxes
    FluxIntegrator = ItemIntegrator(kernel_normalflux, [normalflux(u)]; entities = ON_FACES, resultdim = 1)
    flux4faces = evaluate(FluxIntegrator, sol)
    div4cells = zeros(Float64, num_cells(xgrid))
    cellfacesigns = xgrid[CellFaceSigns]
    facecells = xgrid[FaceCells]
    facenormals = xgrid[FaceNormals]
    cellfaces = xgrid[CellFaces]
    for cell in 1:num_cells(xgrid)
        for j = 1 : 3
            div4cells[cell] += cellfacesigns[j,cell] * flux4faces[cellfaces[j,cell]]
        end
    end
    @info extrema(div4cells)

    ## again by segment integrator
    SI = SegmentIntegrator(Edge1D, kernel_flux, [id(u)]; kwargs...)
    initialize!(SI, sol)
    fill!(div4cells, 0)
    flux = zeros(Float64, 2)
    for cell in 1:num_cells(xgrid)
        face = cellfaces[1,cell]
        SI.integrator(flux, [xgrid[Coordinates][:,j] for j in xgrid[FaceNodes][:,face]], [[0,0.0], [1.0,0]], cell)
        div4cells[cell] += cellfacesigns[1,cell] * dot(flux, facenormals[:,face])
        face = cellfaces[2,cell]
        SI.integrator(flux, [xgrid[Coordinates][:,j] for j in xgrid[FaceNodes][:,face]], [[1,0.0], [0,1.0]], cell)
        div4cells[cell] += cellfacesigns[2,cell] * dot(flux, facenormals[:,face])
        face = cellfaces[3,cell]
        SI.integrator(flux, [xgrid[Coordinates][:,j] for j in xgrid[FaceNodes][:,face]], [[0,1.0], [0,0.0]], cell)
        div4cells[cell] += cellfacesigns[3,cell] * dot(flux, facenormals[:,face])
    end
    @info extrema(div4cells)

    ## again by matrix mode of segment integrator
    initialize!(SI, sol; matrix_mode = true)
    fill!(div4cells, 0)
    num_faces = size(facenormals,2)
    A = ExtendableSparseMatrix{Float64,Int}(2*num_faces, sol[1].FES.ndofs)
    cell::Int = 0
    for face = 1 : num_faces
        cell = facecells[1,face]
        if cellfaces[1,cell] == face
            SI.integrator(A, [xgrid[Coordinates][:,j] for j in xgrid[FaceNodes][:,face]], [[0,0.0], [1.0,0]], cell, face)
        elseif cellfaces[2,cell] == face
            SI.integrator(A, [xgrid[Coordinates][:,j] for j in xgrid[FaceNodes][:,face]], [[1,0.0], [0,1.0]], cell, face)
        elseif cellfaces[3,cell] == face
            SI.integrator(A, [xgrid[Coordinates][:,j] for j in xgrid[FaceNodes][:,face]], [[0,1.0], [0,0.0]], cell, face)
        end
    end
    fluxes = A * view(sol[1])
    for cell in 1:num_cells(xgrid)
        for j = 1 : 3
            face = cellfaces[j,cell]
            div4cells[cell] += cellfacesigns[j,cell] * dot(fluxes[(face-1)*2+1:face*2], facenormals[:,face])
        end
    end

    @info extrema(div4cells)

    pl=GridVisualizer(; Plotter = Plotter, layout = (2,2), clear = true, resolution = (1200,1200))
    scalarplot!(pl[1,1], xgrid, nodevalues_view(sol[u])[1]; Plotter = Plotter)
    scalarplot!(pl[1,2], xgrid, nodevalues_view(sol[u])[2]; Plotter = Plotter)
    gridplot!(pl[2,1], xgrid; Plotter = Plotter)
    scalarplot!(pl[2,2], xgrid, nodevalues(sol[u]; abs = true)[:]; Plotter = Plotter)
    vectorplot!(pl[2,2], xgrid, eval_func(PointEvaluator([id(u)], sol)), spacing = 0.05, clear = false)

    
end

end # module