#= 

# 240 : Compressible Stokes 2D
([source code](SOURCE_URL))

This example solves the compressible Stokes equations where one seeks a (vector-valued) velocity ``\mathbf{u}``, a density ``\varrho`` and a pressure ``p`` such that
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \lambda \nabla(\mathrm{div}(\mathbf{u})) + \nabla p & = \mathbf{f} + \varrho \mathbf{g}\\
\mathrm{div}(\varrho \mathbf{u}) & = 0\\
        p & = eos(\varrho)\\
        \int_\Omega \varrho \, dx & = M\\
        \varrho & \geq 0.
\end{aligned}
```
Here eos ``eos`` is some equation of state function that describes the dependence of the pressure on the density
(and further physical quantities like temperature in a more general setting).
Moreover, ``\mu`` and ``\lambda`` are Lame parameters and ``\mathbf{f}`` and ``\mathbf{g}`` are given right-hand side data.

In this example we solve a analytical toy problem with the prescribed solution
```math
\begin{aligned}
\mathbf{u}(\mathbf{x}) & =0\\
\varrho(\mathbf{x}) & = 1 - (x_2 - 0.5)/c\\
p &= eos(\varrho) := c \varrho^\gamma
\end{aligned}
```
such that ``\mathbf{f} = 0`` and ``\mathbf{g}`` nonzero to match the prescribed solution.

This example is designed to study the well-balanced property of a discretisation. The gradient-robust discretisation
approximates the well-balanced state much better, i.e. has a much smaller L2 velocity error. For larger c the problem gets more incompressible which reduces
the error further as then the right-hand side is a perfect gradient also when evaluated with the (now closer to a constant) discrete density.
See reference below for more details.

!!! reference

    "A gradient-robust well-balanced scheme for the compressible isothermal Stokes problem",\
    M. Akbas, T. Gallouet, A. Gassmann, A. Linke and C. Merdon,\
    Computer Methods in Applied Mechanics and Engineering 367 (2020),\
    [>Journal-Link<](https://doi.org/10.1016/j.cma.2020.113069)
    [>Preprint-Link<](https://arxiv.org/abs/1911.01295)

=#

module Example280_CompressibleStokes

using ExtendableFEM
using ExtendableFEMBase
using ExtendableSparse
using ExtendableGrids
using Triangulate
using SimplexGridFactory
using GridVisualize

function kernel_gravity!(result, ϱ, qpinfo)
    result[1] = 0
    result[2] = -ϱ[1]
end

## everything is wrapped in a main function
function main(; nrefs = 4, M = 1, c = 1, Plotter = nothing, reconstruct = true, μ = 1, order = 1, kwargs...)

	## create grid
	println("Creating grid...")
	xgrid = simplexgrid(Triangulate;
    			points = [0 0; 0.2 0; 0.3 0.2; 0.45 0.05; 0.55 0.35; 0.65 0.2; 0.7 0.3; 0.8 0; 1 0; 1 1 ; 0 1]',
                bfaces = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 11; 11 1]',
                bfaceregions = ones(Int,11),
                regionpoints = [0.5 0.5;]',
                regionnumbers = [1],
                regionvolumes = [4.0^-(nrefs)])

    ## define unknowns
    u = Unknown("u"; name = "velocity", dim = 2)
    ϱ = Unknown("ϱ"; name = "density", dim = 1)
    p = Unknown("p"; name = "pressure", dim = 1)

    ## define reconstruction operator
    id_u = reconstruct ? apply(u, Reconstruct{HDIVRT0{2}, Identity}) : id(u)
    div_u = reconstruct ? apply(u, Reconstruct{HDIVRT0{2}, Divergence}) : div(u)

    ## define first sub-problem: Stokes equations to solve for velocity u
    PD = ProblemDescription("Stokes problem")
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([grad(u)]; factor = μ, store = true, kwargs...))
    assign_operator!(PD, LinearOperator([div_u], [id(ϱ)]; factor = 1, kwargs...))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4, kwargs...))
    assign_operator!(PD, LinearOperator(kernel_gravity!, [id_u], [id(ϱ)]; factor = 1, bonus_quadorder = 2, kwargs...))

    ## FVM for continuity equation
	τ = order * μ / (M*c) # time step for pseudo timestepping
    PDT = ProblemDescription("continuity equation")
    assign_unknown!(PDT, ϱ)
    assign_operator!(PDT, BilinearOperator(ExtendableSparseMatrix{Float64,Int}(0,0), [ϱ], [ϱ], [u]; callback! = assemble_fv_operator!, kwargs...))
    assign_operator!(PDT, BilinearOperator([id(ϱ)]; factor = 1/τ, store = true, kwargs...))
    assign_operator!(PDT, LinearOperator([id(ϱ)], [id(ϱ)]; factor = 1/τ, kwargs...))
    
    ## generate FESpaces and a solution vector for all 3 unknowns
    FETypes = [H1BR{2}, L2P0{1}, L2P0{1}]
    FES = [FESpace{FETypes[j]}(xgrid) for j = 1 : 3]
    sol = FEVector(FES; tags = [u,ϱ,p])

    ## initial guess
    fill!(sol[ϱ],M)

    ## solve the two problems iteratively [1] >> [2] >> [1] >> [2] ...
    sol = iterate_until_stationarity([PD, PDT]; init = sol)

    ## plot
    pl = GridVisualizer(; Plotter = Plotter, layout = (2,1), clear = true, resolution = (800,800))
    scalarplot!(pl[1,1],xgrid, view(nodevalues(sol[u]; abs = true),1,:), levels = 0, colorbarticks = 7)
    vectorplot!(pl[1,1],xgrid, eval_func(PointEvaluator([id(u)], sol)), spacing = 0.25, clear = false, title = "u_h (abs + quiver)")
    scalarplot!(pl[2,1],xgrid, view(nodevalues(sol[ϱ]),1,:), levels = 11, title = "ϱ_h")
end

## pure convection finite volume operator for transport
function assemble_fv_operator!(A, b, sol)

    ## find velocity and transported quantity
    id_u = findfirst(==(:u), [u.identifier for u in sol.tags])
    id_ϱ = findfirst(==(:ϱ), [u.identifier for u in sol.tags])
    if id_u === nothing
        @error "u not found in sol"
    end
    if id_ϱ === nothing
        @error "ϱ not found in sol"
    end

    ## get FESpace and grid
    FES = sol[id_ϱ].FES 
    xgrid = FES.xgrid

    ## matrix
    if size(A) == (FES.ndofs, FES.ndofs)
        fill!(A.cscmatrix.nzval,0)
    else
        A = ExtendableSparseMatrix{Float64,Int}(FES.ndofs, FES.ndofs)
    end

    ## integrate normalfux of velocity
    FluxIntegrator = ItemIntegrator([normalflux(1)]; entities = ON_FACES)
    fluxes::Matrix{Float64} = evaluate(FluxIntegrator,[sol[id_u]])

    ## assemble upwind finite volume fluxes over cell faces
    facecells = xgrid[FaceCells]
    cellfaces = xgrid[CellFaces]
    cellfacesigns = xgrid[CellFaceSigns]
    ncells::Int = num_sources(cellfacesigns)
    nfaces4cell::Int = 0
    face::Int = 0
    flux::Float64 = 0.0
    other_cell::Int = 0
    for cell = 1 : ncells
        nfaces4cell = num_targets(cellfaces,cell)
        for cf = 1 : nfaces4cell
            face = cellfaces[cf,cell]
            other_cell = facecells[1,face]
            if other_cell == cell
                other_cell = facecells[2,face]
            end
            flux = fluxes[face] * cellfacesigns[cf,cell]
            if (other_cell > 0) 
                flux *= 1 // 2 # because it will be accumulated on two cells
            end       
            if flux > 0 # flow from cell to other_cell or out of domain
                _addnz(A,cell,cell,flux,1)
                if other_cell > 0
                    _addnz(A,other_cell,cell,-flux,1)
                    # otherwise flow goes out of domain
                end    
            else # flow from other_cell into cell or into domain
                _addnz(A,cell,cell,1e-16,1) # add zero to keep pattern for LU
                if other_cell > 0 # flow comes from neighbour cell
                    _addnz(A,other_cell,other_cell,-flux,1)
                    _addnz(A,cell,other_cell,flux,1)
                else # flow comes from outside domain
                    # handled in right-hand side loop above
                end 
            end
        end
    end

    flush!(A)
    return A
end


function kernel_inflow!(result, input, qpinfo)
    if input[1] < 0 # if velocity points into domain
        result[1] = 0
    else
        result[1] = 0
    end
end



end