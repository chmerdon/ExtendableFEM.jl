#= 

# 265 : Flow + Transport
([source code](SOURCE_URL))

This example solve the Stokes problem in an Omega-shaped pipe and then uses the velocity in a transport equation for a species with a certain inlet concentration.
Altogether, we are looking for a velocity ``\mathbf{u}``, a pressure ``\mathbf{p}`` and a stationary species concentration ``\mathbf{c}`` such that
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \nabla p & = 0\\
\mathrm{div}(u) & = 0\\
\mathbf{c}_t - \kappa \Delta \mathbf{c} + \mathbf{u} \cdot \nabla \mathbf{c} & = 0
\end{aligned}
```
with some viscosity parameter and diffusion parameter ``\kappa``.

The diffusion coefficient for the species is chosen (almost) zero such that the isolines of the concentration should stay parallel from inlet to outlet. 
For the discretisation of the convection term in the transport equation three possibilities can be chosen:

1. Classical Bernardi--Raugel stationary finite element discretisations ``\mathbf{u}_h \cdot \nabla \mathbf{c}_h``
   [set FVtransport = false, reconstruct = false]
2. As in 1. but with divergence-free reconstruction operator in convection term ``\Pi_\text{reconst} \mathbf{u}_h \cdot \nabla \mathbf{c}_h``
   [set FVtransport = false, reconstruct = true]
3. Time-dependent upwind finite volume discretisation for ``\kappa = 0`` based on normal fluxes along the faces [set FVtransport = true]

Observe that the divergence-free postprocessing helps a lot for mass conservation, but is still not perfect. The finite volume
upwind discretisation ensures mass conservation.

Note, that the transport equation is very convection-dominated and no stabilisation in the finite element discretisations was used here (but instead a nonzero ``\kappa``).
Also note, that only the finite volume discretisation perfectly obeys the maximum principle for the concentration but the isolines do no stay
parallel until the outlet is reached, possibly due to articifial diffusion.
=#

module Example265_FlowTransport

using ExtendableFEM
using ExtendableFEMBase
using ExtendableSparse
using ExtendableGrids
using GridVisualize

## boundary data
function u_inlet!(result, qpinfo)
    x = qpinfo.x
    result[1] = 4*x[2]*(1-x[2])
    result[2] = 0
end
function c_inlet!(result, qpinfo)
    result[1] = (1-qpinfo.x[2])*qpinfo.x[2]
end

function kernel_stokes_standard!(result, u_ops, qpinfo)
    ∇u, p = view(u_ops,1:4), view(u_ops, 5)
    μ = qpinfo.params[1]
    result[1] = μ*∇u[1] - p[1]
    result[2] = μ*∇u[2]
    result[3] = μ*∇u[3]
    result[4] = μ*∇u[4] - p[1]            
    result[5] = -(∇u[1] + ∇u[4])
end

function kernel_convection!(result, ∇T, u, qpinfo)
    result[1] = ∇T[1]*u[1] + ∇T[2]*u[2]
end


## everything is wrapped in a main function
function main(; nrefs = 5, Plotter = nothing, reconstruct = true, FVtransport = true, μ = 1, kwargs...)
    
    ## load mesh and refine
    xgrid = uniform_refine(simplexgrid("assets/2d_grid_upipe.sg"), nrefs)

    ## define unknowns
    u = Unknown("u"; name = "velocity", dim = 2)
    p = Unknown("p"; name = "pressure", dim = 1)
    T = Unknown("T"; name = "temperature", dim = 1)

    ## define first sub-problem: Stokes equations to solve for velocity u
    PD = ProblemDescription("Stokes problem")
    assign_unknown!(PD, u)
    assign_unknown!(PD, p)
    assign_operator!(PD, BilinearOperator(kernel_stokes_standard!, [grad(u), id(p)]; params = [μ], kwargs...))  
    assign_operator!(PD, InterpolateBoundaryData(u, u_inlet!; regions = 4, kwargs...))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = [1,3], kwargs...))

    ## add transport equation of species
    PDT = ProblemDescription("transport problem")
    assign_unknown!(PDT, T)
    if FVtransport ## FVM discretisation of transport equation (pure upwind convection)
        τ = 1e3
        assign_operator!(PDT, BilinearOperator(ExtendableSparseMatrix{Float64,Int}(0,0), [T], [T], [u]; callback! = assemble_fv_operator!, kwargs...))
        assign_operator!(PDT, BilinearOperator([id(T)]; factor = 1/τ, kwargs...))
        assign_operator!(PDT, LinearOperator([id(T)], [id(T)]; factor = 1/τ, kwargs...))
    else ## FEM discretisation of transport equation (with small diffusion term)
        reconstruct_operator = reconstruct ? Reconstruct{HDIVBDM1{2}, Identity} : Identity
        assign_operator!(PDT, BilinearOperator([grad(T)]; factor = 1e-6, kwargs...))
        assign_operator!(PDT, BilinearOperator(kernel_convection!, [id(T)], [grad(T)], [apply(u, reconstruct_operator)]; kwargs...))
        assign_operator!(PDT, InterpolateBoundaryData(T, c_inlet!; regions = [4], kwargs...))
    end
    
    ## generate FESpaces and a solution vector for all 3 unknowns
    FETypes = [H1BR{2}, L2P0{1}, FVtransport ? L2P0{1} : H1P1{1}]
    FES = [FESpace{FETypes[j]}(xgrid) for j = 1 : 3]
    sol = FEVector(FES; tags = [u,p,T])

    ## solve the two problems separately
    sol = solve(PD; init = sol, kwargs...)
    sol = solve(PDT; init = sol, maxiterations = 20, target_residual = 1e-12, kwargs...)

    ## print minimal and maximal concentration to check max principle (shoule be in [0,1])
    println("\n[min(c),max(c)] = [$(minimum(view(sol[T]))),$(maximum(view(sol[T])))]")

    ## plot
    p = GridVisualizer(; Plotter = Plotter, layout = (2,1), clear = true, resolution = (800,800))
    scalarplot!(p[1,1],xgrid, view(nodevalues(sol[u]; abs = true),1,:), levels = 0, colorbarticks = 7)
    vectorplot!(p[1,1],xgrid, eval_func(PointEvaluator([id(u)], sol)), spacing = 0.25, clear = false, title = "u_h (abs + quiver)")
    scalarplot!(p[2,1],xgrid, view(nodevalues(sol[T]),1,:), limits = (0,0.25), levels = 11, title = "c_h")
end

## pure convection finite volume operator for transport
function assemble_fv_operator!(A, b, sol)

    ## find velocity and transported quantity
    id_u = findfirst(==(:u), [u.identifier for u in sol.tags])
    id_T = findfirst(==(:T), [u.identifier for u in sol.tags])
    if id_u === nothing
        @error "u not found in sol"
    end
    if id_T === nothing
        @error "T not found in sol"
    end

    ## get FESpace and grid
    FES = sol[id_T].FES 
    xgrid = FES.xgrid

    ## right-hand side = boundary inflow fluxes if velocity points inward
    BndFluxIntegrator = ItemIntegrator(kernel_inflow!, [normalflux(1)]; entities = ON_BFACES)
    bnd_fluxes::Matrix{Float64} = evaluate(BndFluxIntegrator,[sol[id_u]]) 
    facecells = xgrid[FaceCells]
    bface2face = xgrid[BFaceFaces]
    for bface in 1 : lastindex(bface2face)
        b[1][facecells[1, bface2face[bface]]] -= bnd_fluxes[bface]
    end

    ## matrix
    if size(A) == (FES.ndofs, FES.ndofs)
        return A
    end
    A = ExtendableSparseMatrix{Float64,Int}(FES.ndofs, FES.ndofs)

    ## integrate normalfux of velocity
    FluxIntegrator = ItemIntegrator([normalflux(1)]; entities = ON_FACES)
    fluxes::Matrix{Float64} = evaluate(FluxIntegrator,[sol[id_u]])

    ## assemble upwind finite volume fluxes over cell faces
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
        c_inlet!(result, qpinfo)
        result[1] *= input[1]
    else
        result[1] = 0
    end
end



end