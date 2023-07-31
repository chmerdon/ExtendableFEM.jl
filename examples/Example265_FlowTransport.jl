#= 

# 265 : Flow + Transport
([source code](SOURCE_URL))

This example solve the Stokes problem in a Omega-shaped pipe and then uses the velocity in a transport equation for a species with a certain inlet concentration.
Altogether, we are looking for a velocity ``\mathbf{u}``, a pressure ``\mathbf{p}`` and a species concentration ``\mathbf{c}`` such that
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \nabla p & = 0\\
\mathrm{div}(u) & = 0\\
- \kappa \Delta \mathbf{c} + \mathbf{u} \cdot \nabla \mathbf{c} & = 0
\end{aligned}
```
with some viscosity parameter  and diffusion parameter ``\kappa``.

The diffusion coefficient for the species is chosen (almost) zero such that the isolines of the concentration should stay parallel from inlet to outlet. 
For the discretisation of the convection term in the transport equation two three possibilities can be chosen:

1. Classical finite element discretisations ``\mathbf{u}_h \cdot \nabla \mathbf{c}_h``
2. Pressure-robust finite element discretisation ``\Pi_\text{reconst} \mathbf{u}_h \cdot \nabla \mathbf{c}_h`` with some divergence-free reconstruction operator ``\Pi_\text{reconst}``
3. Upwind finite volume discretisation for ``\kappa = 0`` based on normal fluxes along the faces (also divergence-free in finite volume sense)

Observe that a pressure-robust Bernardi--Raugel discretisation preserves this much better than a classical Bernardi--Raugel method. For comparison also a Taylor--Hood method can be switched on
which is comparable to the pressure-robust lowest-order method in this example. 

Note, that the transport equation is very convection-dominated and no stabilisation in the finite element discretisations was used here (but instead a nonzero ``\kappa``). The results are very sensitive to ``\kappa`` and may be different if a stabilisation is used (work in progress).
Also note, that only the finite volume discretisation perfectly obeys the maximum principle for the concentration but the isolines do no stay parallel until the outlet is reached, possibly due to articifial diffusion.
=#

module Example265_FlowTransport

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using GridVisualize

## boundary data
function u_inlet!(result, qpinfo)
    x = qpinfo.x
    result[1] = 4*x[2]*(1-x[2])
    result[2] = 0
end
function c_inlet!(result, qpinfo)
    result[1] = 1-qpinfo.x[2]
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
function main(; nrefs = 4, Plotter = nothing, postprocess = true, FVtransport = false, μ = 1, kwargs...)
    
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
    if FVtransport ## FVM discretisation of stransport equation
        assign_operator!(PDT, [3,3], FVConvectionDiffusionOperator(1))
    else ## FEM discretisation of transport equation (with small diffusion term)
        postprocess_operator = postprocess ? Reconstruct{HDIVBDM1{2}, Identity} : Identity
        assign_operator!(PDT, BilinearOperator([grad(T)]; factor = 1e-8, kwargs...))
        assign_operator!(PDT, BilinearOperator(kernel_convection!, [id(T)], [grad(T)], [apply(u, postprocess_operator)]; kwargs...))
    end
    ## inlet concentration boundary condition
    assign_operator!(PDT, InterpolateBoundaryData(T, c_inlet!; regions = [4], kwargs...))
    
    ## generate FESpaces and a solution vector for all 3 unknowns
    FETypes = [H1BR{2}, L2P0{1}, FVtransport ? L2P0{1} : H1P1{1}]
    FES = [FESpace{FETypes[j]}(xgrid) for j = 1 : 3]
    sol = FEVector(FES; tags = [u,p,T])

    ## solve the two problems separately
    sol = solve(PD; init = sol)
    sol = solve(PDT; init = sol)

    # ## then solve the transport equation [3] by finite volumes or finite elements
    # if FVtransport == true
    #     ## pseudo-timestepping until stationarity detected, the matrix stays the same in each iteration
    #     TCS = TimeControlSolver(Problem, Solution, BackwardEuler; subiterations = [[3]], skip_update = [-1], timedependent_equations = [3], T_time = Int)
    #     advance_until_stationarity!(TCS, 10000; maxtimesteps = 100, stationarity_threshold = 1e-12)
    # else
    #     ## solve directly
    #     solve!(Solution, Problem; subiterations = [[3]], maxiterations = 5, target_residual = 1e-12)
    # end

    ## print minimal and maximal concentration to check max principle (shoule be in [0,1])
    println("\n[min(c),max(c)] = [$(minimum(view(sol[T]))),$(maximum(view(sol[T])))]")

    ## plot
    p = GridVisualizer(; Plotter = Plotter, layout = (2,1), clear = true, resolution = (800,800))
    scalarplot!(p[1,1],xgrid, view(nodevalues(sol[u]; abs = true),1,:), levels = 0, colorbarticks = 7)
    vectorplot!(p[1,1],xgrid, eval_func(PointEvaluator([id(u)], sol)), spacing = 0.25, clear = false, title = "u_h (abs + quiver)")
    scalarplot!(p[2,1],xgrid, view(nodevalues(sol[T]),1,:), levels = 11, title = "c_h")
end
end