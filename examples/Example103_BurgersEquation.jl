#= 

# 103 : Burger's Equation
([source code](SOURCE_URL))

This example solves the Burger's equation
```math
\begin{aligned}
u_t - \mu \Delta u + \mathrm{div} f(u) & = 0
\end{aligned}
```
with periodic boundary conditions.

=#

module Example103_BurgersEquation

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using DifferentialEquations
using GridVisualize

function kernel_nonlinear!(result, u, qpinfo)
    result[1] = u[1]^2/2
end

function initial_data!(result, qpinfo)
    result[1] = abs(qpinfo.x[1]) < 0.5 ? 1 : 0
end

## everything is wrapped in a main function
function main(;
    ν = 0.01,
    h = 0.005,
    T = 2,
    order = 2,
    τ = 0.01,
    Plotter = nothing,
    use_diffeq = true,
    solver = Rosenbrock23(), 
    kwargs...)

    ## load mesh and exact solution
    xgrid = simplexgrid(-2:h:2)

    ## set finite element types [surface height, velocity]
    FEType = H1Pk{1,1,order}

    ## generate empty PDEDescription for three unknowns (h, u)
    PD = ProblemDescription("Burger's Equation")
    u = Unknown("u"; name = "u")
    assign_unknown!(PD, u)
    assign_operator!(PD, NonlinearOperator(kernel_nonlinear!, [grad(u)], [id(u)]; bonus_quadorder = 2))
    assign_operator!(PD, BilinearOperator([grad(u)]; store = true, factor = ν))
    assign_operator!(PD, CombineDofs(u, u, [1], [num_nodes(xgrid)], [1.0]; kwargs...))

    ## prepare solution vector and initial data
    FES = FESpace{FEType}(xgrid)
    sol = FEVector(FES; tags = PD.unknowns)
    interpolate!(sol[u],  initial_data!)
    SC = SolverConfiguration(PD, [FES]; init = sol, maxiterations = 1, kwargs...)

    ## init plotter and plot u0
    p = GridVisualizer(; Plotter = Plotter, layout = (1,1), clear = true, resolution = (800,800))
    scalarplot!(p[1,1], xgrid, nodevalues_view(sol[u])[1], flimits = (-0.75,2), levels = 0, title = "u_h (t = 0)")

    ## generate mass matrix
    M = FEMatrix(FES)
    assemble!(M, BilinearOperator([id(1)]))

    if (use_diffeq)
        ## generate ODE problem
        prob = generate_ode(DifferentialEquations, SC, (0.0, T); mass_matrix = M.entries.cscmatrix)

        ## solve ODE problem
        de_sol = DifferentialEquations.solve(prob, solver, abstol=1e-6, reltol=1e-3, dt = τ, dtmin = 1e-6, adaptive = true, initializealg=DifferentialEquations.NoInit())
        @info "#tsteps = $(length(de_sol))"

        ## get final solution
        sol.entries .= de_sol[end]
    else
        ## add backward Euler time derivative
        assign_operator!(PD, BilinearOperator(M, [u]; factor = 1/τ, kwargs...))
        assign_operator!(PD, LinearOperator(M, [u], [u]; factor = 1/τ, kwargs...))

        ## iterate tspan
        t = 0
        for it = 1 : Int(floor(T/τ))
            t += τ   
            ExtendableFEM.solve(PD, [FES], SC; time = t)
            scalarplot!(p[1,1], xgrid, nodevalues_view(sol[u])[1], flimits = (-0.75,2), levels = 0, title = "u_h (t = $t)")
        end
    end

    ## plot final state
    scalarplot!(p[1,1], xgrid, nodevalues_view(sol[u])[1], flimits = (-0.75,2), levels = 0, title = "u_h (t = $T)")

end
end