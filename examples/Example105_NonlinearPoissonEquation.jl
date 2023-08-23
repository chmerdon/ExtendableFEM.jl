#= 

# 105 : Nonlinear Poisson Equation
([source code](SOURCE_URL))

This examples solves the nonlinear Poisson problem
```math
\begin{aligned}
- \epsilon \partial^2 u / \partial x^2 + e^u - e^{-u} & = f && \text{in } \Omega
\end{aligned}
```
where
```math
f(x) = \begin{cases}
1 & x \geq 0.5,
-1 & x < 0.5.
\end{cases}
```
on the domain ``\Omega := (0,1)`` with Dirichlet boundary conditions ``u(0) = 0`` and ``u(1) = 1``.
=#

module Example105_NonlinearPoissonEquation

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using GridVisualize

## data and exact solution
function f!(result, qpinfo)
    result[1] = qpinfo.x[1] < 0.5 ? -1 : 1
end

function boundary_data!(result, qpinfo)
    result[1] = qpinfo.x[1]
end

## kernel for the (nonlinear) reaction-convection-diffusion oeprator
function nonlinear_kernel!(result, input, qpinfo)
    ## input = [u,∇u] as a vector of length 2
    result[1] = exp(input[1]) - exp(-input[1])  # (exp(u) - exp(-u), v)
    result[2] = qpinfo.params[1]*input[2]                      # (ϵ∇u, ∇v)
    return nothing
end

## everything is wrapped in a main function
function main(; Plotter = nothing, h = 1e-1, ϵ = 1e-3, order = 2, kwargs...)

    ## problem description
    PD = ProblemDescription("Nonlinear Poisson Equation")
    u = Unknown("u"; name = "u")
    assign_unknown!(PD, u)
    assign_operator!(PD, NonlinearOperator(nonlinear_kernel!, [id(u), grad(u)]; params = [ϵ], kwargs...) )
    assign_operator!(PD, LinearOperator(f!, [id(u)]; kwargs...))
    assign_operator!(PD, InterpolateBoundaryData(u, boundary_data!; kwargs...))

    ## generate coarse and fine mesh
    xgrid = simplexgrid(0:h:1)

    ## choose some finite element type and generate a FESpace for the grid
    ## (here it is a one-dimensional H1-conforming P2 element H1P2{1,1})
    FEType = H1Pk{1,1,order}
    FES = FESpace{FEType}(xgrid)

    ## generate a solution vector and solve
    sol = solve(PD, [FES]; kwargs...)

    ## plot discrete and exact solution (on finer grid)
    p=GridVisualizer(Plotter = Plotter, layout = (1,1))
    scalarplot!(p[1,1], xgrid, nodevalues_view(sol[u])[1], color=(0,0.7,0), label = "u_h", markershape = :x, markersize = 10, markevery = 1)
end

end