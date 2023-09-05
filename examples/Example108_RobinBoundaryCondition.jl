#= 

# 108 : Robin Boundary Condition
([source code](SOURCE_URL))

This demonstrates the assignment of a mixed Robin boundary condition for a nonlinear 1D convection-diffusion-reaction PDE on the unit interval, i.e.
```math
\begin{aligned}
-\partial^2 u / \partial x^2 + u \partial u / \partial x + u & = f && \text{in } \Omega\\
u + \partial u / \partial_x & = g && \text{at } \Gamma_1 = \{ 0 \}\\
u & = u_D && \text{at } \Gamma_2 = \{ 1 \}
\end{aligned}
```
tested with data ``f(x) = e^{2x}``, ``g = 2`` and ``u_D = e`` such that ``u(x) = e^x`` is the exact solution.
=#

module Example108_RobinBoundaryCondition

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using GridVisualize

## data and exact solution
function f!(result, qpinfo)
	x = qpinfo.x
	result[1] = exp(2 * x[1])
end
function u!(result, qpinfo)
	x = qpinfo.x
	result[1] = exp(x[1])
end

## kernel for the (nonlinear) reaction-convection-diffusion oeprator
function nonlinear_kernel!(result, input, qpinfo)
	## input = [u,∇u] as a vector of length 2
	result[1] = input[1] * input[2] + input[1] # convection + reaction (will be multiplied with v)
	result[2] = input[2]                       # diffusion (will be multiplied with ∇v)
	return nothing
end

## kernel for Robin boundary condition
function robin_kernel!(result, input, qpinfo)
	## input = [u]
	result[1] = 2 - input[1] # = g - u (will be multiplied with v)
	return nothing
end

function exact_error!(result, u, qpinfo)
	u!(result, qpinfo)
	result .-= u
	result .= result .^ 2
end

## everything is wrapped in a main function
function main(; Plotter = nothing, h = 1e-1, h_fine = 1e-3, order = 2, kwargs...)

	## problem description
	PD = ProblemDescription()
	u = Unknown("u"; name = "u")
	assign_unknown!(PD, u)
	assign_operator!(PD, NonlinearOperator(nonlinear_kernel!, [id(u), grad(u)]; kwargs...))
	assign_operator!(PD, BilinearOperator(robin_kernel!, [id(u)]; entities = ON_BFACES, regions = [1], kwargs...))
	assign_operator!(PD, LinearOperator(f!, [id(u)]; kwargs...))
	assign_operator!(PD, InterpolateBoundaryData(u, u!; regions = [2], kwargs...))

	## generate coarse and fine mesh
	xgrid = simplexgrid(0:h:1)

	## choose some finite element type and generate a FESpace for the grid
	## (here it is a one-dimensional H1-conforming P2 element H1P2{1,1})
	FEType = H1Pk{1, 1, order}
	FES = FESpace{FEType}(xgrid)

	## generate a solution vector and solve
	sol = solve(PD, FES; kwargs...)

	## compute L2 error
	L2error = ItemIntegrator(exact_error!, [id(u)]; quadorder = 2 * order, kwargs...)
	println("L2error = $(sqrt(sum(evaluate(L2error, sol))))")

	## plot discrete and exact solution (on finer grid)
	p = GridVisualizer(Plotter = Plotter, layout = (1, 1))
	scalarplot!(p[1, 1], xgrid, nodevalues_view(sol[u])[1], color = (0, 0.7, 0), label = "u_h", markershape = :x, markersize = 10, markevery = 1)
	xgrid_fine = simplexgrid(0:h_fine:1)
	scalarplot!(p[1, 1], xgrid_fine, view(nodevalues(xgrid_fine, u!), 1, :), clear = false, color = (1, 0, 0), label = "u", legend = :rb, markershape = :none)
end

end
