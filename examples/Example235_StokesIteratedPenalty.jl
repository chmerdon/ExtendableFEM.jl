#= 

# 235 : Stokes iterated penalty method
([source code](SOURCE_URL))

This example computes a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \nabla p & = \mathbf{0}\\
\mathrm{div}(u) & = 0
\end{aligned}
```
with some μ parameter ``\mu``.

Here we solve the simple Hagen-Poiseuille flow on the two-dimensional unit square domain with the iterated penalty method
suggested in the reference below adapted to the Bernardi--Raugel finite element method.
Given intermediate solutions  ``\mathbf{u}_h`` and  ``p_h`` the next approximations are computed by the two equations

```math
\begin{aligned}
(\nabla \mathbf{u}_h^{next}, \nabla \mathbf{v}_h) + \lambda (\mathrm{div}_h(\mathbf{u}_h) ,\mathrm{div}_h(\mathbf{v}_h)) & = (\mathbf{f},\mathbf{v}_h) + (p_h,\mathrm{div}(\mathbf{v}_h))
&& \text{for all } \mathbf{v}_h \in \mathbf{V}_h\\
(p^{next}_h,q_h) & = (p_h,q_h) - \lambda (\mathrm{div}(\mathbf{u}_h^{next}),q_h) && \text{for all } q_h \in Q_h
\end{aligned}
```

This is done consecutively until the residual of both equations is small enough.
The discrete divergence is computed via a RT0 reconstruction operator that preserves the disrete divergence.
(another way would be to compute B*inv(M)*B' where M is the mass matrix of the pressure and B is the matrix for the div-pressure block).

!!! reference

	"An iterative penalty method for the finite element solution of the stationary Navier-Stokes equations",\
	R. Codina,\
	Computer Methods in Applied Mechanics and Engineering Volume 110, Issues 3–4 (1993),\
	[>Journal-Link<](https://doi.org/10.1016/0045-7825(93)90163-R)

=#

module Example235_StokesIteratedPenalty

using ExtendableFEM
using ExtendableFEMBase
using ExtendableGrids
using ExtendableSparse
using GridVisualize

## data for Hagen-Poiseuille flow
function p!(result, qpinfo)
	x = qpinfo.x
	μ = qpinfo.params[1]
	result[1] = μ * (-2 * x[1] + 1.0)
end
function u!(result, qpinfo)
	x = qpinfo.x
	result[1] = x[2] * (1.0 - x[2])
	result[2] = 0.0
end

function div_projection!(result, input, qpinfo)
	result[1] = input[1] - qpinfo.params[1] * input[2]
end

function exact_error!(result, u, qpinfo)
	u!(result, qpinfo)
	p!(view(result, 3), qpinfo)
	result .-= u
	result .= result .^ 2
end

## everything is wrapped in a main function
function main(; Plotter = nothing, λ = 1e4, μ = 1.0, nrefs = 5, kwargs...)

	## initial grid
	xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefs)

	## Bernardi--Raugel element
	FETypes = (H1BR{2}, L2P0{1})
	PenaltyDivergence = Reconstruct{HDIVRT0{2}, Divergence}


	## generate two problems
	## one for velocity, one for pressure

	u = Unknown("u"; name = "velocity")
	p = Unknown("p"; name = "pressure")
	PDu = ProblemDescription("Stokes IPM - velocity update")
	assign_unknown!(PDu, u)
	assign_operator!(PDu, BilinearOperator([grad(u)]; factor = μ, store = true, kwargs...))
	assign_operator!(PDu, BilinearOperator([apply(u, PenaltyDivergence)]; store = true, factor = λ, kwargs...))
	assign_operator!(PDu, LinearOperator([div(u)], [id(p)]; factor = 1, kwargs...))
	assign_operator!(PDu, InterpolateBoundaryData(u, u!; regions = 1:4, params = [μ], bonus_quadorder = 4, kwargs...))

	PDp = ProblemDescription("Stokes IPM - pressure update")
	assign_unknown!(PDp, p)
	assign_operator!(PDp, BilinearOperator([id(p)]; store = true, kwargs...))
	assign_operator!(PDp, LinearOperator(div_projection!, [id(p)], [id(p), div(u)]; params = [λ], factor = 1, kwargs...))

	## show and solve problem
	FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid)]
	sol = FEVector([FES[1], FES[2]]; tags = [u, p])

	SC1 = SolverConfiguration(PDu; init = sol, maxiterations = 1, target_residual = 1e-8, constant_matrix = true, kwargs...)
	SC2 = SolverConfiguration(PDp; init = sol, maxiterations = 1, target_residual = 1e-8, constant_matrix = true, kwargs...)
	sol, nits = iterate_until_stationarity([SC1, SC2]; init = sol, kwargs...)
	@info "converged after $nits iterations"

	## error calculation
	ErrorIntegratorExact = ItemIntegrator(exact_error!, [id(u), id(p)]; quadorder = 4, params = [μ], kwargs...)
	error = evaluate(ErrorIntegratorExact, sol)
	L2errorU = sqrt(sum(view(error, 1, :)) + sum(view(error, 2, :)))
	L2errorP = sqrt(sum(view(error, 3, :)))
	@info "L2error(u) = $L2errorU"
	@info "L2error(p) = $L2errorP"

	## plot
	pl = GridVisualizer(; Plotter = Plotter, layout = (1, 2), clear = true, size = (1000, 500))
	scalarplot!(pl[1, 1], xgrid, view(nodevalues(sol[u]; abs = true), 1, :), levels = 3)
	vectorplot!(pl[1, 1], xgrid, eval_func(PointEvaluator([id(u)], sol)), spacing = [0.25, 0.1], clear = false, title = "u_h (abs + quiver)")
	scalarplot!(pl[1, 2], xgrid, view(nodevalues(sol[p]), 1, :), levels = 11, title = "p_h")
end
end
