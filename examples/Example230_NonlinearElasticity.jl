#=

# 230 : Nonlinear Elasticity
([source code](@__SOURCE_URL__))

This example computes the displacement field ``u`` of the nonlinear elasticity problem
```math
\begin{aligned}
-\mathrm{div}(\mathbb{C} (\epsilon(u)-\epsilon_T)) & = 0 \quad \text{in } \Omega
\end{aligned}
```
where an isotropic stress tensor ``\mathbb{C}`` is applied to the nonlinear strain ``\epsilon(u) := \frac{1}{2}(\nabla u + (\nabla u)^T + (\nabla u)^T \nabla u)``
and a misfit strain  ``\epsilon_T := \Delta T \alpha`` due to thermal load caused by temperature(s) ``\Delta T`` and thermal expansion coefficients ``\alpha`` (that may be different)
in the two regions of the bimetal.

This example demonstrates how to setup a (parameter- and region-dependent) nonlinear expression and how to assign it to the problem description.

The computed solution for the default parameters looks like this:

![](example230.png)
=#

module Example230_NonlinearElasticity

using ExtendableFEM
using ExtendableGrids
using GridVisualize
using Test #hide

## parameter-dependent nonlinear operator uses a callable struct to reduce allocations
mutable struct nonlinear_operator{T}
	λ::Vector{T}
	μ::Vector{T}
	ϵT::Vector{T}
end

function strain!(result, input)
	result[1] = input[1]
	result[2] = input[4]
	result[3] = input[2] + input[3]

	## add nonlinear part of the strain 1/2 * (grad(u)'*grad(u))
	result[1] += 1 // 2 * (input[1]^2 + input[3]^2)
	result[2] += 1 // 2 * (input[2]^2 + input[4]^2)
	result[3] += input[1] * input[2] + input[3] * input[4]
	return nothing
end

## kernel for nonlinear operator
(op::nonlinear_operator)(result, input, qpinfo) = (
	## input = grad(u) written as a vector
	## compute strain and subtract thermal strain (all in Voigt notation)
	region = qpinfo.region;
	strain!(result, input);
	result[1] -= op.ϵT[region];
	result[2] -= op.ϵT[region];

	## multiply with isotropic stress tensor
	## (stored in input[5:7] using Voigt notation)
	a = op.λ[region] * (result[1] + result[2]) + 2 * op.μ[region] * result[1];
	b = op.λ[region] * (result[1] + result[2]) + 2 * op.μ[region] * result[2];
	c = 2 * op.μ[region] * result[3];

	## write strain into result
	result[1] = a;
	result[2] = c;
	result[3] = c;
	result[4] = b;
	return nothing
)

const op = nonlinear_operator([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])

## everything is wrapped in a main function
function main(;
	ν = [0.3, 0.3],          # Poisson number for each region/material
	E = [2.1, 1.1],          # Elasticity modulus for each region/material
	ΔT = [580, 580],         # temperature for each region/material
	α = [1.3e-5, 2.4e-4],    # thermal expansion coefficients
	scale = [20, 500],       # scale of the bimetal, i.e. [thickness, width]
	nrefs = 0,              # refinement levels
	order = 2,              # finite element order
	periodic = false,       # use periodic boundary conditions?
	Plotter = nothing,
	kwargs...)

	## compute Lame' coefficients μ and λ from ν and E
	## and thermal misfit strain and assign to operator operator
	@. op.μ = E / (2 * (1 + ν ))
	@. op.λ = E * ν / ((1 - 2 * ν) * (1 + ν))
	@. op.ϵT = ΔT * α

	## generate bimetal mesh
	xgrid = bimetal_strip2D(; scale = scale, n = 2 * (nrefs + 1))
	println(stdout, unicode_gridplot(xgrid))

	## create finite element space and solution vector
	FES = FESpace{H1Pk{2, 2, order}}(xgrid)

	## problem description
	PD = ProblemDescription()
	u = Unknown("u"; name = "displacement")
	assign_unknown!(PD, u)
	assign_operator!(PD, NonlinearOperator(op, [grad(u)]; kwargs...))
	if periodic
		## periodic boundary conditions
		## 1) couple dofs left (bregion 1) and right (bregion 3) in y-direction
		dofsX, dofsY, factors = get_periodic_coupling_info(FES, xgrid, 1, 3, (f1, f2) -> abs(f1[2] - f2[2]) < 1e-14; factor_components = [0, 1])
		assign_operator!(PD, CombineDofs(u, u, dofsX, dofsY, factors; kwargs...))
		## 2) find and fix point at [0, scale[1]]
		xCoordinates = xgrid[Coordinates]
		closest::Int = 0
		mindist::Float64 = 1e30
		for j ∈ 1:num_nodes(xgrid)
			dist = xCoordinates[1, j]^2 + (xCoordinates[2, j] - scale[1])^2
			if dist < mindist
				mindist = dist
				closest = j
			end
		end
		assign_operator!(PD, FixDofs(u; dofs = [closest], vals = [0]))
	else
		assign_operator!(PD, HomogeneousBoundaryData(u; regions = [1], mask = [1, 0], kwargs...))
	end

	## solve
	sol = solve(PD, FES; kwargs...)

	## displace mesh and plot
	plt = GridVisualizer(; Plotter = Plotter, layout = (3, 1), clear = true, size = (1000, 1500))
	grad_nodevals = nodevalues(grad(u), sol)
	strain_nodevals = zeros(Float64, 3, num_nodes(xgrid))
	for j in 1:num_nodes(xgrid)
		strain!(view(strain_nodevals, :, j), view(grad_nodevals, :, j))
	end
	scalarplot!(plt[1, 1], xgrid, view(strain_nodevals, 1, :), levels = 3, colorbarticks = 7, xlimits = [-scale[2] / 2 - 10, scale[2] / 2 + 10], ylimits = [-30, scale[1] + 20], title = "ϵ(u)_xx + displacement")
	scalarplot!(plt[2, 1], xgrid, view(strain_nodevals, 2, :), levels = 1, colorbarticks = 7, xlimits = [-scale[2] / 2 - 10, scale[2] / 2 + 10], ylimits = [-30, scale[1] + 20], title = "ϵ(u)_yy + displacement")
	vectorplot!(plt[1, 1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)), rasterpoints = 20, clear = false)
	vectorplot!(plt[2, 1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)), rasterpoints = 20, clear = false)
	displace_mesh!(xgrid, sol[u])
	gridplot!(plt[3, 1], xgrid, linewidth = 1, title = "displaced mesh")
	println(stdout, unicode_gridplot(xgrid))

	return strain_nodevals, plt
end

## grid
function bimetal_strip2D(; scale = [1, 1], n = 2, anisotropy_factor::Int = Int(ceil(scale[2] / (2 * scale[1]))))
	X = linspace(-scale[2] / 2, 0, (n + 1) * anisotropy_factor)
	X2 = linspace(0, scale[2] / 2, (n + 1) * anisotropy_factor)
	append!(X, X2[2:end])
	Y = linspace(0, scale[1], 2 * n + 1)
	xgrid = simplexgrid(X, Y)
	cellmask!(xgrid, [-scale[2] / 2, 0.0], [scale[2] / 2, scale[1] / 2], 1)
	cellmask!(xgrid, [-scale[2] / 2, scale[1] / 2], [scale[2] / 2, scale[1]], 2)
	bfacemask!(xgrid, [-scale[2] / 2, 0.0], [-scale[2] / 2, scale[1] / 2], 1)
	bfacemask!(xgrid, [-scale[2] / 2, scale[1] / 2], [-scale[2] / 2, scale[1]], 1)
	bfacemask!(xgrid, [-scale[2] / 2, 0.0], [scale[2] / 2, 0.0], 2)
	bfacemask!(xgrid, [-scale[2] / 2, scale[1]], [scale[2] / 2, scale[1]], 2)
	bfacemask!(xgrid, [scale[2] / 2, 0.0], [scale[2] / 2, scale[1]], 3)
	return xgrid
end

generateplots = default_generateplots(Example230_NonlinearElasticity, "example230.png") #hide
function runtests() #hide
	strain, plt = main(;) #hide
	@test maximum(strain) ≈ 0.17289633483008537 #hide
end #hide
end
