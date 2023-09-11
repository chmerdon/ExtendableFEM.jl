#= 

# 202 : Poisson-Problem (Mixed)
([source code](SOURCE_URL))

This example computes the solution ``u`` of the two-dimensional Poisson problem
in the mixed form
```math
\begin{aligned}
\sigma + \mu \nabla u = 0\\
\mathrm{div} \sigma & = f \quad \text{in } \Omega
\end{aligned}
```
with right-hand side ``f(x,y) \equiv xy`` and homogeneous Dirichlet boundary conditions
on the unit square domain ``\Omega`` on a given grid.

=#

module Example202_MixedPoissonProblem

using ExtendableFEM
using ExtendableFEMBase
using GridVisualize
using ExtendableGrids


const μ = 0.25          # coefficient for laplacian!
function blf!(result, u_ops, qpinfo)
	σ, divσ, u = view(u_ops, 1:2), view(u_ops, 3), view(u_ops, 4)
	result[1] = σ[1] / μ
	result[2] = σ[2] / μ
	result[3] = -u[1]
	result[4] = divσ[1]
	return nothing
end

function f!(fval, qpinfo)
	fval[1] = qpinfo.x[1] * qpinfo.x[2]
	return nothing
end

## boundary data
function boundarydata!(result, qpinfo)
	result[1] = 0
	return nothing
end

function main(; nrefs = 5, Plotter = nothing, hdivdg = false, kwargs...)

	## problem description
	PD = ProblemDescription()
	σ = Unknown("σ"; name = "pseudostress")
	u = Unknown("u"; name = "potential")
	p = Unknown("p"; name = "LM hdiv continuity") # only_used if hdivdg == true
	assign_unknown!(PD, u)
	assign_unknown!(PD, σ)
	if hdivdg
		assign_unknown!(PD, p)
		assign_operator!(PD, BilinearOperator([jump(normalflux(σ))], [id(p)]; transposed_copy = 1, entities = ON_IFACES, kwargs...))
		assign_operator!(PD, HomogeneousData(p; regions = 1:4))
	end
	assign_operator!(PD, BilinearOperator(blf!, [id(σ), div(σ), id(u)]; kwargs...))
	assign_operator!(PD, LinearOperator(boundarydata!, [normalflux(σ)]; entities = ON_BFACES, regions = 1:4, kwargs...))
	assign_operator!(PD, LinearOperator(f!, [id(u)]; kwargs...))
	assign_operator!(PD, FixDofs(u; dofs = [1], vals = [0]))

	## discretize
	xgrid = uniform_refine(grid_unitsquare_mixedgeometries(), nrefs)
	FES = Dict(u => FESpace{L2P0{1}}(xgrid),
		σ => FESpace{HDIVRT0{2}}(xgrid; broken = hdivdg),
		p => hdivdg ? FESpace{L2P0{1}, ON_FACES}(xgrid) : nothing)

	## solve
	sol = ExtendableFEM.solve(PD, FES; kwargs...)

	## plot
	p = GridVisualizer(; Plotter = Plotter, layout = (1, 2), clear = true, size = (1000, 500))
	scalarplot!(p[1, 1], xgrid, nodevalues(sol[u])[:]; Plotter = Plotter, title = "u")
	scalarplot!(p[1, 2], xgrid, nodevalues(sol[σ]; abs = true)[:]; Plotter = Plotter, title = "σ ≈ -∇u (abs + quiver)")
	vectorplot!(p[1, 2], xgrid, eval_func(PointEvaluator([id(σ)], sol)), vscale = 0.8, clear = false)
end

end # module
