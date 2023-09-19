#= 

# 310 : Div-free RT0 basis
([source code](SOURCE_URL))

This example computes the best-approximation ``\mathbf{\psi}_h`` of a divergence-free velocity
``\mathbf{u} = \mathrm{curl} \mathbf{\psi}`` by solving
for a curl-potential ``\mathbf{\phi}_h \in N_0`` with
```math
\begin{aligned}
(\mathrm{curl} \mathbf{\phi}_h, \mathrm{curl} \mathbf{\theta}_h) & = (\mathbf{u}, \mathrm{curl} \mathbf{\theta}_h) \quad \text{for all } \mathbf{\theta} \in N_0
\end{aligned}
```
Here, ``N_0`` denotes the lowest-order Nedelec space which renders the problem ill-posed unless one selects
a linear independent basis. This is done with the algorithm suggested in the reference below.

!!! reference

	"Decoupling three-dimensional mixed problems using divergence-free finite elements",\
	R. Scheichl,\
	SIAM J. Sci. Comput. 23(5) (2002),\
	[>Journal-Link<](http://www.siam.org/journals/sisc/23-5/37588.html)

=#

module Example310_DivFreeBasis

using ExtendableFEM
using ExtendableFEMBase
using GridVisualize
using ExtendableGrids
using ExtendableSparse
using Symbolics

## exact data for problem by symbolics
function prepare_data()

	@variables x y z

	## stream function ξ
	ξ = [x*y*z,x*y*z,x*y*z]

	## velocity u = curl ξ
	∇ξ = Symbolics.jacobian(ξ, [x, y, z])
	u = [∇ξ[3,2] - ∇ξ[2,3], ∇ξ[1,3] - ∇ξ[3,1], ∇ξ[2,1] - ∇ξ[1,2]]

	## build function
	u_eval = build_function(u, x, y, z, expression = Val{false})

	return u_eval[2]
end

function main(;
	nrefs = 4,                      ## number of refinement levels
	bonus_quadorder = 2,            ## additional quadrature order for data evaluations
	Plotter = nothing,              ## Plotter (e.g. PyPlot)
	kwargs...)

	## prepare problem data
	u_eval = prepare_data()
	exact_u!(result, qpinfo) = (u_eval(result, qpinfo.x[1], qpinfo.x[2], qpinfo.x[3]))

	## prepare plots
	pl = GridVisualizer(; Plotter = Plotter, layout = (2, 2), clear = true, size = (1000, 1000))

	## prepare error calculation
	function exact_error!(result, u, qpinfo)
		exact_u!(view(result, 1:3), qpinfo)
		result .-= u
		result .= result .^ 2
	end
	ErrorIntegratorExact = ItemIntegrator(exact_error!, [curl3(1)]; bonus_quadorder = 2 + bonus_quadorder, kwargs...)
	NDofs = zeros(Int, nrefs)
	L2error = zeros(Float64, nrefs, 5)

	sol = nothing
	xgrid = nothing
	for lvl ∈ 1:nrefs
		## grid
		xgrid = uniform_refine(grid_unitcube(Tetrahedron3D), lvl)

		## get subset of edges, spanning the node graph
		spanning_tree = get_spanning_edge_subset(xgrid)

		## get all other edges = linear independent degrees of freedom
		subset = setdiff(1:num_edges(xgrid), spanning_tree)

		## Generate lowest order Nedelec FESpace
		FES = FESpace{HCURLN0{3}}(xgrid)
		NDofs[lvl] = length(subset)

		## assemble full Nedelec mass matrix
		M = FEMatrix(FES)
		b = FEVector(FES)
		assemble!(M, BilinearOperator([curl3(1)]))
		assemble!(b, LinearOperator(exact_u!, [curl3(1)]; bonus_quadorder = bonus_quadorder))

		## restrict to linear independent basis
		Z = ExtendableSparseMatrix{Float64, Int64}(length(subset), FES.ndofs)
		for j = 1 : length(subset)
			Z[j,subset[j]] = 1
		end
		M2 = Z * (M.entries * Z')
		b2 = Z * b.entries

		## solve
		sol = FEVector(FES)
		sol.entries[subset] .= M2\b2

		## check residual
		@info "residual = $(norm(M.entries * sol.entries - b.entries))"

		## evalute error
		error = evaluate(ErrorIntegratorExact, sol)
		L2error[lvl] = sqrt(sum(view(error, 1, :)) + sum(view(error, 2, :)))
		@info "|| curl(ϕ - ϕ_h) || = $(L2error[lvl])"
	end

	## plot and print convergence history as table
	scalarplot!(pl[1, 1], xgrid, nodevalues(sol[1], Curl3D; abs = true)[1, :]; Plotter = Plotter)
	print_convergencehistory(NDofs, L2error; X_to_h = X -> X .^ (-1 / 3), ylabels = ["|| u - u_h ||", "|| ∇(u - u_h) ||", "|| uR ||", "|| p - p_h ||", "|| div(u + uR) ||"], xlabel = "ndof")
end


## finds a minimal subset (of dimension #nodes - 1) of edges, such that all nodes are connected
function get_spanning_edge_subset(xgrid)
	nnodes = num_nodes(xgrid)
	edgenodes = xgrid[EdgeNodes]
	@time bedgenodes = xgrid[BEdgeNodes]
	bedgeedges = xgrid[BEdgeEdges]

	## boolean arrays to memorize which nodes are visited
	## and which edges belong to the spanning tree
	visited = zeros(Bool, nnodes)
	markededges = zeros(Bool, num_edges(xgrid))

	function find_spanning_tree(edgenodes, remap)
		nodeedges = atranspose(edgenodes)
		function recursive(node)
			visited[node] = true
			nneighbors = num_targets(nodeedges, node)
			for e = 1 : nneighbors
				edge = nodeedges[e, node]
				for k = 1 : 2
					node2 = edgenodes[k, edge]
					if !visited[node2]
						## mark edge
						markededges[remap[edge]] = true
						recursive(node2)
					end
				end
			end
			return nothing
		end
		recursive(edgenodes[1])
	end

	## find spanning tree for Neumann boundary
	## local bedges >> global edge numbers
	find_spanning_tree(bedgenodes, bedgeedges)

	## find spanning tree for remaining part
	other_nodes = setdiff(1:nnodes, unique(view(bedgenodes,:)))
	if length(other_nodes) > 0
		find_spanning_tree(edgenodes, 1 : num_edges(xgrid))
	end

	## return all marked edges
	return findall(==(true), markededges)
end


end # module