function run_nonlinear_operator_tests()

	@testset "Nonlinear Operator" begin
		println("\n")
		println("==========================")
		println("Testing Nonlinear Operator")
		println("==========================")

		@test TestLinearNonlinearOperator() < 1e-14
		@test TestParallelAssemblyNonlinearOperator() < 1e-13
	end
end

## generate a nonlinear operator with a linear kernel
## and compare its jacobian with matrix from BilinearOperator
## for that same kernel, they should be identical
function kernel!(result, u_ops, qpinfo)
	u, ∇u, p = view(u_ops, 1:2), view(u_ops, 3:6), view(u_ops, 7)
	μ = qpinfo.params[1]
	α = qpinfo.params[2]
	result[1] = ∇u[1] + α * u[1] 
	result[2] = ∇u[3] + α * u[2]
	result[3] = μ * ∇u[1] - p[1]
	result[4] = μ * ∇u[2]
	result[5] = μ * ∇u[3]
	result[6] = μ * ∇u[4] - p[1]
	result[7] = -(∇u[1] + ∇u[4])
end
function TestLinearNonlinearOperator(; μ = 0.1, α = 2, sparse = true)

	
	nlop = NonlinearOperator(kernel!, [id(1), grad(1), id(2)]; params = [μ, α], sparse_jacobians = sparse)
	blop = BilinearOperator(kernel!, [id(1), grad(1), id(2)]; params = [μ, α], use_sparsity_pattern = sparse)

	xgrid = uniform_refine(grid_unitsquare(Triangle2D),2)
	FES = [FESpace{H1P2{2,2}}(xgrid), FESpace{H1P1{1}}(xgrid)]
	u = FEVector(FES)
	interpolate!(u[1], (result, qpinfo) -> (result[1] = qpinfo.x[1]^2; result[2] = sum(qpinfo.x);))
	interpolate!(u[2], (result, qpinfo) -> (result[1] = qpinfo.x[2]^2;))
	bnonlin = FEVector(FES)
	Anonlin = FEMatrix(FES, FES)
	Alin = FEMatrix(FES, FES)
	assemble!(Anonlin, bnonlin, nlop, u)
	assemble!(Alin, blop)

	nor = norm(Anonlin.entries.cscmatrix - Alin.entries.cscmatrix)
	@info "norm between jacobian of (linear) nonlinear operator and matrix from (same) bilinear operator = $nor"
	return nor
end

function TestParallelAssemblyNonlinearOperator(; μ = 0.1, α = 2, sparse = true, verbosity = 1)

	
	nlop_seq = NonlinearOperator(kernel!, [id(1), grad(1), id(2)]; params = [μ, α], sparse_jacobians = sparse, parallel = false, verbosity = verbosity)
	nlop_par = NonlinearOperator(kernel!, [id(1), grad(1), id(2)]; params = [μ, α], sparse_jacobians = sparse, parallel = true, verbosity = verbosity)

	## sequential assembly
	xgrid = uniform_refine(grid_unitsquare(Triangle2D), 4)
	FES = [FESpace{H1P2{2,2}}(xgrid), FESpace{H1P1{1}}(xgrid)]
	u = FEVector(FES)
	interpolate!(u[1], (result, qpinfo) -> (result[1] = qpinfo.x[1]^2; result[2] = sum(qpinfo.x);))
	interpolate!(u[2], (result, qpinfo) -> (result[1] = qpinfo.x[2]^2;))
	bseq = FEVector(FES)
	Aseq = FEMatrix(FES, FES)
	assemble!(Aseq, bseq, nlop_seq, u)

	## parallel assembly on same grid
	xgrid = partition(xgrid, PlainMetisPartitioning(npart = 20))
	FES = [FESpace{H1P2{2,2}}(xgrid), FESpace{H1P1{1}}(xgrid)]
	u = FEVector(FES)
	interpolate!(u[1], (result, qpinfo) -> (result[1] = qpinfo.x[1]^2; result[2] = sum(qpinfo.x);))
	interpolate!(u[2], (result, qpinfo) -> (result[1] = qpinfo.x[2]^2;))
	bpar = FEVector(FES)
	Apar = FEMatrix(FES, FES; npartitions = num_partitions(xgrid))
	assemble!(Apar, bpar, nlop_par, u)

	## compare the two matrices
	## since partitioning changes dof enumeration only norms are compared
	nor = abs(norm(Apar.entries.cscmatrix) - norm(Aseq.entries.cscmatrix)) + abs(norm(bpar.entries) - norm(bseq.entries))
	@info "difference between norms of sequantially and parallel assembled jacobians and rhs = $nor"
	return nor
end