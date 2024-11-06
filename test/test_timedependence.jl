function run_dt_tests()

	@testset "TimeDependence" begin
		println("\n")
		println("=========================")
		println("Testing Time-Dependencies")
		println("=========================")

		@test test_heatequation() < 1e-14
	end
end

## Tests heat equation problem with known exact solution
## that is quadratic in space and linear in time with f = 0
function test_heatequation(; nrefs = 2, T = 2.0, τ = 0.5, order = 2, kwargs...)

	## initial state u and exact data
	function exact_u!(result, qpinfo)
		x = qpinfo.x
		t = qpinfo.time
		result[1] = t + (x[1]^2 + x[2]^2) / 4
	end

	## kernel for exact error calculation
	function exact_error!(result, u, qpinfo)
		exact_u!(result, qpinfo)
		result .-= u
		result .= result .^ 2
	end

	## problem description
	PD = ProblemDescription("Heat Equation")
	u = Unknown("u"; name = "temperature")
	assign_unknown!(PD, u)
	assign_operator!(PD, BilinearOperator([grad(u)]; store = true, kwargs...))
	assign_operator!(PD, InterpolateBoundaryData(u, exact_u!; regions = 1:4))

	## grid
	xgrid = uniform_refine(grid_unitsquare(Triangle2D; scale = [4, 4], shift = [-0.5, -0.5]), nrefs)

	## prepare solution vector and initial data u0
	FES = FESpace{H1Pk{1, 2, order}}(xgrid)
	sol = FEVector(FES; tags = PD.unknowns)
	interpolate!(sol[u], exact_u!; bonus_quadorder = 2)
	SC = SolverConfiguration(PD, [FES]; init = sol, constant_matrix = true, kwargs...)

	## compute initial error
	ErrorIntegrator = ItemIntegrator(exact_error!, [id(u)]; quadorder = 2 * order, kwargs...)
	error = evaluate(ErrorIntegrator, sol; time = 0)
	@info "||u-u_h||(t = 0) = $(sqrt(sum(view(error, 1, :))))"

	## generate mass matrix
	M = FEMatrix(FES)
	assemble!(M, BilinearOperator([id(1)]))

	## add backward Euler time derivative
	assign_operator!(PD, BilinearOperator(M, [u]; factor = 1 / τ, kwargs...))
	assign_operator!(PD, LinearOperator(M, [u], [u]; factor = 1 / τ, kwargs...))

	## iterate tspan
	t = 0
	for it ∈ 1:Int(floor(T / τ))
		t += τ
		ExtendableFEM.solve(PD, [FES], SC; time = t)
	end

	## compute error
	ErrorIntegrator = ItemIntegrator(exact_error!, [id(u)]; quadorder = 0, kwargs...)
	error = sqrt(sum(evaluate(ErrorIntegrator, sol; time = T)))
	@info "error heat equation test = $error"
	return error
end