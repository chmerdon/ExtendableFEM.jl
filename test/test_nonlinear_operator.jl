function run_nonlinear_operator_tests()

	@testset "Nonlinear Operator" begin
		println("\n")
		println("==========================")
		println("Testing Nonlinear Operator")
		println("==========================")

		@test TestJacobianSparsityPattern(20) < 1e-16
	end
end

include("../examples/Example226_Thermoforming.jl")

function TestJacobianSparsityPattern(N)


	## run the same simulation with sparse and dense jacobians and compute the difference in the result
	sol1 =  Example226_Thermoforming.main(Plotter=nothing, N=N, sparse_jacobians=true)
	sol2 =  Example226_Thermoforming.main(Plotter=nothing, N=N, sparse_jacobians=false)

	return norm(sol1.entries - sol2.entries)
end

function TestLinearNonlinearOperator

	## generate a nonlinear operator with parameters
	## and run it with parameters that make it linear
	## and compare with the LinearOperator
	function kernel(result, u_ops, qpinfo)
		u, ∇u, p = view(u_ops, 1:2), view(u_ops, 3:6), view(u_ops, 7)
		μ = qpinfo.params[1]
		α = qpinfo.params[2]
		β = qpinfo.params[3]
		result[1] = β * dot(u, view(∇u, 1:2)) + α * u[1] 
		result[2] = β * dot(u, view(∇u, 3:4)) + α * u[2]
		result[3] = μ * ∇u[1] - p[1]
		result[4] = μ * ∇u[2]
		result[5] = μ * ∇u[3]
		result[6] = μ * ∇u[4] - p[1]
		result[7] = -(∇u[1] + ∇u[4])
		return nothing
	end
	nlop = NonlinearOperator(kernel_nonlinear!(; nonlinear = nonlinear), [id(u), grad(u), id(p)]; bonus_quadorder = 2, params = [μ, α], sparse_jacobians = false, kwargs...)
	
			

end
