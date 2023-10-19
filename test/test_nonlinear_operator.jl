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
