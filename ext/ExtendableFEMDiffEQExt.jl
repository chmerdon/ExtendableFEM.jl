module ExtendableFEMDiffEQExt


if isdefined(Base, :get_extension)
	import DifferentialEquations: ODEFunction, ODEProblem
else
	import ..DifferentialEquations: ODEFunction, ODEProblem
end

import ExtendableFEM: SolverConfiguration, generate_ODEProblem

using DifferentialEquations
using LinearAlgebra
using ExtendableFEM
using ExtendableSparse

include("diffeq_interface.jl")

function generate_ODEProblem(SC::SolverConfiguration, tspan; mass_matrix = nothing)
	## generate default mass matrix if needed
	if mass_matrix === nothing
		ops = []
		FES = []
		for u in SC.unknowns
			push!(ops, id(u))
			push!(FES, SC.sol[u].FES)
		end
		M = FEMatrix(FES)
		assemble!(M, BilinearOperator([id(1)]))
		mass_matrix = M.entries.cscmatrix
	end

	## generate ODE problem
	f = DifferentialEquations.ODEFunction(eval_rhs!, jac = eval_jacobian!, jac_prototype = jac_prototype(SC), mass_matrix = mass_matrix)
	prob = DifferentialEquations.ODEProblem(f, SC.sol.entries, tspan, SC)
	return prob
end

end # module
