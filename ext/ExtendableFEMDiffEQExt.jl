module ExtendableFEMDiffEQExt


if isdefined(Base, :get_extension)
	import DifferentialEquations: ODEFunction, ODEProblem
else
	import ..DifferentialEquations: ODEFunction, ODEProblem
end

import ExtendableFEM: SolverConfiguration, generate_ODEProblem
import ExtendableFEMBase: FEMatrix

using DifferentialEquations
using LinearAlgebra
using ExtendableFEM
using ExtendableFEMBase
using ExtendableSparse

include("diffeq_interface.jl")


end # module
