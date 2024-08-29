module ExtendableFEMDiffEQExt


if isdefined(Base, :get_extension)
	import DifferentialEquations: ODEFunction, ODEProblem
else
	import ..DifferentialEquations: ODEFunction, ODEProblem
end

import ExtendableFEM: SolverConfiguration, generate_ODEProblem
import ExtendableFEMBase: FEMatrix

using DifferentialEquations: DifferentialEquations
using ExtendableFEMBase: ExtendableFEMBase, FESpace, fill!, mul!, norm
using ExtendableSparse: ExtendableSparse, ExtendableSparseMatrix, flush!
using LinearAlgebra: LinearAlgebra

include("diffeq_interface.jl")


end # module
