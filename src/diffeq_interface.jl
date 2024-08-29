#############################################
### Interface to DifferentialEquations.jl ###
#############################################

"""
````
function generate_ODEProblem(
	SC::SolverConfiguration,
	tspan;
	mass_matrix = nothing)
	kwargs...)
````

Reframes the ProblemDescription inside the SolverConfiguration into a SciMLBase.ODEProblem,
for DifferentialEquations.jl where tspan is the desired time interval.

If no mass matrix is provided the standard mass matrix for the respective
finite element space(s) for all unknowns is assembled.

Keyword arguments:
$(ExtendableFEM._myprint(ExtendableFEM.default_diffeq_kwargs()))

"""
function generate_ODEProblem(PD::ProblemDescription, FES, tspan; unknowns = PD.unknowns, kwargs...)
	if typeof(FES) <: FESpace
		FES = [FES]
	end
	SC = SolverConfiguration(PD, unknowns, FES, ExtendableFEM.default_diffeq_kwargs(); kwargs...)
	if SC.parameters[:verbosity] > 0
		@info ".... init solver configuration\n"
	end

	return generate_ODEProblem(SC, tspan; mass_matrix = nothing, kwargs...)
end


function generate_ODEProblem(SC::SolverConfiguration, tspan; mass_matrix = nothing, kwargs...)
	## update kwargs
	ExtendableFEM._update_params!(SC.parameters, kwargs)

	## generate default mass matrix if needed
	if mass_matrix === nothing
		if SC.parameters[:verbosity] > 0
			@info ".... generating mass matrix\n"
		end
		FES = [SC.sol[u].FES for u in SC.unknowns]
		M = FEMatrix(FES)
		assemble!(M, BilinearOperator([id(j) for j ∈ 1:length(SC.unknowns)]))
		mass_matrix = M.entries.cscmatrix
	elseif typeof(mass_matrix) <: ExtendableFEMBase.FEMatrix
		flush!(mass_matrix.entries)
		mass_matrix = mass_matrix.entries.cscmatrix
	elseif typeof(mass_matrix) <: ExtendableSparseMatrix
		flush!(mass_matrix)
		mass_matrix = mass_matrix.cscmatrix
	end

	## generate ODE problem
	f = SciMLBase.ODEFunction(eval_rhs!, jac = eval_jacobian!, jac_prototype = jac_prototype(SC), mass_matrix = mass_matrix)
	prob = SciMLBase.ODEProblem(f, SC.sol.entries, tspan, SC)
	return prob
end


function diffeq_assembly!(sys, ctime)
	# unpack
	PD = sys.PD
	A = sys.A
	b = sys.b
	sol = sys.sol

	if sys.parameters[:verbosity] > 0
		@info "DiffEQ-extension: t = $ctime"
	end

	## assemble operators
	if !sys.parameters[:constant_rhs]
		fill!(b.entries, 0)
	end
	if !sys.parameters[:constant_matrix]
		fill!(A.entries.cscmatrix.nzval, 0)
	end
	if sys.parameters[:initialized]
		for op in PD.operators
			ExtendableFEM.assemble!(A, b, sol, op, sys; assemble_matrix = !sys.parameters[:constant_matrix], assemble_rhs = !sys.parameters[:constant_rhs], time = ctime)
		end
	else
		for op in PD.operators
			ExtendableFEM.assemble!(A, b, sol, op, sys; time = ctime)
		end
	end
	flush!(A.entries)

	for op in PD.operators
		ExtendableFEM.apply_penalties!(A, b, sol, op, sys; time = ctime)
	end
	flush!(A.entries)

	## set initialize flag
	sys.parameters[:initialized] = true
end

"""
Provides the rhs function for DifferentialEquations.jl/ODEProblem.
"""
function eval_rhs!(du, x, sys, ctime)
	# (re)assemble system
	if sys.parameters[:verbosity] > 0
		"DiffEQ-extension: evaluating ODE rhs @time = $ctime"
	end

	A = sys.A
	b = sys.b
	sol = sys.sol

	if norm(sol.entries .- x) > sys.parameters[:sametol] || sys.parameters[:initialized] == false
		sol.entries .= x
		diffeq_assembly!(sys, ctime)
	end

	## calculate residual res = A*u - b
	fill!(du, 0)
	mul!(du, A.entries.cscmatrix, x)
	du .= b.entries - du

	nothing
end

"""
Provides the jacobi matrix calculation function for DifferentialEquations.jl/ODEProblem.
"""
function eval_jacobian!(J, u, SC, ctime)
	if SC.parameters[:verbosity] > 0
		@info "DiffEQ-extension: evaluating jacobian @time = $ctime"
	end

	## reassemble if necessary
	sol = SC.sol
	if norm(sol.entries .- u) > SC.parameters[:sametol] || SC.parameters[:initialized] == false
		sol.entries .= u
		diffeq_assembly!(SC, ctime)
	end

	## extract jacobian = system matrix
	A = SC.A
	J .= -A.entries.cscmatrix

	nothing
end

"""
Provides the system matrix as prototype for the jacobian.
"""
function jac_prototype(sys)
	ExtendableSparse.flush!(sys.A[1].entries)
	sys.A[1].entries.cscmatrix
end