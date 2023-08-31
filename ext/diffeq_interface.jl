#
# Interface to DifferentialEquations.jl
# via TimeControlSolver struct

function diffeq_assembly!(sys, ctime)

	# unpack
	PD = sys.PD
	A = sys.A
	b = sys.b
	sol = sys.sol

	if sys.parameters[:verbosity] > 0
		@info "DiffEQ-extension: t = $ctime"
	else
		@debug "DiffEQ-extension: assembly at t = $ctime"
	end

	## assemble operators
	fill!(A.entries.cscmatrix.nzval, 0)
	fill!(b.entries, 0)
	for op in PD.operators
		ExtendableFEM.assemble!(A, b, sol, op, sys; time = ctime)
	end
end

"""
Assume the discrete problem is an ODE problem. Provide the 
rhs function for DifferentialEquations.jl.
"""
function eval_rhs!(du, x, sys, ctime)
	# (re)assemble system
	@debug "DiffEQ-extension: evaluating ODE rhs @time = $ctime"

	PD = sys.PD
	A = sys.A
	b = sys.b
	sol = sys.sol
	residual = sys.res

	sol.entries .= x
	diffeq_assembly!(sys, ctime)

	# calculate residual
	fill!(residual.entries, 0)
	mul!(residual.entries, A.entries.cscmatrix, x)
	residual.entries .-= b.entries

	## set residual as rhs of ODE
	du .= -vec(residual.entries)
	nothing
end

"""
Assume the discrete problem is an ODE problem. Provide the 
jacobi matrix calculation function for DifferentialEquations.jl.
"""
function eval_jacobian!(J, u, SC, ctime)
	@debug "DiffEQ-extension: evaluating jacobian @time = $ctime"
	PD = SC.PD
	A = SC.A
	b = SC.b
	sol = SC.sol
	## assemble operators
	fill!(A.entries.cscmatrix.nzval, 0)
	fill!(b.entries, 0)
	sol.entries .= u
	for op in PD.operators
		assemble!(A, b, sol, op, SC; time = ctime)
	end
	flush!(A.entries)
	J .= -A.entries.cscmatrix
	nothing
end

"""
Provide the system matrix as prototype for the Jacobian.
"""
function jac_prototype(sys)
	@debug "DiffEQ-interface: jac_prototype"
	ExtendableSparse.flush!(sys.A[1].entries)
	sys.A[1].entries.cscmatrix
end
