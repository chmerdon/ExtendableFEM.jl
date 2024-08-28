default_diffeq_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:sametol => (1e-15, "tolerance to identify two solution vectors to be identical (and to skip reassemblies called by DifferentialEquations.jl)"),
	:constant_matrix => (false, "matrix is constant (skips reassembly and refactorization in solver)"),
	:constant_rhs => (false, "right-hand side is constant (skips reassembly)"),
	:init => (nothing, "initial solution (otherwise starts with a zero vector)"),
	:initialized => (false, "linear system in solver configuration is already assembled (turns true after first assembly)"),
	:verbosity => (0, "verbosity level"),
)

"""
````
function generate_ODEProblem(
	PD::ProblemDescription,
	FES,
	tspan;
	mass_matrix = nothing)
	kwargs...)
````

Reframes the ProblemDescription inside the SolverConfiguration into an ODEProblem,
for DifferentialEquations.jl where tspan is the desired time interval.

If no mass matrix is provided the standard mass matrix for the respective
finite element space(s) for all unknowns is assembled.

Additional keyword arguments:
$(_myprint(default_diffeq_kwargs()))

"""
function generate_ODEProblem end # Implementation in extension ExtendableFEMDiffEQExt, see ext subfolder