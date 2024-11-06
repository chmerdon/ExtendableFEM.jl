default_statistics() = Dict{Symbol, Vector{Real}}(
	:assembly_times => [],
	:solver_times => [],
	:assembly_allocations => [],
	:solver_allocations => [],
	:linear_residuals => [],
	:nonlinear_residuals => [],
	:matrix_nnz => [],
	:total_times => [],
	:total_allocations => [],
)

mutable struct SolverConfiguration{AT <: AbstractMatrix, bT, xT}
	PD::ProblemDescription
	A::AT## stores system matrix
	b::bT## stores right-hand side
	sol::xT## stores solution
	tempsol::xT## temporary solution
	res::xT
	freedofs::Vector{Int}## stores indices of free dofs
	LP::LinearProblem
	statistics::Dict{Symbol, Vector{Real}}
	linsolver::Any
	unknown_ids_in_sol::Array{Int, 1}
	unknowns::Array{Unknown, 1}
	unknowns_reduced::Array{Unknown, 1}
	offsets::Array{Int, 1}  ## offset for each unknown that is solved
	parameters::Dict{Symbol, Any} # dictionary with user parameters
end

"""
````
residual(S::SolverConfiguration)
````

returns the residual of the last solve

"""
residual(S::SolverConfiguration) = S.statistics[:nonlinear_residuals][end]


#
# Default context information with help info.
#
default_solver_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:target_residual => (1e-10, "stop if the absolute (nonlinear) residual is smaller than this number"),
	:damping => (0, "amount of damping, value should be between in (0,1)"),
	:abstol => (1e-11, "abstol for linear solver (if iterative)"),
	:reltol => (1e-11, "reltol for linear solver (if iterative)"),
	:time => (0.0, "current time to be used in all time-dependent operators"),
	:init => (nothing, "initial solution (also used to save the new solution)"),
	:spy => (false, "show unicode spy plot of system matrix during solve"),
	:symmetrize => (false, "make system matrix symmetric (replace by (A+A')/2)"),
	:symmetrize_structure => (false, "make the system sparse matrix structurally symmetric (e.g. if [j,k] is also [k,j] must be set, all diagonal entries must be set)"),
	:restrict_dofs => ([], "array of dofs for each unknown that should be solved (default: all dofs)"),
	:check_matrix => (false, "check matrix for symmetry and positive definiteness and largest/smallest eigenvalues"),
	:verbosity => (0, "verbosity level"),
	:show_config => (false, "show configuration at the beginning of solve"),
	:show_matrix => (false, "show system matrix after assembly"),
	:return_config => (false, "solver returns solver configuration (including A and b of last iteration)"),
	:is_linear => ("auto", "linear problem (avoid reassembly of nonlinear operators to check residual)"),
	:inactive => (Array{Unknown, 1}([]), "inactive unknowns (are made available in assembly, but not updated in solve)"),
	:maxiterations => (10, "maximal number of nonlinear iterations/linear solves"),
	:constant_matrix => (false, "matrix is constant (skips reassembly and refactorization in solver)"),
	:constant_rhs => (false, "right-hand side is constant (skips reassembly)"),
	:method_linear => (UMFPACKFactorization(), "any solver or custom LinearSolveFunction compatible with LinearSolve.jl (default = UMFPACKFactorization())"),
	:precon_linear => (nothing, "function that computes preconditioner for method_linear incase an iterative solver is chosen"),
	:initialized => (false, "linear system in solver configuration is already assembled (turns true after first solve)"),
	:plot => (false, "plot all solved unknowns with a (very rough but fast) unicode plot"),
)


function Base.show(io::IO, PD::SolverConfiguration)
	println(io, "\nSOLVER-CONFIGURATION")
	for item in PD.parameters
		print(item.first)
		print(" : ")
		println(item.second)
	end
end


"""
````
function iterate_until_stationarity(
	SolverConfiguration(Problem::ProblemDescription
	[FES::Union{<:FESpace, Vector{<:FESpace}}];
	init = nothing,
	unknowns = Problem.unknowns,
	kwargs...)
````

Returns a solver configuration for the ProblemDescription that can be passed to the solve
function. Here, FES are the FESpaces that should be used to discretize the
selected unknowns. If no FES is provided an initial FEVector (see keyword init) must be provided
(which is used to built the FES).

Keyword arguments:
$(_myprint(default_solver_kwargs()))

"""
function SolverConfiguration(Problem::ProblemDescription; init = nothing, unknowns = Problem.unknowns, kwargs...)
	## try to guess FES from init
	if typeof(init) <: FEVector
		FES = [init[u].FES for u in unknowns]
	end
	SolverConfiguration(Problem, unknowns, FES; kwargs...)
end

function SolverConfiguration(Problem::ProblemDescription, FES; unknowns = Problem.unknowns, kwargs...)
	SolverConfiguration(Problem, unknowns, FES; kwargs...)
end

function SolverConfiguration(Problem::ProblemDescription, unknowns::Array{Unknown, 1}, FES, default_kwargs = default_solver_kwargs(); TvM = Float64, TiM = Int, bT = Float64, kwargs...)
	if typeof(FES) <: FESpace
		FES = [FES]
	end
	@assert length(unknowns) <= length(FES) "length of unknowns and FE spaces must coincide"
	## check if unknowns are part of Problem description
	for u in unknowns
		@assert u in Problem.unknowns "unknown $u is not part of the given ProblemDescription"
	end
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_kwargs)
	_update_params!(parameters, kwargs)
	## compute offsets
	offsets = [0]
	for FE in FES
		push!(offsets, FE.ndofs + offsets[end])
	end

	## storage for full system
	FES_active = FES[1:length(unknowns)]
	A = FEMatrix{TvM, TiM}(FES_active; tags = unknowns, npartitions = num_partitions(FES[1].xgrid))
	b = FEVector{bT}(FES_active; tags = unknowns)
	res = copy(b)

	## initialize solution vector
	if parameters[:init] === nothing
		names = [u.name for u in unknowns]
		append!(names, ["N.N." for j ∈ length(unknowns)+1:length(FES)])
		x = FEVector{bT}(FES; name = names, tags = unknowns)
		unknown_ids_in_sol = 1:length(unknowns)
	else
		x = parameters[:init]
		unknown_ids_in_sol = [findfirst(==(u), x.tags) for u in unknowns]
	end

	## adjustments for using freedofs
	if haskey(parameters, :restrict_dofs)
		if length(parameters[:restrict_dofs]) > 0
			freedofs = Vector{Int}(parameters[:restrict_dofs][1])
			for j ∈ 2:length(parameters[:restrict_dofs])
				parameters[:restrict_dofs][j] .+= FES[j-1].ndofs
				append!(freedofs, parameters[:restrict_dofs][j])
			end
			x_temp = copy(b)
		else
			freedofs = []
			x_temp = x
		end
	else
		freedofs = []
		x_temp = x
	end

	## construct linear problem
	if length(freedofs) > 0
		LP = LinearProblem(A.entries.cscmatrix[freedofs, freedofs], b.entries[freedofs])
	else
		LP = LinearProblem(A.entries.cscmatrix, b.entries)
	end
	return SolverConfiguration{typeof(A), typeof(b), typeof(x)}(Problem, A, b, x, x_temp, res, freedofs, LP, default_statistics(), nothing, unknown_ids_in_sol, unknowns, copy(unknowns), offsets, parameters)
end
