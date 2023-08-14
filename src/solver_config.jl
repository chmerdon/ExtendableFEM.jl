mutable struct SolverConfiguration{AT <: AbstractMatrix, bT, xT}
    PD::ProblemDescription
    A::AT
    b::bT
    sol::xT
    res::xT
    LP::LinearProblem
    linsolver
    unknown_ids_in_sol::Array{Int,1}
    unknowns::Array{Unknown,1}
    unknowns_reduced::Array{Unknown,1}
    offsets::Array{Int,1}  ## offset for each unknown that is solved
    parameters::Dict{Symbol,Any} # dictionary with user parameters
end

#
# Default context information with help info.
#
default_solver_kwargs()=Dict{Symbol,Tuple{Any,String}}(
    :target_residual => (1e-10, "stop if the absolute (nonlinear) residual is smaller than this number"),
    :damping => (0, "amount of damping, value should be between in (0,1)"),
    :abstol => (1e-11, "abstol for linear solver (if iterative)"),
    :reltol => (1e-11, "reltol for linear solver (if iterative)"),
    :time => (0.0, "current time to be used in all time-dependent operators"),
    :init => (nothing, "initial solution (also used to save the new solution)"),
    :spy => (false, "show unicode spy plot of system matrix during solve"),
    :verbosity => (0, "verbosity level"),
    :show_config => (false, "show configuration at the beginning of solve"),
    :show_matrix => (false, "show system matrix after assembly"),
    :return_config => (false, "solver returns solver configuration (including A and b of last iteration)"),
    :is_linear => ("auto", "linear problem (avoid reassembly of nonlinear operators to check residual)"),
    :inactive => (Array{Unknown,1}([]), "inactive unknowns (are made available in assembly, but not updated in solve)"),
    :maxiterations => (10, "maximal number of nonlinear iterations/linear solves"),
    :method_linear => (UMFPACKFactorization(), "any solver or custom LinearSolveFunction compatible with LinearSolve.jl (default = UMFPACKFactorization())"),
    :precon_linear => (nothing, "function that computes preconditioner for method_linear incase an iterative solver is chosen")
)


function Base.show(io::IO, PD::SolverConfiguration)
    println(io, "\nSOLVER-CONFIGURATION")
    for item in PD.parameters
        print(item.first)
        print(" : ")
        println(item.second)
    end
end

function SolverConfiguration(Problem::ProblemDescription, FES; kwargs...)
    SolverConfiguration(Problem, Problem.unknowns, FES; kwargs...)
end

function SolverConfiguration(Problem::ProblemDescription, unknowns::Array{Unknown,1}, FES; TvM = Float64, TiM = Int, bT = Float64, kwargs...)
    @assert length(unknowns) <= length(FES) "length of unknowns and FE spaces must coincide"
    ## check if unknowns are part of Problem description
    for u in unknowns
        @assert u in Problem.unknowns "unknown $u is not part of the given ProblemDescription"
    end
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_solver_kwargs())
    _update_params!(parameters, kwargs)
    ## compute offsets
    offsets = [0]
    for FE in FES
        push!(offsets, FE.ndofs + offsets[end])
    end
    FES_active = FES[1:length(unknowns)]
    A = FEMatrix{TvM, TiM}(FES_active)
    b = FEVector{bT}(FES_active; tags = unknowns)
    if parameters[:init] === nothing
        names = [u.name for u in unknowns]
        append!(names, ["N.N." for j = length(unknowns)+1:length(FES)])
        x = FEVector{bT}(FES; name = names, tags = unknowns)   
        unknown_ids_in_sol = 1:length(unknowns) 
    else   
        x = parameters[:init]
        unknown_ids_in_sol = [findfirst(==(u), x.tags) for u in unknowns]
    end
    res = deepcopy(b)
    LP = LinearProblem(A.entries.cscmatrix, b.entries)
    return SolverConfiguration{typeof(A),typeof(b),typeof(x)}(Problem,A,b,x,res,LP,nothing,unknown_ids_in_sol,unknowns,copy(unknowns),offsets,parameters)
end