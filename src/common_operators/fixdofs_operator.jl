
##################
### FIXED DOFS ###
##################

mutable struct FixDofs{UT, AT, VT} <: AbstractOperator
	u::UT
	dofs::AT
	offset::Int
	vals::VT
	assembler::Any
	parameters::Dict{Symbol, Any}
end

ExtendableFEM.fixed_dofs(O::FixDofs) = O.dofs
fixed_vals(O::FixDofs) = O.vals

default_fixdofs_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:penalty => (1e30, "penalty for fixed degrees of freedom"),
	:name => ("FixDofs", "name for operator used in printouts"),
	:verbosity => (0, "verbosity level"),
)

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::FixDofs)
	return O.u
end

function Base.show(io::IO, O::FixDofs)
	dependencies = dependencies_when_linearized(O)
	print(io, "$(O.parameters[:name])($(ansatz_function(dependencies)), ndofs = $(length(O.dofs)))")
	return nothing
end

"""
````
function FixDofs(u; vals = [], dofs = [], kwargs...)
````

When assembled, all specified dofs of the unknown u will be penalized
to the specified values.

Keyword arguments:
$(_myprint(default_fixdofs_kwargs()))

"""
function FixDofs(u; dofs = [], vals = zeros(Float64, length(dofs)), kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_fixdofs_kwargs())
	_update_params!(parameters, kwargs)
	@assert length(dofs) == length(vals)
	return FixDofs{typeof(u), typeof(dofs), typeof(vals)}(u, dofs, 0, vals, nothing, parameters)
end

function ExtendableFEM.apply_penalties!(A, b, sol, O::FixDofs{UT}, SC::SolverConfiguration; assemble_matrix = true, assemble_rhs = true, kwargs...) where {UT}
	if UT <: Integer
		ind = O.u
	elseif UT <: Unknown
		ind = get_unknown_id(SC, O.u)
	end
	offset = sol[ind].offset
	dofs = O.dofs
	vals = O.vals
	penalty = O.parameters[:penalty]
	if assemble_matrix
		AE = A.entries
		for j ∈ 1:length(dofs)
			dof = dofs[j] + offset
			AE[dof, dof] = penalty
		end
	end
	if assemble_rhs
		BE = b.entries
		for j ∈ 1:length(dofs)
			dof = dofs[j] + offset
			BE[dof] = penalty * vals[j]
		end
	end
	SE = sol.entries
	for j ∈ 1:length(dofs)
		dof = dofs[j] + offset
		SE[dof] = vals[j]
	end
	O.offset = offset
end