
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


default_fixdofs_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:penalty => (1e30, "penalty for fixed degrees of freedom"),
	:name => ("FixDofs", "name for operator used in printouts"),
	:verbosity => (0, "verbosity level"),
)

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::FixDofs)
	return O.u
end

function ExtendableFEM.fixed_dofs(O::FixDofs)
	## assembles operator to full matrix A and b
	return O.dofs .+ O.offset
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

function ExtendableFEM.assemble!(A, b, sol, O::FixDofs{UT}, SC::SolverConfiguration; assemble_matrix = true, assemble_rhs = true, kwargs...) where {UT}
	if UT <: Integer
		ind = O.u
	elseif UT <: Unknown
		ind = get_unknown_id(SC, O.u)
	end
	offset = sol[ind].offset
	dofs = O.dofs
	vals = O.vals
	penalty = O.parameters[:penalty]
	AE = A.entries
	BE = b.entries
	SE = sol.entries
	for j ∈ 1:length(dofs)
		dof = dofs[j] + offset
		if assemble_matrix
			AE[dof, dof] = penalty
		end
		if assemble_rhs
			BE[dof] = penalty * vals[j]
		end
		SE[dof] = vals[j]
	end
	O.offset = offset
end
