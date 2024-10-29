
mutable struct CallbackOperator{UT <: Union{Unknown, Integer}, FT, MT, bT} <: AbstractOperator
	callback::FT
	u_args::Array{UT, 1}
	storage_A::MT
	storage_b::bT
	parameters::Dict{Symbol, Any}
end

default_cbop_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:name => ("CallbackOperator", "name for operator used in printouts"),
	:time_dependent => (false, "operator is time-dependent ?"),
	:linearized_dependencies => (:auto, "[u_ansatz, u_test] when linearized"),
	:modifies_matrix => (true, "callback function modifies the matrix?"),
	:modifies_rhs => (true, "callback function modifies the rhs?"),
	:store => (false, "store matrix and rhs separately (and copy from there when reassembly is triggered)"),
	:verbosity => (0, "verbosity level"),
)

# informs solver when operator needs reassembly
function depends_nonlinearly_on(O::CallbackOperator)
	return O.u_args
end

# informs solver in which blocks the operator assembles to
function dependencies_when_linearized(O::CallbackOperator)
	return O.parameters[:linearized_dependencies]
end

# informs solver when operator needs reassembly in a time dependent setting
function is_timedependent(O::CallbackOperator)
	return O.parameters[:time_dependent]
end

function Base.show(io::IO, O::CallbackOperator)
	dependencies = dependencies_when_linearized(O)
	print(io, "$(O.parameters[:name])($([ansatz_function(dependencies[1][j]) for j = 1 : length(dependencies[1])]), $([test_function(dependencies[2][j]) for j = 1 : length(dependencies[2])]))")
	return nothing
end


"""
````
function CallbackOperator(
	callback!::Function,
	u_args = [];
	kwargs...)
````

Generates an operator that simply passes the matrix and rhs to
a user-specified call back function. The callback function needs to be conform
to the interface

	callback!(A, b, args; assemble_matrix = true, assemble_rhs = true, time = 0, kwargs...)

The u_args argument can be used to specify the arguments of the solution that should be
passed as args (a vector of FEVectorBlocks) to the callback.

Keyword arguments:
$(_myprint(default_cbop_kwargs()))

"""
function CallbackOperator(callback, u_args = []; kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_cbop_kwargs())
	_update_params!(parameters, kwargs)
	if parameters[:linearized_dependencies] == :auto
		parameters[:linearized_dependencies] = [u_args, u_args]
	end
	if parameters[:store] && parameters[:modifies_matrix]
		storage_A = ExtendableSparseMatrix{Float64, Int}(0, 0)
	else
		storage_A = nothing
	end
	if parameters[:store] && parameters[:modifies_rhs]
		storage_b = Vector{Float64}(0)
	else
		storage_b = nothing
	end
	return CallbackOperator{eltype(u_args), typeof(callback), typeof(storage_A), typeof(storage_b)}(callback, u_args, storage_A, storage_b, parameters)
end

function assemble!(A, b, sol, O::CallbackOperator{UT}, SC::SolverConfiguration; time = 0, assemble_matrix = true, assemble_rhs = true, kwargs...) where {UT}
	if O.parameters[:store] && size(A) == size(O.storage)
		add!(A, O.storage_A)
		add!(b, O.storage_b)
	else
		if UT <: Integer
			ind_args = O.u_args
		elseif UT <: Unknown
			ind_args = [findfirst(==(u), sol.tags) for u in O.u_args]
		end
		if O.parameters[:store]
			if O.parameters[:modifies_matrix]
				O.storage_A = ExtendableSparseMatrix{Float64, Int}(size(A, 1), size(A, 2))
			end
			if O.parameters[:modifies_rhs]
				O.storage_b = Vector(Float64)(length(b))
			end
		end
		if O.parameters[:store]
			O.callback(O.storage_A, O.storage_b, [sol[j] for j in ind_args]; time = time, assemble_matrix = assemble_matrix, assemble_rhs = assemble_rhs, kwargs...)
			if O.parameters[:modifies_matrix]
				flush!(O.storage_A)
				add!(A, O.storage_A)
			end
			if O.parameters[:modifies_rhs]
				add!(b, O.storage_b)
			end
		else
			O.callback(A.entries, b.entries, [sol[j] for j in ind_args]; time = time, assemble_matrix = assemble_matrix, assemble_rhs = assemble_rhs, kwargs...)
			flush!(A.entries)
		end
	end
end
