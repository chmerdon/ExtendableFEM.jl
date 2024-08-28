abstract type AbstractReductionOperator end

mutable struct FixbyInterpolation{UT <: Union{Unknown, Integer}, MT} <: AbstractReductionOperator
	u_in::UT
	u_out::UT
	A::MT
	parameters::Dict{Symbol, Any}
end

default_redop_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:entities => (ON_CELLS, "assemble operator on these grid entities (default = ON_CELLS)"),
	:name => ("FixbyInterpolation", "name for operator used in printouts"),
	:factor => (1, "factor that should be multiplied during assembly"),
	:also_remove => ([], "other unknowns besides u_out that are removed by the reduction step"),
	:verbosity => (0, "verbosity level"),
	:regions => ([], "subset of regions where operator should be assembly only"),
)

# informs solver when operator needs reassembly
function depends_nonlinearly_on(O::FixbyInterpolation, u::Unknown)
	return []
end

# informs solver in which blocks the operator assembles to
function dependencies_when_linearized(O::FixbyInterpolation)
	return [O.u_in]
end

function FixbyInterpolation(u_out, A, u_in; kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_redop_kwargs())
	_update_params!(parameters, kwargs)
	return FixbyInterpolation{typeof(u_out), typeof(A)}(u_in, u_out, A, parameters)
end


function apply!(LP, O::FixbyInterpolation{UT}, SC; kwargs...) where {UT}
	A_full = SC.A
	I = O.A
	b_full = SC.b
	sol = SC.sol
	if UT <: Integer
		ind_in = O.u_in
		ind_out = O.u_out
		ind_also = O.parameters[:also_remove]
	elseif UT <: Unknown
		ind_in = get_unknown_id(SC, O.u_in)
		ind_out = get_unknown_id(SC, O.u_out)
		ind_also = get_unknown_id(SC, O.parameters[:also_remove])
	end

	remaining = deleteat!(SC.unknowns_reduced, ind_out)
	remaining = deleteat!(SC.unknowns_reduced, ind_also)
	#remap = union(1:ind_out-1, ind_out+1:length(SC.unknowns_reduced))
	remap = [get_unknown_id(SC, u) for u in remaining]
	@show ind_in, ind_out, remaining, remap
	FES = [sol[j].FES for j ∈ 1:length(sol)]
	FES_reduced = [sol[j].FES for j in remap]
	b = FEVector{eltype(SC.b.entries)}(FES_reduced)
	A = FEMatrix{eltype(SC.b.entries)}(FES_reduced)

	T = ExtendableSparseMatrix{eltype(SC.b.entries), Int}(length(sol.entries), length(b.entries))

	for j ∈ 1:length(remap)
		offset_x = b_full[remap[j]].offset
		offset_y = b[j].offset
		for k ∈ 1:FES_reduced[j].ndofs
			T[offset_x+k, offset_y+k] = 1
		end
	end

	offset_x = b_full[ind_out].offset
	offset_y = b_full[ind_in].offset
	@show size(I), size(T), [FE.ndofs for FE in FES]

	for j ∈ 1:FES[ind_out].ndofs, k ∈ 1:FES[ind_in].ndofs
		T[offset_x+j, offset_y+k] = I[k, j]
	end
	flush!(T)

	A.entries.cscmatrix += T.cscmatrix' * A_full.entries * T.cscmatrix
	@info ".... spy plot of full system matrix:\n$(UnicodePlots.spy(sparse(A_full.entries.cscmatrix)))"
	@info ".... spy plot of trafo matrix:\n$(UnicodePlots.spy(sparse(T.cscmatrix)))"
	@info ".... spy plot of reduced system matrix:\n$(UnicodePlots.spy(sparse(A.entries.cscmatrix)))"
	b.entries .+= T.cscmatrix' * b_full.entries

	## remap reduced matrix

	## return reduced linear problem
	LP_reduced = LinearProblem(A.entries.cscmatrix, b.entries)
	return LP_reduced, A, b
end
