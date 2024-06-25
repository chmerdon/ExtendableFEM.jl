mutable struct FaceInterpolator{Tv, Ti, UT <: Union{Unknown, Integer}, MT, KFT}
	u_args::Array{UT, 1}
	ops_args::Array{DataType, 1}
	coeffs_ops::Array{Array{Ti, 1}, 1}
	kernel::KFT
	FES_args::Any
	FES_target::Any
	BE_args::Any              #::Union{Nothing, Array{FEEvaluator,1}}
	QP_infos::Any             #::Array{QPInfosT,1}
	QF::Any
	L2G::Any
	assembler::Any
	value::Any
	A::MT
	parameters::Dict{Symbol, Any}
end

default_interp_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:order => ("auto", "interpolation order (default: match order of applied finite element space)"),
	:name => ("Projector", "name for operator used in printouts"),
	:parallel_groups => (true, "assemble operator in parallel using CellAssemblyGroups"),
	:only_interior => (false, "only interior faces, interpolation of boundary faces will be zero"),
	:resultdim => (0, "dimension of result field (default = length of arguments)"),
	:params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
	:verbosity => (0, "verbosity level"),
)

function FaceInterpolator(kernel, u_args, ops_args; Tv = Float64, Ti = Int, kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_interp_kwargs())
	_update_params!(parameters, kwargs)
	@assert length(u_args) == length(ops_args)
	coeffs_ops = Array{Array{Ti, 1}, 1}([])
	for op in ops_args
		@assert is_discontinuous(op) "all operators must be of type DiscontinuousFunctionOperator ($(typeof(op)) is not)"
		push!(coeffs_ops, coeffs(op))
	end
	return FaceInterpolator{Tv, Ti, typeof(u_args[1]), ExtendableSparseMatrix{Tv, Int64}, typeof(kernel)}(
		u_args,
		ops_args,
		coeffs_ops,
		kernel,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		ExtendableSparseMatrix{Tv, Int64}(0, 0),
		parameters,
	)
end

function FaceInterpolator(kernel, oa_args::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
	u_args = [oa[1] for oa in oa_args]
	ops_args = [oa[2] for oa in oa_args]
	return FaceInterpolator(kernel, u_args, ops_args; kwargs...)
end

function FaceInterpolator(oa_args::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
	u_args = [oa[1] for oa in oa_args]
	ops_args = [oa[2] for oa in oa_args]
	return FaceInterpolator(ExtendableFEMBase.standard_kernel, u_args, ops_args; kwargs...)
end

function build_assembler!(O::FaceInterpolator{Tv}, FE_args::Array{<:FEVectorBlock, 1}; time = 0.0, kwargs...) where {Tv}

	FES_args = [FE_args[j].FES for j ∈ 1:length(FE_args)]

	if (O.FES_args != FES_args)
		if O.parameters[:verbosity] > 0
			@info ".... building assembler for $(O.parameters[:name])"
		end

		## determine grid
		xgrid = determine_assembly_grid(FES_args)

		## prepare assembly
		AT = ON_CELLS
		Ti = typeof(xgrid).parameters[2]
		itemassemblygroups = xgrid[CellAssemblyGroups]
		itemgeometries = xgrid[CellGeometries]
		itemregions = xgrid[CellRegions]
		itemfaces = xgrid[CellFaces]
		facevolumes = xgrid[FaceVolumes]
		facenormals = xgrid[FaceNormals]
		facecells = xgrid[FaceCells]
		FETypes_args = [eltype(F) for F in FES_args]
		EGs = [itemgeometries[itemassemblygroups[1, j]] for j ∈ 1:num_sources(itemassemblygroups)]

		@assert length(xgrid[UniqueFaceGeometries]) == 1 "currently only grids with single face geometry type are allowed"
		## prepare assembly
		nargs = length(FES_args)
		O.QF = []
		O.BE_args = Array{Array{<:FEEvaluator, 1}, 1}([])
		O.QP_infos = Array{QPInfos, 1}([])
		O.L2G = []
		faceEG = xgrid[UniqueFaceGeometries][1]
		qf_offsets::Array{Int, 1} = [0]
		for EG in EGs
			## quadrature formula for face
			if O.parameters[:order] == "auto"
				polyorder = maximum([get_polynomialorder(FE, faceEG) for FE in FETypes_args])
				minderiv = minimum([ExtendableFEMBase.NeededDerivative4Operator(op) for op in O.ops_args])
				O.parameters[:order] = polyorder - minderiv
			end
			qf = VertexRule(faceEG, O.parameters[:order])

			## generate qf that integrates along full cell boundary
			nfaces = num_faces(EG)
			xref_face_to_cell = xrefFACE2xrefCELL(EG)
			w_cellboundary = Array{eltype(qf.w), 1}([])
			xref_cellboundary = Array{eltype(qf.xref), 1}([])
			for f ∈ 1:nfaces
				append!(w_cellboundary, qf.w)
				push!(qf_offsets, qf_offsets[end] + length(qf.w))
				for i ∈ 1:length(qf.xref)
					push!(xref_cellboundary, xref_face_to_cell[f](qf.xref[i]))
				end
			end
			qf_cellboundary = ExtendableFEMBase.SQuadratureRule{eltype(qf.xref[1]), EG, dim_element(EG), length(qf.xref) * nfaces}("cell boundary rule", xref_cellboundary, w_cellboundary)
			push!(O.QF, qf_cellboundary)

			## FE basis evaluator for EG
			push!(O.BE_args, [FEEvaluator(FES_args[j], StandardFunctionOperator(O.ops_args[j]), O.QF[end]; AT = AT) for j in 1:nargs])

			## L2G map for EG
			push!(O.L2G, L2GTransformer(EG, xgrid, ON_CELLS))

			## parameter structure
			push!(O.QP_infos, QPInfos(xgrid; time = time, params = O.parameters[:params]))
		end


		## prepare operator infos
		op_lengths_args = [size(O.BE_args[1][j].cvals, 1) for j ∈ 1:nargs]
		op_offsets_args = [0]
		append!(op_offsets_args, cumsum(op_lengths_args))
		resultdim::Int = O.parameters[:resultdim]
		if resultdim == 0
			resultdim = op_offsets_args[end]
			O.parameters[:resultdim] = resultdim
		end

		## prepare target FE
		only_interior = O.parameters[:only_interior]
		if O.parameters[:order] <= 0
			FEType_target = L2P0{resultdim}
		else
			FEType_target = H1Pk{resultdim, dim_element(faceEG), O.parameters[:order]}
		end
		FES_target = FESpace{FEType_target, ON_FACES}(xgrid; broken = true)
		b = FEVector(FES_target)
		coffsets = ExtendableFEMBase.get_local_coffsets(FEType_target, ON_CELLS, faceEG)
		#A = FEMatrix(FES_target, FES_target)

		FEATs_args = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[j]), AT) for j ∈ 1:nargs]
		itemdofs_args::Array{Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}}, 1} = [get_dofmap(FES_args[j], xgrid, FEATs_args[j]) for j = 1 : nargs]
		facedofs_target::Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}} = FES_target[CellDofs]

		O.FES_args = FES_args
		O.value = b

		## Assembly loop for fixed geometry
		function assembly_loop(b::AbstractVector{T}, sol::Array{<:FEVectorBlock, 1}, items, EG::ElementGeometries, QF::QuadratureRule, BE_args::Array{<:FEEvaluator, 1}, L2G::L2GTransformer, QPinfos::QPInfos) where {T}

			## prepare parameters
			nfaces = num_faces(EG)
			result_kernel = zeros(Tv, resultdim)
			input_args = zeros(Tv, op_offsets_args[end])
			ndofs_args::Array{Int, 1} = [get_ndofs(ON_CELLS, FE, EG) for FE in FETypes_args]
			weights, xref = QF.w, QF.xref
			left_or_right::Int = 0
			if O.parameters[:order] <= 0
				left_or_right_dofs = [[1], [1]]
			else
				left_or_right_dofs = [1:qf_offsets[2], [2, 1]]
				append!(left_or_right_dofs[2], qf_offsets[2]:-1:3)
			end
			coeffs_ops::Array{Array{Int, 1}, 1} = O.coeffs_ops
			qp::Int = 0


			fill!(b, 0)

			for item::Int in items
				QPinfos.region = itemregions[item]

				## update FE basis evaluators on cell
				for j ∈ 1:nargs
					BE_args[j].citem[] = item
					update_basis!(BE_args[j])
				end
				update_trafo!(L2G, item)

				for localface::Int ∈ 1:nfaces
					face = itemfaces[localface, item]
					if only_interior && facecells[2, face] == 0
						## skip boundary face
						continue
					end
					QPinfos.item = face
					QPinfos.volume = facevolumes[face]
					QPinfos.normal .= view(facenormals,: , face)

					left_or_right = facecells[1, face] == item ? 1 : 2

					## evaluate arguments
					for k ∈ 1:qf_offsets[2]
						qp = left_or_right_dofs[left_or_right][k]
						fill!(input_args, 0)
						for id ∈ 1:nargs
							for j ∈ 1:ndofs_args[id]
								dof_j = itemdofs_args[id][j, item]
								for d ∈ 1:op_lengths_args[id]
									input_args[d+op_offsets_args[id]] += sol[id][dof_j] * BE_args[id].cvals[d, j, qp+qf_offsets[localface]] * coeffs_ops[id][left_or_right]
								end
							end
						end

						## get global x for quadrature point
						eval_trafo!(QPinfos.x, L2G, xref[qp+qf_offsets[localface]])

						# evaluate kernel
						O.kernel(result_kernel, input_args, QPinfos)
						#@show QPinfos.x, result_kernel, left_or_right

						# write into coefficients of target FE
						for d ∈ 1:resultdim
							bdof = facedofs_target[k+coffsets[d], face]
							b[bdof] += result_kernel[d]
						end
					end
				end
			end
		end

		function assembler(b, sol; kwargs...)
			time = @elapsed begin
				for j ∈ 1:length(EGs)
					assembly_loop(b.entries, sol, view(itemassemblygroups, :, j), EGs[j], O.QF[j], O.BE_args[j], O.L2G[j], O.QP_infos[j]; kwargs...)
				end
			end
			if O.parameters[:verbosity] > 1
				@info ".... assembly of $(O.parameters[:name]) took $time s"
			end
		end
		O.assembler = assembler
	else
		## update the time
		for j = 1 : length(O.QP_infos)
			O.QP_infos[j].time = time
		end
	end
end


function ExtendableFEMBase.evaluate!(O::FaceInterpolator{Tv, Ti, UT}, sol; kwargs...) where {Tv, Ti, UT}
	if UT <: Integer
		ind_args = O.u_args
	else
		ind_args = [findfirst(==(u), sol.tags) for u in O.u_args]
	end
	build_assembler!(O, [sol[j] for j in ind_args]; kwargs...)
	O.assembler(O.value, [sol[j] for j in ind_args])
	return O.value
end

