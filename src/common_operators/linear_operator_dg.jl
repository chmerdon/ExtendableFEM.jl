
mutable struct LinearOperatorDG{Tv <: Real, UT <: Union{Unknown, Integer}, KFT <: Function, MT} <: AbstractOperator
	u_test::Array{UT, 1}
	ops_test::Array{DataType, 1}
	u_args::Array{UT, 1}
	ops_args::Array{DataType, 1}
	kernel::KFT
	BE_test_vals::Array{Vector{Matrix{Array{Tv, 3}}}}
	BE_args_vals::Array{Vector{Matrix{Array{Tv, 3}}}}
	FES_test::Any             #::Array{FESpace,1}
	FES_args::Any             #::Array{FESpace,1}
	BE_test::Any              #::Union{Nothing, Array{FEEvaluator,1}}
	BE_args::Any              #::Union{Nothing, Array{FEEvaluator,1}}
	QP_infos::Any             #::Array{QPInfosT,1}
	L2G::Any
	QF::Any
	assembler::Any
	storage::MT
	parameters::Dict{Symbol, Any}
end

default_lfopdg_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:entities => (ON_FACES, "assemble operator on these grid entities (default = ON_FACES)"),
	:name => ("LinearOperatorDG", "name for operator used in printouts"),
	:factor => (1, "factor that should be multiplied during assembly"),
	:params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
	:entry_tolerance => (0, "threshold to add entry to sparse matrix"),
	:parallel_groups => (false, "assemble operator in parallel using CellAssemblyGroups"),
	:time_dependent => (false, "operator is time-dependent ?"),
	:store => (false, "store matrix separately (and copy from there when reassembly is triggered)"),
	:quadorder => ("auto", "quadrature order"),
	:bonus_quadorder => (0, "additional quadrature order added to quadorder"),
	:verbosity => (0, "verbosity level"),
	:regions => ([], "subset of regions where operator should be assembly only"),
)

# informs solver when operator needs reassembly
function ExtendableFEM.depends_nonlinearly_on(O::LinearOperatorDG)
	return unique(O.u_args)
end

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::LinearOperatorDG)
	return [unique(O.u_test)]
end

# informs solver when operator needs reassembly in a time dependent setting
function ExtendableFEM.is_timedependent(O::LinearOperatorDG)
	return O.parameters[:time_dependent]
end

function Base.show(io::IO, O::LinearOperatorDG)
	dependencies = dependencies_when_linearized(O)
	print(io, "$(O.parameters[:name])($([test_function(dependencies[1][j]) for j = 1 : length(dependencies[1])]))")
	return nothing
end


function LinearOperatorDG(kernel::Function, u_test, ops_test; Tv = Float64, kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_lfopdg_kwargs())
	_update_params!(parameters, kwargs)
	@assert length(u_test) == length(ops_test)
	if parameters[:store]
		storage = ExtendableSparseMatrix{Float64, Int}(0, 0)
	else
		storage = nothing
	end
	return LinearOperatorDG{Tv, typeof(u_test[1]), typeof(kernel), typeof(storage)}(
		u_test,
		ops_test,
		[],
		[],
		kernel,
		Array{Vector{Matrix{Array{Tv, 3}}}}(undef, 0),
		Array{Vector{Matrix{Array{Tv, 3}}}}(undef, 0),
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		storage,
		parameters,
	)
end

function LinearOperatorDG(kernel::Function, u_test, ops_test, u_args, ops_args; Tv = Float64, kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_lfopdg_kwargs())
	_update_params!(parameters, kwargs)
	@assert length(u_args) == length(ops_args)
	@assert length(u_test) == length(ops_test)
	if parameters[:store]
		storage = ExtendableSparseMatrix{Float64, Int}(0, 0)
	else
		storage = nothing
	end
	return LinearOperatorDG{Tv, typeof(u_test[1]), typeof(kernel), typeof(storage)}(
		u_test,
		ops_test,
		u_args,
		ops_args,
		kernel,
		Array{Vector{Matrix{Array{Tv, 3}}}}(undef, 0),
		Array{Vector{Matrix{Array{Tv, 3}}}}(undef, 0),
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		nothing,
		storage,
		parameters,
	)
end

function LinearOperatorDG(kernel::Function, oa_test::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
	u_test = [oa[1] for oa in oa_test]
	ops_test = [oa[2] for oa in oa_test]
	return LinearOperatorDG(kernel, u_test, ops_test; kwargs...)
end



"""
````
function LinearOperatorDG(
	[kernel!::Function],
	oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1};
	kwargs...)
````


Generates a linear form that evaluates, in each quadrature point,
the kernel function (if non is provided, a constant function one is used)
and computes the vector product of the result with with the (discontinuous) operator evaluation(s)
of the test function(s). The header of the kernel functions needs to be conform
to the interface

	kernel!(result, qpinfo)

where qpinfo allows to access information at the current quadrature point,
e.g. qpinfo.x are the global coordinates of the quadrature point.

Keyword arguments:
$(_myprint(default_lfopdg_kwargs()))

"""
function LinearOperatorDG(oa_test::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
	u_test = [oa[1] for oa in oa_test]
	ops_test = [oa[2] for oa in oa_test]
	return LinearOperatorDG(ExtendableFEMBase.standard_kernel, u_test, ops_test; kwargs...)
end


"""
````
function LinearOperatorDG(
	kernel::Function,
	oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1},
	oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1};
	kwargs...)
````

Generates a nonlinear linear form that evaluates a kernel function
that depends on the (discontinous) operator evaluations of the current solution.
The result of the kernel function is used in a vector product with the operator evaluation(s)
of the test function(s). Hence, this can be used as a linearization of a
nonlinear operator. The header of the kernel functions needs to be conform
to the interface

    kernel!(result, eval_args, qpinfo)

where qpinfo allows to access information at the current quadrature point.

Operator evaluations are tuples that pair an unknown identifier or integer
with a Function operator.

Keyword arguments:
$(_myprint(default_lfopdg_kwargs()))

"""
function LinearOperatorDG(kernel::Function, oa_test::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}, oa_args::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
	u_test = [oa[1] for oa in oa_test]
	u_args = [oa[1] for oa in oa_args]
	ops_test = [oa[2] for oa in oa_test]
	ops_args = [oa[2] for oa in oa_args]
	return LinearOperatorDG(kernel, u_test, ops_test, u_args, ops_args; kwargs...)
end

function build_assembler!(b, O::LinearOperatorDG{Tv}, FE_test, FE_args::Array{<:FEVectorBlock, 1}; time = 0.0, kwargs...) where {Tv}
	## check if FES is the same as last time
	FES_test = [getFEStest(FE_test[j]) for j ∈ 1:length(FE_test)]
	FES_args = [FE_args[j].FES for j ∈ 1:length(FE_args)]
	if (O.FES_test != FES_test) || (O.FES_args != FES_args)

		if O.parameters[:verbosity] > 0
			@info ".... building assembler for $(O.parameters[:name])"
		end

		## prepare assembly
		AT = O.parameters[:entities]
		@assert AT <: ON_FACES  || AT <: ON_BFACES "only works for entities <: ON_FACES or ON_BFACES"
		xgrid = FES_test[1].xgrid
        if AT <: ON_BFACES
            AT = ON_FACES
            bfaces = xgrid[BFaceFaces]
            itemassemblygroups = zeros(Int, length(bfaces), 1)
            itemassemblygroups[:] .= bfaces
            gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[1]), AT)
        else
            gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[1]), AT)
            itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(gridAT)]
        end
		Ti = typeof(xgrid).parameters[2]
		itemgeometries = xgrid[GridComponentGeometries4AssemblyType(gridAT)]
		itemvolumes = xgrid[GridComponentVolumes4AssemblyType(gridAT)]
		itemregions = xgrid[GridComponentRegions4AssemblyType(gridAT)]
		FETypes_test = [eltype(F) for F in FES_test]
		EGs = xgrid[UniqueCellGeometries]

		coeffs_ops_test = Array{Array{Float64, 1}, 1}([])
		coeffs_ops_args = Array{Array{Float64, 1}, 1}([])
		for op in O.ops_test
			push!(coeffs_ops_test, coeffs(op))
		end
		for op in O.ops_args
			push!(coeffs_ops_args, coeffs(op))
		end

		## prepare assembly
		nargs = length(FES_args)
		ntest = length(FES_test)
		O.QF = []
		O.BE_test = Array{Vector{Matrix{<:FEEvaluator}}, 1}(undef, 0)
		O.BE_args = Array{Vector{Matrix{<:FEEvaluator}}, 1}(undef, 0)
		O.BE_test_vals = Array{Vector{Matrix{Array{Tv, 3}}}, 1}(undef, 0)
		O.BE_args_vals = Array{Array{Array{Tv, 3}, 1}, 1}([])
		O.QP_infos = Array{QPInfos, 1}([])
		O.L2G = []
		for EG in EGs
			## quadrature formula for EG
			polyorder_test = maximum([get_polynomialorder(FETypes_test[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_test[j]) for j ∈ 1:ntest])
			if O.parameters[:quadorder] == "auto"
				quadorder = polyorder_test + O.parameters[:bonus_quadorder]
			else
				quadorder = O.parameters[:quadorder] + O.parameters[:bonus_quadorder]
			end
			if O.parameters[:verbosity] > 1
				@info "...... integrating on $EG with quadrature order $quadorder"
			end

			## generate DG operator
			push!(O.BE_test, [generate_DG_operators(StandardFunctionOperator(O.ops_test[j]), FES_test[j], quadorder, EG) for j ∈ 1:ntest])
			push!(O.BE_args, [generate_DG_operators(StandardFunctionOperator(O.ops_args[j]), FES_args[j], quadorder, EG) for j ∈ 1:nargs])
			push!(O.QF, generate_DG_master_quadrule(quadorder, EG))

			## L2G map for EG
			EGface = facetype_of_cellface(EG, 1)
			push!(O.L2G, L2GTransformer(EGface, xgrid, gridAT))

			## FE basis evaluator for EG
			push!(O.BE_test_vals, [[O.BE_test[end][k][j[1], j[2]].cvals for j in CartesianIndices(O.BE_test[end][k])] for k ∈ 1:ntest])
			push!(O.BE_args_vals, [[O.BE_args[end][k][j[1], j[2]].cvals for j in CartesianIndices(O.BE_args[end][k])] for k ∈ 1:nargs])

			## parameter structure
			push!(O.QP_infos, QPInfos(xgrid; time = time, x = ones(Tv, size(xgrid[Coordinates], 1)), params = O.parameters[:params]))
		end

		## prepare regions
		regions = O.parameters[:regions]
		visit_region = zeros(Bool, maximum(itemregions))
		if length(regions) > 0
			visit_region[regions] .= true
		else
			visit_region .= true
		end

		## prepare operator infos
		op_lengths_test = [size(O.BE_test[1][j][1, 1].cvals, 1) for j ∈ 1:ntest]
		op_lengths_args = [size(O.BE_args[1][j][1, 1].cvals, 1) for j ∈ 1:nargs]

		op_offsets_test = [0]
		op_offsets_args = [0]
		append!(op_offsets_test, cumsum(op_lengths_test))
		append!(op_offsets_args, cumsum(op_lengths_args))
		offsets_test = [FE_test[j].offset for j in 1:length(FES_test)]
		offsets_args = [FE_args[j].offset for j in 1:length(FES_args)]	

		## prepare parallel assembly
		if O.parameters[:parallel_groups]
			Aj = Array{typeof(A), 1}(undef, length(EGs))
			for j ∈ 1:length(EGs)
				Aj[j] = deepcopy(A)
			end
		end

		FEATs_test = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[j]), ON_CELLS) for j ∈ 1:ntest]
		FEATs_args = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[j]), ON_CELLS) for j ∈ 1:nargs]
		itemdofs_test::Array{Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}}, 1} = [FES_test[j][Dofmap4AssemblyType(FEATs_test[j])] for j ∈ 1:ntest]
		itemdofs_args::Array{Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}}, 1} = [FES_args[j][Dofmap4AssemblyType(FEATs_args[j])] for j ∈ 1:nargs]
		factor = O.parameters[:factor]

		## Assembly loop for fixed geometry
		function assembly_loop(
			b::AbstractVector{T},
			sol::Array{<:FEVectorBlock, 1},
			items,
			EG::ElementGeometries,
			QF::QuadratureRule,
			BE_test::Vector{Matrix{<:FEEvaluator}},
			BE_args::Vector{Matrix{<:FEEvaluator}},
			BE_test_vals::Vector{Matrix{Array{Tv, 3}}},
			BE_args_vals::Vector{Matrix{Array{Tv, 3}}},
			L2G::L2GTransformer,
			QPinfos::QPInfos,
		) where {T}

			input_args = zeros(T, op_offsets_args[end])
			result_kernel = zeros(T, op_offsets_test[end])
			itemorientations = xgrid[CellFaceOrientations]
			itemcells = xgrid[FaceCells]
			cellitems = xgrid[CellFaces]

			ndofs_test::Array{Int, 1} = [size(BE[1, 1].cvals, 2) for BE in BE_test]
			ndofs_args::Array{Int, 1} = [size(BE[1, 1].cvals, 2) for BE in BE_args]

			weights, xref = QF.w, QF.xref
			nweights = length(weights)
			cell1::Int = 0
			orientation1::Int = 0
			itempos1::Int = 0

			input_args = [zeros(T, op_offsets_args[end]) for j ∈ 1:nweights]

			for item::Int in items
				QPinfos.region = itemregions[item]
				QPinfos.item = item
				QPinfos.volume = itemvolumes[item]
				update_trafo!(L2G, item)

                boundary_face = itemcells[2, item] == 0
				if AT <: ON_IFACES
					if boundary_face
						continue
					end
				end

				## evaluate arguments at all quadrature points
				for qp ∈ 1:nweights
					fill!(input_args[qp], 0)
					for c1 ∈ 1:2
						cell1 = itemcells[c1, item]
						if (cell1 > 0)
							itempos1 = 1
							while !(cellitems[itempos1, cell1] == item)
								itempos1 += 1
							end
							orientation1 = itemorientations[itempos1, cell1]

							for j ∈ 1:nargs
								BE_args[j][itempos1, orientation1].citem[] = cell1
								update_basis!(BE_args[j][itempos1, orientation1])
							end

							for id ∈ 1:nargs
								for j ∈ 1:ndofs_args[id]
									dof_j = itemdofs_args[id][j, cell1] + offsets_args[id]
									for d ∈ 1:op_lengths_args[id]
										input_args[qp][d+op_offsets_args[id]] += sol[id][dof_j] * BE_args_vals[id][itempos1, orientation1][d, j, qp] * coeffs_ops_args[id][c1]
									end
								end
							end
						end
					end
				end

				for c1 ∈ 1:2
					cell1 = itemcells[c1, item] # current cell of test function
					if (cell1 > 0)
						QPinfos.cell = cell1
						itempos1 = 1
						while !(cellitems[itempos1, cell1] == item)
							itempos1 += 1
						end
						orientation1 = itemorientations[itempos1, cell1]

						## update FE basis evaluators
						for j ∈ 1:ntest
							BE_test[j][itempos1, orientation1].citem[] = cell1
							update_basis!(BE_test[j][itempos1, orientation1])
						end

						## evaluate arguments
						for qp ∈ 1:nweights

							## get global x for quadrature point
							eval_trafo!(QPinfos.x, L2G, xref[qp])

                            # evaluate kernel
                            O.kernel(result_kernel, input_args[qp], QPinfos)
                            result_kernel .*= factor * weights[qp] * itemvolumes[item]

                            # multiply test function operator evaluation on cell 1
                            for idt = 1 : ntest
                                coeff_test = boundary_face ? 1 : coeffs_ops_test[idt][c1]
                                for k ∈ 1:ndofs_test[idt]
                                    dof = itemdofs_test[idt][k, cell1] + offsets_test[idt]
                                    for d ∈ 1:op_lengths_test[idt]
                                        b[dof] += result_kernel[d+op_offsets_test[idt]] * BE_test_vals[idt][itempos1, orientation1][d, k, qp] * coeff_test
                                    end
                                end
                            end
						end
					end
				end
			end
			return
		end
		O.FES_test = FES_test
		O.FES_args = FES_args

		function assembler(b, sol; kwargs...)
			if O.parameters[:store] && size(b) == size(O.storage)
				b .+= O.storage
			else
				if O.parameters[:store]
					s = zeros(eltype(b), length(b))
				else
					s = b
				end
				time = @elapsed begin
					if O.parameters[:parallel_groups]
						Threads.@threads for j ∈ 1:length(EGs)
							fill!(bj[j], 0)
							assembly_loop(bj[j], sol, view(itemassemblygroups, :, j), EGs[j], O.QF[j], O.BE_test[j], O.BE_args[j], O.BE_test_vals[j], O.BE_args_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
						end
						for j ∈ 1:length(EGs)
							s .+= bj[j]
						end
					else
						for j ∈ 1:length(EGs)
							assembly_loop(b, sol, view(itemassemblygroups, :, j), EGs[j], O.QF[j], O.BE_test[j], O.BE_args[j], O.BE_test_vals[j], O.BE_args_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
						end
					end
					if O.parameters[:store]
						b .+= s
						O.storage = s
					end
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

function build_assembler!(b, O::LinearOperatorDG{Tv}, FE_test; time = 0.0, kwargs...) where {Tv}
	## check if FES is the same as last time
	FES_test = [getFEStest(FE_test[j]) for j ∈ 1:length(FE_test)]

	if (O.FES_test != FES_test)

		if O.parameters[:verbosity] > 0
			@info ".... building assembler for $(O.parameters[:name])"
		end
		## prepare assembly
		AT = O.parameters[:entities]
		@assert AT <: ON_FACES  || AT <: ON_BFACES "only works for entities <: ON_FACES or ON_BFACES"
		xgrid = FES_test[1].xgrid
        if AT <: ON_BFACES
            AT = ON_FACES
            bfaces = xgrid[BFaceFaces]
            itemassemblygroups = zeros(Int, length(bfaces), 1)
            itemassemblygroups[:] .= bfaces
            gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[1]), AT)
        else
            gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[1]), AT)
            itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(gridAT)]
        end
		Ti = typeof(xgrid).parameters[2]
		itemgeometries = xgrid[GridComponentGeometries4AssemblyType(gridAT)]
		itemvolumes = xgrid[GridComponentVolumes4AssemblyType(gridAT)]
		itemregions = xgrid[GridComponentRegions4AssemblyType(gridAT)]
		FETypes_test = [eltype(F) for F in FES_test]
		EGs = xgrid[UniqueCellGeometries]

		coeffs_ops_test = Array{Array{Float64, 1}, 1}([])
		for op in O.ops_test
			push!(coeffs_ops_test, coeffs(op))
		end

		## prepare assembly
		ntest = length(FES_test)
		O.QF = []
		O.BE_test = Array{Vector{Matrix{<:FEEvaluator}}, 1}(undef, 0)
		O.BE_test_vals = Array{Vector{Matrix{Array{Tv, 3}}}, 1}(undef, 0)
		O.QP_infos = Array{QPInfos, 1}([])
		O.L2G = []
		for EG in EGs
			## quadrature formula for EG
			polyorder_test = maximum([get_polynomialorder(FETypes_test[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_test[j]) for j ∈ 1:ntest])
			if O.parameters[:quadorder] == "auto"
				quadorder = polyorder_test + O.parameters[:bonus_quadorder]
			else
				quadorder = O.parameters[:quadorder] + O.parameters[:bonus_quadorder]
			end
			if O.parameters[:verbosity] > 1
				@info "...... integrating on $EG with quadrature order $quadorder"
			end

			## generate DG operator
			push!(O.BE_test, [generate_DG_operators(StandardFunctionOperator(O.ops_test[j]), FES_test[j], quadorder, EG) for j ∈ 1:ntest])
			push!(O.QF, generate_DG_master_quadrule(quadorder, EG))

			## L2G map for EG
			EGface = facetype_of_cellface(EG, 1)
			push!(O.L2G, L2GTransformer(EGface, xgrid, gridAT))

			## FE basis evaluator for EG
			push!(O.BE_test_vals, [[O.BE_test[end][k][j[1], j[2]].cvals for j in CartesianIndices(O.BE_test[end][k])] for k ∈ 1:ntest])

			## parameter structure
			push!(O.QP_infos, QPInfos(xgrid; time = time, x = ones(Tv, size(xgrid[Coordinates], 1)), params = O.parameters[:params]))
		end

		## prepare regions
		regions = O.parameters[:regions]
		visit_region = zeros(Bool, maximum(itemregions))
		if length(regions) > 0
			visit_region[regions] .= true
		else
			visit_region .= true
		end

		## prepare operator infos
		op_lengths_test = [size(O.BE_test[1][j][1, 1].cvals, 1) for j ∈ 1:ntest]

		op_offsets_test = [0]
		append!(op_offsets_test, cumsum(op_lengths_test))
		offsets_test = [FE_test[j].offset for j in 1:length(FES_test)]


		## prepare parallel assembly
		if O.parameters[:parallel_groups]
			Aj = Array{typeof(A), 1}(undef, length(EGs))
			for j ∈ 1:length(EGs)
				Aj[j] = deepcopy(A)
			end
		end

		FEATs_test = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[j]), ON_CELLS) for j ∈ 1:ntest]
		itemdofs_test::Array{Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}}, 1} = [FES_test[j][Dofmap4AssemblyType(FEATs_test[j])] for j ∈ 1:ntest]
		factor = O.parameters[:factor]

		## Assembly loop for fixed geometry
		function assembly_loop(
			b::AbstractVector{T},
			items,
			EG::ElementGeometries,
			QF::QuadratureRule,
			BE_test::Vector{Matrix{<:FEEvaluator}},
			BE_test_vals::Vector{Matrix{Array{Tv, 3}}},
			L2G::L2GTransformer,
			QPinfos::QPInfos,
		) where {T}

			result_kernel = zeros(T, op_offsets_test[end])
			itemorientations = xgrid[CellFaceOrientations]
			itemcells = xgrid[FaceCells]
			cellitems = xgrid[CellFaces]

			#ndofs_test::Array{Int,1} = [get_ndofs(ON_CELLS, FE, EG) for FE in FETypes_test]
			ndofs_test::Array{Int, 1} = [size(BE[1, 1].cvals, 2) for BE in BE_test]

			weights, xref = QF.w, QF.xref
			nweights = length(weights)
			cell1::Int = 0
			orientation1::Int = 0
			itempos1::Int = 0

			## loop over faces
			## got into neighbouring cells and evaluate each operator according to
			## facepos and orientation
			for item::Int in items

				QPinfos.region = itemregions[item]
				QPinfos.item = item
				QPinfos.volume = itemvolumes[item]
				update_trafo!(L2G, item)

                boundary_face = itemcells[2, item] == 0
				if AT <: ON_IFACES
					if boundary_face
						continue
					end
				end
				for c1 ∈ 1:2
					cell1 = itemcells[c1, item] # current cell of test function
					if (cell1 > 0)
						QPinfos.cell = cell1 # give cell of input for kernel
						itempos1 = 1
						while !(cellitems[itempos1, cell1] == item)
							itempos1 += 1
						end
						orientation1 = itemorientations[itempos1, cell1]

						## update FE basis evaluators
						for j ∈ 1:ntest
							BE_test[j][itempos1, orientation1].citem[] = cell1
							update_basis!(BE_test[j][itempos1, orientation1])
						end

						## evaluate arguments
						for qp ∈ 1:nweights

							## get global x for quadrature point
							eval_trafo!(QPinfos.x, L2G, xref[qp])

                            # evaluate kernel
                            O.kernel(result_kernel, QPinfos)
                            result_kernel .*= factor * weights[qp] * itemvolumes[item]

                            # multiply test function operator evaluation on cell 1
                            for idt ∈ 1:ntest
                                coeff_test = boundary_face ? 1 : coeffs_ops_test[idt][c1]
                                for k ∈ 1:ndofs_test[idt]
                                    dof = itemdofs_test[idt][k, cell1] + offsets_test[idt]
                                    for d ∈ 1:op_lengths_test[idt]
                                        b[dof] += result_kernel[d+op_offsets_test[idt]] * BE_test_vals[idt][itempos1, orientation1][d, k, qp] * coeff_test
                                    end
                                end
                            end
						end
					end
				end
			end
			return
		end
		O.FES_test = FES_test

		function assembler(b; kwargs...)
			if O.parameters[:store] && size(b) == size(O.storage)
				b .+= O.storage
			else
				if O.parameters[:store]
					s = zeros(eltype(b), length(b))
				else
					s = b
				end
				time = @elapsed begin
					if O.parameters[:parallel_groups]
						Threads.@threads for j ∈ 1:length(EGs)
							fill!(bj[j], 0)
							assembly_loop(bj[j], view(itemassemblygroups, :, j), EGs[j], O.QF[j], O.BE_test[j], O.BE_test_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
						end
						for j ∈ 1:length(EGs)
							s .+= bj[j]
						end
					else
						for j ∈ 1:length(EGs)
							assembly_loop(s, view(itemassemblygroups, :, j), EGs[j], O.QF[j], O.BE_test[j], O.BE_test_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
						end
					end
					if O.parameters[:store]
						b .+= s
						O.storage = s
					end
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

function ExtendableFEM.assemble!(A, b, sol, O::LinearOperatorDG{Tv, UT}, SC::SolverConfiguration; assemble_rhs = true, kwargs...) where {Tv, UT}
	if !assemble_rhs
		return nothing
	end
	if UT <: Integer
		ind_test = O.u_test
		ind_args = O.u_args
	elseif UT <: Unknown
		ind_test = [get_unknown_id(SC, u) for u in O.u_test]
		ind_args = [findfirst(==(u), sol.tags) for u in O.u_args] #[get_unknown_id(SC, u) for u in O.u_args]
	end
	if length(O.u_args) > 0
		build_assembler!(b.entries, O, [b[j] for j in ind_test], [sol[j] for j in ind_args]; kwargs...)
		O.assembler(b.entries, [sol[j] for j in ind_args])
	else
		build_assembler!(b.entries, O, [b[j] for j in ind_test]; kwargs...)
		O.assembler(b.entries)
	end
end

function ExtendableFEM.assemble!(b, O::LinearOperatorDG{Tv, UT}, sol = nothing; assemble_rhs = true, kwargs...) where {Tv, UT}
	if !assemble_rhs
		return nothing
	end
	@assert UT <: Integer
	ind_test = O.u_test
	ind_args = O.u_args
	if length(O.u_args) > 0
		build_assembler!(b.entries, O, [b[j] for j in ind_test], [sol[j] for j in ind_args]; kwargs...)
		O.assembler(b.entries, [sol[j] for j in ind_args])
	else
		build_assembler!(b.entries, O, [b[j] for j in ind_test]; kwargs...)
		O.assembler(b.entries)
	end
end
