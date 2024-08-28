
mutable struct ItemIntegratorDG{Tv <: Real, UT <: Union{Unknown, Integer}, KFT <: Function} <: AbstractOperator
	u_args::Array{UT, 1}
	ops_args::Array{DataType, 1}
	kernel::KFT
	BE_args_vals::Array{Vector{Matrix{Array{Tv, 3}}}}
	FES_args::Any             #::Array{FESpace,1}
	BE_args::Any              #::Union{Nothing, Array{FEEvaluator,1}}
	QP_infos::Any             #::Array{QPInfosT,1}
	L2G::Any
	QF::Any
	assembler::Any
	parameters::Dict{Symbol, Any}
end

default_iiopdg_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:entities => (ON_FACES, "assemble operator on these grid entities (default = ON_CELLS)"),
	:name => ("ItemIntegratorDG", "name for operator used in printouts"),
	:resultdim => (0, "dimension of result field (default = length of arguments)"),
	:params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
	:factor => (1, "factor that should be multiplied during assembly"),
	:piecewise => (true, "returns piecewise integrations, otherwise a global integration"),
	:quadorder => ("auto", "quadrature order"),
	:bonus_quadorder => (0, "additional quadrature order added to quadorder"),
	:verbosity => (0, "verbosity level"),
	:regions => ([], "subset of regions where the item integrator should be evaluated"),
)

# informs solver when operator needs reassembly
function depends_nonlinearly_on(O::ItemIntegratorDG)
	return unique(O.u_args)
end

# informs solver when operator needs reassembly in a time dependent setting
function is_timedependent(O::ItemIntegratorDG)
	return O.parameters[:time_dependent]
end

function Base.show(io::IO, O::ItemIntegratorDG)
	dependencies = dependencies_when_linearized(O)
	print(io, "$(O.parameters[:name])($([test_function(dependencies[1][j]) for j = 1 : length(dependencies[1])]))")
	return nothing
end

function ItemIntegratorDG(kernel::Function, u_args, ops_args; Tv = Float64, kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_iiopdg_kwargs())
	_update_params!(parameters, kwargs)
	@assert length(u_args) == length(ops_args)
	return ItemIntegratorDG{Tv, typeof(u_args[1]), typeof(kernel)}(
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
		parameters,
	)
end

"""
````
function ItemIntegratorDG(
	kernel::Function,
	oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1};
	kwargs...)
````

Generates an ItemIntegrator that evaluates the specified
(discontinuous) operator evaluations,
puts it into the kernel function
and integrates the results over the entities (see kwargs)
along cell boundaries. If no kernel is given, the arguments
are integrated directly. If a kernel is provided it has be conform
to the interface

	kernel!(result, eval_args, qpinfo)

where qpinfo allows to access information at the current quadrature point.
Additionally the length of the result needs to be specified via the kwargs.

Evaluation can be triggered via the evaluate function.

Operator evaluations are tuples that pair an unknown identifier or integer
with a Function operator.

Keyword arguments:
$(_myprint(default_iiopdg_kwargs()))

"""
function ItemIntegratorDG(kernel::Function, oa_args::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
	u_args = [oa[1] for oa in oa_args]
	ops_args = [oa[2] for oa in oa_args]
	return ItemIntegratorDG(kernel, u_args, ops_args; kwargs...)
end

function build_assembler!(O::ItemIntegratorDG{Tv}, FE_args::Array{<:FEVectorBlock, 1}; time = 0.0, kwargs...) where {Tv}
	## check if FES is the same as last time
	FES_args = [FE_args[j].FES for j ∈ 1:length(FE_args)]
	if O.FES_args != FES_args

		if O.parameters[:verbosity] > 0
			@info ".... building assembler for $(O.parameters[:name])"
		end

		## determine grid
		xgrid = determine_assembly_grid(FES_args)

		## prepare assembly
		AT = O.parameters[:entities]
		@assert AT <: ON_FACES  || AT <: ON_BFACES "only works for entities <: ON_FACES or ON_BFACES"
        if AT <: ON_BFACES
            AT = ON_FACES
            bfaces = xgrid[BFaceFaces]
            itemassemblygroups = zeros(Int, length(bfaces), 1)
            itemassemblygroups[:] .= bfaces
            gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[1]), AT)
        else
            gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[1]), AT)
            itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(gridAT)]
        end
		Ti = typeof(xgrid).parameters[2]
		itemgeometries = xgrid[GridComponentGeometries4AssemblyType(gridAT)]
		itemvolumes = xgrid[GridComponentVolumes4AssemblyType(gridAT)]
		itemregions = xgrid[GridComponentRegions4AssemblyType(gridAT)]
		FETypes_args = [eltype(F) for F in FES_args]
		EGs = xgrid[UniqueCellGeometries]

		coeffs_ops_args = Array{Array{Float64, 1}, 1}([])
		for op in O.ops_args
			push!(coeffs_ops_args, coeffs(op))
		end

		## prepare assembly
		nargs = length(FES_args)
		O.QF = []
		O.BE_args = Array{Vector{Matrix{<:FEEvaluator}}, 1}(undef, 0)
		O.BE_args_vals = Array{Array{Array{Tv, 3}, 1}, 1}([])
		O.QP_infos = Array{QPInfos, 1}([])
		O.L2G = []
		for EG in EGs
			## quadrature formula for EG
			polyorder_args = maximum([get_polynomialorder(FETypes_args[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_args[j]) for j ∈ 1:nargs])
			if O.parameters[:quadorder] == "auto"
				quadorder = polyorder_args + O.parameters[:bonus_quadorder]
			else
				quadorder = O.parameters[:quadorder] + O.parameters[:bonus_quadorder]
			end
			if O.parameters[:verbosity] > 1
				@info "...... integrating on $EG with quadrature order $quadorder"
			end

			## generate DG operator
			push!(O.BE_args, [generate_DG_operators(StandardFunctionOperator(O.ops_args[j]), FES_args[j], quadorder, EG) for j ∈ 1:nargs])
			push!(O.QF, generate_DG_master_quadrule(quadorder, EG))

			## L2G map for EG
			EGface = facetype_of_cellface(EG, 1)
			push!(O.L2G, L2GTransformer(EGface, xgrid, gridAT))

			## FE basis evaluator for EG
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
		op_lengths_args = [size(O.BE_args[1][j][1, 1].cvals, 1) for j ∈ 1:nargs]
		piecewise = O.parameters[:piecewise]

		op_offsets_args = [0]
		append!(op_offsets_args, cumsum(op_lengths_args))
		offsets_args = [FE_args[j].offset for j in 1:length(FES_args)]	
		resultdim::Int = O.parameters[:resultdim]
		if resultdim == 0
			resultdim = op_offsets_args[end]
			O.parameters[:resultdim] = resultdim
		end

		FEATs_args = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[j]), ON_CELLS) for j ∈ 1:nargs]
		itemdofs_args::Array{Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}}, 1} = [get_dofmap(FES_args[j], xgrid, FEATs_args[j]) for j = 1 : nargs]
		factor = O.parameters[:factor]

		## Assembly loop for fixed geometry
		function assembly_loop(
			b::AbstractMatrix{T},
			sol::Array{<:FEVectorBlock, 1},
			items,
			EG::ElementGeometries,
			QF::QuadratureRule,
			BE_args::Vector{Matrix{<:FEEvaluator}},
			L2G::L2GTransformer,
			QPinfos::QPInfos,
		) where {T}

			input_args = zeros(T, op_offsets_args[end])
			result_kernel = zeros(Tv, resultdim)
			itemorientations = xgrid[CellFaceOrientations]
			itemcells = xgrid[FaceCells]
			itemnormals = xgrid[FaceNormals]
			cellitems = xgrid[CellFaces]

			ndofs_args::Array{Int, 1} = [size(BE[1, 1].cvals, 2) for BE in BE_args]

			weights, xref = QF.w, QF.xref
			nweights = length(weights)
			cell1::Int = 0
			orientation1::Int = 0
			itempos1::Int = 0

			for item::Int in items
				QPinfos.region = itemregions[item]
				QPinfos.item = item
				QPinfos.normal .= view(itemnormals, :, item)
				QPinfos.volume = itemvolumes[item]
				update_trafo!(L2G, item)

                boundary_face = itemcells[2, item] == 0
				if AT <: ON_IFACES
					if boundary_face
						continue
					end
				end

				for qp ∈ 1:nweights
                    ## evaluate arguments at quadrature points
					fill!(input_args, 0)
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
									dof_j = itemdofs_args[id][j, cell1]
									for d ∈ 1:op_lengths_args[id]
										input_args[d+op_offsets_args[id]] += sol[id][dof_j] * BE_args[id][itempos1, orientation1].cvals[d, j, qp] * coeffs_ops_args[id][c1]
									end
								end
							end
						end
					end

                    ## get global x for quadrature point
                    eval_trafo!(QPinfos.x, L2G, xref[qp])

                    # evaluate kernel
                    O.kernel(result_kernel, input_args, QPinfos)
                    result_kernel .*= factor * weights[qp] * itemvolumes[item]

					# integrate over item
					if piecewise
						b[1:resultdim, item] .+= result_kernel
					else
						b .+= result_kernel
					end
				end

			end
			return
		end
		O.FES_args = FES_args

		function assembler(b, sol; kwargs...)
			time_assembly = @elapsed begin
				for j ∈ 1:length(EGs)
					assembly_loop(b, sol, view(itemassemblygroups, :, j), EGs[j], O.QF[j], O.BE_args[j], O.L2G[j], O.QP_infos[j]; kwargs...)
				end
			end
			if O.parameters[:verbosity] > 1
				@info ".... assembly of $(O.parameters[:name]) took $time_assembly s"
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

"""
````
function evaluate(
	b::AbstractMatrix,
	O::ItemIntegratorDG,
	sol::FEVector;
	time = 0,
	kwargs...)
````

Evaluates the ItemIntegratorDG for the specified solution into the matrix b.
"""
function ExtendableFEMBase.evaluate!(b, O::ItemIntegratorDG, sol::FEVector; kwargs...)
	ind_args = [findfirst(==(u), sol.tags) for u in O.u_args]
	build_assembler!(O, [sol[j] for j in ind_args]; kwargs...)
	O.assembler(b, [sol[j] for j in ind_args])
end

"""
````
function evaluate(
	b::AbstractMatrix,
	O::ItemIntegratorDG,
	sol::Array{FEVEctorBlock};
	time = 0,
	kwargs...)
````

Evaluates the ItemIntegratorDG for the specified solution into the matrix b.
"""
function ExtendableFEMBase.evaluate!(b, O::ItemIntegratorDG, sol::Array{<:FEVectorBlock, 1}; kwargs...)
	ind_args = O.u_args
	build_assembler!(O, [sol[j] for j in ind_args]; kwargs...)
	O.assembler(b, [sol[j] for j in ind_args])
end

"""
````
function evaluate(
	O::ItemIntegratorDG,
	sol;
	time = 0,
	kwargs...)
````

Evaluates the ItemIntegratorDG for the specified solution and returns an matrix of size resultdim x num_items.
"""
function evaluate(O::ItemIntegratorDG{Tv, UT}, sol; kwargs...) where {Tv, UT}
	if UT <: Integer
		ind_args = O.u_args
	elseif UT <: Unknown
		ind_args = [findfirst(==(u), sol.tags) for u in O.u_args]
	end
	build_assembler!(O, [sol[j] for j in ind_args]; kwargs...)
	grid = sol[ind_args[1]].FES.xgrid
	AT = O.parameters[:entities]
	if AT <: ON_CELLS
		nitems = num_cells(grid)
	elseif AT <: ON_FACES
		nitems = size(grid[FaceNodes], 2)
	elseif AT <: ON_EDGES
		nitems = size(grid[EdgeNodes], 2)
	elseif AT <: ON_BFACES
		nitems = size(grid[BFaceNodes], 2)
	elseif AT <: ON_BEDGES
		nitems = size(grid[BEdgeNodes], 2)
	end
	if O.parameters[:piecewise]
		b = zeros(eltype(sol[1].entries), O.parameters[:resultdim], nitems)
	else
		b = zeros(eltype(sol[1].entries), O.parameters[:resultdim])
	end
	O.assembler(b, [sol[j] for j in ind_args])
	return b
end