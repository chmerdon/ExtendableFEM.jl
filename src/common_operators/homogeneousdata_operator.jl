mutable struct HomogeneousData{UT, AT} <: AbstractOperator
	u::UT
	bdofs::Array{Int, 1}
	FES::Any
	assembler::Any
	parameters::Dict{Symbol, Any}
end

ExtendableFEM.fixed_dofs(O::HomogeneousData) = O.bdofs
fixed_vals(O::HomogeneousData) = O.parameters[:value]


default_homdata_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:penalty => (1e30, "penalty for fixed degrees of freedom"),
	:name => ("HomogeneousData", "name for operator used in printouts"),
	:value => (0, "constant value of the data"),
	:mask => ([], "array of zeros/ones to set which components should be set by the operator (only works with componentwise dofs, add a 1 or 0 to mask additional dofs)"),
	:regions => ([], "subset of regions where operator should be assembly only"),
	:verbosity => (0, "verbosity level"),
)

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::HomogeneousData)
	return O.u
end

function Base.show(io::IO, O::HomogeneousData)
	dependencies = dependencies_when_linearized(O)
	print(io, "$(O.parameters[:name])($(ansatz_function(dependencies)), regions = $(O.parameters[:regions]))")
	return nothing
end


"""
````
function HomogeneousData(u; entities = ON_CELLS, kwargs...)
````

When assembled, the unknown u of the Problem will be penalized
to zero on the specifies entities and entity regions (via kwargs).

Keyword arguments:
$(_myprint(default_homdata_kwargs()))

"""
function HomogeneousData(u; entities = ON_CELLS, kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_homdata_kwargs())
	_update_params!(parameters, kwargs)
	return HomogeneousData{typeof(u), entities}(u, zeros(Int, 0), nothing, nothing, parameters)
end

"""
````
function HomogeneousBoundaryData(u; entities = ON_BFACES, kwargs...)
````

When assembled, the unknown u of the Problem will be penalized
to zero on the boundary faces and boundary regions (via kwargs).

Keyword arguments:
$(_myprint(default_homdata_kwargs()))

"""
function HomogeneousBoundaryData(u; entities = ON_BFACES, kwargs...)
	return HomogeneousData(u; entities = entities, kwargs...)
end

function assemble!(O::HomogeneousData{UT, AT}, FES = O.FES; offset = 0, kwargs...) where {UT,AT}
	if O.FES !== FES
		regions = O.parameters[:regions]
		xgrid = FES.dofgrid
		if AT <: ON_BFACES
			itemdofs = FES[ExtendableFEMBase.BFaceDofs]
			itemregions = xgrid[BFaceRegions]
			uniquegeometries = xgrid[UniqueBFaceGeometries]
		elseif AT <: ON_CELLS
			itemdofs = FES[ExtendableFEMBase.CellDofs]
			itemregions = xgrid[CellRegions]
			uniquegeometries = xgrid[UniqueCellGeometries]
		elseif AT <: ON_FACES
			itemdofs = FES[ExtendableFEMBase.FaceDofs]
			itemregions = xgrid[FaceRegions]
			uniquegeometries = xgrid[UniqueFaceGeometries]
		end
		nitems = num_sources(itemdofs)
		ndofs4item = max_num_targets_per_source(itemdofs)
		mask = O.parameters[:mask]
		bdofs = []
		if any(mask .== 0)
			# only some components are Dirichlet
			FEType = get_FEType(FES)
			ncomponents = get_ncomponents(FEType)
			@assert ncomponents <= length(mask) "mask needs to have an entry for each component"
			@assert FEType <: AbstractH1FiniteElement "masks are only allowed for H1FiniteElements"
			@assert length(uniquegeometries) == 1 "masks only work for single geometries for $AT"
			EG = uniquegeometries[1]
			coffsets = ExtendableFEMBase.get_local_coffsets(FEType, AT, EG)
			ndofs = get_ndofs(AT, FEType, EG)
			dofmask = []
			if ndofs > coffsets[end] && length(mask) == length(coffsets) - 1
				@warn "$FEType has additional dofs not associated to single components, add a 0 to the mask if these dofs also should be removed"
			end
			for j ∈ 1:length(mask)
				if j == length(coffsets)
					if mask[end] == 1
						for dof ∈ coffsets[end]+1:ndofs
							push!(dofmask, dof)
						end
					end
				elseif mask[j] == 1
					for dof ∈ coffsets[j]+1:coffsets[j+1]
						push!(dofmask, dof)
					end
				end
			end
			for item ∈ 1:nitems
				if itemregions[item] in regions
					for dof in dofmask
						append!(bdofs, itemdofs[dof, item] + offset)
					end
				end
			end
		else
			for item ∈ 1:nitems
				if itemregions[item] in regions
					ndofs4item = num_targets(itemdofs, item)
					for k ∈ 1:ndofs4item
						dof = itemdofs[k, item]
						push!(bdofs, dof + offset)
					end
				end
			end
		end
		if O.parameters[:verbosity] > 0
			@info ".... $(O.parameters[:name]) penalizes $(length(bdofs)) dofs of '$(O.u.name)' ($AT)"
		end
		O.bdofs = bdofs
		O.FES = FES
	end
end

function ExtendableFEM.assemble!(A, b, sol, O::HomogeneousData{UT, AT}, SC::SolverConfiguration; kwargs...) where {UT, AT}
	if UT <: Integer
		ind = O.u
	elseif UT <: Unknown
		ind = get_unknown_id(SC, O.u)
	end
	FES = sol[ind].FES
	assemble!(O, FES; offset = SC.offsets[ind], kwargs...)
end

function apply!(U::FEVectorBlock, O::HomogeneousData; offset = 0, kwargs...)
	bdofs = O.bdofs
	value = O.parameters[:value]
	UE = U.entries
	UE[bdofs] .= value
end


function ExtendableFEM.apply_penalties!(A, b, sol, O::HomogeneousData{UT}, SC::SolverConfiguration; assemble_matrix = true, assemble_rhs = true, assemble_sol = true, kwargs...) where {UT}
	time = @elapsed begin
		if UT <: Integer
			ind = O.u
			ind_sol = ind
		elseif UT <: Unknown
			ind = get_unknown_id(SC, O.u)
			ind_sol = findfirst(==(O.u), sol.tags)
		end
		bdofs = O.bdofs
		penalty = O.parameters[:penalty]
		value = O.parameters[:value]

		if assemble_matrix
			penalty = O.parameters[:penalty]
			AE = A.entries
			for dof in bdofs
				AE[dof, dof] = penalty
			end
			flush!(AE)
		end
		if assemble_rhs
			BE = b.entries
			BE[bdofs] .= value * penalty
		end
		if assemble_sol
			SE = sol.entries
			SE[bdofs] .= value
		end
	end
	if O.parameters[:verbosity] > 1
		@info ".... applying penalties of $(O.parameters[:name]) took $time s"
	end
end