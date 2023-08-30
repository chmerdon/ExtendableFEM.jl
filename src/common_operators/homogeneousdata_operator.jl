mutable struct HomogeneousData{UT, AT} <: AbstractOperator
	u::UT
	bdofs::Array{Int, 1}
	FES::Any
	assembler::Any
	parameters::Dict{Symbol, Any}
end


default_homdata_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:penalty => (1e30, "penalty for fixed degrees of freedom"),
	:name => ("HomogeneousData", "name for operator used in printouts"),
	:mask => ([], "array of zeros/ones to set which components should be set by the operator (only works with componentwise dofs, add a 1 or 0 to mask additional dofs)"),
	:regions => ([], "subset of regions where operator should be assembly only"),
	:verbosity => (0, "verbosity level"),
)

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::HomogeneousData)
	return O.u
end

function ExtendableFEM.fixed_dofs(O::HomogeneousData)
	## assembles operator to full matrix A and b
	return O.bdofs
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

function ExtendableFEM.assemble!(A, b, sol, O::HomogeneousData{UT, AT}, SC::SolverConfiguration; assemble_matrix = true, assemble_rhs = true, kwargs...) where {UT, AT}
	if UT <: Integer
		ind = O.u
	elseif UT <: Unknown
		ind = get_unknown_id(SC, O.u)
	end
	offset = SC.offsets[ind]
	FES = sol[ind].FES
	regions = O.parameters[:regions]
	bdofs::Array{Int, 1} = O.bdofs
	if O.FES !== FES
		offset = SC.offsets[ind]
		if AT <: ON_BFACES
			itemdofs = sol[ind].FES[ExtendableFEMBase.BFaceDofs]
			itemregions = FES.xgrid[BFaceRegions]
			uniquegeometries = FES.xgrid[UniqueBFaceGeometries]
		elseif AT <: ON_CELLS
			itemdofs = sol[ind].FES[ExtendableFEMBase.CellDofs]
			itemregions = FES.xgrid[CellRegions]
			uniquegeometries = FES.xgrid[UniqueCellGeometries]
		elseif AT <: ON_FACES
			itemdofs = sol[ind].FES[ExtendableFEMBase.FaceDofs]
			itemregions = FES.xgrid[FaceRegions]
			uniquegeometries = FES.xgrid[UniqueFaceGeometries]
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
						append!(bdofs, itemdofs[dof, item])
					end
				end
			end
		else
			for item ∈ 1:nitems
				if itemregions[item] in regions
					for k ∈ 1:ndofs4item
						dof = itemdofs[k, item] + offset
						push!(bdofs, dof)
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
	penalty = O.parameters[:penalty]
	AE = A.entries
	BE = b.entries
	SE = sol.entries
	if assemble_matrix
		for dof in bdofs
			AE[dof, dof] = penalty
		end
		flush!(AE)
	end
	if assemble_rhs
		BE[bdofs] .= 0
	end
	SE[bdofs] .= 0
end
