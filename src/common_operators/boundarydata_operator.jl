
#############################
### INTERPOLATE-DIRICHLET ###
#############################

mutable struct InterpolateBoundaryData{UT,DFT} <: AbstractOperator
    u::UT
    data::DFT
    bdofs::Array{Int,1}
    FES
    bddata
    assembler
    parameters::Dict{Symbol,Any}
end

default_bndop_kwargs()=Dict{Symbol,Tuple{Any,String}}(
    :penalty => (1e30, "penalty for fixed degrees of freedom"),
    :name => ("BoundaryData", "name for operator used in printouts"),
    :bonus_quadorder => (0, "quadrature order shift"),
    :params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
    :mask => ([], "array of zeros/ones to set which components should be set by the operator (only works with componentwise dofs, add a 1 or 0 to mask additional dofs)"),
    :regions => ([], "subset of regions where operator should be assembly only"),
    :verbosity => (0, "verbosity level"),
)

# informs solver when operator needs reassembly
function ExtendableFEM.depends_nonlinearly_on(O::InterpolateBoundaryData, u::Unknown)
    return false
end


# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::InterpolateBoundaryData)
    return O.u
end

function ExtendableFEM.fixed_dofs(O::InterpolateBoundaryData)
    ## assembles operator to full matrix A and b
    return O.bdofs
end

function InterpolateBoundaryData(u, data = nothing; kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_bndop_kwargs())
    _update_params!(parameters, kwargs)
    return InterpolateBoundaryData{typeof(u),typeof(data)}(u, data, zeros(Int,0), nothing, nothing, nothing, parameters)
end

function ExtendableFEM.assemble!(A, b, sol, O::InterpolateBoundaryData{UT}, SC::SolverConfiguration; time = 0, kwargs...) where UT
    if UT <: Integer
        ind = O.u
    elseif UT <: Unknown
        ind = get_unknown_id(SC, O.u)
    end
    offset = SC.offsets[ind]
    FES = sol[ind].FES
    regions = O.parameters[:regions]
    bdofs::Array{Int,1} = O.bdofs
    if O.FES !== FES
        bddata = FEVector(FES)
        bfacedofs::Adjacency{Int32} = sol[ind].FES[ExtendableFEMBase.BFaceDofs]
        bfaceregions = FES.xgrid[BFaceRegions]
        nbfaces = num_sources(bfacedofs)
        ndofs4bface = max_num_targets_per_source(bfacedofs)
        bdofs = []
        for bface = 1 : nbfaces
            if bfaceregions[bface] in regions
                for k = 1 : ndofs4bface
                    dof = bfacedofs[k,bface] + offset
                    push!(bdofs, dof)
                end
            end
        end
        unique!(bdofs)
        O.bdofs = bdofs
        O.bddata = bddata
    end
    time = @elapsed begin
        bddata = O.bddata
        data = O.data
        interpolate!(bddata[1], ON_BFACES, data; time = time, bonus_quadorder = O.parameters[:bonus_quadorder])
        penalty = O.parameters[:penalty]
        AE = A.entries
        BE = b.entries
        SE = sol.entries
        for dof in bdofs
            AE[dof, dof] = penalty
            BE[dof] = penalty * bddata.entries[dof - offset]
            SE[dof] = bddata.entries[dof - offset]
        end
    end
    if O.parameters[:verbosity] > 1
        @info ".... assembly of $(O.parameters[:name]) took $time s"
    end
end


#####################
### HOM-DIRICHLET ###
#####################

mutable struct HomogeneousData{UT,AT} <: AbstractOperator
    u::UT
    bdofs::Array{Int,1}
    FES
    assembler
    parameters::Dict{Symbol,Any}
end

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::HomogeneousData)
    return O.u
end

function ExtendableFEM.fixed_dofs(O::HomogeneousData)
    ## assembles operator to full matrix A and b
    return O.bdofs
end

function HomogeneousData(u; entities = ON_CELLS, kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_bndop_kwargs())
    _update_params!(parameters, kwargs)
    return HomogeneousData{typeof(u),entities}(u, zeros(Int,0), nothing, nothing,parameters)
end

function HomogeneousBoundaryData(u; kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_bndop_kwargs())
    _update_params!(parameters, kwargs)
    return HomogeneousData{typeof(u),ON_BFACES}(u, zeros(Int,0), nothing, nothing,parameters)
end

function ExtendableFEM.assemble!(A, b, sol, O::HomogeneousData{UT,AT}, SC::SolverConfiguration; kwargs...) where {UT,AT}
    if UT <: Integer
        ind = O.u
    elseif UT <: Unknown
        ind = get_unknown_id(SC, O.u)
    end
    offset = SC.offsets[ind]
    FES = sol[ind].FES
    regions = O.parameters[:regions]
    bdofs::Array{Int,1} = O.bdofs
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
            if ndofs > coffsets[end] && length(mask) == length(coffsets)-1
                @warn "$FEType has additional dofs not associated to single components, add a 0 to the mask if these dofs also should be removed"
            end
            for j = 1 : length(mask)
                if j == length(coffsets)
                    if mask[end] == 1
                        for dof = coffsets[end]+1 : ndofs
                            push!(dofmask, dof)
                        end
                    end
                elseif mask[j] == 1
                    for dof = coffsets[j]+1 : coffsets[j+1]
                        push!(dofmask, dof)
                    end
                end
            end
            for item = 1 : nitems
                if itemregions[item] in regions
                    for dof in dofmask
                        append!(bdofs, itemdofs[dof,item])
                    end
                end    
            end 
        else
            for item = 1 : nitems
                if itemregions[item] in regions
                    for k = 1 : ndofs4item
                        dof = itemdofs[k,item] + offset
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
    for dof in bdofs
        AE[dof, dof] = penalty
        BE[dof] = 0
        SE[dof] = 0
    end
    flush!(AE)
end


##################
### FIXED DOFS ###
##################

mutable struct FixDofs{UT, AT, VT} <: AbstractOperator
    u::UT
    dofs::AT
    offset::Int
    vals::VT
    assembler
    parameters::Dict{Symbol,Any}
end

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::FixDofs)
    return O.u
end

function ExtendableFEM.fixed_dofs(O::FixDofs)
    ## assembles operator to full matrix A and b
    return O.dofs .+ O.offset
end

function FixDofs(u; vals = [], dofs =[], kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_bndop_kwargs())
    _update_params!(parameters, kwargs)
    @assert length(dofs) == length(vals)
    return FixDofs{typeof(u),typeof(dofs),typeof(vals)}(u, dofs, 0, vals, nothing, parameters)
end

function ExtendableFEM.assemble!(A, b, sol, O::FixDofs{UT}, SC::SolverConfiguration; kwargs...) where {UT}
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
    for j = 1 : length(dofs)
        dof = dofs[j] + offset
        AE[dof, dof] = penalty
        BE[dof] = penalty * vals[j]
        SE[dof] = vals[j]
    end
    O.offset = offset
end