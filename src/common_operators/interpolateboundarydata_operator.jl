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
    :bonus_quadorder => (0, "additional quadrature order added to the quadorder chosen by the interpolator"),
    :params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
    :regions => ([], "subset of regions where operator should be assembly only"),
    :verbosity => (0, "verbosity level"),
)

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::InterpolateBoundaryData)
    return O.u
end

function ExtendableFEM.fixed_dofs(O::InterpolateBoundaryData)
    ## assembles operator to full matrix A and b
    return O.bdofs
end

# informs solver when operator needs reassembly in a time dependent setting
function ExtendableFEM.is_timedependent(O::InterpolateBoundaryData)
    return O.parameters[:time_dependent]
end

function Base.show(io::IO, O::InterpolateBoundaryData)
    dependencies = dependencies_when_linearized(O)
    print(io, "$(O.parameters[:name])($(ansatz_function(dependencies)))")
    return nothing
end

"""
````
function InterpolateBoundaryData(u, data!::Function; kwargs...)
````

When assembled, the unknown u of the Problem will be penalized
to match the standard interpolation of the provided data! function.
The header of this function needs to be conform to the interface

    data!(result, qpinfo)

where qpinfo allows to access information at the current quadrature point,
e.g. qpinfo.x provides the global coordinates of the quadrature/evaluation point.

Keyword arguments:
$(_myprint(default_bndop_kwargs()))

"""
function InterpolateBoundaryData(u, data = nothing; kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_bndop_kwargs())
    _update_params!(parameters, kwargs)
    return InterpolateBoundaryData{typeof(u),typeof(data)}(u, data, zeros(Int,0), nothing, nothing, nothing, parameters)
end

function ExtendableFEM.assemble!(A, b, sol, O::InterpolateBoundaryData{UT}, SC::SolverConfiguration; time = 0, assemble_matrix = true, assemble_rhs = true, kwargs...) where UT
    if UT <: Integer
        ind = O.u
        inf_sol = ind
    elseif UT <: Unknown
        ind = get_unknown_id(SC, O.u)
        ind_sol = findfirst(==(O.u), sol.tags)
    end
    offset = SC.offsets[ind]
    FES = b[ind].FES
    regions = O.parameters[:regions]
    bdofs::Array{Int,1} = O.bdofs
    if O.FES !== FES
        bddata = FEVector(FES)
        Ti = eltype(FES.xgrid[CellNodes])
        bfacedofs::Adjacency{Ti} = b[ind].FES[ExtendableFEMBase.BFaceDofs]
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
        if assemble_matrix
            for dof in bdofs
                AE[dof, dof] = penalty
            end
            flush!(AE)
        end
        if assemble_rhs
            for dof in bdofs
                BE[dof] = penalty * bddata.entries[dof - offset]
            end
        end
        for dof in bdofs
            sol[ind_sol][dof-offset] = bddata.entries[dof - offset]
        end
    end
    if O.parameters[:verbosity] > 1
        @info ".... assembly of $(O.parameters[:name]) took $time s"
    end
end

