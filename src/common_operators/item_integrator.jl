mutable struct ItemIntegrator{Tv <: Real, UT <: Union{Unknown, Integer}, KFT <: Function}
    u_args::Array{UT,1}
    ops_args::Array{DataType,1}
    kernel::KFT
    FES_args             #::Array{FESpace,1}
    BE_args              #::Union{Nothing, Array{FEEvaluator,1}}
    QP_infos             #::Array{QPInfosT,1}
    L2G             
    QF
    assembler
    parameters::Dict{Symbol,Any}
end

default_iiop_kwargs()=Dict{Symbol,Tuple{Any,String}}(
    :entities => (ON_CELLS, "assemble operator on these grid entities (default = ON_CELLS)"),
    :name => ("ItemIntegrator", "name for operator used in printouts"),
    :resultdim => (0, "dimension of result field (default = length of arguments)"),
    :params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
    :factor => (1, "factor that should be multiplied during assembly"),
    :quadorder => ("auto", "quadrature order"),
    :bonus_quadorder => (0, "additional quadrature order added to quadorder"),
    :verbosity => (0, "verbosity level"),
    :regions => ([], "subset of regions where the item integrator should be evaluated")
)

function l2norm_kernel(result, input, qpinfo)
    result .= input.^2
end

function ItemIntegrator(kernel, u_args, ops_args; Tv = Float64, kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_iiop_kwargs())
    _update_params!(parameters, kwargs)
    @assert length(u_args) == length(ops_args)
    return ItemIntegrator{Tv, typeof(u_args[1]), typeof(kernel)}(u_args, ops_args, kernel, nothing, nothing, nothing, nothing, nothing, nothing, parameters)
end

"""
````
function ItemIntegrator(
    [kernel!::Function],
    oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1};
    kwargs...)
````

Generates an ItemIntegrator that evaluates the specified operator evaluations,
puts it into the kernel function
and integrates the results over the entities (see kwargs). If no kernel is given, the arguments
are integrated directly. If a kernel is provided it has be conform
to the interface

    kernel!(result, eval_args, qpinfo)

where qpinfo allows to access information at the current quadrature point.
Additionally the length of the result needs to be specified via the kwargs.

Evaluation can be triggered via the evaluate function.

Operator evaluations are tuples that pair an unknown identifier or integer
with a Function operator.

Keyword arguments:
$(_myprint(default_iiop_kwargs()))
"""
function ItemIntegrator(kernel, oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1}; kwargs...)
    u_args = [oa[1] for oa in oa_args]
    ops_args = [oa[2] for oa in oa_args]
    return ItemIntegrator(kernel, u_args, ops_args; kwargs...)
end

function ItemIntegrator(oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1}; kwargs...)
    u_args = [oa[1] for oa in oa_args]
    ops_args = [oa[2] for oa in oa_args]
    return ItemIntegrator(ExtendableFEMBase.standard_kernel, u_args, ops_args; kwargs...)
end
function L2NormIntegrator(oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1}; kwargs...)
    u_args = [oa[1] for oa in oa_args]
    ops_args = [oa[2] for oa in oa_args]
    return ItemIntegrator(l2norm_kernel, u_args, ops_args; kwargs...)
end

function build_assembler!(O::ItemIntegrator{Tv}, FE_args::Array{<:FEVectorBlock,1}; time = 0.0) where {Tv}
    ## check if FES is the same as last time
    FES_args = [FE_args[j].FES for j = 1 : length(FE_args)]
    if (O.FES_args != FES_args)

        if O.parameters[:verbosity] > 0
            @info ".... building assembler for $(O.parameters[:name])"
        end

        ## prepare assembly
        AT = O.parameters[:entities]
        gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[1]), AT)
        xgrid = FES_args[1].xgrid
        itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(gridAT)]
        itemgeometries = xgrid[GridComponentGeometries4AssemblyType(gridAT)]
        itemvolumes = xgrid[GridComponentVolumes4AssemblyType(gridAT)]
        itemregions = xgrid[GridComponentRegions4AssemblyType(gridAT)]
        FETypes_args = [eltype(F) for F in FES_args]
        EGs = [itemgeometries[itemassemblygroups[1,j]] for j = 1 : num_sources(itemassemblygroups)]

        ## prepare assembly
        nargs = length(FES_args)
        O.QF = []
        O.BE_args = Array{Array{<:FEEvaluator,1},1}([])
        O.QP_infos = Array{QPInfos,1}([])
        O.L2G = []
        for EG in EGs
            ## quadrature formula for EG
            polyorder_args = maximum([get_polynomialorder(FETypes_args[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_args[j]) for j = 1 : nargs])
            if O.parameters[:quadorder] == "auto"
                quadorder = polyorder_args + O.parameters[:bonus_quadorder]
            else
                quadorder = O.parameters[:quadorder] + O.parameters[:bonus_quadorder]
            end
            if O.parameters[:verbosity] > 1
                @info "...... integrating on $EG with quadrature order $quadorder"
            end
            push!(O.QF, QuadratureRule{Tv, EG}(quadorder))
        
            ## FE basis evaluator for EG
            push!(O.BE_args, [FEEvaluator(FES_args[j], O.ops_args[j], O.QF[end]; AT = AT) for j in 1 : nargs])

            ## L2G map for EG
            push!(O.L2G, L2GTransformer(EG, xgrid, gridAT))

            ## parameter structure
            push!(O.QP_infos, QPInfos(xgrid; time = time, params = O.parameters[:params]))
        end

        ## prepare regions
        regions = O.parameters[:regions]
        visit_region = zeros(Bool, maximum(itemregions))
        if length(regions) > 0
            visit_region[O.regions] = true
        else
            visit_region .= true
        end

        ## prepare parameters

        ## prepare operator infos
        op_lengths_args = [size(O.BE_args[1][j].cvals,1) for j = 1 : nargs]
        
        op_offsets_args = [0]
        append!(op_offsets_args, cumsum(op_lengths_args))
        resultdim::Int = O.parameters[:resultdim]
        if resultdim == 0
            resultdim = op_offsets_args[end]
            O.parameters[:resultdim] = resultdim
        end

        FEATs_args = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[j]), AT) for j = 1 : nargs]
        itemdofs_args::Array{Union{Adjacency{Int32}, SerialVariableTargetAdjacency{Int32}},1} = [FES_args[j][Dofmap4AssemblyType(FEATs_args[j])] for j = 1 : nargs]
        factor = O.parameters[:factor]

        ## Assembly loop for fixed geometry
        function assembly_loop(b::AbstractMatrix{T}, sol::Array{<:FEVectorBlock,1}, items, EG::ElementGeometries, QF::QuadratureRule, BE_args::Array{<:FEEvaluator,1}, L2G::L2GTransformer, QPinfos::QPInfos; time = 0) where {T}

            ## prepare parameters
            result_kernel = zeros(Tv, resultdim)
            input_args = zeros(Tv, op_offsets_args[end])
            ndofs_args::Array{Int,1} = [size(BE.cvals,2) for BE in BE_args]
            weights, xref = QF.w, QF.xref
            nweights = length(weights)
            QPinfos.time = time

            for item::Int in items
                if itemregions[item] > 0
                    if !(visit_region[itemregions[item]])
                        continue
                    end
                end
                QPinfos.region = itemregions[item]
                QPinfos.item = item
                QPinfos.volume = itemvolumes[item]

                ## update FE basis evaluators
                for j = 1 : nargs
                    BE_args[j].citem[] = item
                    update_basis!(BE_args[j]) 
                end
	            update_trafo!(L2G, item)

                ## evaluate arguments
				for qp = 1 : nweights
					fill!(input_args,0)
                    for id = 1 : nargs
                        for j = 1 : ndofs_args[id]
                            dof_j = itemdofs_args[id][j, item]
                            for d = 1 : op_lengths_args[id]
                                input_args[d + op_offsets_args[id]] += sol[id][dof_j] * BE_args[id].cvals[d, j, qp]
                            end
                        end
					end
                
                    ## get global x for quadrature point
                    eval_trafo!(QPinfos.x, L2G, xref[qp])

                    # evaluate kernel
                    O.kernel(result_kernel, input_args, QPinfos)
                    result_kernel .*= factor * weights[qp] * itemvolumes[item]

                    # integrate over item
                    for d = 1 : resultdim
                        b[d, item] += result_kernel[d]
                    end
                end
            end
            return
        end
        O.FES_args = FES_args

        function assembler(b, sol; kwargs...)
            time_assembly = @elapsed begin
                for j = 1 : length(EGs)
                    assembly_loop(b, sol, view(itemassemblygroups,:,j), EGs[j], O.QF[j], O.BE_args[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                end   
            end
            if O.parameters[:verbosity] > 1
                @info ".... assembly of $(O.parameters[:name]) took $time_assembly s"
            end
        end
        O.assembler = assembler
    end
end


"""
````
function evaluate(
    b::AbstractMatrix,
    O::ItemIntegrator,
    sol::FEVector;
    time = 0,
    kwargs...)
````

Evaluates the ItemIntegrator for the specified solution into the matrix b.
"""
function ExtendableFEMBase.evaluate!(b, O::ItemIntegrator, sol::FEVector; time = 0, kwargs...)
    ind_args = [findfirst(==(u), sol.tags) for u in O.u_args]
    build_assembler!(O, [sol[j] for j in ind_args])
    O.assembler(b, [sol[j] for j in ind_args]; time = time)
end

"""
````
function evaluate(
    b::AbstractMatrix,
    O::ItemIntegrator,
    sol::Array{FEVEctorBlock};
    time = 0,
    kwargs...)
````

Evaluates the ItemIntegrator for the specified solution into the matrix b.
"""
function ExtendableFEMBase.evaluate!(b, O::ItemIntegrator, sol::Array{<:FEVectorBlock,1}; time = 0, kwargs...)
    ind_args = O.u_args
    build_assembler!(O, [sol[j] for j in ind_args])
    O.assembler(b, [sol[j] for j in ind_args]; time = time)
end

"""
````
function evaluate(
    O::ItemIntegrator,
    sol;
    time = 0,
    kwargs...)
````

Evaluates the ItemIntegrator for the specified solution and returns an matrix of size resultdim x num_items.
"""
function evaluate(O::ItemIntegrator{Tv,UT}, sol; time = 0, kwargs...) where {Tv,UT}
    if UT <: Integer
        ind_args = O.u_args
    elseif UT <: Unknown
        ind_args = [findfirst(==(u), sol.tags) for u in O.u_args]
    end
    build_assembler!(O, [sol[j] for j in ind_args])
    grid = sol[ind_args[1]].FES.xgrid
    AT = O.parameters[:entities]
    if AT <: ON_CELLS
        nitems = num_cells(grid)
    elseif AT <: ON_FACES
        nitems = size(grid[FaceNodes],2)
    elseif AT <: ON_EDGES
        nitems = size(grid[EdgeNodes],2)
    elseif AT <: ON_BFACES
        nitems = size(grid[BFaceNodes],2)
    elseif AT <: ON_BEDGES
        nitems = size(grid[BEdgeNodes],2)
    end
    b = zeros(eltype(sol[1].entries), O.parameters[:resultdim], nitems)
    O.assembler(b, [sol[j] for j in ind_args]; time = time)
    return b
end