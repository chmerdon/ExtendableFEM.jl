mutable struct LinearOperatorFromVector{UT <: Union{Unknown, Integer}, bT} <: AbstractOperator
    u_test::Array{UT, 1}
    b::bT
    parameters::Dict{Symbol, Any}
end

mutable struct LinearOperatorFromMatrix{UT <: Union{Unknown, Integer}, MT} <: AbstractOperator
    u_test::Array{UT, 1}
    u_args::Array{UT, 1}
    A::MT
    parameters::Dict{Symbol, Any}
end

mutable struct LinearOperator{Tv <: Real, UT <: Union{Unknown, Integer}, KFT, ST} <: AbstractOperator
    u_test::Array{UT, 1}
    ops_test::Array{DataType, 1}
    u_args::Array{UT, 1}
    ops_args::Array{DataType, 1}
    kernel::KFT
    BE_test_vals::Array{Array{Array{Tv, 3}, 1}}
    BE_args_vals::Array{Array{Array{Tv, 3}, 1}}
    FES_test::Any             #::Array{FESpace,1}
    FES_args::Any             #::Array{FESpace,1}
    BE_test::Any              #::Union{Nothing, Array{FEEvaluator,1}}
    BE_args::Any              #::Union{Nothing, Array{FEEvaluator,1}}
    QP_infos::Any             #::Array{QPInfosT,1}
    L2G::Any
    QF::Any
    assembler::Any
    storage::ST
    parameters::Dict{Symbol, Any}
end

default_linop_kwargs() = Dict{Symbol, Tuple{Any, String}}(
    :entities => (ON_CELLS, "assemble operator on these grid entities (default = ON_CELLS)"),
    :name => ("LinearOperator", "name for operator used in printouts"),
    :parallel_groups => (false, "assemble operator in parallel using CellAssemblyGroups"),
    :parallel => (false, "assemble operator in parallel using colors/partitions information"),
    :params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
    :factor => (1, "factor that should be multiplied during assembly"),
    :store => (false, "store matrix separately (and copy from there when reassembly is triggered)"),
    :quadorder => ("auto", "quadrature order"),
    :bonus_quadorder => (0, "additional quadrature order added to quadorder"),
    :time_dependent => (false, "operator is time-dependent ?"),
    :verbosity => (0, "verbosity level"),
    :regions => ([], "subset of regions where operator should be assembly only"),
)

# informs solver when operator needs reassembly
function depends_nonlinearly_on(O::LinearOperator)
    return unique(O.u_args)
end

# informs solver in which blocks the operator assembles to
function dependencies_when_linearized(O::LinearOperator)
    return [unique(O.u_test)]
end

# informs solver when operator needs reassembly in a time dependent setting
function is_timedependent(O::LinearOperator)
    return O.parameters[:time_dependent]
end

function Base.show(io::IO, O::LinearOperator)
    dependencies = dependencies_when_linearized(O)
    print(io, "$(O.parameters[:name])($([test_function(dependencies[1][j]) for j in 1:length(dependencies[1])]))")
    return nothing
end
function Base.show(io::IO, O::Union{LinearOperatorFromVector, LinearOperatorFromMatrix})
    dependencies = dependencies_when_linearized(O)
    print(io, "$(O.parameters[:name])")
    return nothing
end

function LinearOperator(kernel, u_test, ops_test, u_args, ops_args; Tv = Float64, kwargs...)
    parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_linop_kwargs())
    _update_params!(parameters, kwargs)
    @assert length(u_args) == length(ops_args)
    @assert length(u_test) == length(ops_test)
    if parameters[:store]
        storage = zeros(Tv, 0)
    else
        storage = nothing
    end
    return LinearOperator{Tv, typeof(u_test[1]), typeof(kernel), typeof(storage)}(
        u_test,
        ops_test,
        u_args,
        ops_args,
        kernel,
        [[zeros(Tv, 0, 0, 0)]],
        [[zeros(Tv, 0, 0, 0)]],
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

function LinearOperator(kernel, u_test, ops_test::Array{DataType, 1}; Tv = Float64, kwargs...)
    parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_linop_kwargs())
    _update_params!(parameters, kwargs)
    @assert length(u_test) == length(ops_test)
    if parameters[:store]
        storage = zeros(Tv, 0)
    else
        storage = nothing
    end
    return LinearOperator{Tv, typeof(u_test[1]), typeof(kernel), typeof(storage)}(u_test, ops_test, [], [], kernel, [[zeros(Tv, 0, 0, 0)]], [[zeros(Tv, 0, 0, 0)]], nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, storage, parameters)
end

"""
````
function LinearOperator(
	[kernel!::Function],
	oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1};
	kwargs...)
````

Generates a linear form that evaluates, in each quadrature point,
the kernel function (if non is provided, a constant function one is used)
and computes the vector product of the result with with the operator evaluation(s)
of the test function(s). The header of the kernel functions needs to be conform
to the interface

	kernel!(result, qpinfo)

where qpinfo allows to access information at the current quadrature point,
e.g. qpinfo.x are the global coordinates of the quadrature point.

Operator evaluations are tuples that pair an unknown identifier or integer
with a Function operator.

Example: LinearOperator(kernel!, [id(1)]; kwargs...) generates the right-hand side
for a Poisson problem, where kernel! evaluates the right-hand side.

Keyword arguments:
$(_myprint(default_linop_kwargs()))

"""
function LinearOperator(kernel, oa_test::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
    u_test = [oa[1] for oa in oa_test]
    ops_test = [oa[2] for oa in oa_test]
    return LinearOperator(kernel, u_test, ops_test; kwargs...)
end

function LinearOperator(oa_test::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
    u_test = [oa[1] for oa in oa_test]
    ops_test = [oa[2] for oa in oa_test]
    return LinearOperator(ExtendableFEMBase.constant_one_kernel, u_test, ops_test; kwargs...)
end


"""
````
function LinearOperator(
	b,
	u_test;
	kwargs...)
````

Generates a linear form from a user-provided vector b, which can be an AbstractVector or a FEVector with
multiple blocks. The argument u_test specifies where to put the (blocks of the) vector in the system right-hand side.

"""
function LinearOperator(b, u_test; kwargs...)
    parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_linop_kwargs())
    _update_params!(parameters, kwargs)
    return LinearOperatorFromVector{typeof(u_test[1]), typeof(b)}(u_test, b, parameters)
end

"""
````
function LinearOperator(
	A,
	u_test,
	u_args;
	kwargs...)
````

Generates a linear form from a user-provided matrix A, which can be an AbstractMatrix or a FEMatrix with
multiple blocks. The arguments u_args specify which coefficients of the current solution
should be multiplied with the matrix and u_test specifies where to put the
(blocks of the) resulting vector in the system right-hand side.

"""
function LinearOperator(A::AbstractMatrix, u_test::Array{<:Union{Unknown, Int}, 1}, u_args::Array{<:Union{Unknown, Int}, 1}; kwargs...)
    parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_linop_kwargs())
    _update_params!(parameters, kwargs)
    return LinearOperatorFromMatrix{typeof(u_test[1]), typeof(A)}(u_test, u_args, A, parameters)
end


"""
````
function LinearOperator(
	kernel!::Function,
	oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1},
	oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1};
	kwargs...)
````

Generates a nonlinear linear form that evaluates a kernel function
that depends on the operator evaluations of the current solution. The result of the
kernel function is used in a vector product with the operator evaluation(s)
of the test function(s). Hence, this can be used as a linearization of a
nonlinear operator. The header of the kernel functions needs to be conform
to the interface

	kernel!(result, eval_args, qpinfo)

where qpinfo allows to access information at the current quadrature point.

Operator evaluations are tuples that pair an unknown identifier or integer
with a Function operator.

Keyword arguments:
$(_myprint(default_linop_kwargs()))

"""
function LinearOperator(kernel, oa_test::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}, oa_args::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
    u_test = [oa[1] for oa in oa_test]
    u_args = [oa[1] for oa in oa_args]
    ops_test = [oa[2] for oa in oa_test]
    ops_args = [oa[2] for oa in oa_args]
    return LinearOperator(kernel, u_test, ops_test, u_args, ops_args; kwargs...)
end

function LinearOperator(oa_test::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}, oa_args::Array{<:Tuple{Union{Unknown, Int}, DataType}, 1}; kwargs...)
    u_test = [oa[1] for oa in oa_test]
    u_args = [oa[1] for oa in oa_args]
    ops_test = [oa[2] for oa in oa_test]
    ops_args = [oa[2] for oa in oa_args]
    return LinearOperator(ExtendableFEMBase.standard_kernel, u_test, ops_test, u_args, ops_args; kwargs...)
end

function build_assembler!(b, O::LinearOperator{Tv}, FE_test, FE_args; time = 0.0, kwargs...) where {Tv}
    ## check if FES is the same as last time
    FES_test = [FE_test[j].FES for j in 1:length(FE_test)]
    FES_args = [FE_args[j].FES for j in 1:length(FE_args)]
    return if (O.FES_test != FES_test) || (O.FES_args != FES_args)

        if O.parameters[:verbosity] > 0
            @info "$(O.parameters[:name]) : building assembler"
        end

        ## determine grid
        xgrid = determine_assembly_grid(FES_test, FES_args)
        AT = O.parameters[:entities]
        if xgrid == FES_test[1].dofgrid
            gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[1]), AT)
        else
            gridAT = AT
        end

        ## prepare assembly
        Ti = typeof(xgrid).parameters[2]
        itemgeometries = xgrid[GridComponentGeometries4AssemblyType(gridAT)]
        itemvolumes = xgrid[GridComponentVolumes4AssemblyType(gridAT)]
        itemregions = xgrid[GridComponentRegions4AssemblyType(gridAT)]
        if num_pcolors(xgrid) > 1 && gridAT == ON_CELLS
            maxnpartitions = maximum(num_partitions_per_color(xgrid))
            pc = xgrid[PartitionCells]
            itemassemblygroups = [pc[j]:(pc[j + 1] - 1) for j in 1:num_partitions(xgrid)]
            # assuming here that all cells of one partition have the same geometry
        else
            itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(gridAT)]
            itemassemblygroups = [view(itemassemblygroups, :, j) for j in 1:num_sources(itemassemblygroups)]
        end
        Ti = typeof(xgrid).parameters[2]
        has_normals = true
        if AT <: ON_FACES
            itemnormals = xgrid[FaceNormals]
        elseif AT <: ON_BFACES
            itemnormals = xgrid[FaceNormals][:, xgrid[BFaceFaces]]
        else
            has_normals = false
        end
        FETypes_test = [eltype(F) for F in FES_test]
        FETypes_args = [eltype(F) for F in FES_args]
        EGs = [itemgeometries[itemassemblygroups[j][1]] for j in 1:length(itemassemblygroups)]

        ## prepare assembly
        nargs = length(FES_args)
        ntest = length(FES_test)
        O.QF = []
        O.BE_test = Array{Array{<:FEEvaluator, 1}, 1}([])
        O.BE_args = Array{Array{<:FEEvaluator, 1}, 1}([])
        O.BE_test_vals = Array{Array{Array{Tv, 3}, 1}, 1}([])
        O.BE_args_vals = Array{Array{Array{Tv, 3}, 1}, 1}([])
        O.QP_infos = Array{QPInfos, 1}([])
        O.L2G = []
        for EG in EGs
            ## quadrature formula for EG
            polyorder_args = maximum([get_polynomialorder(FETypes_args[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_args[j]) for j in 1:nargs])
            polyorder_test = maximum([get_polynomialorder(FETypes_test[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_test[j]) for j in 1:ntest])
            if O.parameters[:quadorder] == "auto"
                quadorder = polyorder_args + polyorder_test + O.parameters[:bonus_quadorder]
            else
                quadorder = O.parameters[:quadorder] + O.parameters[:bonus_quadorder]
            end
            if O.parameters[:verbosity] > 1
                @info "...... integrating on $EG with quadrature order $quadorder"
            end
            push!(O.QF, QuadratureRule{Tv, EG}(quadorder))

            ## FE basis evaluator for EG
            push!(O.BE_test, [FEEvaluator(FES_test[j], O.ops_test[j], O.QF[end]; AT = AT) for j in 1:ntest])
            push!(O.BE_args, [FEEvaluator(FES_args[j], O.ops_args[j], O.QF[end]; AT = AT) for j in 1:nargs])
            push!(O.BE_test_vals, [BE.cvals for BE in O.BE_test[end]])
            push!(O.BE_args_vals, [BE.cvals for BE in O.BE_args[end]])

            ## L2G map for EG
            push!(O.L2G, L2GTransformer(EG, xgrid, ON_CELLS))

            ## parameter structure
            push!(O.QP_infos, QPInfos(xgrid; time = time, params = O.parameters[:params]))
        end

        ## prepare regions
        regions = O.parameters[:regions]
        visit_region = zeros(Bool, maximum(itemregions))
        if length(regions) > 0
            visit_region[regions] .= true
        else
            visit_region .= true
        end

        ## prepare parameters

        ## prepare operator infos
        op_lengths_test = [size(O.BE_test[1][j].cvals, 1) for j in 1:ntest]
        op_lengths_args = [size(O.BE_args[1][j].cvals, 1) for j in 1:nargs]

        op_offsets_test = [0]
        op_offsets_args = [0]
        append!(op_offsets_test, cumsum(op_lengths_test))
        append!(op_offsets_args, cumsum(op_lengths_args))

        ## prepare parallel assembly
        if O.parameters[:parallel_groups]
            bj = Array{typeof(b), 1}(undef, length(EGs))
            for j in 1:length(EGs)
                bj[j] = copy(b)
            end
        end

        FEATs_test = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[j]), AT) for j in 1:ntest]
        FEATs_args = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[j]), AT) for j in 1:nargs]
        itemdofs_test::Array{Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}}, 1} = [get_dofmap(FES_test[j], xgrid, FEATs_test[j]) for j in 1:ntest]
        itemdofs_args::Array{Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}}, 1} = [get_dofmap(FES_args[j], xgrid, FEATs_args[j]) for j in 1:nargs]
        factor = O.parameters[:factor]

        ## Assembly loop for fixed geometry
        function assembly_loop(
                b::AbstractVector{T},
                sol::Array{<:FEVectorBlock, 1},
                items,
                EG::ElementGeometries,
                QF::QuadratureRule,
                BE_test::Array{<:FEEvaluator, 1},
                BE_args::Array{<:FEEvaluator, 1},
                BE_test_vals::Array{Array{Tv, 3}, 1},
                BE_args_vals::Array{Array{Tv, 3}, 1},
                L2G::L2GTransformer,
                QPinfos::QPInfos,
            ) where {T}

            ## prepare parameters
            input_args = zeros(Tv, op_offsets_args[end])
            result_kernel = zeros(Tv, op_offsets_test[end])
            offsets_test = [FE_test[j].offset for j in 1:length(FES_test)]

            ndofs_test::Array{Int, 1} = [get_ndofs(AT, FE, EG) for FE in FETypes_test]
            ndofs_args::Array{Int, 1} = [get_ndofs(AT, FE, EG) for FE in FETypes_args]
            weights, xref = QF.w, QF.xref
            nweights = length(weights)

            for item::Int in items
                if itemregions[item] > 0
                    if !(visit_region[itemregions[item]]) || AT == ON_IFACES
                        continue
                    end
                end
                QPinfos.region = itemregions[item]
                QPinfos.item = item
                if has_normals
                    QPinfos.normal .= view(itemnormals, :, item)
                end
                QPinfos.volume = itemvolumes[item]

                ## update FE basis evaluators
                for j in 1:ntest
                    BE_test[j].citem[] = item
                    update_basis!(BE_test[j])
                end
                for j in 1:nargs
                    BE_args[j].citem[] = item
                    update_basis!(BE_args[j])
                end
                update_trafo!(L2G, item)

                ## evaluate arguments
                for qp in 1:nweights
                    fill!(input_args, 0)
                    for id in 1:nargs
                        for j in 1:ndofs_args[id]
                            dof_j = itemdofs_args[id][j, item]
                            for d in 1:op_lengths_args[id]
                                input_args[d + op_offsets_args[id]] += sol[id][dof_j] * BE_args_vals[id][d, j, qp]
                            end
                        end
                    end

                    ## get global x for quadrature point
                    eval_trafo!(QPinfos.x, L2G, xref[qp])

                    # evaluate kernel
                    O.kernel(result_kernel, input_args, QPinfos)
                    result_kernel .*= factor * weights[qp] * itemvolumes[item]

                    # multiply test function operator evaluation
                    for idt in 1:ntest
                        for k in 1:ndofs_test[idt]
                            dof = itemdofs_test[idt][k, item] + offsets_test[idt]
                            for d in 1:op_lengths_test[idt]
                                b[dof] += result_kernel[d + op_offsets_test[idt]] * BE_test_vals[idt][d, k, qp]
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
            time = @elapsed begin
                if O.parameters[:parallel]
                    pcp = xgrid[PColorPartitions]
                    ncolors = length(pcp) - 1
                    if O.parameters[:verbosity] > 0
                        @info "$(O.parameters[:name]) : assembling in parallel with $ncolors colors, $(length(EGs)) partitions and $(Threads.nthreads()) threads"
                    end
                    for color in 1:ncolors
                        Threads.@threads for part in pcp[color]:(pcp[color + 1] - 1)
                            assembly_loop(b, sol, itemassemblygroups[part], EGs[part], O.QF[part], O.BE_test[part], O.BE_args[part], O.BE_test_vals[part], O.BE_args_vals[part], O.L2G[part], O.QP_infos[part]; kwargs...)
                        end
                    end
                elseif O.parameters[:parallel_groups]
                    Threads.@threads for j in 1:length(EGs)
                        fill!(bj[j], 0)
                        assembly_loop(bj[j], sol, itemassemblygroups[j], EGs[j], O.QF[j], O.BE_test[j], O.BE_args[j], O.BE_test_vals[j], O.BE_args_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                    end
                    for j in 1:length(EGs)
                        b .+= bj[j]
                    end
                else
                    for j in 1:length(EGs)
                        assembly_loop(b, sol, itemassemblygroups[j], EGs[j], O.QF[j], O.BE_test[j], O.BE_args[j], O.BE_test_vals[j], O.BE_args_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                    end
                end
            end
            return if O.parameters[:verbosity] > 0
                @info "$(O.parameters[:name]) : assembly took $time s"
            end
        end
        O.assembler = assembler
    else
        ## update the time
        for j in 1:length(O.QP_infos)
            O.QP_infos[j].time = time
        end
    end
end

function build_assembler!(b, O::LinearOperator{Tv}, FE_test::Array{<:FEVectorBlock, 1}; time = 0.0, kwargs...) where {Tv}
    ## check if FES is the same as last time
    FES_test = [FE_test[j].FES for j in 1:length(FE_test)]
    return if (O.FES_test != FES_test)

        if O.parameters[:verbosity] > 0
            @info "$(O.parameters[:name]) : building assembler"
        end

        ## determine grid
        AT = O.parameters[:entities]
        xgrid = determine_assembly_grid(FES_test)
        if xgrid == FES_test[1].dofgrid
            gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[1]), AT)
        else
            gridAT = AT
        end

        ## prepare assembly
        Ti = typeof(xgrid).parameters[2]
        itemgeometries = xgrid[GridComponentGeometries4AssemblyType(gridAT)]
        itemvolumes = xgrid[GridComponentVolumes4AssemblyType(gridAT)]
        itemregions = xgrid[GridComponentRegions4AssemblyType(gridAT)]
        if num_pcolors(xgrid) > 1 && gridAT == ON_CELLS
            maxnpartitions = maximum(num_partitions_per_color(xgrid))
            pc = xgrid[PartitionCells]
            itemassemblygroups = [pc[j]:(pc[j + 1] - 1) for j in 1:num_partitions(xgrid)]
            # assuming here that all cells of one partition have the same geometry
        else
            itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(gridAT)]
            itemassemblygroups = [view(itemassemblygroups, :, j) for j in 1:num_sources(itemassemblygroups)]
        end
        has_normals = true
        if AT <: ON_FACES
            itemnormals = xgrid[FaceNormals]
        elseif AT <: ON_BFACES
            itemnormals = xgrid[FaceNormals][:, xgrid[BFaceFaces]]
        else
            has_normals = false
        end
        FETypes_test = [eltype(F) for F in FES_test]
        EGs = [itemgeometries[itemassemblygroups[j][1]] for j in 1:length(itemassemblygroups)]

        ## prepare assembly
        ntest = length(FES_test)
        O.QF = []
        O.BE_test = Array{Array{<:FEEvaluator, 1}, 1}([])
        O.BE_test_vals = Array{Array{Array{Tv, 3}, 1}, 1}([])
        O.QP_infos = Array{QPInfos, 1}([])
        O.L2G = []
        for EG in EGs
            ## quadrature formula for EG
            polyorder_test = maximum([get_polynomialorder(FETypes_test[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_test[j]) for j in 1:ntest])
            if O.parameters[:quadorder] == "auto"
                quadorder = polyorder_test + O.parameters[:bonus_quadorder]
            else
                quadorder = O.parameters[:quadorder] + O.parameters[:bonus_quadorder]
            end
            if O.parameters[:verbosity] > 1
                @info "...... integrating on $EG with quadrature order $quadorder"
            end
            push!(O.QF, QuadratureRule{Tv, EG}(quadorder))

            ## FE basis evaluator for EG
            push!(O.BE_test, [FEEvaluator(FES_test[j], O.ops_test[j], O.QF[end]; AT = AT) for j in 1:ntest])
            push!(O.BE_test_vals, [BE.cvals for BE in O.BE_test[end]])

            ## L2G map for EG
            push!(O.L2G, L2GTransformer(EG, xgrid, AT))

            ## parameter structure
            push!(O.QP_infos, QPInfos(xgrid; time = time, params = O.parameters[:params]))
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
        op_lengths_test = [size(O.BE_test[1][j].cvals, 1) for j in 1:ntest]

        op_offsets_test = [0]
        append!(op_offsets_test, cumsum(op_lengths_test))

        ## prepare parallel assembly
        if O.parameters[:parallel_groups]
            bj = Array{typeof(b), 1}(undef, length(EGs))
            for j in 1:length(EGs)
                bj[j] = copy(b)
            end
        end

        FEATs_test = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[j]), AT) for j in 1:ntest]
        itemdofs_test::Array{Union{Adjacency{Ti}, SerialVariableTargetAdjacency{Ti}}, 1} = [get_dofmap(FES_test[j], xgrid, FEATs_test[j]) for j in 1:ntest]
        factor = O.parameters[:factor]

        ## Assembly loop for fixed geometry
        function assembly_loop(b::AbstractVector{T}, items, EG::ElementGeometries, QF::QuadratureRule, BE_test::Array{<:FEEvaluator, 1}, BE_test_vals::Array{Array{Tv, 3}, 1}, L2G::L2GTransformer, QPinfos::QPInfos) where {T}

            ## prepare parameters
            result_kernel = zeros(Tv, op_offsets_test[end])
            offsets_test = [FE_test[j].offset for j in 1:length(FES_test)]

            ndofs_test::Array{Int, 1} = [get_ndofs(AT, FE, EG) for FE in FETypes_test]
            weights, xref = QF.w, QF.xref
            nweights = length(weights)

            for item::Int in items
                if itemregions[item] > 0
                    if !(visit_region[itemregions[item]]) || AT == ON_IFACES
                        continue
                    end
                else
                    if length(regions) > 0
                        continue
                    end
                end
                QPinfos.region = itemregions[item]
                QPinfos.item = item
                if has_normals
                    QPinfos.normal .= view(itemnormals, :, item)
                end
                QPinfos.volume = itemvolumes[item]

                ## update FE basis evaluators
                for j in 1:ntest
                    BE_test[j].citem[] = item
                    update_basis!(BE_test[j])
                end
                update_trafo!(L2G, item)

                ## evaluate arguments
                for qp in 1:nweights

                    ## get global x for quadrature point
                    eval_trafo!(QPinfos.x, L2G, xref[qp])

                    # evaluate kernel
                    O.kernel(result_kernel, QPinfos)
                    result_kernel .*= factor * weights[qp] * itemvolumes[item]

                    # multiply test function operator evaluation
                    for idt in 1:ntest
                        for k in 1:ndofs_test[idt]
                            dof = itemdofs_test[idt][k, item] + offsets_test[idt]
                            for d in 1:op_lengths_test[idt]
                                b[dof] += result_kernel[d + op_offsets_test[idt]] * BE_test_vals[idt][d, k, qp]
                            end
                        end
                    end
                end
            end
            return
        end
        O.FES_test = FES_test

        function assembler(b; kwargs...)
            time = @elapsed begin
                if O.parameters[:store] && size(b) == size(O.storage)
                    b .+= O.storage
                else
                    if O.parameters[:store]
                        s = zeros(eltype(b), length(b))
                    else
                        s = b
                    end
                    if O.parameters[:parallel]
                        pcp = xgrid[PColorPartitions]
                        ncolors = length(pcp) - 1
                        if O.parameters[:verbosity] > 0
                            @info "$(O.parameters[:name]) : assembling in parallel with $ncolors colors, $(length(EGs)) partitions and $(Threads.nthreads()) threads"
                        end
                        for color in 1:ncolors
                            Threads.@threads for part in pcp[color]:(pcp[color + 1] - 1)
                                assembly_loop(s, itemassemblygroups[part], EGs[part], O.QF[part], O.BE_test[part], O.BE_test_vals[part], O.L2G[part], O.QP_infos[part]; kwargs...)
                            end
                        end
                    elseif O.parameters[:parallel_groups]
                        Threads.@threads for j in 1:length(EGs)
                            fill!(bj[j], 0)
                            assembly_loop(bj[j], itemassemblygroups[j], EGs[j], O.QF[j], O.BE_test[j], O.BE_test_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                        end
                        for j in 1:length(EGs)
                            s .+= bj[j]
                        end
                    else
                        for j in 1:length(EGs)
                            assembly_loop(s, itemassemblygroups[j], EGs[j], O.QF[j], O.BE_test[j], O.BE_test_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                        end
                    end
                    if O.parameters[:store]
                        b .+= s
                        O.storage = s
                    end
                end
            end

            return if O.parameters[:verbosity] > 0
                @info "$(O.parameters[:name]) : assembly took $time s"
            end
        end
        O.assembler = assembler
    else
        ## update the time
        for j in 1:length(O.QP_infos)
            O.QP_infos[j].time = time
        end
    end
end

function assemble!(A, b, sol, O::LinearOperator{Tv, UT}, SC::SolverConfiguration; assemble_rhs = true, kwargs...) where {Tv, UT}
    if !assemble_rhs
        return
    end
    if UT <: Integer
        ind_test = O.u_test
        ind_args = O.u_args
    elseif UT <: Unknown
        ind_test = [get_unknown_id(SC, u) for u in O.u_test]
        ind_args = [findfirst(==(u), sol.tags) for u in O.u_args] # [get_unknown_id(SC, u) for u in O.u_args]
    end
    return if length(O.u_args) > 0
        build_assembler!(b.entries, O, [b[j] for j in ind_test], [sol[j] for j in ind_args]; kwargs...)
        O.assembler(b.entries, [sol[j] for j in ind_args])
    else
        build_assembler!(b.entries, O, [b[j] for j in ind_test]; kwargs...)
        O.assembler(b.entries)
    end
end


function assemble!(b::FEVector, O::LinearOperator{Tv, UT}, sol = nothing; assemble_rhs = true, kwargs...) where {Tv, UT}
    if !assemble_rhs
        return
    end
    ind_test = O.u_test
    ind_args = O.u_args
    return if length(O.u_args) > 0
        build_assembler!(b.entries, O, [b[j] for j in ind_test], [sol[j] for j in ind_args]; kwargs...)
        O.assembler(b.entries, [sol[j] for j in ind_args])
    else
        build_assembler!(b.entries, O, [b[j] for j in ind_test]; kwargs...)
        O.assembler(b.entries)
    end
end


function assemble!(A, b, sol, O::LinearOperatorFromVector{UT, bT}, SC::SolverConfiguration; assemble_rhs = true, kwargs...) where {UT, bT}
    if !assemble_rhs
        return
    end
    if UT <: Integer
        ind_test = O.u_test
    elseif UT <: Unknown
        ind_test = [get_unknown_id(SC, u) for u in O.u_test]
    end
    return if bT <: FEVector
        for (j, ij) in enumerate(ind_test)
            addblock!(b[j], O.b[ij]; factor = O.parameters[:factor])
        end
    else
        @assert length(ind_test) == 1
        addblock!(b[ind_test[1]], O.b; factor = O.parameters[:factor])
    end
end


function assemble!(A, b, sol, O::LinearOperatorFromMatrix{UT, MT}, SC::SolverConfiguration; assemble_rhs = true, kwargs...) where {UT, MT}
    if !assemble_rhs
        return
    end
    if UT <: Integer
        ind_test = O.u_test
        ind_args = O.u_args
    elseif UT <: Unknown
        ind_test = [get_unknown_id(SC, u) for u in O.u_test]
        ind_args = [get_unknown_id(SC, u) for u in O.u_args]
    end
    return if MT <: FEMatrix
        for (j, ij) in enumerate(ind_test), k in ind_args
            addblock_matmul!(b[j], O.A[ij, k], sol[k]; factor = O.parameters[:factor])
        end
    else
        @assert length(ind_test) == 1 && length(ind_args) == 1
        addblock_matmul!(b[ind_test[1]], O.A, sol[ind_args[1]]; factor = O.parameters[:factor])
    end
end
