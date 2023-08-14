
mutable struct BilinearOperatorFromMatrix{UT <: Union{Unknown, Integer}, MT} <: AbstractOperator
    u_test::Array{UT,1}
    u_ansatz::Array{UT,1}
    u_args::Array{UT,1}
    A::MT
    parameters::Dict{Symbol,Any}
end

# informs solver when operator needs reassembly
function ExtendableFEM.depends_nonlinearly_on(O::BilinearOperatorFromMatrix)
    return O.u_args
end

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::BilinearOperatorFromMatrix)
    return [O.u_ansatz, O.u_test]
end

# informs solver when operator needs reassembly in a time dependent setting
function ExtendableFEM.is_timedependent(O::BilinearOperatorFromMatrix)
    return O.parameters[:time_dependent]
end

function Base.show(io::IO, O::BilinearOperatorFromMatrix)
    dependencies = dependencies_when_linearized(O)
    print(io, "$(O.parameters[:name])($([ansatz_function(dependencies[1][j]) for j = 1 : length(dependencies[1])]), $([test_function(dependencies[2][j]) for j = 1 : length(dependencies[2])]))")
    return nothing
end


mutable struct BilinearOperator{Tv <: Real, UT <: Union{Unknown, Integer}, KFT <: Function, MT} <: AbstractOperator
    u_test::Array{UT,1}
    ops_test::Array{DataType,1}
    u_ansatz::Array{UT,1}
    ops_ansatz::Array{DataType,1}
    u_args::Array{UT,1}
    ops_args::Array{DataType,1}
    kernel::KFT
    BE_test_vals::Array{Array{Array{Tv,3},1}}
    BE_ansatz_vals::Array{Array{Array{Tv,3},1}}
    BE_args_vals::Array{Array{Array{Tv,3},1}}
    FES_test             #::Array{FESpace,1}
    FES_ansatz           #::Array{FESpace,1}
    FES_args             #::Array{FESpace,1}
    BE_test              #::Union{Nothing, Array{FEEvaluator,1}}
    BE_ansatz            #::Union{Nothing, Array{FEEvaluator,1}}
    BE_args              #::Union{Nothing, Array{FEEvaluator,1}}
    QP_infos             #::Array{QPInfosT,1}
    L2G             
    QF
    assembler
    storage::MT
    parameters::Dict{Symbol,Any}
end

default_blfop_kwargs()=Dict{Symbol,Tuple{Any,String}}(
    :entities => (ON_CELLS, "assemble operator on these grid entities (default = ON_CELLS)"),
    :name => ("BilinearOperator", "name for operator used in printouts"),
    :transposed_copy => (0, "assemble a transposed copy of that operator into the transposed matrix block(s), 0 = no, 1 = symmetric, -1 = skew-symmetric"),
    :factor => (1, "factor that should be multiplied during assembly"),
    :lump => (false, "lump the operator (= only assemble the diagonal)"),
    :params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
    :entry_tolerance => (0, "threshold to add entry to sparse matrix"),
    :use_sparsity_pattern => ("auto", "read sparsity pattern of jacobian of kernel to find out which components couple"),
    :parallel_groups => (false, "assemble operator in parallel using CellAssemblyGroups"),
    :time_dependent => (false, "operator is time-dependent ?"),
    :callback! => (nothing, "function with interface (A, b, sol) that is called in each assembly step"),
    :store => (false, "store matrix separately (and copy from there when reassembly is triggered)"),
    :quadorder => ("auto", "quadrature order"),
    :bonus_quadorder => (0, "additional quadrature order added to quadorder"),
    :verbosity => (0, "verbosity level"),
    :regions => ([], "subset of regions where operator should be assembly only")
)

getFEStest(FVB::FEVectorBlock) = FVB.FES
getFESansatz(FVB::FEVectorBlock) = FVB.FES
getFEStest(FMB::FEMatrixBlock) = FMB.FES
getFESansatz(FMB::FEMatrixBlock) = FMB.FESY

# informs solver when operator needs reassembly
function ExtendableFEM.depends_nonlinearly_on(O::BilinearOperator)
    return unique(O.u_args)
end

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::BilinearOperator)
    return [unique(O.u_ansatz), unique(O.u_test)]
end

# informs solver when operator needs reassembly in a time dependent setting
function ExtendableFEM.is_timedependent(O::BilinearOperator)
    return O.parameters[:time_dependent]
end

function Base.show(io::IO, O::BilinearOperator)
    dependencies = dependencies_when_linearized(O)
    print(io, "$(O.parameters[:name])($([ansatz_function(dependencies[1][j]) for j = 1 : length(dependencies[1])]), $([test_function(dependencies[2][j]) for j = 1 : length(dependencies[2])]); entities = $(O.parameters[:entities]))")
    return nothing
end


"""
````
function BilinearOperator(
    A::AbstractMatrix,
    u_test,
    u_ansatz = u_test;
    kwargs...)
````

Generates a bilinear form from a user-provided matrix, which can be a sparse matrix or a FEMatrix with
multiple blocks. The arguments u_test and u_ansatz
specify where to put the (blocks of the) matrix in the system.

"""
function BilinearOperator(A::AbstractMatrix, u_test, u_ansatz = u_test, u_args = []; kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_blfop_kwargs())
    _update_params!(parameters, kwargs)
    return BilinearOperatorFromMatrix{typeof(u_test[1]), typeof(A)}(u_test, u_ansatz, u_args, A, parameters)
end


function BilinearOperator(kernel::Function, u_test, ops_test, u_ansatz = u_test, ops_ansatz = ops_test; Tv = Float64, kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_blfop_kwargs())
    _update_params!(parameters, kwargs)
    @assert length(u_ansatz) == length(ops_ansatz)
    @assert length(u_test) == length(ops_test)
    if parameters[:store]
        storage = ExtendableSparseMatrix{Float64,Int}(0,0)
    else
        storage = nothing
    end
    return BilinearOperator{Tv, typeof(u_test[1]), typeof(kernel), typeof(storage)}(u_test, ops_test, u_ansatz, ops_ansatz, [], [], kernel, [[zeros(Tv, 0, 0, 0)]], [[zeros(Tv, 0, 0, 0)]], [[zeros(Tv, 0, 0, 0)]],nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, storage, parameters)
end

function BilinearOperator(kernel::Function, u_test, ops_test, u_ansatz, ops_ansatz, u_args, ops_args; Tv = Float64, kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_blfop_kwargs())
    _update_params!(parameters, kwargs)
    @assert length(u_args) == length(ops_args)
    @assert length(u_ansatz) == length(ops_ansatz)
    @assert length(u_test) == length(ops_test)
    if parameters[:store]
        storage = ExtendableSparseMatrix{Float64,Int}(0,0)
    else
        storage = nothing
    end
    return BilinearOperator{Tv, typeof(u_test[1]), typeof(kernel), typeof(storage)}(u_test, ops_test, u_ansatz, ops_ansatz, u_args, ops_args, kernel, [[zeros(Tv, 0, 0, 0)]], [[zeros(Tv, 0, 0, 0)]], [[zeros(Tv, 0, 0, 0)]], nothing, nothing, nothing, nothing, nothing, nothing,nothing, nothing, nothing, nothing, storage, parameters)
end

function BilinearOperator(kernel::Function, oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1}, oa_ansatz::Array{<:Tuple{Union{Unknown,Int}, DataType},1} = oa_test; kwargs...)
    u_test = [oa[1] for oa in oa_test]
    u_ansatz = [oa[1] for oa in oa_ansatz]
    ops_test = [oa[2] for oa in oa_test]
    ops_ansatz = [oa[2] for oa in oa_ansatz]
    return BilinearOperator(kernel, u_test, ops_test, u_ansatz, ops_ansatz; kwargs...)
end



"""
````
function BilinearOperator(
    [kernel!::Function],
    oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1},
    oa_ansatz::Array{<:Tuple{Union{Unknown,Int}, DataType},1} = oa_test;
    kwargs...)
````

Generates a bilinear form that evaluates the vector product of the
operator evaluation(s) of the test function(s) with the operator evaluation(s)
of the ansatz function(s). If a function is provided in the first argument,
the ansatz function evaluations can be customized by the kernel function
and its result vector is then used in a dot product with the test function evaluations.
In this case the header of the kernel functions needs to be conform
to the interface

    kernel!(result, eval_ansatz, qpinfo)

where qpinfo allows to access information at the current quadrature point.

Operator evaluations are tuples that pair an unknown identifier or integer
with a Function operator.

Example: BilinearOperator([grad(1)], [grad(1)]; kwargs...) generates a weak Laplace operator.

Keyword arguments:
$(_myprint(default_blfop_kwargs()))

"""
function BilinearOperator(oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1}, oa_ansatz::Array{<:Tuple{Union{Unknown,Int}, DataType},1} = oa_test; kwargs...)
    u_test = [oa[1] for oa in oa_test]
    u_ansatz = [oa[1] for oa in oa_ansatz]
    ops_test = [oa[2] for oa in oa_test]
    ops_ansatz = [oa[2] for oa in oa_ansatz]
    return BilinearOperator(ExtendableFEMBase.standard_kernel, u_test, ops_test, u_ansatz, ops_ansatz; kwargs...)
end


"""
````
function BilinearOperator(
    kernel::Function,
    oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1},
    oa_ansatz::Array{<:Tuple{Union{Unknown,Int}, DataType},1},
    oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1};
    kwargs...)
````

Generates a nonlinear bilinear form that evaluates a kernel function
that depends on the operator evaluation(s) of the ansatz function(s)
and the operator evaluations of the current solution. The result of the
kernel function is used in a vector product with the operator evaluation(s)
of the test function(s). Hence, this can be used as a linearization of a
nonlinear operator. The header of the kernel functions needs to be conform
to the interface

    kernel!(result, eval_ansatz, eval_args, qpinfo)

where qpinfo allows to access information at the current quadrature point.

Operator evaluations are tuples that pair an unknown identifier or integer
with a Function operator.

Example: BilinearOperator([grad(1)], [grad(1)]; kwargs...) generates a weak Laplace operator.

Keyword arguments:
$(_myprint(default_blfop_kwargs()))

"""
function BilinearOperator(kernel::Function, oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1}, oa_ansatz::Array{<:Tuple{Union{Unknown,Int}, DataType},1}, oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1}; kwargs...)
    u_test = [oa[1] for oa in oa_test]
    u_ansatz = [oa[1] for oa in oa_ansatz]
    u_args = [oa[1] for oa in oa_args]
    ops_test = [oa[2] for oa in oa_test]
    ops_ansatz = [oa[2] for oa in oa_ansatz]
    ops_args = [oa[2] for oa in oa_args]
    return BilinearOperator(kernel, u_test, ops_test, u_ansatz, ops_ansatz, u_args, ops_args; kwargs...)
end

function build_assembler!(A, O::BilinearOperator{Tv}, FE_test, FE_ansatz, FE_args::Array{<:FEVectorBlock,1}; time = 0.0) where {Tv}
    ## check if FES is the same as last time
    FES_test = [getFEStest(FE_test[j]) for j = 1 : length(FE_test)]
    FES_ansatz = [getFESansatz(FE_ansatz[j]) for j = 1 : length(FE_ansatz)]
    FES_args = [FE_args[j].FES for j = 1 : length(FE_args)]
    if (O.FES_test != FES_test) || (O.FES_args != FES_args)

        if O.parameters[:verbosity] > 0
            @info ".... building assembler for $(O.parameters[:name])"
        end

        ## prepare assembly
        AT = O.parameters[:entities]
        xgrid = FES_test[1].xgrid
        gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[1]), AT)
        itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(AT)]
        itemgeometries = xgrid[GridComponentGeometries4AssemblyType(AT)]
        itemvolumes = xgrid[GridComponentVolumes4AssemblyType(AT)]
        itemregions = xgrid[GridComponentRegions4AssemblyType(AT)]
        FETypes_test = [eltype(F) for F in FES_test]
        FETypes_ansatz = [eltype(F) for F in FES_ansatz]
        FETypes_args = [eltype(F) for F in FES_args]
        EGs = [itemgeometries[itemassemblygroups[1,j]] for j = 1 : num_sources(itemassemblygroups)]

        ## prepare assembly
        nargs = length(FES_args)
        ntest = length(FES_test)
        nansatz = length(FES_ansatz)
        O.QF = []
        O.BE_test = Array{Array{<:FEEvaluator,1},1}([])
        O.BE_ansatz = Array{Array{<:FEEvaluator,1},1}([])
        O.BE_args = Array{Array{<:FEEvaluator,1},1}([])
        O.BE_test_vals = Array{Array{Array{Tv,3},1},1}([])
        O.BE_ansatz_vals = Array{Array{Array{Tv,3},1},1}([])
        O.BE_args_vals = Array{Array{Array{Tv,3},1},1}([])
        O.QP_infos = Array{QPInfos,1}([])
        O.L2G = []
        for EG in EGs
            ## quadrature formula for EG
            polyorder_ansatz = maximum([get_polynomialorder(FETypes_ansatz[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_ansatz[j]) for j = 1 : nansatz])
            polyorder_test = maximum([get_polynomialorder(FETypes_test[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_test[j]) for j = 1 : ntest])
            if O.parameters[:quadorder] == "auto"
                quadorder = polyorder_ansatz + polyorder_test + O.parameters[:bonus_quadorder]
            else
                quadorder = O.parameters[:quadorder] + O.parameters[:bonus_quadorder]
            end
            if O.parameters[:verbosity] > 1
                @info "...... integrating on $EG with quadrature order $quadorder"
            end
            push!(O.QF, QuadratureRule{Tv, EG}(quadorder))

            ## L2G map for EG
            push!(O.L2G, L2GTransformer(EG, xgrid, gridAT))
        
            ## FE basis evaluator for EG
            push!(O.BE_test, [FEEvaluator(FES_test[j], O.ops_test[j], O.QF[end]; AT = AT, L2G = O.L2G[end]) for j in 1 : ntest])
            push!(O.BE_ansatz, [FEEvaluator(FES_ansatz[j], O.ops_ansatz[j], O.QF[end]; AT = AT, L2G = O.L2G[end]) for j in 1 : nansatz])
            push!(O.BE_args, [FEEvaluator(FES_args[j], O.ops_args[j], O.QF[end]; AT = AT, L2G = O.L2G[end]) for j in 1 : nargs])
            push!(O.BE_test_vals, [BE.cvals for BE in O.BE_test[end]])
            push!(O.BE_ansatz_vals, [BE.cvals for BE in O.BE_ansatz[end]])
            push!(O.BE_args_vals, [BE.cvals for BE in O.BE_args[end]])

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

        ## prepare operator infos
        op_lengths_test = [size(O.BE_test[1][j].cvals,1) for j = 1 : ntest]
        op_lengths_ansatz = [size(O.BE_ansatz[1][j].cvals,1) for j = 1 : nansatz]
        op_lengths_args = [size(O.BE_args[1][j].cvals,1) for j = 1 : nargs]
        
        op_offsets_test = [0]
        op_offsets_ansatz = [0]
        op_offsets_args = [0]
        append!(op_offsets_test, cumsum(op_lengths_test))
        append!(op_offsets_ansatz, cumsum(op_lengths_ansatz))
        append!(op_offsets_args, cumsum(op_lengths_args))
        offsets_test = [FE_test[j].offset for j in 1 : length(FES_test)]
        offsets_ansatz = [FE_ansatz[j].offset for j in 1 : length(FES_ansatz)]

        ## prepare sparsity pattern
        use_sparsity_pattern = O.parameters[:use_sparsity_pattern]
        if use_sparsity_pattern == "auto"
             use_sparsity_pattern = false
        end
        coupling_matrix::Matrix{Bool} = ones(Bool, nansatz, ntest)
        if use_sparsity_pattern
            kernel_params = (result, input) -> (O.kernel(result, input, O.QP_infos[1]);)
            sparsity_pattern = SparseMatrixCSC{Float64,Int}(Symbolics.jacobian_sparsity(kernel_params, zeros(Tv, op_offsets_test[end]), zeros(Tv, op_offsets_ansatz[end])))

            ## find out which test and ansatz functions couple
            for id = 1 : nansatz
                for idt = 1 : ntest
                    couple = false
                    for j = 1 : op_lengths_ansatz[id]
                        for k = 1 : op_lengths_test[idt]
                            if sparsity_pattern[k + op_offsets_test[idt], j + op_offsets_ansatz[id]] > 0
                                couple = true
                            end
                        end
                    end
                    coupling_matrix[id, idt] = couple
                end
            end
        end
        couples_with::Vector{Vector{Int}} = [findall(==(true), view(coupling_matrix,j,:)) for j = 1 : nansatz]

        ## prepare parallel assembly
        if O.parameters[:parallel_groups]
            Aj = Array{typeof(A),1}(undef, length(EGs))
            for j = 1 : length(EGs)
                Aj[j] = deepcopy(A)
            end
        end

        FEATs_test = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[j]), AT) for j = 1 : ntest]
        FEATs_ansatz = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_ansatz[j]), AT) for j = 1 : nansatz]
        FEATs_args = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_args[j]), AT) for j = 1 : nargs]
        itemdofs_test::Array{Union{Adjacency{Int32}, SerialVariableTargetAdjacency{Int32}},1} = [FES_test[j][Dofmap4AssemblyType(FEATs_test[j])] for j = 1 : ntest]
        itemdofs_ansatz::Array{Union{Adjacency{Int32}, SerialVariableTargetAdjacency{Int32}},1} = [FES_ansatz[j][Dofmap4AssemblyType(FEATs_ansatz[j])] for j = 1 : nansatz]
        itemdofs_args::Array{Union{Adjacency{Int32}, SerialVariableTargetAdjacency{Int32}},1} = [FES_args[j][Dofmap4AssemblyType(FEATs_args[j])] for j = 1 : nargs]
        factor = O.parameters[:factor]
        transposed_copy = O.parameters[:transposed_copy]
        entry_tol = O.parameters[:entry_tolerance]
        lump = O.parameters[:lump]

        ## Assembly loop for fixed geometry
        function assembly_loop(A::AbstractSparseArray{T}, sol::Array{<:FEVectorBlock,1}, items, EG::ElementGeometries, QF::QuadratureRule, BE_test::Array{<:FEEvaluator,1}, BE_ansatz::Array{<:FEEvaluator,1}, BE_args::Array{<:FEEvaluator,1}, BE_test_vals::Array{Array{Tv,3},1}, BE_ansatz_vals::Array{Array{Tv,3},1}, BE_args_vals::Array{Array{Tv,3},1}, L2G::L2GTransformer, QPinfos::QPInfos) where {T}

            input_ansatz = zeros(T, op_offsets_ansatz[end])
            input_args = zeros(T, op_offsets_args[end])
            result_kernel = zeros(T, op_offsets_test[end])

            ndofs_test::Array{Int,1} = [size(BE.cvals,2) for BE in BE_test]
            ndofs_ansatz::Array{Int,1} = [size(BE.cvals,2) for BE in BE_ansatz]
            ndofs_args::Array{Int,1} = [size(BE.cvals,2) for BE in BE_args]

            Aloc = Matrix{Matrix{T}}(undef, ntest, nansatz)
            for j = 1 : ntest, k = 1 : nansatz
                Aloc[j,k] = zeros(T, ndofs_test[j], ndofs_ansatz[k])
            end
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
                QPinfos.volume = itemvolumes[item]

                ## update FE basis evaluators
                for j = 1 : ntest
                    BE_test[j].citem[] = item
                    update_basis!(BE_test[j]) 
                end
                for j = 1 : nansatz
                    BE_ansatz[j].citem[] = item
                    update_basis!(BE_ansatz[j]) 
                end
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
                                input_args[d + op_offsets_args[id]] += sol[id][dof_j] * BE_args_vals[id][d, j, qp]
                            end
                        end
					end
                
                    ## get global x for quadrature point
                    eval_trafo!(QPinfos.x, L2G, xref[qp])

                    # update matrix
                    for id = 1 : nansatz
                        for j = 1 : ndofs_ansatz[id]
                            # evaluat kernel for ansatz basis function
                            fill!(input_ansatz, 0)
                            for d = 1 : op_lengths_ansatz[id]
                                input_ansatz[d + op_offsets_ansatz[id]] += BE_ansatz_vals[id][d,j,qp]
                            end

                            # evaluate kernel
                            O.kernel(result_kernel, input_ansatz, input_args, QPinfos)
                            result_kernel .*= factor * weights[qp]

                            # multiply test function operator evaluation
                            if lump
                                for d = 1 : op_lengths_test[id]
                                    Aloc[id,id][j,j] += result_kernel[d + op_offsets_test[id]] * BE_test_vals[id][d,j,qp]
                                end
                            else
                                for idt in couples_with[id]
                                    for k = 1 : ndofs_test[idt]
                                        for d = 1 : op_lengths_test[idt]
                                            Aloc[idt,id][k,j] += result_kernel[d + op_offsets_test[idt]] * BE_test_vals[idt][d,k,qp]
                                        end
                                    end
                                end
                            end
                        end
                    end 
                end

                ## add local matrices to global matrix
                for id = 1 : nansatz, idt = 1 : ntest
                    Aloc[idt,id] .*= itemvolumes[item]
                    for j = 1 : ndofs_test[idt]
                        dof_j = itemdofs_test[idt][j, item] + offsets_test[idt]
                        for k = 1 : ndofs_ansatz[id]
                            dof_k = itemdofs_ansatz[id][k, item] + offsets_ansatz[id]
                            if abs(Aloc[idt,id][j,k]) > entry_tol
                                rawupdateindex!(A, +, Aloc[idt,id][j,k], dof_j, dof_k)
                            end
                        end
                    end
                end
                if transposed_copy != 0
                    for id = 1 : nansatz, idt = 1 : ntest
                        Aloc[idt,id] .*= transposed_copy
                        for j = 1 : ndofs_test[idt]
                            dof_j = itemdofs_test[idt][j, item] + offsets_test[idt]
                            for k = 1 : ndofs_ansatz[id]
                                dof_k = itemdofs_ansatz[id][k, item] + offsets_ansatz[id]
                                if abs(Aloc[idt,id][j,k]) > entry_tol
                                    rawupdateindex!(A, +, Aloc[idt,id][j,k], dof_k, dof_j)
                                end
                            end
                        end
                    end
                end
        
                for id = 1 : nansatz, idt = 1 : ntest
                    fill!(Aloc[idt,id], 0)
                end
            end
            flush!(A)
            return
        end
        O.FES_test = FES_test
        O.FES_ansatz = FES_ansatz
        O.FES_args = FES_args

        function assembler(A, b, sol; kwargs...)
            time = @elapsed begin
                if O.parameters[:parallel_groups]
                    Threads.@threads for j = 1 : length(EGs)
                        fill!(Aj[j].cscmatrix.nzval,0)
                        assembly_loop(Aj[j], sol, view(itemassemblygroups,:,j), EGs[j], O.QF[j], O.BE_test[j], O.BE_ansatz[j], O.BE_args[j], O.BE_test_vals[j], O.BE_ansatz_vals[j], O.BE_args_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                    end
                    for j = 1 : length(EGs)
                        A.cscmatrix += Aj[j].cscmatrix
                    end
                    flush!(A)
                else
                    for j = 1 : length(EGs)
                        assembly_loop(A, sol, view(itemassemblygroups,:,j), EGs[j], O.QF[j], O.BE_test[j], O.BE_ansatz[j], O.BE_args[j], O.BE_test_vals[j], O.BE_ansatz_vals[j], O.BE_args_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                    end
                end   
            end
            if O.parameters[:verbosity] > 1
                @info ".... assembly of $(O.parameters[:name]) took $time s"
            end
        end
        O.assembler = assembler
    end
end

function build_assembler!(A, O::BilinearOperator{Tv}, FE_test, FE_ansatz; time = 0.0) where {Tv}
    ## check if FES is the same as last time
    FES_test = [getFEStest(FE_test[j]) for j = 1 : length(FE_test)]
    FES_ansatz = [getFESansatz(FE_ansatz[j]) for j = 1 : length(FE_ansatz)]

    if (O.FES_test != FES_test) || (O.FES_ansatz != FES_ansatz)

        if O.parameters[:verbosity] > 0
            @info ".... building assembler for $(O.parameters[:name])"
        end
        ## prepare assembly
        AT = O.parameters[:entities]
        xgrid = FES_test[1].xgrid
        gridAT = ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[1]), AT)
        itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(gridAT)]
        itemgeometries = xgrid[GridComponentGeometries4AssemblyType(gridAT)]
        itemvolumes = xgrid[GridComponentVolumes4AssemblyType(gridAT)]
        itemregions = xgrid[GridComponentRegions4AssemblyType(gridAT)]
        FETypes_test = [eltype(F) for F in FES_test]
        FETypes_ansatz = [eltype(F) for F in FES_ansatz]
        EGs = [itemgeometries[itemassemblygroups[1,j]] for j = 1 : num_sources(itemassemblygroups)]

        ## prepare assembly
        ntest = length(FES_test)
        nansatz = length(FES_ansatz)
        O.QF = []
        O.BE_test = Array{Array{<:FEEvaluator,1},1}([])
        O.BE_ansatz = Array{Array{<:FEEvaluator,1},1}([])
        O.BE_test_vals = Array{Array{Array{Tv,3},1},1}([])
        O.BE_ansatz_vals = Array{Array{Array{Tv,3},1},1}([])
        O.QP_infos = Array{QPInfos,1}([])
        O.L2G = []
        for EG in EGs
            ## quadrature formula for EG
            polyorder_ansatz = maximum([get_polynomialorder(FETypes_ansatz[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_ansatz[j]) for j = 1 : nansatz])
            polyorder_test = maximum([get_polynomialorder(FETypes_test[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_test[j]) for j = 1 : ntest])
            if O.parameters[:quadorder] == "auto"
                quadorder = polyorder_ansatz + polyorder_test + O.parameters[:bonus_quadorder]
            else
                quadorder = O.parameters[:quadorder] + O.parameters[:bonus_quadorder]
            end
            if O.parameters[:verbosity] > 1
                @info "...... integrating on $EG with quadrature order $quadorder"
            end
            push!(O.QF, QuadratureRule{Tv, EG}(quadorder))

            ## L2G map for EG
            push!(O.L2G, L2GTransformer(EG, xgrid, gridAT))
        
            ## FE basis evaluator for EG
            push!(O.BE_test, [FEEvaluator(FES_test[j], O.ops_test[j], O.QF[end]; AT = AT, L2G = O.L2G[end]) for j in 1 : ntest])
            push!(O.BE_ansatz, [FEEvaluator(FES_ansatz[j], O.ops_ansatz[j], O.QF[end]; AT = AT, L2G = O.L2G[end]) for j in 1 : nansatz])
            push!(O.BE_test_vals, [BE.cvals for BE in O.BE_test[end]])
            push!(O.BE_ansatz_vals, [BE.cvals for BE in O.BE_ansatz[end]])

            ## parameter structure
            push!(O.QP_infos, QPInfos(xgrid; time = time, x = ones(Tv, size(xgrid[Coordinates],1)), params = O.parameters[:params]))
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
        op_lengths_test = [size(O.BE_test[1][j].cvals,1) for j = 1 : ntest]
        op_lengths_ansatz = [size(O.BE_ansatz[1][j].cvals,1) for j = 1 : nansatz]
        
        op_offsets_test = [0]
        op_offsets_ansatz = [0]
        append!(op_offsets_test, cumsum(op_lengths_test))
        append!(op_offsets_ansatz, cumsum(op_lengths_ansatz))
        offsets_test = [FE_test[j].offset for j in 1 : length(FES_test)]
        offsets_ansatz = [FE_ansatz[j].offset for j in 1 : length(FES_ansatz)]

        ## prepare sparsity pattern
        use_sparsity_pattern = O.parameters[:use_sparsity_pattern]
         if use_sparsity_pattern == "auto"
             use_sparsity_pattern = ntest > 1
        end
        coupling_matrix::Matrix{Bool} = ones(Bool, nansatz, ntest)
        if use_sparsity_pattern
            kernel_params = (result, input) -> (O.kernel(result, input, O.QP_infos[1]);)
            sparsity_pattern = SparseMatrixCSC{Float64,Int}(Symbolics.jacobian_sparsity(kernel_params, zeros(Tv, op_offsets_test[end]), zeros(Tv, op_offsets_ansatz[end])))

            ## find out which test and ansatz functions couple
            for id = 1 : nansatz
                for idt = 1 : ntest
                    couple = false
                    for j = 1 : op_lengths_ansatz[id]
                        for k = 1 : op_lengths_test[idt]
                            if sparsity_pattern[k + op_offsets_test[idt], j + op_offsets_ansatz[id]] > 0
                                couple = true
                            end
                        end
                    end
                    coupling_matrix[id, idt] = couple
                end
            end
        end
        couples_with::Vector{Vector{Int}} = [findall(==(true), view(coupling_matrix,j,:)) for j = 1 : nansatz]

        ## prepare parallel assembly
        if O.parameters[:parallel_groups]
            Aj = Array{typeof(A),1}(undef, length(EGs))
            for j = 1 : length(EGs)
                Aj[j] = deepcopy(A)
            end
        end

        FEATs_test = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_test[j]), AT) for j = 1 : ntest]
        FEATs_ansatz = [ExtendableFEMBase.EffAT4AssemblyType(get_AT(FES_ansatz[j]), AT) for j = 1 : nansatz]
        itemdofs_test::Array{Union{Adjacency{Int32}, SerialVariableTargetAdjacency{Int32}},1} = [FES_test[j][Dofmap4AssemblyType(FEATs_test[j])] for j = 1 : ntest]
        itemdofs_ansatz::Array{Union{Adjacency{Int32}, SerialVariableTargetAdjacency{Int32}},1} = [FES_ansatz[j][Dofmap4AssemblyType(FEATs_ansatz[j])] for j = 1 : nansatz]
        factor = O.parameters[:factor]
        transposed_copy = O.parameters[:transposed_copy]
        entry_tol = O.parameters[:entry_tolerance]
        lump = O.parameters[:lump]

        ## Assembly loop for fixed geometry
        function assembly_loop(A::AbstractSparseArray{T}, items, EG::ElementGeometries, QF::QuadratureRule, BE_test::Array{<:FEEvaluator,1}, BE_ansatz::Array{<:FEEvaluator,1}, BE_test_vals::Array{Array{Tv,3},1}, BE_ansatz_vals::Array{Array{Tv,3},1}, L2G::L2GTransformer, QPinfos::QPInfos) where {T}

            input_ansatz = zeros(T, op_offsets_ansatz[end])
            result_kernel = zeros(T, op_offsets_test[end])

            #ndofs_test::Array{Int,1} = [get_ndofs(ON_CELLS, FE, EG) for FE in FETypes_test]
            #ndofs_ansatz::Array{Int,1} = [get_ndofs(ON_CELLS, FE, EG) for FE in FETypes_ansatz]
            ndofs_test::Array{Int,1} = [size(BE.cvals,2) for BE in BE_test]
            ndofs_ansatz::Array{Int,1} = [size(BE.cvals,2) for BE in BE_ansatz]
            
            Aloc = Matrix{Matrix{T}}(undef, ntest, nansatz)
            for j = 1 : ntest, k = 1 : nansatz
                Aloc[j,k] = zeros(T, ndofs_test[j], ndofs_ansatz[k])
            end
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
                QPinfos.volume = itemvolumes[item]

                ## update FE basis evaluators
                for j = 1 : ntest
                    BE_test[j].citem[] = item
                    update_basis!(BE_test[j]) 
                end
                for j = 1 : nansatz
                    BE_ansatz[j].citem[] = item
                    update_basis!(BE_ansatz[j]) 
                end
	            update_trafo!(L2G, item)

                ## evaluate arguments
				for qp = 1 : nweights
                
                    ## get global x for quadrature point
                    eval_trafo!(QPinfos.x, L2G, xref[qp])

                    # update matrix
                    for id = 1 : nansatz
                        for j = 1 : ndofs_ansatz[id]
                            # evaluat kernel for ansatz basis function
                            fill!(input_ansatz, 0)
                            for d = 1 : op_lengths_ansatz[id]
                                input_ansatz[d + op_offsets_ansatz[id]] = BE_ansatz_vals[id][d,j,qp]
                            end

                            # evaluate kernel
                            O.kernel(result_kernel, input_ansatz, QPinfos)
                            result_kernel .*= factor * weights[qp]

                            # multiply test function operator evaluation
                            if lump
                                for d = 1 : op_lengths_test[id]
                                    Aloc[id,id][j,j] += result_kernel[d + op_offsets_test[id]] * BE_test_vals[id][d,j,qp]
                                end
                            else
                                for idt in couples_with[id]
                                    for k = 1 : ndofs_test[idt]
                                        for d = 1 : op_lengths_test[idt]
                                            Aloc[idt,id][k,j] += result_kernel[d + op_offsets_test[idt]] * BE_test_vals[idt][d,k,qp]
                                        end
                                    end
                                end
                            end
                        end
                    end 
                end

                ## add local matrices to global matrix
                for id = 1 : nansatz, idt in couples_with[id]
                    Aloc[idt,id] .*= itemvolumes[item]
                    for j = 1 : ndofs_test[idt]

                        dof_j = itemdofs_test[idt][j, item] + offsets_test[idt]
                        for k = 1 : ndofs_ansatz[id]
                            dof_k = itemdofs_ansatz[id][k, item] + offsets_ansatz[id]
                            if abs(Aloc[idt,id][j,k]) > entry_tol
                                rawupdateindex!(A, +, Aloc[idt,id][j,k], dof_j, dof_k)
                            end
                        end
                    end
                end
                if transposed_copy != 0
                    for id = 1 : nansatz, idt in couples_with[id]
                        Aloc[idt,id] .*= transposed_copy
                        for j = 1 : ndofs_test[idt]
                            dof_j = itemdofs_test[idt][j, item] + offsets_test[idt]
                            for k = 1 : ndofs_ansatz[id]
                                dof_k = itemdofs_ansatz[id][k, item] + offsets_ansatz[id]
                                if abs(Aloc[idt,id][j,k]) > entry_tol
                                    rawupdateindex!(A, +, Aloc[idt,id][j,k], dof_k, dof_j)
                                end
                            end
                        end
                    end
                end
        
                for id = 1 : nansatz, idt = 1 : ntest
                    fill!(Aloc[idt,id], 0)
                end
            end
            flush!(A)
            return
        end
        O.FES_test = FES_test
        O.FES_ansatz = FES_ansatz

        function assembler(A, b; kwargs...)
            if O.parameters[:store] && size(A) == size(O.storage)
                A.cscmatrix += O.storage.cscmatrix
            else
                if O.parameters[:store]
                    S = ExtendableSparseMatrix{Float64,Int}(size(A,1), size(A,2))
                else
                    S = A
                end
                time = @elapsed begin
                    if O.parameters[:parallel_groups]
                        Threads.@threads for j = 1 : length(EGs)
                            fill!(Aj[j].cscmatrix.nzval,0)
                            assembly_loop(Aj[j], view(itemassemblygroups,:,j), EGs[j], O.QF[j], O.BE_test[j], O.BE_ansatz[j], O.BE_test_vals[j], O.BE_ansatz_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                        end
                        for j = 1 : length(EGs)
                            S.cscmatrix += Aj[j].cscmatrix
                        end
                        flush!(S)
                    else
                        for j = 1 : length(EGs)
                            assembly_loop(S, view(itemassemblygroups,:,j), EGs[j], O.QF[j], O.BE_test[j], O.BE_ansatz[j], O.BE_test_vals[j], O.BE_ansatz_vals[j], O.L2G[j], O.QP_infos[j]; kwargs...)
                        end
                    end   
                end
                if O.parameters[:callback!] !== nothing
                    S = O.parameters[:callback!](S, b, sol)
                end
                if O.parameters[:verbosity] > 1
                    @info ".... assembly of $(O.parameters[:name]) took $time s"
                end
                if O.parameters[:store]
                    A.cscmatrix += S.cscmatrix
                    O.storage = S
                end
            end
        end
        O.assembler = assembler
    end
end

function ExtendableFEM.assemble!(A, b, sol, O::BilinearOperator{Tv,UT}, SC::SolverConfiguration; kwargs...) where {Tv,UT}
    if UT <: Integer
        ind_test = O.u_test
        ind_ansatz = O.u_ansatz
        ind_args = O.u_args
    elseif UT <: Unknown
        ind_test = [get_unknown_id(SC, u) for u in O.u_test]
        ind_ansatz = [get_unknown_id(SC, u) for u in O.u_ansatz]
        ind_args = [findfirst(==(u), sol.tags) for u in O.u_args] #[get_unknown_id(SC, u) for u in O.u_args]
    end
    if length(O.u_args) > 0
        build_assembler!(A.entries, O, [A[j,j] for j in ind_test], [A[j,j] for j in ind_ansatz], [sol[j] for j in ind_args])
        O.assembler(A.entries, b.entries, [sol[j] for j in ind_args])
    else
        build_assembler!(A.entries, O, [A[j,j] for j in ind_test], [A[j,j] for j in ind_ansatz])
        O.assembler(A.entries, b.entries)
    end
end

function ExtendableFEM.assemble!(A::FEMatrix, O::BilinearOperator{Tv,UT}, sol = nothing; kwargs...) where {Tv,UT}
    @assert UT <: Integer
    ind_test = O.u_test
    ind_ansatz = O.u_ansatz
    ind_args = O.u_args
    if length(O.u_args) > 0
        build_assembler!(A.entries, O, [A[j,j] for j in ind_test], [A[j,j] for j in ind_ansatz], [sol[j] for j in ind_args])
        O.assembler(A.entries, [sol[j] for j in ind_args])
    else
        build_assembler!(A.entries, O, [A[j,j] for j in ind_test], [A[j,j] for j in ind_ansatz])
        O.assembler(A.entries, nothing)
    end
end


function ExtendableFEM.assemble!(A, b, sol, O::BilinearOperatorFromMatrix{UT,MT}, SC::SolverConfiguration; kwargs...) where {UT,MT}
    if UT <: Integer
        ind_test = O.u_test
        ind_ansatz = O.u_ansatz
    elseif UT <: Unknown
        ind_test = [get_unknown_id(SC, u) for u in O.u_test]
        ind_ansatz = [get_unknown_id(SC, u) for u in O.u_ansatz]
    end
    if O.parameters[:callback!] !== nothing
        O.A = O.parameters[:callback!](O.A, [b[j] for j in ind_test], sol)
    end
    if MT <: FEMatrix
        for (j,ij) in enumerate(ind_test), (k,ik) in enumerate(ind_ansatz)
            addblock!(A[j,k], O.A[ij,ik]; factor = O.parameters[:factor])
        end
    else
        @assert length(ind_test) == 1
        @assert length(ind_ansatz) == 1
        addblock!(A[ind_test[1],ind_ansatz[1]], O.A; factor = O.parameters[:factor])
        if O.parameters[:transposed_copy] != 0
            addblock!(A[ind_test[1],ind_ansatz[1]], O.A; transpose = true, factor = O.parameters[:factor] * O.parameters[:transposed_copy])
        end
    end
end