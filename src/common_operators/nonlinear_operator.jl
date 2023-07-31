
struct KernelEvaluator{Tv <: Real, QPInfosT <: QPInfos, JT, VT, DRT, CFGT, KPT <: Function}
    input_args::Array{Tv,1}
    params::QPInfosT
    result_kernel::Array{Tv,1}
    jac::JT
    value::VT
    Dresult::DRT
    cfg::CFGT
    kernel::KPT
end


mutable struct NonlinearOperator{Tv <: Real, UT <: Union{Unknown, Integer}, KFT, JFT} <: AbstractOperator
    u_test::Array{UT,1}
    ops_test::Array{DataType,1}
    u_args::Array{UT,1}
    ops_args::Array{DataType,1}
    kernel::KFT
    jacobian::JFT
    BE_test_vals::Array{Array{Array{Tv,3},1}}
    BE_args_vals::Array{Array{Array{Tv,3},1}}
    FES_test             #::Array{FESpace,1}
    FES_args             #::Array{FESpace,1}
    BE_test              #::Union{Nothing, Array{FEEvaluator,1}}
    BE_args              #::Union{Nothing, Array{FEEvaluator,1}}
    L2G             
    QF
    assembler
    parameters::Dict{Symbol,Any}
end

default_nlop_kwargs()=Dict{Symbol,Tuple{Any,String}}(
    :entities => (ON_CELLS, "assemble operator on these grid entities (default = ON_CELLS)"),
    :name => ("NonlinearOperator", "name for operator used in printouts"),
    :parallel_groups => (false, "assemble operator in parallel using CellAssemblyGroups"),
    :factor => (1, "factor that should be multiplied during assembly"),
    :sparse_jacobians => (true, "use sparse jacobians"),
    :entry_tolerance => (0, "threshold to add entry to sparse matrix"),
    :params => (nothing, "array of parameters that should be made available in qpinfo argument of kernel function"),
    :quadorder => ("auto", "quadrature order"),
    :bonus_quadorder => (0, "additional quadrature order added to quadorder"),
    :time_dependent => (false, "operator is time-dependent ?"),
    :verbosity => (0, "verbosity level"),
    :regions => ([], "subset of regions where operator should be assembly only")
)

# informs solver when operator needs reassembly
function ExtendableFEM.depends_nonlinearly_on(O::NonlinearOperator)
    return unique(O.u_args)
end

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::NonlinearOperator)
    return [unique(O.u_test), unique(O.u_args)]
end

# informs solver when operator needs reassembly in a time dependent setting
function ExtendableFEM.is_timedependent(O::NonlinearOperator)
    return O.parameters[:time_dependent]
end

function Base.show(io::IO, O::NonlinearOperator)
    nl_dependencies = depends_nonlinearly_on(O)
    dependencies = dependencies_when_linearized(O)
    print(io, "$(O.parameters[:name])($([ansatz_function(nl_dependencies[j]) for j = 1 : length(nl_dependencies)]); $([ansatz_function(dependencies[1][j]) for j = 1 : length(dependencies[1])]), $([test_function(dependencies[2][j]) for j = 1 : length(dependencies[2])]))")
    return nothing
end


function NonlinearOperator(kernel, u_test, ops_test, u_args = u_test, ops_args = ops_test; Tv = Float64, jacobian = nothing, kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_nlop_kwargs())
    _update_params!(parameters, kwargs)
    @assert length(u_args) == length(ops_args)
    @assert length(u_test) == length(ops_test)
    return NonlinearOperator{Tv, typeof(u_test[1]), typeof(kernel), typeof(jacobian)}(u_test, ops_test, u_args, ops_args, kernel, jacobian, [[zeros(Tv, 0, 0, 0)]], [[zeros(Tv, 0, 0, 0)]], nothing, nothing, nothing, nothing, nothing, nothing, nothing, parameters)
end


"""
````
function NonlinearOperator(
    [kernel!::Function],
    oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1},
    oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1} = oa_test;
    jacobian = nothing,
    kwargs...)
````

Generates a nonlinear form for the specified kernel function, test function operators,
and argument operators evaluations. Operator evaluations are tuples that pair an unknown identifier or integer
with a FunctionOperator. The header of the kernel functions needs to be conform
to the interface

    kernel!(result, input, qpinfo)

where qpinfo allows to access information at the current quadrature point.

During assembly the Newton update is computed via local jacobians of the kernel
which are calculated by automatic differentiation or
by the user-provided jacobian function with interface

    jacobian!(jac, input_args, params)


Keyword arguments:
$(_myprint(default_nlop_kwargs()))

"""
function NonlinearOperator(kernel, oa_test::Array{<:Tuple{Union{Unknown,Int}, DataType},1}, oa_args::Array{<:Tuple{Union{Unknown,Int}, DataType},1} = oa_test; kwargs...)
    u_test = [oa[1] for oa in oa_test]
    u_args = [oa[1] for oa in oa_args]
    ops_test = [oa[2] for oa in oa_test]
    ops_args = [oa[2] for oa in oa_args]
    return NonlinearOperator(kernel, u_test, ops_test, u_args, ops_args; kwargs...)
end

function build_assembler!(A, b, O::NonlinearOperator{Tv}, FE_test::Array{<:FEVectorBlock,1}, FE_args::Array{<:FEVectorBlock,1}; time = 0.0, kwargs...) where {Tv}
    ## check if FES is the same as last time
    FES_test = [FE_test[j].FES for j = 1 : length(FE_test)]
    FES_args = [FE_args[j].FES for j = 1 : length(FE_args)]
    _update_params!(O.parameters, kwargs)
    if (O.FES_test != FES_test) || (O.FES_args != FES_args)

        if O.parameters[:verbosity] > 0
            @info ".... building assembler for $(O.parameters[:name])"
        end

        ## prepare assembly
        AT = O.parameters[:entities]
        xgrid = FES_test[1].xgrid
        itemassemblygroups = xgrid[GridComponentAssemblyGroups4AssemblyType(AT)]
        itemgeometries = xgrid[GridComponentGeometries4AssemblyType(AT)]
        itemvolumes = xgrid[GridComponentVolumes4AssemblyType(AT)]
        itemregions = xgrid[GridComponentRegions4AssemblyType(AT)]
        FETypes_test = [eltype(F) for F in FES_test]
        FETypes_args = [eltype(F) for F in FES_args]
        EGs = [itemgeometries[itemassemblygroups[1,j]] for j = 1 : num_sources(itemassemblygroups)]

        ## prepare assembly
        nargs = length(FES_args)
        ntest = length(FES_test)
        O.QF = []
        O.BE_test = Array{Array{<:FEEvaluator{Tv},1},1}([])
        O.BE_args = Array{Array{<:FEEvaluator{Tv},1},1}([])
        O.BE_test_vals = Array{Array{Array{Tv,3},1},1}([])
        O.BE_args_vals = Array{Array{Array{Tv,3},1},1}([])
        O.L2G = []
        for EG in EGs
            ## quadrature formula for EG
            polyorder_args = maximum([get_polynomialorder(FETypes_args[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_args[j]) for j = 1 : nargs])
            polyorder_test = maximum([get_polynomialorder(FETypes_test[j], EG) - ExtendableFEMBase.NeededDerivative4Operator(O.ops_test[j]) for j = 1 : ntest])
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
            push!(O.BE_test, [FEEvaluator(FES_test[j], O.ops_test[j], O.QF[end]) for j in 1 : ntest])
            push!(O.BE_args, [FEEvaluator(FES_args[j], O.ops_args[j], O.QF[end]) for j in 1 : nargs])
            push!(O.BE_test_vals, [BE.cvals for BE in O.BE_test[end]])
            push!(O.BE_args_vals, [BE.cvals for BE in O.BE_args[end]])

            ## L2G map for EG
            push!(O.L2G, L2GTransformer(EG, xgrid, ON_CELLS))

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
        op_lengths_args = [size(O.BE_args[1][j].cvals,1) for j = 1 : nargs]
        
        op_offsets_test = [0]
        op_offsets_args = [0]
        append!(op_offsets_test, cumsum(op_lengths_test))
        append!(op_offsets_args, cumsum(op_lengths_args))
        offsets_test = [FE_test[j].offset for j in 1 : length(FES_test)]
        offsets_args = [FE_args[j].offset for j in 1 : length(FES_args)]

        Kj = Array{KernelEvaluator,1}([])

        sparse_jacobians = O.parameters[:sparse_jacobians]
        use_autodiff = O.jacobian === nothing
        for EG in EGs
            ## prepare parameters
            QPj = QPInfos(xgrid; time = time, x = ones(Tv, size(xgrid[Coordinates],1)), params = O.parameters[:params])
            kernel_params = (result, input) -> (O.kernel(result, input, QPj);)
            if sparse_jacobians
                input_args = zeros(Tv, op_offsets_args[end])
                result_kernel = zeros(Tv, op_offsets_test[end])
                sparsity_pattern = Symbolics.jacobian_sparsity(kernel_params,result_kernel,input_args)
                jac = Float64.(sparse(sparsity_pattern))
                value = zeros(Tv, op_offsets_test[end])
                colors = matrix_colors(jac)
                Dresult = nothing
                cfg = ForwardColorJacCache(kernel_params,input_args,nothing;
                            dx = nothing,
                            colorvec = colors,
                            sparsity = sparsity_pattern)
            else
                input_args = zeros(Tv, op_offsets_args[end])
                result_kernel = zeros(Tv, op_offsets_test[end])
                Dresult = DiffResults.JacobianResult(result_kernel, input_args)
                jac = DiffResults.jacobian(Dresult)
                value = DiffResults.value(Dresult)
                cfg = ForwardDiff.JacobianConfig(kernel_params, result_kernel, input_args, ForwardDiff.Chunk{op_offsets_args[end]}())
            end
            push!(Kj, KernelEvaluator(input_args, QPj, result_kernel, jac, value, Dresult, cfg, kernel_params))
        end

        ## prepare parallel assembly
        if O.parameters[:parallel_groups]
            Aj = Array{typeof(A),1}(undef, length(EGs))
            bj = Array{typeof(b),1}(undef, length(EGs))
            for j = 1 : length(EGs)
                Aj[j] = deepcopy(A)
                bj[j] = deepcopy(b)
            end
        end
        
        FEATs_test = [EffAT4AssemblyType(get_AT(FES_test[j]), AT) for j = 1 : ntest]
        FEATs_args = [EffAT4AssemblyType(get_AT(FES_args[j]), AT) for j = 1 : nargs]
        itemdofs_test::Array{Union{Adjacency{Int32}, SerialVariableTargetAdjacency{Int32}},1} = [FES_test[j][Dofmap4AssemblyType(FEATs_test[j])] for j = 1 : ntest]
        itemdofs_args::Array{Union{Adjacency{Int32}, SerialVariableTargetAdjacency{Int32}},1} = [FES_args[j][Dofmap4AssemblyType(FEATs_args[j])] for j = 1 : nargs]
        factor = O.parameters[:factor]
        entry_tol = O.parameters[:entry_tolerance]

        ## Assembly loop for fixed geometry
        function assembly_loop(A::AbstractSparseArray{T}, b::AbstractVector{T}, sol::Array{<:FEVectorBlock{T,Tv,Ti},1}, items, EG::ElementGeometries, QF::QuadratureRule, BE_test::Array{<:FEEvaluator,1}, BE_args::Array{<:FEEvaluator,1}, BE_test_vals::Array{Array{Tv,3},1}, BE_args_vals::Array{Array{Tv,3},1}, L2G::L2GTransformer, K::KernelEvaluator) where {T,Tv,Ti}

            ## extract kernel properties
            params = K.params
            input_args = K.input_args
            result_kernel = K.result_kernel
            cfg = K.cfg
            Dresult = K.Dresult
            jac = K.jac
            value = K.value
            kernel_params = K.kernel

            ndofs_test::Array{Int,1} = [get_ndofs(ON_CELLS, FE, EG) for FE in FETypes_test]
            ndofs_args::Array{Int,1} = [get_ndofs(ON_CELLS, FE, EG) for FE in FETypes_args]
            Aloc = Matrix{Matrix{T}}(undef, ntest, nargs)
            for j = 1 : ntest, k = 1 : nargs
                Aloc[j,k] = zeros(T, ndofs_test[j], ndofs_args[k])
            end
            weights, xref = QF.w, QF.xref
            nweights = length(weights)
            tempV = zeros(T, op_offsets_test[end])
            dof_j::Int, dof_k::Int = 0, 0

            for item::Int in items
                if itemregions[item] > 0
                    if !(visit_region[itemregions[item]])
                        continue
                    end
                else
                    params.region = itemregions[item]
                    params.item = item
                    params.volume = itemvolumes[item]
                end

                ## update FE basis evaluators
                for j = 1 : ntest
                    BE_test[j].citem[] = item
                    update_basis!(BE_test[j]) 
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
                
                    ## evaluate jacobian
                    ## get global x for quadrature point
                    eval_trafo!(params.x, L2G, xref[qp])
                    if use_autodiff
                        if sparse_jacobians
                            forwarddiff_color_jacobian!(jac, kernel_params, input_args, cfg)
                            kernel_params(value, input_args)
                        else
                            ForwardDiff.chunk_mode_jacobian!(Dresult, kernel_params, result_kernel, input_args, cfg)
                        end
                    else
                        O.jacobian(jac, input_args, params)
                        O.kernel(value, input_args, params)
                    end

                    # update matrix
                    for id = 1 : nargs
                        for j = 1 : ndofs_args[id]
                            # multiply ansatz function with local jacobian
                            fill!(tempV,0)
                            if sparse_jacobians
                                rows = rowvals(jac)
                                jac_vals = jac.nzval
                                for col = 1 : op_lengths_args[id]
                                    for r in nzrange(jac, col + op_offsets_args[id])
                                        tempV[rows[r]] += jac_vals[r] * BE_args_vals[id][col,j,qp]
                                    end
                                end
                            else
                                for d = 1 : op_lengths_args[id]
                                    for k = 1 : op_offsets_test[end]
                                        tempV[k] += jac[k,d + op_offsets_args[id]] * BE_args_vals[id][d,j,qp]
                                    end
                                end
                            end

                            # multiply test function operator evaluation
                            for idt = 1 : ntest
                                for k = 1 : ndofs_test[idt]
                                    for d = 1 : op_lengths_test[idt]
                                        Aloc[idt,id][k,j] += tempV[d + op_offsets_test[idt]] * BE_test_vals[idt][d,k,qp] * weights[qp]
                                    end
                                end
                            end
                        end
                    end 

                    # update rhs
                    mul!(tempV, jac, input_args)
                    tempV .-= value
                    tempV .*= factor * weights[qp] * itemvolumes[item] 
                    for idt = 1 : ntest
                        for j = 1 : ndofs_test[idt]
                            dof = itemdofs_test[idt][j, item] + offsets_test[idt]
                            for d = 1 : op_lengths_test[idt]
                                b[dof] += tempV[d + op_offsets_test[idt]] * BE_test_vals[idt][d,j,qp]
                            end
                        end
                    end
                end

                ## add local matrices to global matrix
                for id = 1 : nargs, idt = 1 : ntest
                    Aloc[idt,id] .*= factor * itemvolumes[item]
                    for j = 1 : ndofs_test[idt]
                        dof_j = itemdofs_test[idt][j, item] + offsets_test[idt]
                        for k = 1 : ndofs_args[id]
                            dof_k = itemdofs_args[id][k, item] + offsets_args[id]
                            if abs(Aloc[idt,id][j,k]) > entry_tol
                                rawupdateindex!(A, +, Aloc[idt,id][j,k], dof_j, dof_k)
                            end
                        end
                    end
                end
        
                for id = 1 : nargs, idt = 1 : ntest
                    fill!(Aloc[idt,id], 0)
                end
            end
            flush!(A)
            return
        end
        O.FES_test = FES_test
        O.FES_args = FES_args

        function assembler(A, b, sol; kwargs...)
            time = @elapsed begin
                    if O.parameters[:parallel_groups]
                    Threads.@threads for j = 1 : length(EGs)
                        fill!(bj[j],0)
                        fill!(Aj[j].cscmatrix.nzval,0)
                        assembly_loop(Aj[j], bj[j], sol, view(itemassemblygroups,:,j), EGs[j], O.QF[j], O.BE_test[j], O.BE_args[j], O.BE_test_vals[j], O.BE_args_vals[j], O.L2G[j], Kj[j]; kwargs...)
                    end
                    for j = 1 : length(EGs)
                        A.cscmatrix += Aj[j].cscmatrix
                        b .+= bj[j]
                    end
                    flush!(A)
                else
                    for j = 1 : length(EGs)
                        assembly_loop(A, b, sol, view(itemassemblygroups,:,j), EGs[j], O.QF[j], O.BE_test[j], O.BE_args[j], O.BE_test_vals[j], O.BE_args_vals[j], O.L2G[j], Kj[j]; kwargs...)
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

function ExtendableFEM.assemble!(A, b, sol, O::NonlinearOperator{Tv,UT}, SC::SolverConfiguration; kwargs...) where {Tv,UT}
    if UT <: Integer
        ind_test = O.u_test
       ind_args = O.u_args
    elseif UT <: Unknown
        ind_test = [get_unknown_id(SC, u) for u in O.u_test]
        ind_args = [findfirst(==(u), sol.tags) for u in O.u_args] #[get_unknown_id(SC, u) for u in O.u_args]
    end
    build_assembler!(A.entries, b.entries, O, [sol[j] for j in ind_test], [sol[j] for j in ind_args]; kwargs...)
    O.assembler(A.entries, b.entries, [sol[j] for j in ind_args])
end