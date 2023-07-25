
###########################################
### COMBINE DOFS (e.g. for periodicity) ###
###########################################

mutable struct CombineDofs{XT,UT,YT,FT} <: AbstractOperator
    uX::UT                  # component nr for dofsX
    uY::UT                  # component nr for dofsY
    dofsX::XT                    # dofsX that should be the same as dofsY in Y component
    dofsY::YT
    factors::FT
    FESX
    FESY
    assembler
    parameters::Dict{Symbol,Any}
end

default_combop_kwargs()=Dict{Symbol,Tuple{Any,String}}(
    :penalty => (1e30, "penalty for fixed degrees of freedom"),
    :verbosity => (0, "verbosity level"),
)

# informs solver in which blocks the operator assembles to
function ExtendableFEM.dependencies_when_linearized(O::CombineDofs)
    return [O.uX,O.uY]
end

function ExtendableFEM.fixed_dofs(O::CombineDofs)
    ## assembles operator to full matrix A and b
    return O.dofsY
end

function CombineDofs(uX, uY, dofsX, dofsY, factors; kwargs...)
    parameters=Dict{Symbol,Any}( k => v[1] for (k,v) in default_combop_kwargs())
    _update_params!(parameters, kwargs)
    @assert length(dofsX) == length(dofsY)
    return CombineDofs(uX, uY, dofsX, dofsY, factors, nothing, nothing, nothing, parameters)
end

function ExtendableFEM.assemble!(A, b, sol, O::CombineDofs{Tv,UT}, SC::SolverConfiguration; kwargs...) where {Tv,UT}
    if UT <: Integer
        ind = [O.ux, O.uY]
    elseif UT <: Unknown
        ind = [get_unknown_id(SC, O.uX), get_unknown_id(SC, O.uY)]
    end
    build_assembler!(O, [sol[j] for j in ind])
    O.assembler(A.entries, b.entries)
end


function build_assembler!(O::CombineDofs{Tv}, FE::Array{<:FEVectorBlock,1}; time = 0.0) where {Tv}
    ## check if FES is the same as last time
    FESX, FESY = FE[1].FES, FE[2].FES
    if (O.FESX != FESX) || (O.FESY != FESY)

        dofsX = O.dofsX
        dofsY = O.dofsY
        offsetX = FE[1].offset
        offsetY = FE[2].offset
        factors = O.factors
        if O.parameters[:verbosity] > 0
            @info ".... combining $(length(dofsX)) dofs"
        end
        
        function assemble(A::AbstractSparseArray{T}, b::AbstractVector{T}) where {T}

            #if !only_rhs 
            targetrow::Int = 0
            sourcerow::Int = 0
            targetcol::Int = 0
            sourcecol::Int = 0
            val::Float64 = 0
            ncols::Int = size(A,2)
            nrows::Int = size(A,1)
            for gdof = 1 : length(dofsX)
                # copy source row (for dofY) to target row (for dofX)
                targetrow = dofsX[gdof] + offsetX
                sourcerow = offsetY + dofsY[gdof]
                for sourcecol = 1 : ncols
                    targetcol = sourcecol - offsetY + offsetX
                    val = A[sourcerow,sourcecol]
                    _addnz(A, targetrow, targetcol, factors[gdof] * val,1)
                    A[sourcerow,sourcecol] = 0
                end

                # replace source row (of dofY) with equation for coupling the two dofs
                sourcecol = dofsY[gdof] + offsetY
                targetcol = dofsX[gdof] + offsetX
                sourcerow = offsetY + dofsY[gdof]
                _addnz(A, sourcerow, targetcol, 1,1)
                _addnz(A, sourcerow, sourcecol, -factors[gdof],1)
            end
            #end

            for gdof = 1 : length(dofsX)
                sourcerow = offsetY + dofsY[gdof]
                targetrow = offsetX + dofsX[gdof]
                b[targetrow] += b[sourcerow]
                b[sourcerow] = 0
            end
            flush!(A)
        end
        O.assembler = assemble
        O.FESX = FESX
        O.FESY = FESY
    end
end