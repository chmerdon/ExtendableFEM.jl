abstract type AbstractOperator end

# informs solver when operator needs reassembly
function depends_nonlinearly_on(O::AbstractOperator)
    return []
end

# informs solver in which blocks the operator assembles to
function dependencies_when_linearized(O::AbstractOperator)
    return nothing # Array{Symbol,1} (linear forms) or Array{Array{Symbol,1},1} (bilinearform)
end

function Base.show(io::IO, O::AbstractOperator)
    print(io, "AbstractOperator")
    return nothing # Array{Symbol,1} (linear forms) or Array{Array{Symbol,1},1} (bilinearform)
end

# informs solver when operator needs reassembly in a time dependent setting
function is_timedependent(O::AbstractOperator)
    return false
end

function fixed_dofs(O::AbstractOperator)
    ## assembles operator to full matrix A and b
    return []
end

function assemble!(A::AbstractMatrix, b::AbstractVector, O::AbstractOperator, sol; time = 0)
    ## assembles operator to full matrix A and b
    return nothing
end


function standard_kernel(result, input, qpinfo)
    result .= input
    return nothing
end

function constant_one_kernel(result, qpinfo)
    result .= 1
    return nothing
end