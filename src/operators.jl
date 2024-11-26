"""
	AbstractOperator

Root type for all operators
"""
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

function assemble!(A, b, sol, O::AbstractOperator, SC; kwargs...)
    ## assembles operator to full matrix A and b
    return nothing
end

function apply_penalties!(A, b, sol, O::AbstractOperator, SC; kwargs...)
    ## applies penalties to full matrix A and b and also sets values in sol
    return nothing
end
