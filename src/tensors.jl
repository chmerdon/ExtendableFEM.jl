
"""
    TensorDescription{R,D}

General type for an `R`-tensor of dimension/extent `D`.
Mathematically, this describes the shape of an element 
in ``\\underbrace{\\mathbb{R}^D\\times\\cdots\\times\\mathbb{R}^D}_{R} \\text{ times}}``.

See also: 
[TDScalar{D}](@ref),
[TDVector{D}](@ref),
[TDMatrix{D}](@ref),
[TDRank3{D}](@ref),
[TDRank4{D}](@ref)
"""
struct TensorDescription{R,D}
end
TensorDescription(R,D) = TensorDescription{R,D}()

"""
````
function tensor_view(input,i::Int,::TensorDescription{rank,dim})
````

Returns a view of `input[i]` and subsequent entries, 
reshaped as a `rank`-tensor of dimension `dim`.

Note that this general implementation is a fallback for `rank>4`
that will likely produce allocations and slow assembly 
times if used in a kernel function.
"""
function tensor_view(input, i::Int, ::TensorDescription{rank,dim}) where {rank,dim}
    @warn "tensor_view for rank > 4 is a general implementation that needs allocations!"
    return reshape(view(input, i:(i+(dim^rank)-1)),ntuple(i->dim,rank))
end


"""
    TDScalar{D}

Specification for a 0-tensor or scalar,
i.e. `TensorDescription{0,D}`, to improve readability.

Note that in this case `D` has no greater effect 
and is only provided to have a matching interface 
between all the specifications.
"""
const TDScalar{D} = TensorDescription{0,D}
TDScalar(D) = TDScalar{D}()
TDScalar() = TDScalar{0}()

"""
    TDVector{D}

Specification for a 1-tensor or vector,
i.e. `TensorDescription{1,D}`, to improve readability.
"""
const TDVector{D} = TensorDescription{1,D}
TDVector(D) = TDVector{D}()

"""
    TDMatrix{D}

Specification for a 2-tensor or matrix,
i.e. `TensorDescription{2,D}`, to improve readability.
"""
const TDMatrix{D} = TensorDescription{2,D}
TDMatrix(D) = TDMatrix{D}()


"""
    TDRank3{D}

Specification for a 3-tensor,
i.e. `TensorDescription{3,D}`, to improve readability.
"""
const TDRank3{D} = TensorDescription{3,D}
TDRank3(D) = TDRank3{D}()


"""
    TDRank4{D}

Specification for a 4-tensor,
i.e. `TensorDescription{4,D}`, to improve readability.
"""
const TDRank4{D} = TensorDescription{4,D}
TDRank4(D) = TDRank4{D}()


"""
````
function tensor_view(input,i::Int,::TensorDescription{0,dim})
````

Returns a view of `input[i]` reshaped as a vector of length 1.
"""
function tensor_view(input,i::Int, ::TensorDescription{0,dim}) where dim
    return view(input, i:i)
end


"""
````
function tensor_view(input,i::Int,::TensorDescription{1,dim})
````

Returns a view of `input[i:i+dim-1]` reshaped as a vector of length dim.
"""
function tensor_view(input,i::Int, ::TensorDescription{1,dim}) where dim
    return view(input, i:i+dim-1)
end

"""
````
function tensor_view(input,i::Int,::TensorDescription{2,dim})
````

Returns a view of `input[i:i+dim^2-1]` reshaped as a `(dim,dim)` matrix.
"""
function tensor_view(input,i::Int, ::TensorDescription{2,dim}) where dim
    return reshape(view(input, i:(i+(dim*dim)-1)), (dim,dim))
end

"""
````
function tensor_view(input,i::Int,::TensorDescription{3,dim})
````

Returns a view of `input[i:i+dim^3-1]` reshaped as a `(dim,dim,dim)` 3-tensor.
"""
function tensor_view(input,i::Int, ::TensorDescription{3,dim}) where dim
    return reshape(view(input, i:(i+(dim^3)-1)), (dim, dim,dim))
end

"""
````
function tensor_view(input,i::Int,::TensorDescription{4,dim})
````

Returns a view of `input[i:i+dim^4-1]` reshaped as `(dim,dim,dim,dim)` 4-tensor.
"""
function tensor_view(input,i::Int, ::TensorDescription{4,dim}) where dim
    return reshape(view(input, i:(i+(dim^4)-1)), (dim,dim,dim,dim))
end


