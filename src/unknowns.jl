mutable struct Unknown{IT}
    name::String
    identifier::IT
    dimension::Int
    ansatz_name::String
    test_name::String
    algebraic_constraint::Bool
end

function Unknown(identifier; name = String(identifier), ansatz = "", test = "", dim = 1, algebraic_constraint = false)
    return Unknown(name,Symbol(identifier),dim,"","",algebraic_constraint)
end

function Base.show(io::IO, u::Unknown)
    println(io, "$(u.identifier) (dim = $(u.dimension), name = $(u.name), ansatz = $(u.ansatz_name), test = $(u.test_name))")
end

## remapping of all function operators
FO(u) = (u, FO)
jump(o::Tuple{Union{Unknown,Int}, DataType}) = (o[1], Jump{o[2]})
average(o::Tuple{Union{Unknown,Int}, DataType}) = (o[1], Average{o[2]})

## some aliases
id(u) = (u, Identity)
grad(u) = (u, Gradient)
ExtendableFEMBase.div(u) = (u, Divergence)
normalflux(u) = (u, NormalFlux)

Δ(u) = (u, Laplacian)
apply(u, FO::Type{<:AbstractFunctionOperator}) = (u, FO)




