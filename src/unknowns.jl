"""
````
struct Unknown
````

Structure holding information for an unknwon with the following fields:

$(TYPEDFIELDS)
"""
mutable struct Unknown{IT}
	"""
	The name of the unknown used for printout messages.
	"""
	name::String
	"""
	The identifier of the unknown used for assignments to operators.
	"""
	identifier::IT
	"""
	Further properties of the unknown can be stored in a Dict, see constructor.
	"""
	parameters::Dict{Symbol, Any}
end


default_unknown_kwargs() = Dict{Symbol, Tuple{Any, String}}(
	:dimension => (nothing, "dimension of the unknown"),
	:symbol_ansatz => (nothing, "symbol for ansatz functions of this unknown in printouts"),
	:symbol_test => (nothing, "symbol for test functions of this unknown in printouts"),
	:algebraic_constraint => (nothing, "is this unknown an algebraic constraint?"),
)

test_function(u::Unknown) = u.parameters[:symbol_test] === nothing ? "X($(u.identifier))" : u.parameters[:symbol_test]
ansatz_function(u::Unknown) = u.parameters[:symbol_ansatz] === nothing ? u.identifier : u.parameters[:symbol_ansatz]


"""
````
function Unknown(
	u::String;
	identifier = Symbol(u),
	name = u,
	kwargs...)
````

Generates and returns an Unknown with the specified name, identifier and other traits.

Example: BilinearOperator([grad(1)], [grad(1)]; kwargs...) generates a weak Laplace operator.

Keyword arguments:
$(_myprint(default_unknown_kwargs()))

"""
function Unknown(u::String; identifier = Symbol(u), name = u, kwargs...)
	parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_unknown_kwargs())
	_update_params!(parameters, kwargs)
	return Unknown(name, identifier, parameters)
end

function Base.show(io::IO, u::Unknown)
	print(io, "$(u.identifier) (name = $(u.name))")
end

## remapping of all function operators
FO(u) = (u, FO)
jump(o::Tuple{Union{Unknown, Int}, DataType}) = (o[1], Jump{o[2]})
average(o::Tuple{Union{Unknown, Int}, DataType}) = (o[1], Average{o[2]})
this(o::Tuple{Union{Unknown, Int}, DataType}) = (o[1], Left{o[2]})
other(o::Tuple{Union{Unknown, Int}, DataType}) = (o[1], Right{o[2]})

## some aliases
id(u) = (u, Identity)
curl1(u) = (u, CurlScalar)
curl2(u) = (u, Curl2D)
curl3(u) = (u, Curl3D)
grad(u) = (u, Gradient)
laplace(u) = (u, Laplacian)
hessian(u) = (u, Hessian)
ExtendableFEMBase.div(u) = (u, Divergence)
normalflux(u) = (u, NormalFlux)

Δ(u) = (u, Laplacian)
apply(u, FO::Type{<:AbstractFunctionOperator}) = (u, FO)




