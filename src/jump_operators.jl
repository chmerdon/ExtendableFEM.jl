
"""
	DiscontinuousFunctionOperator

Subtype of AbstractFunctionOperator dedicated to evaluations of discontinuous quantities on faces
like jumps, averages etc.
"""
abstract type DiscontinuousFunctionOperator <: AbstractFunctionOperator end
"""
	Jump{StandardFunctionOperator}

evaluates the jump of a StandardFunctionOperator
"""
abstract type Jump{O} <: DiscontinuousFunctionOperator where {O <: StandardFunctionOperator} end # calculates the jump between both sides of the face
"""
	Average{StandardFunctionOperator}

evaluates the average of a StandardFunctionOperator
"""
abstract type Average{O} <: DiscontinuousFunctionOperator where {O <: StandardFunctionOperator} end # calculates the average between both sides of the face
"""
	Left{StandardFunctionOperator}

evaluates the left (w.r.t. orientation of the face) value of a StandardFunctionOperator
"""
abstract type Left{O} <: DiscontinuousFunctionOperator where {O <: StandardFunctionOperator} end # calculates the value on left side of the face
"""
	Average{StandardFunctionOperator}

evaluates the right (w.r.t. orientation of the face) value of a StandardFunctionOperator
"""
abstract type Right{O} <: DiscontinuousFunctionOperator where {O <: StandardFunctionOperator} end # calculates the value on right side of the face


StandardFunctionOperator(::Type{Jump{O}}) where {O} = O
StandardFunctionOperator(::Type{Average{O}}) where {O} = O
StandardFunctionOperator(::Type{Left{O}}) where {O} = O
StandardFunctionOperator(::Type{Right{O}}) where {O} = O
StandardFunctionOperator(::Type{O}) where {O <: StandardFunctionOperator} = O
coeffs(::Type{<:Jump}) = [1, -1]
coeffs(::Type{<:Average}) = [0.5, 0.5]
coeffs(::Type{<:Left}) = [1, 0]
coeffs(::Type{<:Right}) = [0, 1]
coeffs(::Type{<:AbstractFunctionOperator}) = [1, 0]
is_discontinuous(::Type{<:StandardFunctionOperator}) = false
is_discontinuous(::Type{<:DiscontinuousFunctionOperator}) = true

ExtendableFEMBase.NeededDerivative4Operator(::Type{Jump{O}}) where {O} = NeededDerivative4Operator(O)
ExtendableFEMBase.Length4Operator(::Type{Jump{O}}, xdim, nc) where {O} = Length4Operator(O, xdim, nc)
ExtendableFEMBase.DefaultName4Operator(::Type{Jump{O}}) where {O} = "[[" * DefaultName4Operator(O) * "]]"
ExtendableFEMBase.NeededDerivative4Operator(::Type{Average{O}}) where {O} = NeededDerivative4Operator(O)
ExtendableFEMBase.Length4Operator(::Type{Average{O}}, xdim, nc) where {O} = Length4Operator(O, xdim, nc)
ExtendableFEMBase.DefaultName4Operator(::Type{Average{O}}) where {O} = "{{" * DefaultName4Operator(O) * "}}"
ExtendableFEMBase.NeededDerivative4Operator(::Type{Left{O}}) where {O} = NeededDerivative4Operator(O)
ExtendableFEMBase.Length4Operator(::Type{Left{O}}, xdim, nc) where {O} = Length4Operator(O, xdim, nc)
ExtendableFEMBase.DefaultName4Operator(::Type{Left{O}}) where {O} = DefaultName4Operator(O) * "|_Left"
ExtendableFEMBase.NeededDerivative4Operator(::Type{Right{O}}) where {O} = NeededDerivative4Operator(O)
ExtendableFEMBase.Length4Operator(::Type{Right{O}}, xdim, nc) where {O} = Length4Operator(O, xdim, nc)
ExtendableFEMBase.DefaultName4Operator(::Type{Right{O}}) where {O} = DefaultName4Operator(O) * "|_Right"


## additional infrastructure for discontinuous operators of broken face-continuous spaces

#struct DuplicateCValView{T,FT} <: AbstractArray{T,3}
#    cvals::Array{T,3}
#    j2dofindex::Array{Int,1}
#    factors::Array{FT,1}
#end

#Base.getindex(SCV::DuplicateCValView,i::Int,j::Int,k::Int) = SCV.cvals[i,SCV.j2dofindex[j],k] * SCV.factors[j]
#Base.size(SCV::DuplicateCValView) = [size(SCV.cvals,1), 2*size(SCV.cvals,2), size(SCV.cvals,3)]
#Base.size(SCV::DuplicateCValView,i) = (i == 2) ? 2 * size(SCV.cvals,i) : size(SCV.cvals,i)

struct FEEvaluatorDisc{T, TvG, TiG, FEType, FEBType, O <: DiscontinuousFunctionOperator} <: FEEvaluator{T, TvG, TiG}
	citem::Base.RefValue{Int}                   # current item
	FE::FESpace{TvG, TiG, FEType}       # link to full FE (e.g. for coefficients)
	FEB::FEBType                     # first FEBasisEvaluator
	coeffs::Array{T, 1}
	cvals::Array{T, 3}
end

function FEEvaluator(
	FE::FESpace{TvG, TiG, FEType, FEAPT},
	operator::Type{<:DiscontinuousFunctionOperator},
	qrule::QuadratureRule{TvR, EG};
	T = Float64,
	kwargs...) where {TvG, TiG, TvR, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEAPT <: AssemblyType}

	FEB = FEEvaluator(FE, StandardFunctionOperator(operator), qrule; T = T, kwargs...)
	ndofs = size(FEB.cvals, 2)
	cvals = reshape(repeat(FEB.cvals, 2), (size(FEB.cvals, 1), 2 * ndofs, size(FEB.cvals, 3)))
	return FEEvaluatorDisc{T, TvG, TiG, FEType, typeof(FEB), operator}(FEB.citem, FE, FEB, coeffs(operator), cvals)
end


function ExtendableFEMBase.update_basis!(FEBE::FEEvaluatorDisc)
	ExtendableFEMBase.update_basis!(FEBE.FEB)
	cvals_std = FEBE.FEB.cvals
	cvals = FEBE.cvals
	coeffs = FEBE.coeffs
	for d ∈ 1:size(cvals_std, 1), j ∈ size(cvals_std, 2), qp ∈ size(cvals_std, 3)
		cvals[d, j, qp] = coeffs[1] * cvals_std[d, j, qp]
		cvals[d, j+size(cvals_std, 2), qp] = coeffs[2] * cvals_std[d, j, qp]
	end
end

function ExtendableFEMBase.update_basis!(FEBE::FEEvaluatorDisc, item)
	if FEBE.citem[] == item
	else
		FEBE.citem[] = item
		update_basis!(FEBE.FEB, item)
	end
end


#### DG operators


function generate_DG_master_quadrule(quadorder, EG; T = Float64)
	EGface = facetype_of_cellface(EG, 1)
	nfaces4cell = num_faces(EG)
	for j ∈ 1:nfaces4cell
		@assert facetype_of_cellface(EG, j) == EGface "all faces of cell must have the same face geometry!"
	end

	return QuadratureRule{T, EGface}(quadorder)
end

function generate_DG_operators(operator, FE, quadorder, EG; T = Float64)
	## prototype quadrature rule on face geometry
	qf4face = generate_DG_master_quadrule(quadorder, EG; T = T)

	EGface = facetype_of_cellface(EG, 1)
	nfaces4cell = num_faces(EG)

	# generate new quadrature rules on cell
	# where quadrature points of face are mapped to quadrature points of cells
	xrefFACE2CELL = xrefFACE2xrefCELL(EG)
	xrefFACE2OFACE = xrefFACE2xrefOFACE(EGface)
	norientations = length(xrefFACE2OFACE)
	basisevaler4EG = Array{FEEvaluator, 2}(undef, nfaces4cell, norientations)
	xrefdim = length(qf4face.xref)
	qf4cell = ExtendableFEMBase.SQuadratureRule{T, EG, xrefdim, length(qf4face.xref)}(qf4face.name * " (shape faces)", Array{Array{T, 1}, 1}(undef, length(qf4face.xref)), qf4face.w)
	for f ∈ 1:nfaces4cell, orientation ∈ 1:norientations
		## modify quadrature rule for this local face and local orientation
		for i ∈ 1:length(qf4face.xref)
			qf4cell.xref[i] = xrefFACE2CELL[f](xrefFACE2OFACE[orientation](qf4face.xref[i]))
		end
		basisevaler4EG[f, orientation] = FEEvaluator(FE, operator, deepcopy(qf4cell); T = T, AT = ON_CELLS)
	end
	return basisevaler4EG
end
