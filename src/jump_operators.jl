

abstract type DiscontinuousFunctionOperator <: AbstractFunctionOperator end
abstract type Jump{O} <: DiscontinuousFunctionOperator where {O<:StandardFunctionOperator} end # calculates the jump between both sides of the face
abstract type Average{O} <: DiscontinuousFunctionOperator where {O<:StandardFunctionOperator} end # calculates the average between both sides of the face
abstract type Left{O} <: DiscontinuousFunctionOperator where {O<:StandardFunctionOperator} end # calculates the value on left side of the face
abstract type Right{O} <: DiscontinuousFunctionOperator where {O<:StandardFunctionOperator} end # calculates the value on right side of the face


StandardFunctionOperator(::Type{Jump{O}}) where {O} = O
StandardFunctionOperator(::Type{Average{O}}) where {O} = O
StandardFunctionOperator(::Type{Left{O}}) where {O} = O
StandardFunctionOperator(::Type{Right{O}}) where {O} = O
coeffs(::Type{<:Jump}) = [1,-1]
coeffs(::Type{<:Average}) = [0.5,0.5]
coeffs(::Type{<:Left}) = [1,0]
coeffs(::Type{<:Right}) = [0,1]
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


##### additional infrastructure for pairs of FE evaluators

struct DuplicateCValView{T,FT} <: AbstractArray{T,3}
    cvals::Array{T,3}
    j2dofindex::Array{Int,1}
    factors::Array{FT,1}
end

Base.getindex(SCV::DuplicateCValView,i::Int,j::Int,k::Int) = SCV.cvals[i,SCV.j2dofindex[j],k] * SCV.factors[j]
Base.size(SCV::DuplicateCValView) = [size(SCV.cvals,1), 2*size(SCV.cvals,2), size(SCV.cvals,3)]
Base.size(SCV::DuplicateCValView,i) = (i == 2) ? 2 * size(SCV.cvals,i) : size(SCV.cvals,i)

struct FEEvaluatorDisc{T,TvG,TiG,FEType,FEBType,O<:DiscontinuousFunctionOperator} <: FEEvaluator{T,TvG,TiG}
    citem::Base.RefValue{Int}                   # current item
    FE::FESpace{TvG,TiG,FEType}       # link to full FE (e.g. for coefficients)
    FEB::FEBType                     # first FEBasisEvaluator
    cvals::DuplicateCValView{T}          # view that doubles cvals of FEB and weights it with proper factors to evaluate jump, average etc.
end

function FEEvaluator(FE::FESpace{TvG,TiG,FEType,FEAPT}, operator::Type{<:DiscontinuousFunctionOperator}, qrule::QuadratureRule{TvR,EG}; T = Float64, AT = ON_CELLS) where {TvG, TiG, TvR, FEType <: AbstractFiniteElement, EG <: AbstractElementGeometry, FEAPT <: AssemblyType}
    FEB = FEEvaluator(FE, StandardFunctionOperator(operator), qrule; T = T, AT = AT)
    ndofs = size(FEB.cvals, 2)
    j2dofindex = zeros(Int, 2*ndofs)
    factors = zeros(eltype(FEB.cvals), 2*ndofs)
    j2dofindex[1:ndofs] = 1:ndofs
    j2dofindex[ndofs+1:2*ndofs] = 1:ndofs
    op_coeffs = coeffs(operator)
    factors[1:ndofs] .= op_coeffs[1]
    factors[ndofs+1:2*ndofs] .= op_coeffs[2]
    cvals = DuplicateCValView(FEB.cvals,j2dofindex,factors)
    return FEEvaluatorDisc{T,TvG,TiG,FEType,typeof(FEB),operator}(FEB.citem, FE, FEB, cvals)
end


function ExtendableFEMBase.update_basis!(FEBE::FEEvaluatorDisc)
    ExtendableFEMBase.update_basis!(FEBE.FEB)
end

function ExtendableFEMBase.update_basis!(FEBE::FEEvaluatorDisc, item)
    if FEBE.citem[] == item
    else
        FEBE.citem[] = item
        update_basis!(FEBE.FEB, item)
    end
end



