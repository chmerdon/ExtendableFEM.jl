

"""
````
function get_periodic_coupling_info(FES, xgrid, b1, b2, is_opposite::Function; factor_vectordofs = "auto")
````

computes the dofs that have to be coupled for periodic boundary conditions on the given xgrid for boundary regions b1, b2.
The is_opposite function evaluates if two provided face midpoints are on opposite sides to each other (the mesh xgrid should be appropriate).
For vector-valued FETypes the user can provide factor_vectordofs to incorporate a sign change if needed.
This is automatically done for all Hdiv-conforming elements and (for the normal-weighted face bubbles of) the Bernardi-Raugel element H1BR. 

"""
function get_periodic_coupling_info(
	FES::FESpace,
	xgrid::ExtendableGrid,
	b1,
	b2,
	is_opposite::Function;
	factor_vectordofs = "auto",
	factor_components = "auto")

	FEType = eltype(FES)
	ncomponents = get_ncomponents(FEType)
	if factor_vectordofs == "auto"
		if FEType <: AbstractHdivFiniteElement || FEType <: H1BR
			factor_vectordofs = -1
		else
			factor_vectordofs = 1
		end
	end
	if factor_components == "auto"
		factor_components = ones(Int, ncomponents)
	end


	@assert FEType <: AbstractH1FiniteElement "not yet working for non H1-conforming elements"
	xBFaceRegions = xgrid[BFaceRegions]
	xBFaceNodes = xgrid[BFaceNodes]
	xBFaceFaces = xgrid[BFaceFaces]
	xCoordinates = xgrid[Coordinates]
	nbfaces = size(xBFaceNodes, 2)
	nnodes = num_nodes(xgrid)
	nnodes4bface = size(xBFaceNodes, 1)
	EG = xgrid[UniqueBFaceGeometries][1]
	xdim = size(xCoordinates, 1)
	nedges4bface = xdim == 3 ? num_faces(EG) : 0
	xBFaceMidPoints = zeros(Float64, xdim, nbfaces)
	for bface ∈ 1:nbfaces, j ∈ 1:xdim, bn ∈ 1:nnodes4bface
		xBFaceMidPoints[j, bface] += xCoordinates[j, xBFaceNodes[bn, bface]] / nnodes4bface
	end
	if xdim == 3
		xEdgeMidPoint = zeros(Float64, xdim)
		xEdgeMidPoint2 = zeros(Float64, xdim)
		xEdgeNodes = xgrid[EdgeNodes]
		xFaceEdges = xgrid[FaceEdges]
		if FEType <: H1P1
			nedgedofs = 0
		elseif FEType <: H1P2
			nedgedofs = 1
		elseif FEType <: H1P3
			nedgedofs = 2
		else
			@warn "get_periodic_coupling_info not yet working for non H1-conforming elements"
		end
	end
	@assert FEType <: AbstractH1FiniteElement "get_periodic_coupling_info not yet working for non H1-conforming elements"
	xBFaceDofs = FES[BFaceDofs]
	dofsX, dofsY, factors = Int[], Int[], Int[]
	counterface = 0
	nfb = 0
	partners = zeros(Int, xdim)
	coffsets = ExtendableFEMBase.get_local_coffsets(FEType, ON_BFACES, EG)
	nedgedofs = 0

	for bface ∈ 1:nbfaces
		counterface = 0
		nfb = num_targets(xBFaceDofs, bface)
		if xBFaceRegions[bface] == b1
			for bface2 ∈ 1:nbfaces
				if xBFaceRegions[bface2] == b2
					if is_opposite(view(xBFaceMidPoints, :, bface), view(xBFaceMidPoints, :, bface2))
						counterface = bface2
						break
					end
				end
			end
		end
		if counterface > 0

			# couple first two node dofs in opposite order due to orientation
			for c ∈ 1:ncomponents
				if factor_components[c] == 0
					continue
				end
				nfbc = coffsets[c+1] - coffsets[c] # total dof count for this component

				# couple nodes
				for nb ∈ 1:nnodes4bface
					## find node partner on other side that evaluates true in is_ooposite function
					for nc ∈ 1:nnodes4bface
						if is_opposite(view(xCoordinates, :, xBFaceDofs[nb, bface]), view(xCoordinates, :, xBFaceDofs[nc, counterface]))
							partners[nb] = nc
							break
						end
					end
					## couple node dofs (to be skipped for e.g. Hdiv, Hcurl elements)
					push!(dofsX, xBFaceDofs[coffsets[c]+nb, bface])
					push!(dofsY, xBFaceDofs[coffsets[c]+partners[nb], counterface])
				end
				# @info "matching face $bface (nodes = $(xBFaceNodes[:,bface]), dofs = $(xBFaceDofs[:,bface])) with face $counterface (nodes = $(xBFaceNodes[:,counterface]), dofs = $(xBFaceDofs[:,counterface])) with partner node order $partners"

				## couple edges
				if nedges4bface > 0 && FEType <: H1P2 || FEType <: H1P3
					# todo: for H1P3 edge orientation place a role !!!
					for nb ∈ 1:nedges4bface
						fill!(xEdgeMidPoint, 0)
						for j ∈ 1:xdim, k ∈ 1:2
							xEdgeMidPoint[j] += xCoordinates[j, xEdgeNodes[k, xFaceEdges[nb, xBFaceFaces[bface]]]] / 2
						end
						## find edge partner on other side that evaluates true at edge midpoint in is_opposite function
						for nc ∈ 1:nnodes4bface
							fill!(xEdgeMidPoint2, 0)
							for j ∈ 1:xdim, k ∈ 1:2
								xEdgeMidPoint2[j] += xCoordinates[j, xEdgeNodes[k, xFaceEdges[nc, xBFaceFaces[counterface]]]] / 2
							end
							if is_opposite(xEdgeMidPoint, xEdgeMidPoint2)
								partners[nb] = nc
								break
							end
						end

						## couple edge dofs (local orientation information is needed for more than one dof on each edge !!! )
						for k ∈ 1:nedgedofs
							push!(dofsX, xBFaceDofs[coffsets[c]+nnodes4bface+nb+(k-1)*nedgedofs, bface])
							push!(dofsY, xBFaceDofs[coffsets[c]+nnodes4bface+partners[nb]+(k-1)*nedgedofs, counterface])
						end
					end
				end

				## couple face dofs (interior dofs of bface)
				for nb ∈ 1:nfbc-nnodes4bface-nedges4bface*nedgedofs
					push!(dofsX, xBFaceDofs[coffsets[c]+nnodes4bface+nedges4bface*nedgedofs+nb, bface])
					push!(dofsY, xBFaceDofs[coffsets[c]+nnodes4bface+nfbc-nnodes4bface-nedges4bface*nedgedofs+1-nb, counterface]) # couple face dofs in opposite order due to orientation (works in 2D at least)
				end
				append!(factors, ones(nfbc) * factor_components[c])
			end

			## couple remaining dofs (should be vector dofs)
			for dof ∈ coffsets[end]+1:nfb
				push!(dofsX, xBFaceDofs[dof, bface])
				push!(dofsY, xBFaceDofs[nfb-coffsets[end]+dof-1, counterface]) # couple face dofs in opposite order due to orientation (works in 2D at least, e.g. for Bernardi--Raugel)
				push!(factors, factor_vectordofs)
			end
		end
	end

	return dofsX, dofsY, factors
end

## determines a common assembly grid for the given arrays of finite element spaces
function determine_assembly_grid(FES_test, FES_ansatz = [], FES_args = [])
	xgrid = FES_test[1].xgrid
	dofgrid = FES_test[1].dofgrid
	all_same_xgrid = true
	all_same_dofgrid = true
	for j = 2 : length(FES_test)
		if xgrid !== FES_test[j].xgrid
			all_same_xgrid = false
		end
		if dofgrid !== FES_test[j].dofgrid
			all_same_dofgrid = false
		end
	end
	for j = 1 : length(FES_ansatz)
		if xgrid !== FES_ansatz[j].xgrid
			all_same_xgrid = false
		end
		if dofgrid !== FES_ansatz[j].dofgrid
			all_same_dofgrid = false
		end
	end
	for j = 1 : length(FES_args)
		if xgrid !== FES_args[j].xgrid
			all_same_xgrid = false
		end
		if dofgrid !== FES_args[j].dofgrid
			all_same_dofgrid = false
		end
	end
	if all_same_dofgrid
		return dofgrid
	elseif all_same_xgrid
		return xgrid
	else
		@warn "detected non-matching grids for involved finite element spaces, trying assembly on grid of first testfunction argument"
		return xgrid
	end
	return xgrid
end

## gets the dofmap for the FESpace FES fr the assemblygrid xgrid and the assembly type AT
function get_dofmap(FES, xgrid, AT)
	DM = Dofmap4AssemblyType(AT)
	if FES.dofgrid !== xgrid && FES.xgrid !== xgrid
		@warn "warning assembly grid does neither match FES dofgrid or parent grid!"
		return FES[DM]
	end	
	FES[DM]
	return FES.dofgrid === xgrid ? FES[DM] : FES[ParentDofmap4Dofmap(DM)]
end


# """
# ````
# function tensor_view(input, i, rank, dim)
# ````

# Returns a view of input[i] and following entries 
# reshaped as a tensor of rank `rank`.
# The parameter `dim` specifies the size of a tensor in each direction, 
# e.g. a 1-Tensor (Vector) of length(dim) or a dim x dim 2-Tensor (matrix).
# As an example `tensor_view(v,5,2,5)` returns a view of `v(5:29)`  
# as a 5x5 matrix.

# """
# function tensor_view(input, i::Int, rank::Int, dim::Int)
#     if rank == 0
#         return view(input, i:i)
#     elseif rank == 1
#         return view(input, i:i+dim-1)
#     elseif rank == 2
#         return reshape(view(input, i:(i+(dim*dim)-1)), (dim,dim))
#     elseif rank == 3
#         return reshape(view(input, i:(i+(dim*dim*dim)-1)), (dim, dim,dim))
#     else
#         @warn "tensor_view for rank > 3 is a general implementation that needs allocations!"
#         return reshape(view(input, i:(i+(dim^rank)-1)),ntuple(i->dim,rank))
#     end
# end


"""
````
function tmul!(y,A,x,α=1.0,β=0.0)
````

Combined inplace  matrix-vector multiply-add ``A^T x α + y β``.
The result is stored in `y` by overwriting it.  Note that `y` must not be
aliased with either `A` or `x`.

"""
function tmul!(y, A, x, α = 1.0, β = 0.0)
    for i in eachindex(y)
        y[i] *= β
        for j in eachindex(x)
            y[i] += α * A[j, i] * x[j]
        end
    end
end

"""
````
function tmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α=1.0, β=0.0) where {T<:AbstractFloat}
````

Overload of the generic function for types supported by 
`LinearAlgebra.BLAS.gemv!` to avoid slow run times for large inputs.
"""
function tmul!(
    y::AbstractVector{T},
    A::AbstractMatrix{T},
    x::AbstractVector{T},
    α=1.0,
    β=0.0) where {T<:AbstractFloat}
    LinearAlgebra.BLAS.gemv!('T', α, A, x, β, y)
end
