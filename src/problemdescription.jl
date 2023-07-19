

mutable struct ProblemDescription
    name::String
    unknowns::Array{Unknown,1}
    operators::Array{AbstractOperator,1}
    reduction_operators::Array{AbstractReductionOperator,1}
end

function ProblemDescription(name = "N.N.")
    return ProblemDescription(name, Array{Unknown,1}(undef,0), Array{AbstractOperator,1}(undef,0), Array{AbstractReductionOperator,1}(undef,0))
end

function assign_unknown!(PD::ProblemDescription, u::Unknown)
    push!(PD.unknowns, u)
end

function assign_operator!(PD::ProblemDescription, u::AbstractOperator)
    push!(PD.operators, u)
end

function assign_reduction!(PD::ProblemDescription, u::AbstractReductionOperator)
    push!(PD.reduction_operators, u)
end

function Base.show(io::IO, PD::ProblemDescription)
    println(io, "\nPDE-DESCRIPTION")
    println(io, "    • name = $(PD.name)")
    println(io, "\n  <<<UNKNOWNS>>>") 
    for u in PD.unknowns
        print(io, "    • $u")
    end

    println(io, "\n  <<<OPERATORS>>>") 
    for o in PD.operators
        println(io, "    • $(o.parameters[:name])")
    end

    #println(io, " reductions =") 
    #for o in PD.reduction_operators
    #    println(io, "    • $o")
    #end
end


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
    nbfaces = size(xBFaceNodes,2)
    nnodes = num_nodes(xgrid)
    nnodes4bface = size(xBFaceNodes,1)
    EG = xgrid[UniqueBFaceGeometries][1]
    xdim = size(xCoordinates,1)
    nedges4bface = xdim == 3 ? num_faces(EG) : 0
    xBFaceMidPoints = zeros(Float64,xdim,nbfaces)
    for bface = 1 : nbfaces, j = 1 : xdim, bn = 1 : nnodes4bface 
        xBFaceMidPoints[j,bface] += xCoordinates[j,xBFaceNodes[bn,bface]] / nnodes4bface 
    end
    if xdim == 3
        xEdgeMidPoint = zeros(Float64,xdim)
        xEdgeMidPoint2 = zeros(Float64,xdim)
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
    
    for bface = 1 : nbfaces
        counterface = 0
        nfb = num_targets(xBFaceDofs, bface)
        if xBFaceRegions[bface] == b1
            for bface2 = 1 : nbfaces
                if xBFaceRegions[bface2] == b2
                    if is_opposite(view(xBFaceMidPoints,:,bface), view(xBFaceMidPoints,:,bface2))
                        counterface = bface2
                        break
                    end
                end
            end
        end
        if counterface > 0

            # couple first two node dofs in opposite order due to orientation
            for c = 1 : ncomponents
                if factor_components[c] == 0
                    continue
                end
                nfbc = coffsets[c+1] - coffsets[c] # total dof count for this component

                # couple nodes
                for nb = 1 : nnodes4bface 
                    ## find node partner on other side that evaluates true in is_ooposite function
                    for nc = 1 : nnodes4bface 
                        if is_opposite(view(xCoordinates,:,xBFaceDofs[nb,bface]), view(xCoordinates,:,xBFaceDofs[nc,counterface]))
                            partners[nb] = nc
                            break
                        end 
                    end
                    ## couple node dofs (to be skipped for e.g. Hdiv, Hcurl elements)
                    push!(dofsX, xBFaceDofs[coffsets[c]+nb,bface])
                    push!(dofsY, xBFaceDofs[coffsets[c]+partners[nb],counterface])
                end
                # @info "matching face $bface (nodes = $(xBFaceNodes[:,bface]), dofs = $(xBFaceDofs[:,bface])) with face $counterface (nodes = $(xBFaceNodes[:,counterface]), dofs = $(xBFaceDofs[:,counterface])) with partner node order $partners"
            
                ## couple edges
                if nedges4bface > 0 && FEType <: H1P2 || FEType <: H1P3
                    # todo: for H1P3 edge orientation place a role !!!
                    for nb = 1 : nedges4bface 
                        fill!(xEdgeMidPoint,0)
                        for j = 1 : xdim, k = 1 : 2
                            xEdgeMidPoint[j] += xCoordinates[j,xEdgeNodes[k,xFaceEdges[nb,xBFaceFaces[bface]]]] / 2 
                        end
                        ## find edge partner on other side that evaluates true at edge midpoint in is_opposite function
                        for nc = 1 : nnodes4bface 
                            fill!(xEdgeMidPoint2,0)
                            for j = 1 : xdim, k = 1 : 2
                                xEdgeMidPoint2[j] += xCoordinates[j,xEdgeNodes[k,xFaceEdges[nc,xBFaceFaces[counterface]]]] / 2
                            end
                            if is_opposite(xEdgeMidPoint, xEdgeMidPoint2)
                                partners[nb] = nc
                                break
                            end 
                        end

                        ## couple edge dofs (local orientation information is needed for more than one dof on each edge !!! )
                        for k = 1 : nedgedofs
                            push!(dofsX, xBFaceDofs[coffsets[c]+nnodes4bface+nb+(k-1)*nedgedofs,bface])
                            push!(dofsY, xBFaceDofs[coffsets[c]+nnodes4bface+partners[nb]+(k-1)*nedgedofs,counterface])
                        end
                    end
                end

                ## couple face dofs (interior dofs of bface)
                for nb = 1 : nfbc-nnodes4bface-nedges4bface*nedgedofs
                    push!(dofsX, xBFaceDofs[coffsets[c]+nnodes4bface+nedges4bface*nedgedofs+nb,bface])
                    push!(dofsY, xBFaceDofs[coffsets[c]+nnodes4bface+nfbc-nnodes4bface-nedges4bface*nedgedofs + 1 - nb,counterface]) # couple face dofs in opposite order due to orientation (works in 2D at least)
                end
                append!(factors, ones(nfbc) * factor_components[c])
            end
            
            ## couple remaining dofs (should be vector dofs)
            for dof = coffsets[end]+1:nfb
                push!(dofsX, xBFaceDofs[dof,bface])
                push!(dofsY, xBFaceDofs[nfb-coffsets[end]+dof-1,counterface]) # couple face dofs in opposite order due to orientation (works in 2D at least, e.g. for Bernardi--Raugel)
                push!(factors, factor_vectordofs)
            end
        end
    end

    return dofsX, dofsY, factors
end