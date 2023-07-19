module Example240_StokesSVRTEnrichment

using ExtendableFEM
using ExtendableFEMBase
using GridVisualize
using ExtendableGrids
using ExtendableSparse
using Triangulate
using SimplexGridFactory

const μ = 1

function rhs!(result, qpinfo)
    t = qpinfo.time
    x = qpinfo.x
    ## Laplacian
    result[1] =  2*π^2*sin(π*x[2])*cos(π*x[2])*(cos(π*x[1])^2 - sin(π*x[1])^2) - 4*π^2*sin(π*x[1])^2*sin(π*x[2])*cos(π*x[2]);
    result[2] = -2*π^2*sin(π*x[1])*cos(π*x[1])*(cos(π*x[2])^2 - sin(π*x[2])^2) + 4*π^2*sin(π*x[2])^2*sin(π*x[1])*cos(π*x[1]);
    result .*= - μ * 2*π*sin(π*t);
    ## ∇p
    result[1] += 40*x[1]*x[2]*sin(π*t);
    result[2] += 20*x[1]^2*sin(π*t);
    return nothing
end

function exact_u!(result, qpinfo)
    t = qpinfo.time
    x = qpinfo.x
    result[1] = sin(π*x[1])^2*sin(π*x[2])*cos(π*x[2]);
    result[2] = -sin(π*x[2])^2*sin(π*x[1])*cos(π*x[1]);
    result .*= 2*π*sin(π*t);
    return nothing;
end

function exact_p!(result, qpinfo)
    t = qpinfo.time
    x = qpinfo.x
    result[1] = 20*sin(π*t)*(x[1]^2*x[2] - 1/6)
    return nothing;
end

function exact_∇u!(result, qpinfo)
    t = qpinfo.time
    x = qpinfo.x
    result[1] = 2*pi*sin(π*x[1])*cos(π*x[1])*sin(π*x[2])*cos(π*x[2]);
    result[2] = π*sin(π*x[1])^2*(cos(π*x[2])^2 - sin(π*x[2])^2);
    result[3] = -π*sin(π*x[2])^2*(cos(π*x[1])^2 - sin(π*x[1])^2);
    result[4] = -2*pi*sin(π*x[1])*cos(π*x[1])*sin(π*x[2])*cos(π*x[2]);
    result .*= 2*π*sin(π*t);
    return nothing;
end

function exact_error!(result, u, qpinfo)
    exact_u!(result, qpinfo)
    exact_∇u!(view(result,3:6), qpinfo)
    exact_p!(view(result,7), qpinfo)
    result .-= u
    result .= result.^2
end

function kernel_stokes_standard!(result, u_ops, params)
    ∇u, p = view(u_ops,1:4), view(u_ops, 5)
    result[1] = μ*∇u[1] - p[1]
    result[2] = μ*∇u[2]
    result[3] = μ*∇u[3]
    result[4] = μ*∇u[4] - p[1]            
    result[5] = -(∇u[1] + ∇u[4])
    return nothing
end

function get_grid2D(nref; uniform = false, barycentric = false)
    if uniform || barycentric
        gen_ref = 0
    else
        gen_ref = nref
    end
    grid = simplexgrid(Triangulate;
                points=[0 0 ; 0 1 ; 1 1 ; 1 0]',
                bfaces=[1 2 ; 2 3 ; 3 4 ; 4 1 ]',
                bfaceregions=[1, 2, 3, 4],
                regionpoints=[0.5 0.5;]',
                regionnumbers=[1],
                regionvolumes=[4.0^(-gen_ref-1)])
    if uniform
        grid = uniform_refine(grid, nref)
    end
	if barycentric
		grid = barycentric_refine(grid)
	end
	return grid
end


function main(; nrefs = 4, order = 2, Plotter = nothing, enrich = true, reduce = true, time = 0.5, bonus_quadorder = 5, kwargs...)

    ## grid
    xgrid = get_grid2D(nrefs)

    ## define and assign unknowns
    PD = ProblemDescription("Stokes problem")
    u = Unknown("u"; name = "velocity", dim = 2)
    p = Unknown("p"; name = "pressure", dim = 1)
    uR = Unknown("uR"; name = "velocity enrichment", dim = 2) # only used if enrich == true
    assign_unknown!(PD, u)
    assign_unknown!(PD, p)

    ## FESpaces
    if order == 1
        FES_enrich = FESpace{HDIVRT0{2}}(xgrid)
    else
        FES_enrich = FESpace{HDIVRTkENRICH{2,order-1,reduce}}(xgrid)
    end
    FES = Dict(u => FESpace{H1Pk{2,2,order}}(xgrid),
               p => FESpace{order == 1 || reduce ? L2P0{1} : H1Pk{1,2,order-1}}(xgrid; broken = true),
               uR => enrich ? FES_enrich : nothing)

    ## define operators
    assign_operator!(PD, LinearOperator(rhs!, [id(u)]; bonus_quadorder = 5, kwargs...)) 
    assign_operator!(PD, BilinearOperator(kernel_stokes_standard!, [grad(u), id(p)]; kwargs...))  
    if enrich
        if reduce
            @time if order == 1
                @info "... preparing condensation of RT0 dofs"
                AR = FEMatrix(FES_enrich)
                BR = FEMatrix(FES[p], FES_enrich)
                bR = FEVector(FES_enrich)
                assemble!(AR, BilinearOperator([div(1)]; lump = true, factor = μ, kwargs...))
                for bface in xgrid[BFaceFaces]
                    AR.entries[bface,bface] = 1e60
                end
                flush!(AR.entries)
                assemble!(BR, BilinearOperator([id(1)], [div(1)]; factor = -1, kwargs...))
                assemble!(bR, LinearOperator(rhs!, [id(1)]; bonus_quadorder = 5, kwargs...); time = time)
                ## invert AR (diagonal matrix)
                AR.entries.cscmatrix.nzval .= 1 ./ AR.entries.cscmatrix.nzval
                C = -BR.entries.cscmatrix * AR.entries.cscmatrix * BR.entries.cscmatrix'
                c = -BR.entries.cscmatrix * AR.entries.cscmatrix * bR.entries
                assign_operator!(PD, BilinearOperator(C, [p], [p]; kwargs...))
                assign_operator!(PD, LinearOperator(c, [p]; kwargs...))
            else
                @info "... preparing removal of enrichment dofs"
                BR = FEMatrix(FES[p], FES_enrich)
                A1R = FEMatrix(FES_enrich, FES[u])
                bR = FEVector(FES_enrich)
                assemble!(BR, BilinearOperator([id(1)], [div(1)]; factor = -1, kwargs...))
                assemble!(bR, LinearOperator(rhs!, [id(1)]; bonus_quadorder = 5, kwargs...); time = time)
                assemble!(A1R, BilinearOperator([id(1)],[Δ(1)]; factor = -μ, kwargs...))
                F = div_projector(xgrid, FES[u], FES_enrich)
                C = F.entries.cscmatrix * A1R.entries.cscmatrix
                assign_operator!(PD, BilinearOperator(C, [u], [u]; factor = 1, transposed_copy = -1, kwargs...))
                assign_operator!(PD, LinearOperator(F.entries.cscmatrix * bR.entries, [u]; kwargs...))
            end
        else
            assign_unknown!(PD, uR)
            assign_operator!(PD, LinearOperator(rhs!, [id(uR)]; bonus_quadorder = 5, kwargs...))
            assign_operator!(PD, BilinearOperator([id(p)], [div(uR)]; transposed_copy = 1, factor = -1, kwargs...))
            if order == 1
                assign_operator!(PD, BilinearOperator([div(uR)]; lump = true, factor = μ, kwargs...))
                assign_operator!(PD, HomogeneousBoundaryData(uR; regions = 1:4))
            else
                assign_operator!(PD, BilinearOperator([Δ(u)], [id(uR)]; factor = μ, transposed_copy = -1, kwargs...))
            end
        end
    end
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4))
    assign_operator!(PD, FixDofs(p; dofs = [1], vals = [0]))

    ## solve
	sol = solve!(PD, FES; time = time, kwargs...)

    ## move integral mean of pressure
    pintegrate = ItemIntegrator([id(p)])
    pmean = sum(evaluate!(pintegrate, sol)) / sum(xgrid[CellVolumes])
    view(sol[p]) .-= pmean

    ## postprocess
    if enrich && reduce
        append!(sol, FES_enrich; tag = uR)
        if order == 1
            ## compute enrichment part of velocity
            view(sol[uR]) .= AR.entries.cscmatrix * (bR.entries - BR.entries.cscmatrix' * view(sol[p]))
        else
            ## compute enrichment part of velocity
            view(sol[uR]) .= F.entries.cscmatrix' * view(sol[u])
        end
    end

    ## error calculation
    ErrorIntegratorExact = ItemIntegrator(exact_error!, [id(u), grad(u), id(p)]; quadorder = 2*(order+1), kwargs...)
    error = evaluate!(ErrorIntegratorExact, sol; time = time)
    L2errorU = sqrt(sum(view(error,1,:)) + sum(view(error,2,:)))
    H1errorU = sqrt(sum(view(error,3,:)) + sum(view(error,4,:)) + sum(view(error,5,:)) + sum(view(error,6,:)))
    L2errorP = sqrt(sum(view(error,7,:)))
    @info "L2error(u) = $L2errorU"
    @info "L2error(∇u) = $H1errorU"
    @info "L2error(p) = $L2errorP"
    if enrich
        evaluate!(error, L2NormIntegrator([id(uR)]; quadorder = 2*order), sol)
        L2normUR = sqrt(sum(view(error,1,:)) + sum(view(error,2,:)))
        @info "L2norm(uR) = $L2normUR"
    end

    ## prepare plots
    pl=GridVisualizer(; Plotter = Plotter, layout = (1,3), clear = true, resolution = (1200,400))
    scalarplot!(pl[1,1], xgrid, nodevalues(sol[u]; abs = true)[1,:]; Plotter = Plotter)
    scalarplot!(pl[1,2], xgrid, nodevalues(sol[p])[1,:]; Plotter = Plotter)
    if order == 1 && enrich
        scalarplot!(pl[1,3], xgrid, nodevalues(sol[uR]; abs = true)[1,:]; Plotter = Plotter)
    end
end


function div_projector(xgrid, V1, VR)

    ## setup interpolation matrix
    celldofs_V1::VariableTargetAdjacency{Int32} = V1[CellDofs]
    celldofs_VR::VariableTargetAdjacency{Int32} = VR[CellDofs]
    ndofs_V1 = max_num_targets_per_source(celldofs_V1)
    ndofs_VR = max_num_targets_per_source(celldofs_VR)

    DD_RR = FEMatrix(VR)
    assemble!(DD_RR, BilinearOperator([div(1)]))
    DD_RRE::ExtendableSparseMatrix{Float64,Int64} = DD_RR.entries
    DD_1R = FEMatrix(V1, VR)
    assemble!(DD_1R, BilinearOperator([div(1)]))
    DD_1RE::ExtendableSparseMatrix{Float64,Int64} = DD_1R.entries
    Ap = zeros(Float64,ndofs_VR,ndofs_VR)
    bp = zeros(Float64,ndofs_VR)
    xp = zeros(Float64,ndofs_VR)
    dof::Int = 0
    dof2::Int = 0
    ncells::Int = num_cells(xgrid)
    F = FEMatrix(V1, VR)
    FE::ExtendableSparseMatrix{Float64,Int64} = F.entries
    for cell = 1 : ncells

        ## solve local pressure reconstruction for RTk part
        for dof_j = 1 : ndofs_VR
            dof = celldofs_VR[dof_j, cell]
            for dof_k = 1 : ndofs_VR
                dof2 = celldofs_VR[dof_k, cell]
                Ap[dof_j,dof_k] = DD_RRE[dof,dof2]
            end
        end
        
        for dof_j = 1 : ndofs_V1
            dof = celldofs_V1[dof_j, cell]
            for dof_k = 1: ndofs_VR
                dof2 = celldofs_VR[dof_k, cell]
                bp[dof_k] = -DD_1RE[dof, dof2]
            end

            xp = Ap \ bp

            for dof_k = 1: ndofs_VR
                dof2 = celldofs_VR[dof_k, cell]
                FE[dof, dof2] = xp[dof_k]
            end
        end
    end
    flush!(FE)
    return F
end

end # module