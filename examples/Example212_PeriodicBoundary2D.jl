#=

# 212 : Periodic Boundary 2D
([source code](@__SOURCE_URL__))

This is a simple demonstration and validation of the generic periodic boundary operator.

We construct an unstructured periodic 2D grid and solve a simple linear elastic problem
with periodic boundary coupling along the x-axis.

![](example212.png)
=#

module Example212_PeriodicBoundary2D

using ExtendableFEM
using ExtendableGrids
using SimplexGridFactory
using GridVisualize
using Triangulate
using UnicodePlots
using StaticArrays
using LinearAlgebra
using Test #hide
using SparseArrays

## enumerate the boundary regions
const reg_left = 4
const reg_right = 2
const reg_dirichlet = 1
const reg_default = 3

## 2D reduction of the material used in Example 312
## in Voigt notation
function material_tensor()
    c11 = 396.0
    c12 = 137.0
    c44 = 116.0

    return @SArray [
        c11 c12 0
        c12 c11 0
        0   0   c44
    ]
end

## generate the kernels for the linear problem
## 𝐂: Hooke tensor, 𝑓: body force
function make_kernels(𝐂, 𝑓)

    ## linear stress-strain mapping
    bilinear_kernel!(σ, εv, qpinfo) = mul!(σ, 𝐂, εv)

    ## plain body force
    linear_kernel!(result, qpinfo) = (result .= 𝑓)

    return bilinear_kernel!, linear_kernel!
end


"""
    create a 2D grid with Dirichlet boundary region at the bottom center
"""
function create_grid(; h, height, width)
    builder = SimplexGridBuilder(; Generator = Triangulate)

    ## bottom points
    b1 = point!(builder, 0, 0)
    b2 = point!(builder, 0.45 * width, 0)
    b3 = point!(builder, 0.5 * width, 0)
    b4 = point!(builder, 0.55 * width, 0)
    b5 = point!(builder, width, 0)

    ## top points
    t1 = point!(builder, 0, height)
    t2 = point!(builder, width / 2, height)
    t3 = point!(builder, width, height)

    ## default faces
    facetregion!(builder, reg_default)
    facet!(builder, b1, b2)
    facet!(builder, b4, b5)
    facet!(builder, t1, t2)
    facet!(builder, t2, t3)

    ## left face
    facetregion!(builder, reg_left)
    facet!(builder, b1, t1)

    ## right face
    facetregion!(builder, reg_right)
    facet!(builder, b5, t3)

    ## Dirichlet face
    facetregion!(builder, reg_dirichlet)
    facet!(builder, b3, b4)
    facet!(builder, b2, b3)

    ## divider
    facetregion!(builder, reg_default)
    facet!(builder, t2, b3)


    cellregion!(builder, 1)
    maxvolume!(builder, h)
    regionpoint!(builder, width / 3, height / 2)

    cellregion!(builder, 2)
    # much finer grid on the right half to make periodic coupling non-trivial
    maxvolume!(builder, 0.1 * h)
    regionpoint!(builder, 2width / 3, height / 2)

    return simplexgrid(builder)
end

function main(;
        order = 1,
        periodic = true,
        Plotter = nothing,
        force = 10.0,
        h = 1.0e-2,
        width = 6.0,
        height = 1.0,
        kwargs...
    )
    xgrid = create_grid(; h, width, height)

    ## create finite element space and solution vector
    if order == 1
        FES = FESpace{H1P1{2}}(xgrid)
    elseif order == 2
        FES = FESpace{H1P2{2, 2}}(xgrid)
    end

    ## problem description
    PD = ProblemDescription()
    u = Unknown("u"; name = "displacement")
    assign_unknown!(PD, u)

    𝐂 = material_tensor()
    𝑓 = force * [0, 1]

    bilinear_kernel!, linear_kernel! = make_kernels(𝐂, 𝑓)
    assign_operator!(PD, BilinearOperator(bilinear_kernel!, [εV(u, 1.0)]; kwargs...))
    assign_operator!(PD, LinearOperator(linear_kernel!, [id(u)]; kwargs...))

    assign_operator!(PD, HomogeneousBoundaryData(u; regions = [reg_dirichlet]))

    if periodic
        function give_opposite!(y, x)
            y .= x
            y[1] = width - x[1]
            return nothing
        end

        coupling_matrix = get_periodic_coupling_matrix(FES, xgrid, reg_left, reg_right, give_opposite!)
        display(coupling_matrix)
        assign_operator!(PD, CombineDofs(u, u, coupling_matrix; kwargs...))
    end

    sol = solve(PD, FES)

    plt = GridVisualizer(; Plotter, size = (1300, 800))

    magnification = 1
    displaced_grid = deepcopy(xgrid)
    displace_mesh!(displaced_grid, sol[1], magnify = magnification)
    gridplot!(plt, displaced_grid, linewidth = 1, title = "displaced mesh, $(magnification)x magnified", scene3d = :LScene)

    return sol, plt
end

generateplots = ExtendableFEM.default_generateplots(Example212_PeriodicBoundary2D, "example212.png") #hide
function runtests()                                                                                  #hide
    sol, plt = main()                                                                                #hide
    @test maximum(view(sol[1])) ≈ 1.3447465095618172                                                 #hide
    return nothing                                                                                   #hide
end                                                                                                  #hide

end # module
