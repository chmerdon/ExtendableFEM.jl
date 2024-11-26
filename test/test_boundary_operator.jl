function run_boundary_operator_tests()

    return @testset "Boundary Operator" begin
        println("\n")
        println("==========================")
        println("Testing Boundary Operator")
        println("==========================")

        for broken in [false, true]
            TestInterpolateBoundary(H1P1{2}, 1, broken)
            TestInterpolateBoundary(H1P2{2, 2}, 2, broken)
            TestInterpolateBoundary(H1P3{2, 2}, 3, broken)
            TestInterpolateBoundary(HDIVRT0{2}, 0, broken)
            TestInterpolateBoundary(HDIVRT1{2}, 1, broken)
            TestInterpolateBoundary(HDIVBDM1{2}, 1, broken)
            TestInterpolateBoundary(HDIVBDM2{2}, 2, broken)
        end
    end
end

function TestInterpolateBoundary(FEType, order = get_polynomialorder(FEType, Triangle2D), broken = false)
    ## tests if boundary data for polynomial in ansatz space matches its interpolation
    ## which should be reliable (since verified by tests in ExtendableFEMBase)
    ncomponents = get_ncomponents(FEType)
    function u!(result, qpinfo)
        for j in 1:ncomponents
            result[j] = qpinfo.x[1]^order + j * qpinfo.x[2]^order
        end
        return
    end

    xgrid = reference_domain(Triangle2D) #uniform_refine(grid_unitsquare(Triangle2D),0)
    FES = FESpace{FEType}(xgrid; broken = broken)

    ## first test: interpolate a smooth function from the discrete space and check if its interpolation has no jumps
    uh = FEVector(FES)
    interpolate!(uh[1], u!; bonus_quadorder = order)
    uh2 = FEVector(FES)
    boundary_operator = InterpolateBoundaryData(1, u!; bonus_quadorder = order, regions = 1:4)
    assemble!(boundary_operator, FES)
    boundary_dofs = fixed_dofs(boundary_operator)
    apply!(uh2[1], boundary_operator)
    error = norm(uh.entries[boundary_dofs] - uh2.entries[boundary_dofs])
    @info "Error for boundary operator for $FEType (broken = $broken) = $error"
    return @test error < 1.0e-15
end
