function run_dgblf_tests()

    return @testset "BilinearOperatorDG" begin
        println("\n")
        println("==========================")
        println("Testing BilinearOperatorDG")
        println("==========================")

        @test TestParallelAssemblyDGBLF() < 5.0e-10 # windows/macOS-14 tests on github need greater tolerance

        for operator in [jump(grad(1)), jump(id(1))]
            TestDGBLF(H1Pk{1, 2, 1}, 1, operator)
            TestDGBLF(H1Pk{1, 2, 2}, 2, operator)
            TestDGBLF(H1Pk{1, 2, 3}, 3, operator)
            TestDGBLF(HDIVBDM1{2}, 1, operator)
            TestDGBLF(HDIVRT1{2}, 1, operator)
            TestDGBLF(HDIVBDM2{2}, 2, operator)
        end
    end
end

## stab_kernel!
function stab_kernel!(result, input, qpinfo)
    return result .= input / qpinfo.volume
end

function TestDGBLF(FEType = H1Pk{1, 2, 3}, order = get_polynomialorder(FEType, Triangle2D), operator = jump(grad(1)), tol = 1.0e-12)
    ## tests if jumps of polynomial in ansatz space is zero
    ncomponents = get_ncomponents(FEType)
    function u!(result, qpinfo)
        for j in 1:ncomponents
            result[j] = qpinfo.x[1]^order + j * qpinfo.x[2]^order
        end
        return
    end

    xgrid = uniform_refine(grid_unitsquare(Triangle2D), 2)
    FES = FESpace{FEType}(xgrid)

    ## first test: interpolate a smooth function from the discrete space and check if its interpolation has no jumps
    uh = FEVector(FES)
    interpolate!(uh[1], u!)
    A = FEMatrix(FES, FES)

    dgblf = BilinearOperatorDG(stab_kernel!, [operator]; entities = ON_IFACES, quadorder = 2 * order, factor = 1.0e-2)
    assemble!(A, dgblf)

    error = uh.entries' * A.entries * uh.entries
    @info "DG jump test for FEType=$FEType with order = $order yields error = $error"
    @test error < tol

    ## second test: solve best-approximatio problem for the same problem with jump penalization and see if the correct function is found
    PD = ProblemDescription()
    u = Unknown("u")
    assign_unknown!(PD, u)
    assign_operator!(PD, BilinearOperator([id(u)]))
    assign_operator!(PD, BilinearOperator(A, [u], [u]))
    assign_operator!(PD, LinearOperator(u!, [id(u)]; bonus_quadorder = order))
    assign_operator!(PD, InterpolateBoundaryData(u, u!; regions = 1:4, bonus_quadorder = order))

    uh = solve(PD, FES)
    error = uh.entries' * A.entries * uh.entries
    @info "DG jump test 2 for FEType=$FEType with order = $order yields error = $error"
    @test error < tol


    return error
end

function TestParallelAssemblyDGBLF(FEType = H1Pk{1, 2, 3}, order = get_polynomialorder(FEType, Triangle2D), operator = jump(grad(1)), verbosity = 1)

    dgblf_seq = BilinearOperatorDG(stab_kernel!, [operator]; entities = ON_IFACES, quadorder = 2 * order, factor = 1.0e-2, parallel = false, verbosity = verbosity)
    dgblf_par = BilinearOperatorDG(stab_kernel!, [operator]; entities = ON_IFACES, quadorder = 2 * order, factor = 1.0e-2, parallel = true, verbosity = verbosity)

    ## sequential assembly
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), 4)
    FES = FESpace{FEType}(xgrid)
    Aseq = FEMatrix(FES, FES)
    assemble!(Aseq, dgblf_seq)

    ## parallel assembly on same grid
    xgrid = partition(xgrid, PlainMetisPartitioning(npart = 20), edges = true)
    FES = FESpace{FEType}(xgrid)
    Apar = FEMatrix(FES, FES; npartitions = num_partitions(xgrid))
    assemble!(Apar, dgblf_par)

    ## compare the two matrices
    ## since partitioning changes dof enumeration only norms are compared
    nor = abs(norm(Apar.entries.cscmatrix) - norm(Aseq.entries.cscmatrix))
    @info "difference between norms of sequantially and parallel assembled DG matrix = $nor"
    return nor
end
