function run_dgblf_tests()
    
    @testset "ItemIntegrator" begin
        println("\n")
        println("==========================")
        println("Testing BilinearOperatorDG")
        println("==========================")
            
        @test TestDGBLF(1) < 1e-14
    end
end

## stab_kernel!
function stab_kernel!(result, ∇u, qpinfo)
    result .= ∇u .* qpinfo.volume^2
end

function TestDGBLF(order = 3)
    ## tests if jumps of gradient of interpolation of polynomial in ansatz space is zero
    function u!(result, qpinfo)
        result[1] = qpinfo.x[1]^order
    end

    FEType = H1Pk{1,2,order}
    xgrid = grid_unitsquare(Triangle2D)
    FES = FESpace{FEType}(xgrid)
    uh = FEVector(FES)
    interpolate!(uh[1], u!)
    A = FEMatrix(FES, FES)

    dgblf = BilinearOperatorDG(stab_kernel!, [jump(grad(1))]; entities = ON_IFACES, quadorder = 0, factor = 1e-2)
    assemble!(A, dgblf)

    error = uh.entries' * A.entries * uh.entries

    return error
end