function run_itemintegrator_tests()
    
    @testset "ItemIntegrator" begin
        println("\n")
        println("======================")
        println("Testing ItemIntegrator")
        println("======================")
            
        @test TestInterpolationErrorIntegration(3) < 1e-15
    end
end


function TestInterpolationErrorIntegration(order = 3)
    function u!(result, qpinfo)
        x =qpinfo.x
        result[1] = x[1]^2 + x[2]
    end
    function exact_error!(result, u, qpinfo)
        u!(result, qpinfo)
        result .-= u
        result .= result.^2
    end
    ErrorIntegrator = ItemIntegrator(exact_error!, [id(1)]; quadorder = 2*order)

    FEType = H1Pk{1,2,order}
    xgrid = grid_unitsquare(Triangle2D)
    FES = FESpace{FEType}(xgrid)
    uh = FEVector(FES)
    interpolate!(uh[1], u!)

    return maximum(evaluate(ErrorIntegrator, uh)[:])
end