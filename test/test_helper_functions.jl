# check if coords are opposite to each other
function test_coords(X, Y, xgrid, is_opposite)
    coords = xgrid[Coordinates]
    for (x, y) in zip(X, Y)
        if @views !is_opposite(coords[:, x], coords[:, y])
            @show coords[:, x] coords[:, y]
            return false
        end
    end
    return true
end


@testset "get_periodic_coupling_info" begin

    xgrid = simplexgrid(0:100, 0:100)
    # couple left <-> right
    let
        is_opposite(x, y) = abs(x[2] - y[2]) < 1.0e-12
        FES = FESpace{H1Pk{1, 2, 1}}(xgrid)
        X, Y, F = get_periodic_coupling_info(FES, xgrid, 4, 2, is_opposite)
        @test test_coords(X, Y, xgrid, is_opposite)
    end
    # couple top <-> bottom
    let
        is_opposite(x, y) = abs(x[1] - y[1]) < 1.0e-12
        FES = FESpace{H1Pk{1, 2, 1}}(xgrid)
        X, Y, F = get_periodic_coupling_info(FES, xgrid, 1, 3, is_opposite)
        @test test_coords(X, Y, xgrid, is_opposite)
    end
end


# check if matrix is a proper coupling matrix
# for testing, we assume that the FES on the "from" boundary is
# contained in the FES in the "to" boundary
function test_matrix(matrix; structured_grid = true)

    if structured_grid
        # only 1.0 entries!
        if findfirst(≉(1.0, atol = 1.0e-8), matrix.nzval) !== nothing
            @show "found entries ≉ 1.0 "
            return false
        end

        # at most one entry in each col
        for i in 1:size(matrix, 2)
            if sum(matrix[:, i]) > 1.0 + 1.0e-8
                @show sum(matrix[:, i]) i
                return false
            end
        end
    end

    # row sum is 0.0 or 1.0
    for i in 1:size(matrix, 1)
        row_sum = sum(matrix[i, :])
        if !(row_sum == 0.0 || row_sum ≈ 1.0)
            @show row_sum i
            return false
        end
    end

    return true
end

@testset "get_periodic_coupling_matrix" begin

    # combine left/right at x=0/1
    function give_opposite!(y, x)
        y .= x
        y[1] = 1 - x[1]
        return nothing
    end

    let # 3D P1
        xgrid = simplexgrid(0:0.1:1.0, 0:0.1:1.0, 0:0.1:1.0)
        FES = FESpace{H1P1{1}}(xgrid)
        A = get_periodic_coupling_matrix(FES, xgrid, 4, 2, give_opposite!, sparsity_tol = 1.0e-8)
        @test test_matrix(A)
    end

    let # 3D P2 with 2 components
        xgrid = simplexgrid(0:0.5:1.0, 0:0.5:1.0, 0:0.5:1.0)
        FES = FESpace{H1P2{2, 3}}(xgrid)
        A = get_periodic_coupling_matrix(FES, xgrid, 4, 2, give_opposite!, sparsity_tol = 1.0e-8)
        @test test_matrix(A)
    end

    let # 2D unstructured
        b = SimplexGridBuilder(Generator = Triangulate)
        grid1 = simplexgrid(0:1.0, 0:1.0)
        grid2 = simplexgrid(0:1.0, 0:0.5:1.0)
        bregions!(b, grid1, 1 => 1, 3 => 3, 4 => 4)
        bregions!(b, grid2, 2 => 2)
        xgrid = simplexgrid(b)

        FES = FESpace{H1P1{1}}(xgrid)
        A = get_periodic_coupling_matrix(FES, xgrid, 4, 2, give_opposite!, sparsity_tol = 1.0e-8)
        @test test_matrix(A; structured_grid = false)
    end
end
