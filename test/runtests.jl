using Test
using ExtendableGrids
using ExtendableFEMBase
using ExtendableFEM

include("test_itemintegrator.jl")

function run_all_tests()
    run_itemintegrator_tests()
end

run_all_tests()
