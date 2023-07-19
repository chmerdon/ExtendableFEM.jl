using Test
using ExtendableGrids
using ExtendableFEMBase
using ExtendableFEM


include("test_segmentintegrator.jl")
include("test_pointevaluator.jl")
include("test_itemintegrator.jl")

function run_all_tests()
    run_itemintegrator_tests()
    run_segmentintegrator_tests()
    run_pointevaluator_tests()
end

run_all_tests()
