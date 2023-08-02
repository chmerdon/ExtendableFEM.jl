using Test
using ExtendableGrids
using ExtendableFEMBase
using ExtendableFEM

include("test_dgblf.jl")
include("test_itemintegrator.jl")

function run_all_tests()
    run_dgblf_tests()
    run_itemintegrator_tests()
end

run_all_tests()
