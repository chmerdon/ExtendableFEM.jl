using Test
using ExtendableGrids
using ExtendableFEMBase
using ExtendableFEM

include("test_dgblf.jl")
include("test_itemintegrator.jl")
include("test_timedependence.jl")

function run_all_tests()
	run_dgblf_tests()
	run_itemintegrator_tests()
	run_dt_tests()
end

run_all_tests()
