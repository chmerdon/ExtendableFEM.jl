using Test
using ExtendableGrids
using ExtendableFEMBase
using ExtendableFEM

include("test_dgblf.jl")
include("test_itemintegrator.jl")
include("test_timedependence.jl")
include("test_nonlinear_operator.jl")

function run_all_tests()
	run_dgblf_tests()
	run_itemintegrator_tests()
	run_dt_tests()
	run_nonlinear_operator_tests()
end

run_all_tests()
