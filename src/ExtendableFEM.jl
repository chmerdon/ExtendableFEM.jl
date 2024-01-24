module ExtendableFEM

using ExtendableFEMBase
using ExtendableSparse
using ExtendableGrids
using SparseArrays
using CommonSolve
using LinearAlgebra
using LinearSolve
using Symbolics
using GridVisualize
using ForwardDiff
using SparseDiffTools
using DiffResults
using Printf
using DocStringExtensions

if !isdefined(Base, :get_extension)
	using Requires
end

## reexport stuff from ExtendableFEMBase
export FESpace, FEMatrix, FEVector
export H1P1, H1P2, H1P3, H1Pk
export H1Q1, H1Q2
export H1CR, H1BR, H1P2B, H1MINI,H1P1TEB, H1BUBBLE
export HDIVRT0, HDIVRT1, HDIVRTk
export HDIVBDM1, HDIVBDM2
export HCURLN0, HCURLN1
export HDIVRTkENRICH
export L2P0, L2P1
export nodevalues, nodevalues!, nodevalues_view, nodevalues_subset!
export interpolate!, lazy_interpolate!
export PointEvaluator, evaluate, evaluate!, evaluate_bary!, eval_func, eval_func_bary
export SegmentIntegrator, integrate_segment!, initialize!
export integrate!, integrate, QuadratureRule
export unicode_gridplot, unicode_scalarplot
export CellDofs, BFaceDofs, FaceDofs, EdgeDofs, BEdgeDofs
export get_polynomialorder
export displace_mesh, displace_mesh!
export Reconstruct, Identity, Divergence, Gradient
export _addnz
export addblock!, addblock_matmul!

## reexport stuff from GridVisualize
export reveal, save

include("io.jl")
export print_convergencehistory
export print_table

include("unknowns.jl")
export Unknown
export grid
export id, grad, hessian, div, normalflux, Δ, apply, curl1, curl2, curl3, laplace
export id_jump, grad_jump, normalflux_jump

include("operators.jl")
export AbstractOperator
export AssemblyInformation
export assemble!, apply_penalties!

include("common_operators/reduction_operator.jl")
export AbstractReductionOperator
export FixbyInterpolation

include("problemdescription.jl")
export ProblemDescription
export assign_unknown!
export assign_operator!
export replace_operator!
export assign_reduction!

include("helper_functions.jl")
export get_periodic_coupling_info

include("solver_config.jl")
export SolverConfiguration
export residual

include("solvers.jl")
export solve
export iterate_until_stationarity
export get_unknown_id

include("solvers_diffeq.jl")
export generate_ODEProblem

include("jump_operators.jl")
export DiscontinuousFunctionOperator
export Jump, Average, Left, Right
export jump, average, this, other
export is_discontinuous

include("common_operators/item_integrator.jl")
export ItemIntegrator
export evaluate, evaluate!
export L2NormIntegrator
include("common_operators/linear_operator.jl")
export LinearOperator
include("common_operators/linear_operator_dg.jl")
export LinearOperatorDG
include("common_operators/bilinear_operator.jl")
export BilinearOperator
include("common_operators/bilinear_operator_dg.jl")
export BilinearOperatorDG
include("common_operators/nonlinear_operator.jl")
export NonlinearOperator
include("common_operators/callback_operator.jl")
export CallbackOperator
include("common_operators/combinedofs.jl")
export CombineDofs
export get_periodic_coupling_info
include("common_operators/interpolateboundarydata_operator.jl")
export InterpolateBoundaryData
export apply!
include("common_operators/homogeneousdata_operator.jl")
export HomogeneousBoundaryData
export HomogeneousData
export assemble!, fixed_dofs, fixed_vals
include("common_operators/fixdofs_operator.jl")
export FixDofs
include("common_operators/discface_interpolator.jl")
export FaceInterpolator

include("plots.jl")
export plot_convergencehistory!
export scalarplot!
export vectorplot!
export plot, plot!
export default_generateplots
export plot_unicode

@static if !isdefined(Base, :get_extension)
	function __init__()
		@require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" begin
			include("../ext/ExtendableFEMDiffEQExt.jl")
		end
	end
end

end #module
