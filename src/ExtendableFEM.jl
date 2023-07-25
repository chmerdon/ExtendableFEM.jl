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
using UnicodePlots
using Printf
using DocStringExtensions

include("io.jl")
export print_convergencehistory

include("unknowns.jl")
export Unknown

include("operators.jl")
export AbstractOperator
export AssemblyInformation
export id, grad, div, normalflux, Δ, apply
export id_jump, grad_jump, normalflux_jump
export assemble!

include("common_operators/reduction_operator.jl")
export AbstractReductionOperator
export FixbyInterpolation

include("problemdescription.jl")
export ProblemDescription
export assign_unknown!
export assign_operator!
export assign_reduction!

include("helper_functions.jl")
export get_periodic_coupling_info

include("solver_config.jl")
export SolverConfiguration

include("solvers.jl")
export solve
export get_unknown_id

include("jump_operators.jl")
export DiscontinuousFunctionOperator
export Jump, Average, Left, Right
export jump, average
export is_discontinuous

include("common_operators/item_integrator.jl")
export ItemIntegrator
export evaluate, evaluate!
export L2NormIntegrator
include("common_operators/linear_operator.jl")
export LinearOperator
include("common_operators/bilinear_operator.jl")
export BilinearOperator
include("common_operators/nonlinear_operator.jl")
export NonlinearOperator
include("common_operators/combinedofs.jl")
export CombineDofs
export get_periodic_coupling_info
include("common_operators/interpolateboundarydata_operator.jl")
export InterpolateBoundaryData
include("common_operators/homogeneousdata_operator.jl")
export HomogeneousBoundaryData
export HomogeneousData
include("common_operators/fixdofs_operator.jl")
export FixDofs
include("common_operators/discface_interpolator.jl")
export FaceInterpolator

include("common_operators/segment_integrator.jl")
export SegmentIntegrator, initialize!, integrate_segment!

include("common_operators/point_evaluator.jl")
export PointEvaluator, evaluate!, eval_func


include("lazy_interpolate.jl")
export lazy_interpolate!

include("plots.jl")
export plot_convergencehistory!


end #module