module ExtendableFEM

using CommonSolve: CommonSolve
using DiffResults: DiffResults
using DocStringExtensions: DocStringExtensions, TYPEDFIELDS
using ExtendableFEMBase: ExtendableFEMBase, AbstractFiniteElement,
	AbstractFunctionOperator, AbstractH1FiniteElement,
	AbstractHdivFiniteElement, BEdgeDofs, BFaceDofs,
	CellDofs, Curl2D, Curl3D, CurlScalar,
	DefaultName4Operator, Divergence, Dofmap4AssemblyType,
	EdgeDofs, EffAT4AssemblyType, FEEvaluator, FEMatrix,
	FEMatrixBlock, FESpace, FEVector, FEVectorBlock,
	FaceDofs, Gradient, H1BR, H1BUBBLE, H1CR, H1MINI,
	H1P1, H1P1TEB, H1P2, H1P2B, H1P3, H1Pk, H1Q1, H1Q2,
	HCURLN0, HCURLN1, HDIVBDM1, HDIVBDM2, HDIVRT0,
	HDIVRT1, HDIVRTk, HDIVRTkENRICH, Hessian, Identity,
	L2P0, L2P1, Laplacian, Length4Operator,
	NeededDerivative4Operator, NormalFlux,
	ParentDofmap4Dofmap, PointEvaluator, QPInfos,
	QuadratureRule, Reconstruct, SegmentIntegrator,
	StandardFunctionOperator, TangentFlux,
	TangentialGradient, VertexRule, _addnz, add!,
	addblock!, addblock_matmul!, displace_mesh,
	displace_mesh!, eval_func, eval_func_bary, evaluate!,
	evaluate_bary!, fill!, get_AT, get_FEType,
	get_ncomponents, get_ndofs, get_polynomialorder,
	initialize!, integrate, integrate!,
	integrate_segment!, lazy_interpolate!, nodevalues,
	nodevalues!, nodevalues_subset!, nodevalues_view,
	norms, unicode_gridplot, unicode_scalarplot,
	update_basis!
using ExtendableGrids: ExtendableGrids, AT_NODES, AbstractElementGeometry,
	Adjacency, AssemblyType, BEdgeNodes, BFaceFaces,
	BFaceNodes, BFaceRegions, CellAssemblyGroups,
	CellFaceOrientations, CellFaces, CellGeometries,
	CellNodes, CellRegions, Coordinates, EdgeNodes,
	ElementGeometries, ExtendableGrid, FaceCells, FaceEdges,
	FaceNodes, FaceNormals, FaceRegions, FaceVolumes,
	PColorPartitions, PartitionCells, PartitionEdges,
	GridComponentAssemblyGroups4AssemblyType,
	GridComponentGeometries4AssemblyType,
	GridComponentRegions4AssemblyType,
	GridComponentVolumes4AssemblyType, L2GTransformer,
	ON_BEDGES, ON_BFACES, ON_CELLS, ON_EDGES, ON_FACES,
	ON_IFACES, SerialVariableTargetAdjacency,
	UniqueBFaceGeometries, UniqueCellGeometries,
	UniqueFaceGeometries, append!, dim_element, eval_trafo!,
	facetype_of_cellface, interpolate!,
	max_num_targets_per_source, num_cells, num_faces,
	num_nodes, num_sources, num_targets, simplexgrid,
	num_pcolors, num_partitions, num_partitions_per_color,
	unique, update_trafo!, xrefFACE2xrefCELL,
	xrefFACE2xrefOFACE
using ExtendableSparse: ExtendableSparse, ExtendableSparseMatrix, flush!,
	MTExtendableSparseMatrixCSC,
	rawupdateindex!
using ForwardDiff: ForwardDiff
using GridVisualize: GridVisualize, GridVisualizer, gridplot!, reveal, save,
	scalarplot!, vectorplot!
using LinearAlgebra: LinearAlgebra, copyto!, isposdef, mul!, norm
using LinearSolve: LinearSolve, LinearProblem, UMFPACKFactorization, deleteat!,
	init, solve
using Printf: Printf, @printf
using SparseArrays: SparseArrays, AbstractSparseArray, SparseMatrixCSC, nnz,
	nzrange, rowvals, sparse
using SparseDiffTools: SparseDiffTools, ForwardColorJacCache,
	forwarddiff_color_jacobian!, matrix_colors
using Symbolics: Symbolics
using SciMLBase: SciMLBase
using UnicodePlots: UnicodePlots

if !isdefined(Base, :get_extension)
	using Requires
end

## reexport stuff from ExtendableFEMBase and ExtendableGrids
export FESpace, FEMatrix, FEVector
export H1P1, H1P2, H1P3, H1Pk
export H1Q1, H1Q2
export H1CR, H1BR, H1P2B, H1MINI, H1P1TEB, H1BUBBLE
export HDIVRT0, HDIVRT1, HDIVRTk
export HDIVBDM1, HDIVBDM2
export HCURLN0, HCURLN1
export HDIVRTkENRICH
export L2P0, L2P1
export ON_FACES, ON_BFACES, ON_EDGES, ON_FACES, ON_CELLS, AT_NODES
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
export unicode_gridplot, unicode_scalarplot

## reexport stuff from GridVisualize
export reveal, save

include("io.jl")
export print_convergencehistory
export print_table

include("unknowns.jl")
export Unknown
export grid, dofgrid
export id, grad, hessian, div, normalflux, tangentialflux, Δ, apply, curl1, curl2, curl3, laplace, tangentialgrad

include("operators.jl")
export AbstractOperator
export assemble!, apply_penalties!

include("common_operators/reduction_operator.jl")
export AbstractReductionOperator
export FixbyInterpolation

include("problemdescription.jl")
export ProblemDescription
export assign_unknown!
export assign_operator!
export replace_operator!

include("helper_functions.jl")
export get_periodic_coupling_info
export tmul!

include("tensors.jl")
export TensorDescription
export TDScalar
export TDVector
export TDMatrix
export TDRank3
export TDRank4
export tensor_view


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
include("common_operators/item_integrator_dg.jl")
export ItemIntegratorDG
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

include("diffeq_interface.jl")
export generate_ODEProblem

end #module
