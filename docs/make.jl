using Documenter
using ExampleJuggler
using CairoMakie
using ExtendableFEM

function make_all(; with_examples::Bool = true, modules = :all, run_examples::Bool = true, run_notebooks::Bool = false)

	module_examples = []

	if with_examples
		DocMeta.setdocmeta!(ExampleJuggler, :DocTestSetup, :(using ExampleJuggler); recursive = true)

		example_dir = joinpath(@__DIR__, "..", "examples")

		if modules === :all
			modules = [
				"Example103_BurgersEquation.jl",
				"Example105_NonlinearPoissonEquation.jl",
				"Example106_NonlinearDiffusion.jl",
				"Example108_RobinBoundaryCondition.jl",
				"Example201_PoissonProblem.jl",
				"Example202_MixedPoissonProblem.jl",
				"Example203_PoissonProblemDG.jl",
				"Example204_LaplaceEVProblem.jl",
				"Example205_HeatEquation.jl",
				"Example206_CoupledSubGridProblems.jl",
				"Example207_AdvectionUpwindDG.jl",
				"Example210_LshapeAdaptivePoissonProblem.jl",
				"Example211_LshapeAdaptiveEQPoissonProblem.jl",
				"Example220_ReactionConvectionDiffusion.jl",
				"Example225_ObstacleProblem.jl",
				"Example226_Thermoforming.jl",
				"Example230_NonlinearElasticity.jl",
				"Example235_StokesIteratedPenalty.jl",
				"Example240_SVRTEnrichment.jl",
				"Example245_NSEFlowAroundCylinder.jl",
				"Example250_NSELidDrivenCavity.jl",
				"Example252_NSEPlanarLatticeFlow.jl",
				"Example260_AxisymmetricNavierStokesProblem.jl",
				"Example265_FlowTransport.jl",
				"Example270_NaturalConvectionProblem.jl",
				"Example275_OptimalControlStokes.jl",
				"Example280_CompressibleStokes.jl",
				"Example282_IncompressibleMHD.jl",
				"Example284_LevelSetMethod.jl",
				"Example285_CahnHilliard.jl",
				"Example290_PoroElasticity.jl",
				"Example301_PoissonProblem.jl",
				"Example310_DivFreeBasis.jl",
				"Example330_HyperElasticity.jl",
			]
		end

		#notebooks = ["PlutoTemplate.jl"
		#             "Example with Graphics" => "ExamplePluto.jl"]

		cleanexamples()

		module_examples = @docmodules(example_dir, modules, Plotter = CairoMakie)
		#html_examples = @docplutonotebooks(example_dir, notebooks, iframe=false)
		#pluto_examples = @docplutonotebooks(example_dir, notebooks, iframe=true)
	end

	makedocs(
		modules = [ExtendableFEM],
		sitename = "ExtendableFEM.jl",
		authors = "Christian Merdon, Jan Philipp Thiele",
		format = Documenter.HTML(; repolink = "https://github.com/chmerdon/ExtendableFEM.jl", mathengine = MathJax3()),
		clean = false,
		checkdocs = :all,
		warnonly = false,
		doctest = true,
		pages = [
			"Home" => "index.md",
			"Index" => "package_index.md",
			"Problem Description" => [
				"problemdescription.md",
                "tensordescription.md",
				"nonlinearoperator.md",
				"bilinearoperator.md",
				"linearoperator.md",
				"interpolateboundarydata.md",
				"homogeneousdata.md",
				"fixdofs.md",
				"combinedofs.md",
				"callbackoperator.md",
			],
			"Solving" => Any[
				"pdesolvers.md",
				"pdesolvers_dt.md",
				"parallel_assembly.md"
			],
			"Postprocessing" => Any[
				"postprocessing.md",
				"itemintegrators.md",
			],
			#"Tutorial Notebooks" => notebooks,
			"Examples" => module_examples,
		],
	)

	cleanexamples()

end

#make_all(; with_examples = true, run_examples = true, run_notebooks = true)
make_all(; with_examples = true, run_examples = true, run_notebooks = false)

deploydocs(
	repo = "github.com/chmerdon/ExtendableFEM.jl",
)
