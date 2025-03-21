using Test
using ExtendableGrids
using ExtendableFEMBase
using ExtendableFEM
using ExampleJuggler
using ExplicitImports
using Metis
using Aqua
using Triangulate
using SimplexGridFactory


include("test_dgblf.jl")
include("test_boundary_operator.jl")
include("test_itemintegrator.jl")
include("test_timedependence.jl")
include("test_nonlinear_operator.jl")
include("test_helper_functions.jl")

function run_examples()
    ExampleJuggler.verbose!(true)

    example_dir = joinpath(@__DIR__, "..", "examples")

    modules = [
        "Example103_BurgersEquation.jl",
        "Example105_NonlinearPoissonEquation.jl",
        "Example106_NonlinearDiffusion.jl",
        "Example108_RobinBoundaryCondition.jl",
        "Example201_PoissonProblem.jl",
        "Example202_MixedPoissonProblem.jl",
        "Example203_PoissonProblemDG.jl",
        #"Example204_LaplaceEVProblem.jl",
        "Example205_HeatEquation.jl",
        "Example207_AdvectionUpwindDG.jl",
        "Example210_LshapeAdaptivePoissonProblem.jl",
        "Example211_LshapeAdaptiveEQPoissonProblem.jl",
        "Example212_PeriodicBoundary2D.jl",
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
        #"Example284_LevelSetMethod.jl",
        #"Example285_CahnHilliard.jl",
        "Example290_PoroElasticity.jl",
        "Example301_PoissonProblem.jl",
        "Example310_DivFreeBasis.jl",
        "Example312_PeriodicBoundary3D.jl",
    ]

    return @testset "module examples" begin
        @testmodules(example_dir, modules)
    end
end

function run_all_tests()
    @testset "Aqua.jl" begin
        Aqua.test_all(
            ExtendableFEM;
            ambiguities = false,
            piracies = false,
        )
        Aqua.test_ambiguities(ExtendableFEM)
    end

    @testset "ExplicitImports" begin
        @test ExplicitImports.check_no_implicit_imports(ExtendableFEM) === nothing
        @test ExplicitImports.check_no_stale_explicit_imports(ExtendableFEM) === nothing
    end

    if isdefined(Docs, :undocumented_names) # >=1.11
        @testset "UndocumentedNames" begin
            @test isempty(Docs.undocumented_names(ExtendableFEM))
        end
    end


    run_boundary_operator_tests()
    run_dgblf_tests()
    run_nonlinear_operator_tests()
    run_itemintegrator_tests()
    run_dt_tests()
    run_test_helper_functions()

    return nothing
end

run_all_tests()
run_examples()
