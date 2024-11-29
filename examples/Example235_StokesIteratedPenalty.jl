#=

# 235 : Stokes iterated penalty method
([source code](@__SOURCE_URL__))

This example computes a velocity ``\mathbf{u}`` and pressure ``\mathbf{p}`` of the incompressible Stokes problem
```math
\begin{aligned}
- \mu \Delta \mathbf{u} + \nabla p & = \mathbf{0}\\
\mathrm{div}(\mathbf{u}) & = 0
\end{aligned}
```
with some μ parameter ``\mu``.

Here we solve the simple Hagen-Poiseuille flow on the two-dimensional unit square domain with the iterated penalty method
suggested in the reference below adapted to the Bernardi--Raugel finite element method.
Given intermediate solutions ``\mathbf{u}_h`` and ``p_h`` the next approximations are computed by the two equations

```math
\begin{aligned}
(\nabla \mathbf{u}_h^{next}, \nabla \mathbf{v}_h) + \lambda (\mathrm{div}_h(\mathbf{u}^{next}_h) ,\mathrm{div}_h(\mathbf{v}_h)) & = (\mathbf{f},\mathbf{v}_h) + (p_h,\mathrm{div}(\mathbf{v}_h))
&& \text{for all } \mathbf{v}_h \in \mathbf{V}_h\\
(p^{next}_h,q_h) & = (p_h,q_h) - \lambda (\mathrm{div}(\mathbf{u}_h^{next}),q_h) && \text{for all } q_h \in Q_h
\end{aligned}
```

This is done consecutively until the residual of both equations is small enough.
The discrete divergence is computed via a RT0 reconstruction operator that preserves the discrete divergence.
(another way would be to compute ``B M^{-1} B^T`` where ``M`` is the mass matrix of the pressure and ``B`` is the matrix for the div-pressure block).

!!! reference

	"An iterative penalty method for the finite element solution of the stationary Navier-Stokes equations",\
	R. Codina,\
	Computer Methods in Applied Mechanics and Engineering Volume 110, Issues 3–4 (1993),\
	[>Journal-Link<](https://doi.org/10.1016/0045-7825(93)90163-R)


The computed solution for the default parameters looks like this:

![](example235.png)
=#

module Example235_StokesIteratedPenalty

using ExtendableFEM
using ExtendableGrids
using Test #hide

## data for Hagen-Poiseuille flow
function p!(result, qpinfo)
    x = qpinfo.x
    μ = qpinfo.params[1]
    return result[1] = μ * (-2 * x[1] + 1.0)
end
function u!(result, qpinfo)
    x = qpinfo.x
    result[1] = x[2] * (1.0 - x[2])
    return result[2] = 0.0
end
## kernel for div projection
function div_projection!(result, input, qpinfo)
    return result[1] = input[1] - qpinfo.params[1] * input[2]
end

## everything is wrapped in a main function
function main(; Plotter = nothing, λ = 1.0e4, μ = 1.0, nrefs = 5, kwargs...)

    ## initial grid
    xgrid = uniform_refine(grid_unitsquare(Triangle2D), nrefs)

    ## Bernardi--Raugel element with reconstruction operator
    FETypes = (H1BR{2}, L2P0{1})
    PenaltyDivergence = Reconstruct{HDIVRT0{2}, Divergence}

    ## generate two problems
    ## one for velocity, one for pressure
    u = Unknown("u"; name = "velocity")
    p = Unknown("p"; name = "pressure")
    PDu = ProblemDescription("Stokes IPM - velocity update")
    assign_unknown!(PDu, u)
    assign_operator!(PDu, BilinearOperator([grad(u)]; factor = μ, store = true, kwargs...))
    assign_operator!(PDu, BilinearOperator([apply(u, PenaltyDivergence)]; store = true, factor = λ, kwargs...))
    assign_operator!(PDu, LinearOperator([div(u)], [id(p)]; factor = 1, kwargs...))
    assign_operator!(PDu, InterpolateBoundaryData(u, u!; regions = 1:4, params = [μ], bonus_quadorder = 4, kwargs...))

    PDp = ProblemDescription("Stokes IPM - pressure update")
    assign_unknown!(PDp, p)
    assign_operator!(PDp, BilinearOperator([id(p)]; store = true, kwargs...))
    assign_operator!(PDp, LinearOperator(div_projection!, [id(p)], [id(p), div(u)]; params = [λ], factor = 1, kwargs...))

    ## show and solve problem
    FES = [FESpace{FETypes[1]}(xgrid), FESpace{FETypes[2]}(xgrid)]
    sol = FEVector([FES[1], FES[2]]; tags = [u, p])
    SC1 = SolverConfiguration(PDu; init = sol, maxiterations = 1, target_residual = 1.0e-8, constant_matrix = true, kwargs...)
    SC2 = SolverConfiguration(PDp; init = sol, maxiterations = 1, target_residual = 1.0e-8, constant_matrix = true, kwargs...)
    sol, nits = iterate_until_stationarity([SC1, SC2]; init = sol, kwargs...)
    @info "converged after $nits iterations"

    ## plot
    plt = plot([id(u), id(p)], sol; Plotter = Plotter)

    return sol, plt
end

generateplots = ExtendableFEM.default_generateplots(Example235_StokesIteratedPenalty, "example235.png") #hide
function exact_error!(result, u, qpinfo) #hide
    u!(result, qpinfo) #hide
    p!(view(result, 3), qpinfo) #hide
    return result .= (result .- u) .^ 2 #hide
end #hide
function runtests(; μ = 1.0) #hide
    sol, plt = main(; μ = μ) #hide
    ErrorIntegratorExact = ItemIntegrator(exact_error!, [id(1), id(2)]; quadorder = 4, params = [μ]) #hide
    error = evaluate(ErrorIntegratorExact, sol) #hide
    error_u = sqrt(sum(view(error, 1, :)) + sum(view(error, 2, :))) #hide
    error_p = sqrt(sum(view(error, 3, :))) #hide
    @test error_u ≈ 3.990987355891888e-5 #hide
    return @test error_p ≈ 0.010437891104305222 #hide
end #hide
end
