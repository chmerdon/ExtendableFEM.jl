# Time-dependent Solvers

For time-dependent (non-stationary) problems the user currently has these options:
- add custom time derivatives to the problem (i.e. a mass matrix as a BilinearOperator and necessary LinearOperators for evaluating the previous time step(s), if more than one previous time step needs to be remembered, further unknowns must be registered)
- reframe the ProblemDescription as an ODE problem and evolve it via DifferentialEquations with the following extension

## Extension ExtendableFEMDiffEQExt.jl

This extension is automatically loaded when also DifferentialEquations.jl is used. It allows to easily reframe
the ProblemDescription as the right-hand side of an ODE. Here, the ProblemDescription contains
the right-hand side description of the ODE
```math
\begin{aligned}
M u_t(t) & = b(u(t)) - A(u(t))
\end{aligned}
```
where A and b coresspond to the assembled system matrix and right-hand sides
of the operators stored in the ProblemDescription. The matrix M is the mass matrix
and can be customized somewhat (as long as it stays constant).


```@autodocs
Modules = [ExtendableFEM]
Pages = ["solvers_diffeq.jl"]
Order   = [:type, :function]
```

!!! note

    The solvers of DifferentialEquations should be run with the autodiff=false option
    as it is currently not possible to differentiate the right-hand side of the generated
    ODEProblem with respect to time.

## Example : 2D Heat equation

The following ProblemDescription yields the space discretisation of the
heat equation (including homogeneous boundary conditions and equivalent to the Poisson equation).
```julia
PD = ProblemDescription("Heat Equation")
u = Unknown("u"; name = "temperature")
assign_unknown!(PD, u)
assign_operator!(PD, BilinearOperator([grad(u)]; store = true, kwargs...))
assign_operator!(PD, HomogeneousBoundaryData(u))
```
Given a finite element space FES and an initial FEVector sol for the unknown, the
ODEProblem for some time interval (0,T) can be generated and solved via
```julia
prob = generate_ODEProblem(PD, FES, (0, T); init = sol)
DifferentialEquations.solve(prob, Rosenbrock23(autodiff = false), dt = 1e-3, dtmin = 1e-6, adaptive = true)
```