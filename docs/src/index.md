# ExtendableFEM.jl

This package offers a toolkit to easily setup (mostly low-order, standard and non-standard) finite element methods for multiphysics problems in Julia
and to run fixed-point iterations to solve them.

The implementation is based on [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl) (for meshing and administration) and [ExtendableFEMBase.jl](https://github.com/chmerdon/ExtendableFEMBase.jl) (for quadrature and FEM basis functions).

Also note, that this package is part of the meta-package [PDELIB.jl](https://github.com/WIAS-BERLIN/PDELib.jl)

!!! note

    This package is still in an early development stage and features and interfaces might change in future updates.
    

#### Dependencies on other Julia packages

[ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl)\
[ExtendableSparse.jl](https://github.com/j-fu/ExtendableSparse.jl)\
[ExtendableFEMBase.jl](https://github.com/chmerdon/ExtendableFEMBase.jl)\
[GridVisualize.jl](https://github.com/j-fu/GridVisualize.jl)\
[DocStringExtensions.jl](https://github.com/JuliaDocs/DocStringExtensions.jl)\
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)\
[DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl)\


## Getting started

The general work-flow is as follows:

#### 1. Geometry description / meshing

The geometry description and meshing is not really separated.
For meshes of rectangular domains, there are simple constructors available in [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl), e.g.
```julia
using ExtendableGrids
## unit square criss-cross into Triangles
xgrid1 = uniform_refine(grid_unitsquare(Triangle2D), 4)

## uniform rectangular grid
xgrid2 = simplexgrid(0:0.1:1, 0:0.2:2)
```
Note that these grids generate boundary regions from 1 to 4 (bottom, left, top, right) that can be used
to assign boundary conditions.

More complex grids can be created via the mesh generators in [SimplexGridFactory.jl](https://github.com/j-fu/SimplexGridFactory.jl),
see e.g. [Example 245](examples/Example245_NSEFlowAroundCylinder) or [Example 265](examples/Example265_FlowTransport), or by loading a Gmsh grid file via the corresponding [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl) extension.

#### 2. Problem description

Before discretizing the user has the option to pose his problems
in form of a [Problem Description](@ref). Note, that usually no grid
has to be defined at this point, but region numbers correspond
to regions defined in the grid. Here is a short example:

```julia
# a simple Poisson problem with right-hand side f(x,y) = x*y and u = 0 along boundary
PD = ProblemDescription()
u = Unknown("u"; name = "potential")
assign_unknown!(PD, u)
assign_operator!(PD, BilinearOperator([grad(u)]; factor = 1e-3))
f! = (result, qpinfo) -> (result[1] = qpinfo.x[1] * qpinfo.x[2])
assign_operator!(PD, LinearOperator(f!, [id(u)]))
assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4))
```


#### 3. Discretization

In this step, the user chooses suitable finite element types for the unknowns of the problem,
and generates finite element spaces on the grid (and probably already a solution vector
to preoccupy it with an initial solution).
```julia
# cubic H1 element in 2D with one component
FES = FESpace{H1Pk{1,2,3}}(xgrid) 
# alternatively: create solution vector and tag blocks with problem unknowns
sol = FEVector(FES; tags = PD.unknowns) 
# fill block corresponding to unknown u with initial values
fill(sol[u], 1)
# interpolate some given function u!(result, qpinfo)
interpolate!(sol[u], u!)
```

#### 4. Solve

Here, we solve the problem. If the problem is nonlinear, several
additional arguments allow to steer the fixed-point iteration,
see [Stationary Solvers](@ref). In the simplest case, the user
only needs to call:

```julia
# solve problem with finite element space(s)
# (in case of more than one unknown, provide a vector of FESpaces)
sol = solve(PD, FES; init = sol)
```

For time-dependent problem, the user can add the necessary
operators for the time derivative manually. Alternatively,
the problem description in space can be turned into an ODE
and solve via DifferentialEquations.jl, see
[Time-dependent Solvers](@ref) for details.

Also note, that the use can bypass the problem description
and control the algebraic level manually via
assembling the operators directly into an FEMatrix,
see e.g. [Example310](examples/Example310_DivFreeBasis).
It is also possible to take control over the low-level
assembly of the operators, see [ExtendableFEMBase.jl](https://github.com/chmerdon/ExtendableFEMBase.jl)
for details.



#### 5. Plot and postprocess

After solving, the user can postprocess the solution,
calculate quantities of interest or plot components.


## Gradient-robustness

This package offers some ingredients to build gradient-robust schemes via reconstruction operators or divergence-free elements.
Gradient-robustness is a feature of discretisations that exactly balance gradient forces in the momentum balance. In the case of the incompressible Navier--Stokes equations this means that the discrete velocity does not depend on the exact pressure. Divergence-free finite element methods have this property but are usually expensive and difficult to contruct. However, also non-divergence-free classical finite element methods can be made pressure-robust with the help of reconstruction operators applied to testfunctions in certain terms of the momentum balance, see e.g. references [1,2] below.

Recently gradient-robustness was also connected to the design of well-balanced schemes e.g. in the context of (nearly) compressible flows, see e.g. reference [3] below.

#### References

- [1]   "On the divergence constraint in mixed finite element methods for incompressible flows",\
        V. John, A. Linke, C. Merdon, M. Neilan and L. Rebholz,\
        SIAM Review 59(3) (2017), 492--544,\
        [>Journal-Link<](https://doi.org/10.1137/15M1047696),
        [>Preprint-Link<](http://www.wias-berlin.de/publications/wias-publ/run.jsp?template=abstract&type=Preprint&year=2015&number=2177)
- [2]   "Pressure-robustness and discrete Helmholtz projectors in mixed finite element methods for the incompressible Navier--Stokes equations",\
        A. Linke and C. Merdon,
        Computer Methods in Applied Mechanics and Engineering 311 (2016), 304--326,\
        [>Journal-Link<](http://dx.doi.org/10.1016/j.cma.2016.08.018)
        [>Preprint-Link<](http://www.wias-berlin.de/publications/wias-publ/run.jsp?template=abstract&type=Preprint&year=2016&number=2250)
- [3]   "A gradient-robust well-balanced scheme for the compressible isothermal Stokes problem",\
        M. Akbas, T. Gallouet, A. Gassmann, A. Linke and C. Merdon,\
        Computer Methods in Applied Mechanics and Engineering 367 (2020),\
        [>Journal-Link<](https://doi.org/10.1016/j.cma.2020.113069)
        [>Preprint-Link<](https://arxiv.org/abs/1911.01295)

