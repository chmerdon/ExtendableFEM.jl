[![Build status](https://github.com/chmerdon/ExtendableFEM.jl/workflows/linux-macos-windows/badge.svg)](https://github.com/chmerdon/ExtendableFEM.jl/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chmerdon.github.io/ExtendableFEM.jl/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chmerdon.github.io/ExtendableFEM.jl/dev/index.html)
[![DOI](https://zenodo.org/badge/668345991.svg)](https://zenodo.org/doi/10.5281/zenodo.10563834)

# ExtendableFEM
High Level API Finite Element Methods based on [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl) (for grid management)
and [ExtendableFEMBase.jl](https://github.com/chmerdon/ExtendableFEMBase.jl) (for finite element basis functions and dof management). 
It offers a ProblemDescription interface, that basically involves assigning Unknowns and Operators. Such operators usually stem from a weak formulation of the problem and mainly consist of three types that can be customized via kernel functions:

- BilinearOperator,
- LinearOperator,
- NonlinearOperator (that automatically assemble Newton's method by automatic differentiation)

### Quick Example

The following minimal example demonstrates how to setup a Poisson problem.

```julia
using ExtendableFEM
using ExtendableGrids

# build/load any grid (here: a uniform-refined 2D unit square into triangles)
xgrid = uniform_refine(grid_unitsquare(Triangle2D), 4)

# create empty PDE description
PD = ProblemDescription()

# create and assign unknown
u = Unknown("u"; name = "potential")
assign_unknown!(PD, u)

# assign Laplace operator
assign_operator!(PD, BilinearOperator([grad(u)]; factor = 1e-3))

# assign right-hand side data
function f!(fval, qpinfo)
    x = qpinfo.x # global coordinates of quadrature point
    fval[1] = x[1] * x[2]
end
assign_operator!(PD, LinearOperator(f!, [id(u)]))

# assing boundary data (here: u = 0)
assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:4))

# discretise = choose FEVector with appropriate FESpaces
FEType = H1Pk{1,2,3} # cubic H1-conforming element with 1 component in 2D
FES = FESpace{FEType}(xgrid)

# solve
sol = solve!(Problem, [FES])

# plot
using PyPlot
plot(id(u), sol; Plotter = PyPlot)
```