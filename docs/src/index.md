[![Build status](https://github.com/chmerdon/ExtendableFEM.jl/workflows/linux-macos-windows/badge.svg)](https://github.com/chmerdon/ExtendableFEM.jl/actions)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://chmerdon.github.io/ExtendableFEM.jl/stable/index.html)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chmerdon.github.io/ExtendableFEM.jl/dev/index.html)
[![DOI](https://zenodo.org/badge/229078096.svg)](https://zenodo.org/badge/latestdoi/229078096)

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

1. Mesh the domain of computation, possibly using one of the constructors by ExtendableGrid.jl or via mesh generators in [SimplexGridFactory.jl](https://github.com/j-fu/SimplexGridFactory.jl).
2. Describe your PDE system with the help of the [Problem Description](@ref) and [Operators](@ref).
3. Discretise, i.e. choose suitable finite element ansatz spaces for the unknowns of your PDE system.
4. Solve
5. Postprocess

Please have a look at the Examples.




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

