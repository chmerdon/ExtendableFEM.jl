# Other packages

A lot of functionality is already provided by the bases packages, e.g.:
- [ExtendableGrids.jl](https://github.com/WIAS-PDELib/ExtendableGrids.jl) offers an interface to [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl) which can be used, e.g. in combination with nodevalues interpolations or (piecewise constant) item integrator results. There is also the CellFinder that can be used to find the right cell for a certain point of the domain.
- [ExtendableFEMBase.jl](https://github.com/WIAS-PDELib/ExtendableFEMBase.jl) offers a PointEvaluator and a SegmentIntegrator to evaluate solutions at arbitrary points of the domain or integrating along 1D line intersections with the cells of the triangulation. It also provides some basic unicode plots.
- [GridVisualize.jl](https://github.com/WIAS-PDELib/GridVisualize.jl) provides grid and scalar piecewise linear function plotting for various plotting backends on simplicial grids in one, two or three space dimensions. The main supported backends are PyPlot, GLMakie and PlutoVista.


## Plots and Tables

Some convenient plotting shortcuts are avaiables via these functions:


```@autodocs
Modules = [ExtendableFEM]
Pages = ["plots.jl", "io.jl"]
Order   = [:type, :function]
```
