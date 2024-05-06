# Item Integrators

Item integrators compute certain quantities of the Solution, like a posteriori errors estimators, norms, drag/lift coefficients or other statistics.


```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/item_integrator.jl"]
Order   = [:type, :function]
```


## ItemIntegratorDG

ItemIntegratorDG is intended for quantities that involve jumps of discontinuous quantities
on faces whose assembly requires evaluation of all degrees of freedom on the neighbouring cells,
e.g. gradient jumps for H1 conforming functions or jumps of broken FESpaces.
In this case the assembly loop triggers
integration along the boundary of the cells.

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/item_integrator_dg.jl"]
Order   = [:type, :function]
```
