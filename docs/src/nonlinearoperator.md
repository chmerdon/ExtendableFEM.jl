
# NonlinearOperator

A nonlinear operator automatically assembles all necessary term for the
next Newton step. Other linearisations of a nonlinear operator can be
constructed with special constructors for BilinearOperator or LinearOperator.

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/nonlinear_operator.jl"]
Order   = [:type, :function]
```
