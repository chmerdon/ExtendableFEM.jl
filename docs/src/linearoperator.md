
# LinearOperator

A linear operator allows to add vectors to the right-hand side of the system that usually refer to
right-hand side data or linearisations of PDE operators (see remark in NonlinearOperator example).

## Constructors

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/linear_operator.jl"]
Order   = [:type, :function]
```

## Example - right-hand side

For a right-hand side operator of a Poisson problem with some given function ```f(x)```
a kernel could look like
```julia
function kernel!(result, qpinfo)
    result[1] = f(qpinfo.x)
end
```
and the coressponding LinearOperator constructor call reads
```julia
u = Unknown("u")
NonlinearOperator(kernel!, [id(u)])
```
The second argument triggers that the ```result``` vector of the kernel is multiplied with the Identity evaluation of the test function.


## DG LinearOperator

LinearOperatorDG is intended for bilinear forms that involves jumps of discontinuous quantities
on faces whose assembly requires evaluation of all degrees of freedom on the neighbouring cells,
e.g. gradient jumps for H1 conforming functions.

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/linear_operator_dg.jl"]
Order   = [:type, :function]
```