
# LinearOperator

A linear operator allows to add vectors to the right-hand side of the system that usually refer to
right-hand side data or linearisations of PDE operators (see remark in NonlinearOperator example).
If the linear operator
lives on face entities, also jumps of operators can be involved, if they are naturally
continuous for the finite element space in operation (also jumps for broken spaces)
and only involve degrees of freedom on the face, e.g.
normal jumps for Hdiv spaces or jumps for H1-conforming spaces or tangential jumps
of Hcurl spaces. For all other discontinuous operator evaluations
(that needs to evaluate more than the degrees of freedom on the face)
there is the possibility to use LinearOperatorDG.
It is also possible to assign a vector assembled by the user as a LinearOperator.

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


## LinearOperatorDG

LinearOperatorDG is intended for linear forms that involves jumps of discontinuous quantities
on faces whose assembly requires evaluation of all degrees of freedom on the neighbouring cells,
e.g. gradient jumps for H1 conforming functions. In this case the assembly loop triggers
integration along the boundary of the cells.

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/linear_operator_dg.jl"]
Order   = [:type, :function]
```