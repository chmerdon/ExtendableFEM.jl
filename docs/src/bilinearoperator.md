
# BilinearOperator

A bilinear operator allows to add matrices to the system matrix that usually refer to
linearisations of the PDE operators or stabilisations. If the bilinear operator
lives on face entities, also jumps of operators can be involved, if they are naturally
continuous for the ground finite element space (also jumps for broken spaces)
and only involve degrees of freedom on the face, e.g.
normal jumps for Hdiv spaces or jumps for H1-conforming spaces. For all other
discontinuous operator evaluations there is the possibility to use BilinerOperatorDG.
It is also possible to assign a matrix assembled by the user as a BilinearOperator.

## Constructors

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/bilinear_operator.jl"]
Order   = [:type, :function]
```

## DG BilinearOperator

BilinearOperatorDG is intended for bilinear forms that involves jumps of discontinuous quantities
on faces whose assembly requires evaluation of all degrees of freedom on the neighbouring cells,
e.g. gradient jumps for H1 conforming functions.

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/bilinear_operator_dg.jl"]
Order   = [:type, :function]
```

## Examples

Below two examples illustrate some use cases.

### Example - Stokes operator

For the linear operator of a Stokes problem a kernel could look like
```julia
μ = 0.1 # viscosity parameter
function kernel!(result, input, qpinfo)
    ∇u, p = view(input,1:4), view(input, 5)
    result[1] = μ*∇u[1] - p[1]
    result[2] = μ*∇u[2]
    result[3] = μ*∇u[3]
    result[4] = μ*∇u[4] - p[1]
    result[5] = -(∇u[1] + ∇u[4])
    return nothing
end
```
and the coressponding BilinearOperator constructor call reads
```julia
u = Unknown("u"; name = "velocity")
p = Unknown("p"; name = "pressure")
BilinearOperator(kernel!, [grad(u), id(p)]; use_sparsity_pattern = true)
```
The additional argument causes that the zero pressure-pressure block of the matrix is not (even tried to be) assembled,
since ```input[5]``` does not couple with ```result[5]```.


### Example - interior penalty stabilization

A popular convection stabilization is based on the jumps of the gradient, which can be realised with
the kernel
```julia
function stab_kernel!(result, input, qpinfo)
    result .= input .* qpinfo.volume^2
end
```
and the BilinearOperatorDG constructor call
```julia
u = Unknown("u")
assign_operator!(PD, BilinearOperatorDG(stab_kernel!, [jump(grad(u))]; entities = ON_IFACES, factor = 0.01))
```