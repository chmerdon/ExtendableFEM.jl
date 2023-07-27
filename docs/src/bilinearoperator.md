
# BilinearOperator

A bilinear operator allows to add matrices to the system matrix that usually refer to
linearisations of the PDE operators or stabilisations.

## Constructors

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/bilinear_operator.jl"]
Order   = [:type, :function]
```

## Example - Stokes operator

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
