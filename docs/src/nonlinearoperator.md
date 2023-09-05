
# NonlinearOperator

A nonlinear operator automatically assembles all necessary terms for the
Newton method. Other linearisations of a nonlinear operator can be
constructed with special constructors for BilinearOperator or LinearOperator.

## Constructor

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/nonlinear_operator.jl"]
Order   = [:type, :function]
```

## Example - NSE convection operator

For the 2D Navier--Stokes equations, a kernel function for the convection operator could look like
```julia
function kernel!(result, input, qpinfo)
    u, ∇u = view(input, 1:2), view(input,3:6)
    result[1] = dot(u, view(∇u,1:2))
    result[2] = dot(u, view(∇u,3:4))
end
```
and the coressponding NonlinearOperator constructor call reads
```julia
u = Unknown("u"; name = "velocity")
NonlinearOperator(kernel!, [id(u)], [id(u),grad(u)])
```
The second argument triggers that the evaluation of the Identity and Gradient operator of the
current velocity iterate at each quadrature point go (in that order) into the ```input``` vector (of length 6) of the kernel, while the third argument
triggers that the ```result``` vector of the kernel is multiplied with the Identity evaluation of the velocity test function.

!!! remark

    Also note, that the same kernel could be used for a fully explicit linearisation of the convection term as a LinearOperator via
    ```julia
    u = Unknown("u"; name = "velocity")
    LinearOperator(kernel!, [id(u)], [id(u),grad(u)])
    ```
    For a Picard iteration of the convection term, a BilinearOperator can be used with a slightly modified kernel
    that separates the operator evaluations of the ansatz function and the current solution, i.e.,
    ```julia
    function kernel_picard!(result, input_ansatz, input_args, qpinfo)
        a, ∇u = view(input_args, 1:2), view(input_ansatz,1:4)
        result[1] = dot(a, view(∇u,1:2))
        result[2] = dot(a, view(∇u,3:4))
    end
    u = Unknown("u"; name = "velocity")
    BilinearOperator(kernel_picard!, [id(u)], [grad(u)], [id(u)])
    ```

!!! note

    Kernels are allowed to depend on region numbers, space and time coordinates via the qpinfo argument.


## Newton by local jacobians of kernel

To demonstrate the general approach consider a model problem with a nonlinear operator that
has the weak formulation that seeks some function ``u(x) \in X`` in some finite-dimensional
space ``X`` with ``N := \mathrm{dim} X``, i.e., some coefficient
vector ``x \in \mathbb{R}^N``, such that
```math
\begin{aligned}
F(x) := \int_\Omega A(L_1u(x)(y)) \cdot L_2v(y) \,\textit{dy} & = 0 \quad \text{for all } v \in X
\end{aligned}
```
for some given nonlinear kernel function ``A : \mathbb{R}^m \rightarrow \mathbb{R}^n``
where ``m`` is the dimension of the input ``L_1 u(x)(y) \in \mathbb{R}^m``
and ``n`` is the dimension of the result ``L_2 v(y) \in \mathbb{R}^n``.
Here, ``L_1`` and ``L_2`` are linear operators, e.g. primitive differential
operator evaluations of ``u`` or ``v``.

Let us consider the Newton scheme to find a root of the residual function ``F : \mathbb{R}^N \rightarrow \mathbb{R}^N``,
which iterates
```math
\begin{aligned}
x_{n+1} = x_{n} - D_xF(x_n)^{-1} F(x_n)
\end{aligned}
```
or, equivalently, solves
```math
\begin{aligned}
D_xF(x_n) \left(x_{n+1} - x_{n}\right) = -F(x_n)
\end{aligned}
```


To compute the jacobian of ``F``, observe that its discretisation on a mesh ``\mathcal{T}`` and some quadrature rule
``(x_{qp}, w_{qp})`` leads to
```math
\begin{aligned}
F(x) =  \sum_{T \in \mathcal{T}} \lvert T \rvert \sum_{x_{qp}} A(L_1u_h(x)(x_{qp})) \cdot L_2v_h(x_{qp}) w_{qp} & = 0 \quad \text{in } \Omega
\end{aligned}
```
Now, by linearity of everything involved other than ``A``, we can evaluate the jacobian by
```math
\begin{aligned}
D_xF(x) =  \sum_{T \in \mathcal{T}} \lvert T \rvert \sum_{x_{qp}} DA(L_1 u_h(x)(x_{qp})) \cdot L_2 v_h(x_{qp}) w_{qp} & = 0 \quad \text{in } \Omega
\end{aligned}
```
Hence, assembly only requires to evaluate the low-dimensional jacobians ``DA \in \mathbb{R}^{m \times n}`` of ``A``
at ``L_1 u_h(x)(x_{qp})``. These jacobians are computed by automatic differentiation via ForwardDiff.jl (or via the user-given jacobian function).
If ``m`` and ``n`` are a little larger, e.g. when more operator evaluations ``L_1`` and ``L_2``
or more unknowns are involved, there is the option
to use sparse_jacobians (using the sparsity detection of Symbolics.jl).