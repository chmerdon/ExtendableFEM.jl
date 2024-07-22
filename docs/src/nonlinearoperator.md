
# NonlinearOperator

A nonlinear operator automatically assembles all necessary terms for the
Newton method. Other linearisations of a nonlinear operator can be
constructed with special constructors for BilinearOperator or LinearOperator.

## Constructor

To describe a NonlinearOperator we have to specify a kernel function. 
These functions are 'flat' in the sense that the input and output vector 
contain the components of the test-function values and derivatives
as specified by `oa_test` and `oa_args` respectively.
The assembly of the local matrix will be done internally 
by multiplying the subvectors of result with its test-function counterparts.
For a more detailed explanation of this see the following

```@autodocs
Modules = [ExtendableFEM]
Pages = ["common_operators/nonlinear_operator.jl"]
Order   = [:type, :function]
```

## Example - NSE convection operator


For the Navier--Stokes equations, we need a kernel function for the nonlinear
convection term
```math
\begin{equation}
(v,u\cdot\nabla u) = (v,\nabla u^T u)
\end{equation}
```
In 2D the input (as specified below) will contain the two
components of ``u=(u_1,u_2)'`` and the four components of the gradient 
``\nabla u = \begin{pmatrix} u_{11} & u_{12} \\ u_{21} & u_{22}\end{pmatrix}``
in order, i.e. ``(u_1,u_2,u_{11},u_{12},u_{21},u_{22})``.
As the convection term is tested with ``v``, 
the ouptut vector ``o`` only has to contain what should be tested with each component
of ``v``, i.e.
```math
\begin{equation}
    A_\text{local} = (v_1,v_2)^T(o_1,o_2) = 
        \begin{pmatrix}
            v_1o_1 & v_1o_2\\
            v_2o_1 & v_2o_2
        \end{pmatrix}.
\end{equation}
```
To construct the kernel there are two options, 
component-wise and based on `tensor_view`.
For the first we have to write the convection term as individual components
```math
\begin{equation}
o = 
    \begin{pmatrix}
        u_1\cdot u_{11}+u_2\cdot u_{12}\\
        u_1\cdot u_{21}+u_2\cdot u_{22}\\
    \end{pmatrix}
= 
\begin{pmatrix}
    u\cdot (u_11,u_12)^T\\
    u\cdot (u_21,u_22)^T
\end{pmatrix}.
\end{equation}
```
To make our lives a bit easier we will extract the subcompontents of 
`input` as views, such that `∇u[3]` actually accesses `input[5]`,
which corresponds to the third entry ``u_{21}`` of ``\nabla u``. 
```julia
function kernel!(result, input, qpinfo)
    u, ∇u = view(input, 1:2), view(input,3:6)
    result[1] = dot(u, view(∇u,1:2))
    result[2] = dot(u, view(∇u,3:4))
    return nothing
end
```
To improve readability of the kernels and to make them easier to understand,
we provide the function `tensor_view` which constructs a view and reshapes 
it into an object matching the given `TensorDescription`.
See the [table](@ref "Which tensor for which unknown?") 
to see which tensor size is needed for which derivative of a scalar, vector 
or matrix-valued variable.
```julia
function kernel!(result, input, qpinfo)
    u = tensor_view(input,1,TDVector(2))
    v = tensor_view(result,1,TDVector(2))
    ∇u = tensor_view(input,3,TDMatrix(2))
    tmul!(v,∇u,u)
    return nothing
end
```

The coressponding NonlinearOperator constructor call is the same in both cases 
and reads
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

!!! note "Dimension independent kernels"

    If done correctly, the operator-based approach allows us to write a kernel 
    that is 'independent' of the spatial dimension, 
    i.e. one instead of up to three kernels.
    Assuming `dim` is a known variable we can re-write the kernel from above as
    ```julia
    function kernel!(result, input, qpinfo)
        u = tensor_view(input,1,TDVector(dim))
        v = tensor_view(result,1,TDVector(dim))
        ∇u = tensor_view(input,1+dim,TDMatrix(dim))
        tmul!(v,∇u,u)
        return nothing
    end
    ```

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