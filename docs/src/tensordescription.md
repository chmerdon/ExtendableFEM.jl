
# Tensor Description

To be able to construct [reshaped views](@ref "Reshaped views") 
of the test functions and their derivates, we can describe the 
shape of the view through a [`TensorDescription{R,D}`](@ref ExtendableFEM.TensorDescription{R,D}) 
where `R` is the *rank* of the tensor and `D` is the dimension 
or extent of the tensor in each of the `R` directions. 
That means a real valued `R`-tensor is an element of 
``\underbrace{\mathbb{R}^D\times\cdots\times\mathbb{R}^D}_{R \text{ times}}``. 
Specifically, we can identify the following mathematical objects with 
tensors of different ranks:

| math. object                                 | `R`-Tensor | 
| :------------------------------------------- | :--------- |
| scalar ``\in\mathbb{R}``                     | 0-Tensor   | 
| vector ``\in\mathbb{R}^D``                   | 1-Tensor   | 
| matrix ``\in\mathbb{R}^D\times\mathbb{R}^D`` | 2-Tensor   | 

For finite elements, `D` usually matches the spatial dimension of 
the problem we want to solve, i.e. `D=2` for 2D and `D=3` for 3D.

## Tensor Types

```@docs
ExtendableFEM.TensorDescription
ExtendableFEM.TDScalar
ExtendableFEM.TDVector
ExtendableFEM.TDMatrix
ExtendableFEM.TDRank3
ExtendableFEM.TDRank4
```


## Reshaped views

```@autodocs
Modules = [ExtendableFEM]
Pages = ["tensors.jl"]
Order   = [:function]
```

## Which tensor for which unknown?
For an unknown variable `u` of tensor rank `r` 
a derivative of order `n` has rank `r+n`,
i.e. the hessian (n=2) of a scalar unknown (rank 0)
and the gradient (n=1) of a vector valued (rank 1) 
variable are both matrices (rank 2).

For a more comprehensive list see the following table

| derivative order   | scalar-valued    | vector-valued            | matrix-valued            |
| :----------------- | :--------------- | :----------------------- | :----------------------- |
| 0 (value/`id`)     | `TDScalar(D)`    | `TDVector(D)`            | `TDMatrix(D)`            |
| 1 (`grad`)         | `TDVector(D)`    | `TDMatrix(D)`            | `TDRank3(D)`             |
| 2 (`hessian`)      | `TDMatrix(D)`    | `TDRank3(D)`             | `TDRank4(D)`             |
| 3                  | `TDRank3(D)`     | `TDRank4(D)`             | `TensorDescription(5,D)` |
| 4                  | `TDRank4(D)`     | `TensorDescription(5,D)` | `TensorDescription(6,D)` |
| ``\vdots``         | ``\vdots``       | ``\vdots``               | ``\vdots``               | 



## Helpers


```@docs
tmul!
```
