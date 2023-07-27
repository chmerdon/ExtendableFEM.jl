#= 

# 270 : Natural convection
([source code](SOURCE_URL))

Seek velocity ``u``, pressure ``p`` and temperature ``\theta`` such that
```math
\begin{aligned}
	- \mu \Delta u + (u \cdot \nabla) u + \nabla p & = Ra \, \theta \, g \\
       - \Delta \theta + u \cdot \nabla \theta & = 0
\end{aligned}
```
on a given domain ``\Omega`` (here a triangle) and boundary conditions
```math
\begin{aligned}
	u & = 0 && \quad \text{along } \partial \Omega\\
 	T & = T_\text{bottom} &&\quad \text{along } y = 0\\
	T & = 0 &&\quad \text{along } x = 0
\end{aligned}
```

The weak formulation seeks ``(u,p,\theta) \in V \times Q \times X \subseteq H^1_0(\Omega)^2 \times L^2_0(\Omega) \times H^1_D(\Omega)`` such that
```math
\begin{aligned}
	(\mu \nabla u, \nabla v) + ((u \cdot \nabla) u, v) - (\mathrm{div} v, p) & = (v, Ra g \, \theta) && \quad \text{for all } v \in V,\\
(\mathrm{div} u, q) & = 0 && \quad \text{for all } q \in Q,\\
       (\nabla \theta, \nabla \varphi) + (u \cdot \nabla \theta, \varphi) & = 0
 && \quad \text{for all } \varphi \in X.
\end{aligned} 
```

=#


module Example270_NaturalConvectionProblem

using ExtendableFEM
using ExtendableFEMBase
using GridVisualize
using ExtendableGrids

const μ = 1.0
const ϵ = 1.0
const Ra_final = 1.0e6

function kernel_nonlinear!(result, u_ops, qpinfo)
    u, ∇u, p, ∇T, T = view(u_ops, 1:2), view(u_ops,3:6), view(u_ops, 7), view(u_ops, 8:9), view(u_ops, 10)
    Ra = qpinfo.params[1]
    result[1] = dot(u, view(∇u,1:2))
    result[2] = dot(u, view(∇u,3:4)) - Ra*T[1]
    result[3] = μ*∇u[1] - p[1]
    result[4] = μ*∇u[2]
    result[5] = μ*∇u[3]
    result[6] = μ*∇u[4] - p[1]
    result[7] = -(∇u[1] + ∇u[4])
    result[8] = ϵ*∇T[1]
    result[9] = ϵ*∇T[2]
    result[10] = dot(u, ∇T)
    return nothing
end

function T_bottom!(result, qpinfo)
    x = qpinfo.x
    result[1] = 2*(1-cos(2*π*x[1]))
end

function main(; nrefs = 5, Plotter = nothing, kwargs...)

    ## problem description
    PD = ProblemDescription()
    u = Unknown("u"; name = "velocity")
    p = Unknown("p"; name = "pressure")
    T = Unknown("p"; name = "temperature")
    assign_unknown!(PD, u)
    assign_unknown!(PD, p)
    assign_unknown!(PD, T)
    assign_operator!(PD, NonlinearOperator(kernel_nonlinear!, [apply(u, ReconstructionIdentity{HDIVBDM1{2}}),grad(u),id(p),grad(T),id(T)]; kwargs...))#; jacobian = kernel_jacobian!))
    assign_operator!(PD, HomogeneousBoundaryData(u; regions = 1:3))
    assign_operator!(PD, HomogeneousBoundaryData(T; regions = 3))
    assign_operator!(PD, InterpolateBoundaryData(T, T_bottom!; regions = 1))

    ## grid
    xgrid = uniform_refine(reference_domain(Triangle2D), nrefs)

    ## FESpaces
    FES = Dict(u => FESpace{H1BR{2}}(xgrid),
               p => FESpace{L2P0{1}}(xgrid),
               T => FESpace{H1P1{1}}(xgrid))

    ## solve by Ra embedding
	extra_params = Array{Float64,1}([min(Ra_final, 4000)])
    sol, SC = solve(PD, FES; return_config = true, target_residual = 1e-6, params = extra_params, maxiterations = 20, kwargs...)
	step = 0
	while (true)
		if extra_params[1] >= Ra_final
			break
		end
		extra_params[1] = min(Ra_final, extra_params[1]*3)
		step += 1
		@info "Step $step : solving for Ra=$(extra_params[1])"
        
        sol, SC = ExtendableFEM.solve(PD, FES, SC; init = sol, return_config = true, target_residual = 1e-7, params = extra_params, kwargs...)
        scalarplot(xgrid, nodevalues(sol[T])[1,:]; Plotter = Plotter)
	end
end

end # module