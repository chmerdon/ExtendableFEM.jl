function GridVisualize.scalarplot!(p, op::Tuple{Unknown, DataType}, sol; abs = false, component = 1, title = String(op[1].identifier), kwargs...)
	GridVisualize.scalarplot!(p, sol[op[1]].FES.dofgrid, view(nodevalues(sol[op[1]], op[2]; abs = abs), component, :); title = title, kwargs...)
end
function GridVisualize.scalarplot!(p, op::Tuple{Int, DataType}, sol; abs = false, component = 1, title = sol[op[1]].name, kwargs...)
	GridVisualize.scalarplot!(p, sol[op[1]].FES.dofgrid, view(nodevalues(sol[op[1]], op[2]; abs = abs), component, :); title = title, kwargs...)
end
function GridVisualize.vectorplot!(p, op::Tuple{Unknown, DataType}, sol; title = String(op[1].identifier), kwargs...)
	GridVisualize.vectorplot!(p, sol[op[1]].FES.dofgrid, eval_func_bary(PointEvaluator([op], sol)); title = title, kwargs...)
end
function GridVisualize.vectorplot!(p, op::Tuple{Int, DataType}, sol; title = sol[op[1]].name, kwargs...)
	GridVisualize.vectorplot!(p, sol[op[1]].FES.dofgrid, eval_func_bary(PointEvaluator([op], sol)); title = title, kwargs...)
end

function plot!(p::GridVisualizer, ops, sol; rasterpoints = 10, keep = [], ncols = size(p.subplots, 2), do_abs = true, do_vector_plots = true, title_add = "", kwargs...)
	col, row, id = 0, 1, 0
	for op in ops
		col += 1
		id += 1
		if col == ncols + 1
			col, row = 1, row + 1
		end
		while id in keep
			col += 1
			id += 1
			if col == ncols + 1
				col, row = 1, row + 1
			end
		end
		if op[2] == "grid"
			gridplot!(p[row, col], sol[op[1]].FES.xgrid; kwargs...)
		elseif op[2] == "dofgrid"
			gridplot!(p[row, col], sol[op[1]].FES.dofgrid; kwargs...)
		else
			ncomponents = get_ncomponents(sol[op[1]])
			edim = size(sol[op[1]].FES.xgrid[Coordinates], 1)
			resultdim = Length4Operator(op[2], edim, ncomponents)
			if typeof(op[1]) <: Unknown
				title = op[2] == Identity ? String(op[1].identifier) : "$(op[2])(" * String(op[1].identifier) * ")"
			else
				title = op[2] == Identity ? "$(sol[op[1]].name)" : "$(op[2])($(sol[op[1]].name))"
			end
			if resultdim == 1
				GridVisualize.scalarplot!(p[row, col], sol[op[1]].FES.dofgrid, view(nodevalues(sol[op[1]], op[2]; abs = false), 1, :), title = title * title_add; kwargs...)
			elseif do_abs == true
				GridVisualize.scalarplot!(p[row, col], sol[op[1]].FES.dofgrid, view(nodevalues(sol[op[1]], op[2]; abs = true), 1, :), title = "|" * title * "|" * title_add; kwargs...)
			else
				nv = nodevalues(sol[op[1]], op[2]; abs = false)
				for k ∈ 1:resultdim
					if k > 1
						col += 1
						if col == ncols + 1
							col, row = 1, row + 1
						end
					end
					GridVisualize.scalarplot!(p[row, col], sol[op[1]].FES.dofgrid, view(nv, k, :), title = title * " (component $k)" * title_add, kwargs...)
				end
			end
			if resultdim > 1 && do_vector_plots && do_abs == true && edim > 1
				GridVisualize.vectorplot!(p[row, col], sol[op[1]].FES.dofgrid, eval_func_bary(PointEvaluator([op], sol)); rasterpoints = rasterpoints, title = "|" * title * "|" * " + quiver" * title_add, clear = false, kwargs...)
			end
		end
	end
	return p
end

function plot(ops, sol; add = 0, Plotter = nothing, ncols = min(2, length(ops) + add), do_abs = true, width = (length(ops) + add) == 1 ? 400 : 800, height = 0, kwargs...)
	nplots = length(ops) + add
	for op in ops
		ncomponents = get_ncomponents(sol[op[1]])
		edim = size(sol[op[1]].FES.xgrid[Coordinates], 1)
		if !(op[2] in ["grid", "dofgrid"])
			resultdim = Length4Operator(op[2], edim, ncomponents)
			if resultdim > 1 && do_abs == false
				nplots += resultdim - 1
			end
		end
	end
	nrows = Int(ceil(nplots / ncols))
	if height == 0
		height = width / ncols * nrows
	end
	p = GridVisualizer(; Plotter = Plotter, layout = (nrows, ncols), clear = true, resolution = (width, height))
	plot!(p, ops, sol; do_abs = do_abs, kwargs...)
end

function plot_unicode(sol; kwargs...)
	for u ∈ 1:length(sol)
		println(stdout, unicode_scalarplot(sol[u]; title = sol[u].name, kwargs...))
	end
end

function GridVisualize.vectorplot!(p, xgrid, op::Tuple{Union{Unknown, Int}, DataType}, sol; title = sol[op[1]].name, kwargs...)
	GridVisualize.vectorplot!(p, xgrid, eval_func(PointEvaluator([op], sol)); title = title, kwargs...)
end


function plot_convergencehistory!(
	target,
	X,
	Y;
	add_h_powers = [],
	X_to_h = X -> X,
	colors = [:blue, :green, :red, :magenta, :lightblue],
	title = "convergence history",
	legend = :best,
	ylabel = "",
	ylabels = [],
	xlabel = "ndofs",
	markershape = :circle,
	markevery = 1,
	clear = true,
	args...,
)
	for j ∈ 1:size(Y, 2)
		Xk = []
		Yk = []
		for k ∈ 1:length(X)
			if Y[k, j] > 0
				push!(Xk, X[k])
				push!(Yk, Y[k, j])
			end
		end
		if length(ylabels) >= j
			label = ylabels[j]
		else
			label = "Data $j"
		end
		scalarplot!(
			target,
			simplexgrid(Xk),
			Yk;
			xlabel = xlabel,
			ylabel = ylabel,
			color = length(colors) >= j ? colors[j] : :black,
			clear = j == 1 ? clear : false,
			markershape = markershape,
			markevery = markevery,
			xscale = :log,
			yscale = :log,
			label = label,
			legend = legend,
			title = title,
			args...,
		)
	end
	for p in add_h_powers
		label = "h^$p"
		scalarplot!(target, simplexgrid(X), X_to_h(X) .^ p; linestyle = :dot, xlabel = xlabel, ylabel = ylabel, color = :gray, clear = false, markershape = :none, xscale = :log, yscale = :log, label = label, legend = legend, title = title, args...)
	end
end

function plot_convergencehistory(X, Y; Plotter = nothing, size = (800, 600), add_h_powers = [], X_to_h = X -> X, colors = [:blue, :green, :red, :magenta, :lightblue], legend = :best, ylabel = "", ylabels = [], xlabel = "ndofs", clear = true, args...)
	p = GridVisualizer(; Plotter = Plotter, layout = (1, 1), clear = true, size = size)
	plot_convergencehistory!(p[1, 1], X, Y; add_h_powers = add_h_powers, X_to_h = X_to_h, colors = colors, legend = legend, ylabel = ylabel, ylabels = ylabels, xlabel = xlabel, clear = clear, args...)
end


function ExtendableFEMBase.nodevalues(op, sol; kwargs...)
	return nodevalues(sol[op[1]], op[2]; kwargs...)
end

## default function for generateplots for ExampleJuggler.jl
function default_generateplots(example_module, filename; kwargs...)
	function closure(dir = pwd(); Plotter = nothing, kwargs...)
		~, plt = example_module.main(; Plotter = Plotter, kwargs...)
		scene = GridVisualize.reveal(plt)
		GridVisualize.save(joinpath(dir, filename), scene; Plotter = Plotter)
	end
end
