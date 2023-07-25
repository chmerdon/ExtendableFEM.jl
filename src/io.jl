
#
# Print default dict for solver parameters into docstrings
#
function _myprint(dict::Dict{Symbol,Tuple{Any,String}})
    lines_out=IOBuffer()
    for (k,v) in dict
        if typeof(v[1]) <: String
            println(lines_out,"  - $(k): $(v[2]). Default: ''$(v[1])''\n")
        else
            println(lines_out,"  - $(k): $(v[2]). Default: $(v[1])\n")
        end
    end
    String(take!(lines_out))
end


function center_string(S::String, L::Int = 8)
    if length(S) > L
        S = S[1:L]
    end
    while length(S) < L-1
        S = " " * S * " "
    end
    if length(S) < L
        S = " " * S
    end
    return S
end


function print_convergencehistory(X, Y; X_to_h = X -> X, ylabels = [], xlabel = "ndofs", latex_mode = false, seperator = latex_mode ? "&" : "|", order_seperator = latex_mode ? "&" : "")
    xlabel = center_string(xlabel,12)
    if latex_mode
        tabular_argument = "c"
        for j = 1 : size(Y,2)
            tabular_argument *= "|cc"
        end
        @printf("\\begin{tabular}{%s}",tabular_argument)
    end
    @printf("\n%s%s",xlabel,seperator)
    for j = 1 : size(Y,2)
        if length(ylabels) < j
            push!(ylabels, "DATA $j")
        end
        if j == size(Y,2)
            @printf("%s %s order %s", center_string(ylabels[j],18), order_seperator, latex_mode ? "" : seperator)
        else
            @printf("%s %s order %s", center_string(ylabels[j],18), order_seperator, seperator)
        end
    end
    @printf("\n")
    if latex_mode
        @printf("\\\\\\hline")
    else
        @printf("============|")
        for j = 1 : size(Y,2)
            @printf("==========================|")
        end
    end
    @printf("\n")
    order = 0
    for j=1:length(X)
        @printf("   %7d  %s",X[j],seperator);
        for k = 1 : size(Y,2)
            if j > 1
                order = -log(Y[j-1,k]/Y[j,k]) / (log(X_to_h(X[j])/X_to_h(X[j-1])))
            end
            if k == size(Y,2)
                @printf("     %.3e  %s    %.2f  %s",Y[j,k], order_seperator, order, latex_mode ? "" : seperator)
            else
                @printf("     %.3e  %s    %.2f  %s",Y[j,k], order_seperator, order, seperator)
            end
        end
        if latex_mode
            @printf("\\\\")
        end
        @printf("\n")
    end
    if latex_mode
        @printf("\\end{tabular}")
    end
end

function print_table(X, Y; ylabels = [], xlabel = "ndofs")
    xlabel = center_string(xlabel,12)
    @printf("\n%s|",xlabel)
    for j = 1 : size(Y,2)
        if length(ylabels) < j
            push!(ylabels, "DATA $j")
        end
        @printf(" %s |", center_string(ylabels[j],20))
    end
    @printf("\n")
    @printf("============|")
    for j = 1 : size(Y,2)
        @printf("======================|")
    end
    @printf("\n")
    for j=1:length(X)
        if eltype(X) <: Int
            @printf("   %7d  |",X[j]);
        else
            @printf("  %.2e  |",X[j]);
        end
        for k = 1 : size(Y,2)
            @printf("    %.8e    |",Y[j,k])
        end
        @printf("\n")
    end
end
