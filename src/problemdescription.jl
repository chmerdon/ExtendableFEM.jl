"""
````
struct ProblemDescription
````

Structure holding data for a problem description with the following fields:

$(TYPEDFIELDS)
"""
mutable struct ProblemDescription
    """
    The name of the problem used for printout messages. Default: "My Problem"
    """
    name::String
    """
    A vector of Unknowns that are involved in the problem.
    """
    unknowns::Array{Unknown,1}
    """
    A vector of operators that are involved in the problem.
    """
    operators::Array{AbstractOperator,1}
    #reduction_operators::Array{AbstractReductionOperator,1}
end

"""
````
ProblemDescription(name = "My problem")
````

Generates an empty ProblemDescription with the given name.

"""
function ProblemDescription(name = "My problem")
    return ProblemDescription(name, Array{Unknown,1}(undef,0), Array{AbstractOperator,1}(undef,0))
end


"""
````
assign_unknown!(PD::ProblemDescription, u::Unknown)
````

Assigns the Unknown u to the ProblemDescription PD
and returns its position in the unknowns array of the ProblemDescription.

"""
function assign_unknown!(PD::ProblemDescription, u::Unknown)
    if u in PD.unknowns
        @warn "This unknown was already assigned to the problem description! Ignoring this call."
        return find(==(u), PD.unknowns)
    else
        push!(PD.unknowns, u)
        return length(PD.unknowns)
    end
end


"""
````
assign_operator!(PD::ProblemDescription, o::AbstractOperator)
````

Assigns the AbstractOperator o to the ProblemDescription PD
and returns its position in the operators array of the ProblemDescription.

"""
function assign_operator!(PD::ProblemDescription, o::AbstractOperator)
    push!(PD.operators, o)
    return length(PD.operators)
end


"""
````
replace_operator!(PD::ProblemDescription, j::Int, o::AbstractOperator)
````

Replaces the j-th operator of the ProblemDescription PD by the new operator o.
Here, j is the position in operator array returned by the assign_operator! function.
Nothing is returned (as the new operator gets position j).

"""
function replace_operator!(PD::ProblemDescription, j, o::AbstractOperator)
    PD.operators[j] = o
    return nothing
end

#function assign_reduction!(PD::ProblemDescription, u::AbstractReductionOperator)
#    push!(PD.reduction_operators, u)
#end

function Base.show(io::IO, PD::ProblemDescription)
    println(io, "\nPDE-DESCRIPTION")
    println(io, "    • name = $(PD.name)")
    println(io, "\n  <<<UNKNOWNS>>>") 
    for u in PD.unknowns
        print(io, "    • $u")
    end

    println(io, "\n  <<<OPERATORS>>>") 
    for o in PD.operators
        println(io, "    • $(o)")
    end

    #println(io, " reductions =") 
    #for o in PD.reduction_operators
    #    println(io, "    • $o")
    #end
end
