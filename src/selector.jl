mutable struct VarSelector
    vars::Dict{Symbol, UnitRange{Int}}
    num_vars::Int

    function VarSelector()
        new(Dict{Symbol, UnitRange{Int}}(), 0)
    end
end

(vs::VarSelector)(name::Symbol) = vs.vars[name]
(vs::VarSelector)(x::AbstractArray{T}, name::Symbol) where T = x[vs.vars[name]]

function add_var!(selector::VarSelector, name::Symbol, size::Int)
    if haskey(selector.vars, name)
        throw(ArgumentError("Variable name '$name' already exists"))
    end
    if size < 1
        throw(ArgumentError("Variable size must be greater than 0"))
    end
    selector.vars[name] = selector.num_vars .+ (1:size)
    selector.num_vars += size
    
    selector.num_vars
end

mutable struct ConSelector
    eqs::Dict{Symbol, UnitRange{Int}}
    num_eqs::Int
    ineqs::Dict{Symbol, UnitRange{Int}}
    num_ineqs::Int
    
    function ConSelector()
        new(Dict{Symbol, UnitRange{Int}}(), 0, Dict{Symbol, UnitRange{Int}}(), 0)
    end
end

function add_eq!(selector::ConSelector, name::Symbol, size::Int)
    if haskey(selector.eqs, name) || haskey(selector.ineqs, name)
        throw(ArgumentError("Constraint name '$name' already exists"))
    end
    if size < 1
        throw(ArgumentError("Constraint size must be greater than 0"))
    end
    selector.eqs[name] = selector.num_eqs .+ (1:size)
    selector.num_eqs += size
    
    selector.num_eqs
end

function add_ineq!(selector::ConSelector, name::Symbol, size::Int)
    if haskey(selector.eqs, name) || haskey(selector.ineqs, name)
        throw(ArgumentError("Constraint name '$name' already exists"))
    end
    if size < 1
        throw(ArgumentError("Constraint size must be greater than 0"))
    end
    selector.ineqs[name] = selector.num_ineqs .+ (1:size)
    selector.num_ineqs += size
    
    selector.num_ineqs
end