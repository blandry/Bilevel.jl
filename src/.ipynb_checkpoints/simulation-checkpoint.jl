struct SimData
    mechanism::Mechanism
    env::Environment
    state_cache::Vector{StateCache}
    envj_cache::Vector{EnvironmentJacobianCache}
    Δt::Real
    vs::VariableSelector
    cs::ConstraintSelector
    generate_solver_fn::Symbol
    extract_sol::Symbol
    lower_vs::Vector{VariableSelector}
    lower_cs::Vector{ConstraintSelector}
    lower_options::Vector{Dict}
    normal_vs::Vector{VariableSelector}
    normal_cs::Vector{ConstraintSelector}
    normal_options::Vector{Dict}
    fric_vs::Vector{VariableSelector}
    fric_cs::Vector{ConstraintSelector}
    fric_options::Vector{Dict}
    N::Int
    con_fns::Vector{Tuple{Symbol,Any}}
    obj_fns::Vector{Tuple{Symbol,Any}}
    fric_osqp_models::Vector{OSQP.Model}
    normal_osqp_models::Vector{OSQP.Model}
end

function add_eq!(sim_data::SimData, name::Symbol, size::Int, fun)
    add_eq!(sim_data.cs, name, size)
    push!(sim_data.con_fns, (name, fun))

    sim_data.con_fns
end

function add_ineq!(sim_data::SimData, name::Symbol, size::Int, fun)
    add_ineq!(sim_data.cs, name, size)
    push!(sim_data.con_fns, (name, fun))

    sim_data.con_fns
end

function add_obj!(sim_data::SimData, name::Symbol, fun)
    push!(sim_data.obj_fns, (name, fun))

    sim_data.obj_fns
end

function add_box_con!(sim_data::SimData, name::Symbol, var_name::Symbol, min::AbstractArray{T}, max::AbstractArray{T}) where T
    min_name = Symbol(name, "_min")
    add_ineq!(sim_data.cs, min_name, length(min))
    push!(sim_data.con_fns, (min_name, x -> min - sim_data.vs(x, var_name)))

    max_name = Symbol(name, "_max")
    add_ineq!(sim_data.cs, max_name, length(max))
    push!(sim_data.con_fns, (max_name, x -> sim_data.vs(x, var_name) - max))

    sim_data.con_fns
end

function simulate(sim_data::SimData,control!,state0::MechanismState,N::Int;
                  opt_tol=1e-6,major_feas=1e-6,minor_feas=1e-6,verbose=0)

    results = zeros(sim_data.vs.num_vars, N)
    results[sim_data.vs(:qnext), 1] = configuration(state0)
    results[sim_data.vs(:vnext), 1] = velocity(state0)

    x_ctrl = MechanismState(sim_data.mechanism)
    u0 = zeros(num_velocities(sim_data.mechanism))

    for i in 2:N
        x = results[:,i-1]
        q0 = sim_data.vs(x, :qnext)
        v0 = sim_data.vs(x, :vnext)

        set_configuration!(x_ctrl, q0)
        set_velocity!(x_ctrl, v0)
        control!(u0, (i-2)*sim_data.Δt, x_ctrl)

        solver_fn = eval(sim_data.generate_solver_fn)(sim_data, q0, v0, u0)

        options = Dict{String, Any}()
        options["Derivative option"] = 1
        options["Verify level"] = -1 # -1 => 0ff, 0 => cheap
        options["Major optimality tolerance"] = opt_tol
        options["Major feasibility tolerance"] = major_feas
        options["Minor feasibility tolerance"] = minor_feas

        xopt, info = snopt(solver_fn, sim_data.cs.num_eqs, sim_data.cs.num_ineqs, x, options)

        if verbose >= 1
            println(info)
        end

        results[:,i] .= xopt
    end

    sol = eval(sim_data.extract_sol)(sim_data, results)

    sol
end
