struct SimData
    mechanism::Mechanism
    x0_cache::StateCache
    xnext_cache::StateCache
    env_cache::Environment
    Δt::Real
    relax_comp::Bool
    vs::VariableSelector
    cs::ConstraintSelector
    generate_solver_fn::Symbol
end

function simulate(sim_data::SimData,control!,state0::MechanismState,N::Int;
                  opt_tol=1e-6,major_feas=1e-6,minor_feas=1e-6,verbose=0)
    x_L = -1e19 * ones(sim_data.num_xn)
    x_U = 1e19 * ones(sim_data.num_xn)
    results = zeros(sim_data.num_xn)
    results[1:sim_data.num_q] = configuration(state0)
    results[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = velocity(state0)

    x_ctrl = MechanismState(sim_data.mechanism)
    u0 = zeros(sim_data.num_v)

    for i in 1:N
        x = results[:,end]
        q0 = x[1:sim_data.num_q]
        v0 = x[sim_data.num_q+1:sim_data.num_q+sim_data.num_v]

        set_configuration!(x_ctrl,q0)
        set_velocity!(x_ctrl,v0)
        setdirty!(x_ctrl)
        control!(u0, (i-1)*sim_data.Δt, x_ctrl)

        solver_fn = eval(sim_data.generate_solver_fn)(sim_data,q0,v0,u0)

        options = Dict{String, Any}()
        options["Derivative option"] = 1
        options["Verify level"] = -1 # -1 = 0ff, 0 = cheap
        options["Major optimality tolerance"] = opt_tol
        options["Major feasibility tolerance"] = major_feas
        options["Minor feasibility tolerance"] = minor_feas

        xopt, fopt, info = snopt(solver_fn, x, x_L, x_U, options)
        
        if verbose >= 1
            println(info)
        end

        results = hcat(results,xopt)
    end

    results
end