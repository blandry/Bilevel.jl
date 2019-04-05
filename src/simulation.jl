struct SimData
    mechanism::Mechanism
    env::Environment
    x0_cache::StateCache
    xn_cache::StateCache
    envj_cache::EnvironmentJacobianCache
    Δt::Real
    vs::VariableSelector
    cs::ConstraintSelector
    generate_solver_fn::Symbol
end

function simulate(sim_data::SimData,control!,state0::MechanismState,N::Int;
                  opt_tol=1e-6,major_feas=1e-6,minor_feas=1e-6,verbose=0)
    
    results = zeros(sim_data.vs.num_vars)
    results[sim_data.vs(:qnext)] = configuration(state0)
    results[sim_data.vs(:vnext)] = velocity(state0)

    x_ctrl = MechanismState(sim_data.mechanism)
    u0 = zeros(num_velocities(sim_data.mechanism))

    for i in 1:N
        x = results[:,end]
        q0 = sim_data.vs(x, :qnext)
        v0 = sim_data.vs(x, :vnext)

        set_configuration!(x_ctrl, q0)
        set_velocity!(x_ctrl, v0)
        control!(u0, (i-1)*sim_data.Δt, x_ctrl)

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

        results = hcat(results,xopt)
    end

    results
end