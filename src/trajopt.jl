function trajopt(sim_data::SimData;
                 x0=nothing,quaternion_state=false,
                 opt_tol=1e-6,major_feas=1e-6,minor_feas=1e-6,verbose=0)

    solver_fn = eval(sim_data.generate_solver_fn)(sim_data)

    if isa(x0, Nothing)
        # should start with some linear interpolation here
        
        # need a better way to do this...
        x0 = zeros(sim_data.vs.num_vars)
        if quaternion_state
            for n = 1:sim_data.N
                x0[sim_data.vs(Symbol("q", n))] .= [1., 0., 0., 0., 0., 0., 0.]
            end
        end
    end

    options = Dict{String, Any}()
    options["Derivative option"] = 1
    options["Verify level"] = -1 # -1 => 0ff, 0 => cheap
    options["Major optimality tolerance"] = opt_tol
    options["Major feasibility tolerance"] = major_feas
    options["Minor feasibility tolerance"] = minor_feas

    xopt, info = snopt(solver_fn, sim_data.cs.num_eqs, sim_data.cs.num_ineqs, x0, options)

    if verbose >= 1
        println(info)
    end
    
    sol = eval(sim_data.extract_sol)(sim_data, xopt)
    
    sol
end