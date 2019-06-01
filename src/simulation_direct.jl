function get_sim_data_direct(mechanism::Mechanism,env::Environment,Δt::Real;
                             relax_comp=false)
    num_contacts = length(env.contacts)

    vs = VariableSelector()
    add_var!(vs, :qnext, num_positions(mechanism))
    add_var!(vs, :vnext, num_velocities(mechanism))
    if relax_comp
        add_var!(vs, :slack, 1)
    end

    cs = ConstraintSelector()
    add_eq!(cs, :kin, num_positions(mechanism))
    add_eq!(cs, :dyn, num_velocities(mechanism))

    state_cache = [StateCache(mechanism) for n = 1:2]
    envj_cache = [EnvironmentJacobianCache(env) for n = 1:2]

    generate_solver_fn = :generate_solver_fn_sim_direct
    extract_sol = :extract_sol_sim_direct

    normal_vs = VariableSelector()
    for i = 1:length(env.contacts)
        add_var!(normal_vs, Symbol("c_n", i), 1)
    end

    normal_cs = ConstraintSelector()
    for i = 1:length(env.contacts)
        add_ineq!(normal_cs, Symbol("c_n_pos", i), 1)
        add_ineq!(normal_cs, Symbol("ϕ", i), 1)
    end

    normal_options = Dict{String, Any}()
    normal_options["num_fosteps"] = 1
    normal_options["num_sosteps"] = 10
    normal_options["c"] = 10
    normal_options["c_fos"] = 10
    normal_options["c_sos"] = 1

    fric_vs = VariableSelector()
    for i = 1:length(env.contacts)
        β_dim = size(env.contacts[i].obstacle.basis,2)
        add_var!(fric_vs, Symbol("β", i), β_dim)
    end

    fric_cs = ConstraintSelector()
    for i = 1:length(env.contacts)
        β_dim = size(env.contacts[i].obstacle.basis,2)
        add_ineq!(fric_cs, Symbol("β_pos", i), β_dim)
        add_ineq!(fric_cs, Symbol("fric_cone", i), 1)
    end

    fric_options = Dict{String, Any}()
    fric_options["num_fosteps"] = 1
    fric_options["num_sosteps"] = 10
    fric_options["c"] = 10
    fric_options["c_fos"] = 10
    fric_options["c_sos"] = 1

    SimData(mechanism,env,
            state_cache,envj_cache,
            Δt,vs,cs,generate_solver_fn,extract_sol,
            [],[],[],[normal_vs],[normal_cs],[normal_options],[fric_vs],[fric_cs],[fric_options],1,[],[])
end

function extract_sol_sim_direct(sim_data::SimData, results::AbstractArray{T,2}) where T
    vs = sim_data.vs
    relax_comp = haskey(vs.vars, :slack)
    env = sim_data.env
    N = size(results,2)

    qtraj = Array{Array{Float64,1},1}(undef, 0)
    vtraj = Array{Array{Float64,1},1}(undef, 0)
    utraj = Array{Array{Float64,1},1}(undef, 0)
    contact_traj = Array{Array{Float64,1},1}(undef, 0)
    slack_traj = Array{Array{Float64,1},1}(undef, 0)
    x = MechanismState(sim_data.mechanism)
    for n = 1:N
        set_configuration!(x, vs(results[:,n], Symbol("qnext")))
        normalize_configuration!(x)
        qnext = configuration(x)
        push!(qtraj, qnext)
        push!(vtraj, vs(results[:,n], Symbol("vnext")))
        if relax_comp
            push!(slack_traj, vs(results[:,n], Symbol("slack")))
        end
        # TODO compute contact force for comparison purposes
    end

    # some other usefull vectors
    ttraj = [(i-1)*sim_data.Δt for i = 1:N]
    qv_mat = vcat(hcat(qtraj...),hcat(vtraj...))

    qtraj, vtraj, utraj, contact_traj, slack_traj, ttraj, qv_mat, results
end

function generate_solver_fn_sim_direct(sim_data,q0,v0,u0)
    x0 = sim_data.state_cache[1][Float64]
    envj = sim_data.envj_cache[1][Float64]
    Δt = sim_data.Δt
    vs = sim_data.vs
    cs = sim_data.cs

    relax_comp = haskey(vs.vars, :slack)
    num_contacts = length(sim_data.env.contacts)
    num_vel = num_velocities(sim_data.mechanism)

    set_configuration!(x0, q0)
    set_velocity!(x0, v0)
    setdirty!(x0)
    H = mass_matrix(x0)
    Hi = inv(H)

    contact_jacobian!(envj, x0)
    dyn_bias0 = dynamics_bias(x0) # TODO preallocate

    function eval_obj(x::AbstractArray{T}) where T
        f = 0.

        if relax_comp
            slack = vs(x, :slack)
            f += .5 * slack' * slack
        end

        f
    end

    function eval_cons(x::AbstractArray{T}) where T
        xn = sim_data.state_cache[2][T]

        normal_bias = Vector{T}(undef, num_vel)
        contact_bias = Vector{T}(undef, num_vel)
        g = Vector{T}(undef, cs.num_eqs + cs.num_ineqs) # TODO preallocate

        qnext = vs(x, :qnext)
        vnext = vs(x, :vnext)
        if relax_comp
            slack = vs(x, :slack)
        end

        set_configuration!(xn, qnext)
        set_velocity!(xn, vnext)
        setdirty!(xn)

        normalize_configuration!(xn)

        if (num_contacts > 0)
            # compute normal forces
            x_normal = contact_normal_τ_direct!(normal_bias, sim_data, sim_data.Δt, Hi, envj, dyn_bias0, u0, v0, x, 1)

            # compute friction forces
            # contact_friction_τ_direct!(contact_bias, sim_data, sim_data.Δt, Hi, envj, dyn_bias0, u0, v0, x, x_normal, 1)
            contact_friction_τ_direct_osqp!(contact_bias, sim_data, sim_data.Δt, Hi, envj, dyn_bias0, u0, v0, x, x_normal, 1)
        end
        config_derivative = configuration_derivative(xn) # TODO preallocate
        dyn_bias = dynamics_bias(xn) # TODO preallocate

        g[cs(:kin)] .= qnext .- q0 .- Δt .* config_derivative
        g[cs(:dyn)] .= H * (vnext - v0) .- Δt .* (u0 .- dyn_bias .- contact_bias)

        g
    end

    generate_autodiff_solver_fn(eval_obj,eval_cons,cs.eqs,cs.ineqs,vs.num_vars)
end
