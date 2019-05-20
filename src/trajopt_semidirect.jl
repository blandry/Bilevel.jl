function get_trajopt_data_semidirect(mechanism::Mechanism,env::Environment,Δt::Real,N::Int;
                                    relax_comp=false)
    vs = VariableSelector()

    for n = 1:N
        add_var!(vs, Symbol("q", n), num_positions(mechanism))
        add_var!(vs, Symbol("v", n), num_velocities(mechanism))
        if n < N
            add_var!(vs, Symbol("u", n), num_velocities(mechanism))
            add_var!(vs, Symbol("h", n), 1)
            for i = 1:length(env.contacts)
                if relax_comp
                    add_var!(vs, Symbol("slack", i, "_", n), 1)
                end
                add_var!(vs, Symbol("c_n", i, "_", n), 1)
            end
        end
    end

    cs = ConstraintSelector()

    for n = 1:N-1
        add_eq!(cs, Symbol("kin", n), num_positions(mechanism))
        add_eq!(cs, Symbol("dyn", n), num_velocities(mechanism))
        add_ineq!(cs, Symbol("h_pos", n), 1)
        for i = 1:length(env.contacts)
            if relax_comp
                add_ineq!(cs, Symbol("slack_pos", i, "_", n), 1)
                add_ineq!(cs, Symbol("ϕ_c_n_comp", i, "_", n), 1)
            else
                add_eq!(cs, Symbol("ϕ_c_n_comp", i, "_", n), 1)
            end
            add_ineq!(cs, Symbol("c_n_pos", i, "_", n), 1)
            add_ineq!(cs, Symbol("ϕ_pos", i, "_", n), 1)
        end
    end

    state_cache = [StateCache(mechanism) for n = 1:2]
    envj_cache = [EnvironmentJacobianCache(env) for n = 1:2]

    generate_solver_fn = :generate_solver_fn_trajopt_semidirect
    extract_sol = :extract_sol_trajopt_semidirect

    fric_vs = []
    fric_cs = []
    fric_options = []

    f_options = Dict{String, Any}()
    f_options["num_fosteps"] = 1
    f_options["num_sosteps"] = 10
    f_options["c"] = 1.
    f_options["c_fos"] = 1.
    f_options["c_sos"] = 1.

    for n = 1:N-1
        f_vs = VariableSelector()
        for i = 1:length(env.contacts)
            β_dim = size(env.contacts[i].obstacle.basis,2)
            add_var!(f_vs, Symbol("β", i), β_dim)
        end

        f_cs = ConstraintSelector()
        for i = 1:length(env.contacts)
            β_dim = size(env.contacts[i].obstacle.basis,2)
            add_ineq!(f_cs, Symbol("β_pos", i), β_dim)
            add_ineq!(f_cs, Symbol("fric_cone", i), 1)
        end

        push!(fric_vs, f_vs)
        push!(fric_cs, f_cs)
        push!(fric_options, f_options)
    end

    sim_data = SimData(mechanism,env,
                       state_cache,envj_cache,
                       Δt,vs,cs,generate_solver_fn,extract_sol,
                       [],[],[],
                       [],[],[],
                       fric_vs,fric_cs,fric_options,
                       N,[],[])

    sim_data
end

function extract_sol_trajopt_semidirect(sim_data::SimData, xopt::AbstractArray{T}) where T
    N = sim_data.N
    vs = sim_data.vs
    env = sim_data.env
    relax_comp = haskey(vs.vars, :slack1)

    qtraj = Array{Array{Float64,1},1}(undef, 0)
    vtraj = Array{Array{Float64,1},1}(undef, 0)
    utraj = Array{Array{Float64,1},1}(undef, 0)
    htraj = Array{Float64,1}(undef, 0)
    contact_traj = Array{Array{Float64,1},1}(undef, 0)
    slack_traj = Array{Array{Float64,1},1}(undef, 0)
    for n = 1:N
        push!(qtraj, vs(xopt, Symbol("q", n)))
        push!(vtraj, vs(xopt, Symbol("v", n)))
        if n < N
            push!(utraj, vs(xopt, Symbol("u", n)))
            push!(htraj, vs(xopt, Symbol("h", n))[1])
            if relax_comp
                push!(slack_traj, vs(xopt, Symbol("slack", n)))
            end
            contact_sol = []
            for i = 1:length(env.contacts)
                contact_sol = vcat(contact_sol,
                                   vs(xopt, Symbol("c_n", i, "_", n)))
            end
            push!(contact_traj, contact_sol)
        end
    end

    # some other useful vectors
    ttraj = vcat(0., cumsum(htraj)...)
    qv_mat = vcat(hcat(qtraj...),hcat(vtraj...))

    qtraj, vtraj, utraj, htraj, contact_traj, slack_traj, ttraj, qv_mat
end

function generate_solver_fn_trajopt_semidirect(sim_data::SimData)
    N = sim_data.N
    vs = sim_data.vs
    cs = sim_data.cs

    num_contacts = length(sim_data.env.contacts)
    num_vel = num_velocities(sim_data.mechanism)
    world = root_body(sim_data.mechanism)
    world_frame = default_frame(world)
    relax_comp = haskey(vs.vars, :slack1)

    function eval_obj(x::AbstractArray{T}) where T
        f = 0.

        if relax_comp
            for n = 1:N-1
                for i = 1:num_contacts
                    slack = vs(x, Symbol("slack", i, "_", n))
                    f += sum(slack)
                end
            end
        end

        # extra user-defined objective
        for i = 1:length(sim_data.obj_fns)
            obj_name, obj_fn = sim_data.obj_fns[i]
            f += obj_fn(x)
        end

        f
    end

    function eval_cons(x::AbstractArray{T}) where T
        g = Vector{T}(undef, cs.num_eqs + cs.num_ineqs) # TODO preallocate

        # @threads for n = 1:(N-1)
        for n = 1:(N-1)
            q0 = vs(x, Symbol("q", n))
            v0 = vs(x, Symbol("v", n))
            u0 = vs(x, Symbol("u", n))
            h = vs(x, Symbol("h", n))
            qnext = vs(x, Symbol("q", n+1))
            vnext = vs(x, Symbol("v", n+1))
            if relax_comp
                slack = vs(x, Symbol("slack", n))
            end

            x0 = sim_data.state_cache[1][T]
            xn = sim_data.state_cache[2][T]
            envj = sim_data.envj_cache[2][T]

            set_configuration!(x0, q0)
            set_velocity!(x0, v0)
            setdirty!(x0)
            set_configuration!(xn, qnext)
            set_velocity!(xn, vnext)
            setdirty!(xn)

            H = mass_matrix(x0)
            Hi = inv(H)
            config_derivative = configuration_derivative(xn) # TODO preallocate
            dyn_bias = dynamics_bias(xn) # TODO preallocate

            contact_jacobian!(envj, xn)
            contact_bias = Vector{T}(undef, num_vel)
            if (num_contacts > 0)
                # compute friction forces
                contact_friction_τ_direct!(contact_bias, sim_data, Hi, envj, dyn_bias, u0, v0, x, n)
            end

            g[cs(Symbol("kin", n))] .= qnext .- q0 .- h .* config_derivative
            g[cs(Symbol("dyn", n))] .= H * (vnext - v0) .- h .* (u0 .- dyn_bias .- contact_bias)
            g[cs(Symbol("h_pos", n))] .= -h
            for i = 1:num_contacts
                c_n = vs(x, Symbol("c_n", i, "_", n))
                g[cs(Symbol("c_n_pos", i, "_", n))] .= -c_n
                g[cs(Symbol("ϕ_pos", i, "_", n))] .= -envj.contact_jacobians[i].ϕ
                if !relax_comp
                    g[cs(Symbol("ϕ_c_n_comp", i, "_", n))] .= envj.contact_jacobians[i].ϕ .* c_n
                else
                    slack = vs(x, Symbol("slack", i, "_", n))
                    g[cs(Symbol("slack_pos", i, "_", n))] .= -slack
                    g[cs(Symbol("ϕ_c_n_comp", i, "_", n))] .= envj.contact_jacobians[i].ϕ .* c_n .- slack
                end
            end
        end

        # extra user-defined constraints
        for i = 1:length(sim_data.con_fns)
            con_name, con_fn = sim_data.con_fns[i]
            g[cs(con_name)] .= con_fn(x)
        end

        g
    end

    generate_autodiff_solver_fn(eval_obj,eval_cons,cs.eqs,cs.ineqs,vs.num_vars)
end
