function get_trajopt_data_direct(mechanism::Mechanism,env::Environment,Δt::Real,N::Int;
                                 relax_comp=false)
    vs = VariableSelector()
        
    for n = 1:N
        add_var!(vs, Symbol("q", n), num_positions(mechanism))
        add_var!(vs, Symbol("v", n), num_velocities(mechanism))
        if n < N
            add_var!(vs, Symbol("u", n), num_velocities(mechanism))
            if relax_comp
                add_var!(vs, Symbol("slack", n), 1)
            end
            for i = 1:length(env.contacts)
                add_var!(vs, Symbol("c_n", i, "_", n), 1)
            end
        end
    end

    cs = ConstraintSelector()

    for n = 1:N-1
        add_eq!(cs, Symbol("kin", n), num_positions(mechanism))
        add_eq!(cs, Symbol("dyn", n), num_velocities(mechanism))
        for i = 1:length(env.contacts)
            add_ineq!(cs, Symbol("c_n_pos", i, "_", n), 1)
            add_ineq!(cs, Symbol("ϕ_pos", i, "_", n), 1)
            if relax_comp
                add_ineq!(cs, Symbol("ϕ_c_n_comp", i, "_", n), 1)
            else
                add_eq!(cs, Symbol("ϕ_c_n_comp", i, "_", n), 1)
            end
        end
    end
    
    x0_cache = StateCache(mechanism)
    xn_cache = StateCache(mechanism)
    envj_cache = EnvironmentJacobianCache(env)

    generate_solver_fn = :generate_solver_fn_trajopt_direct
    extract_sol = :extract_sol_trajopt_direct

    lower_vs = []
    lower_cs = []
    lower_options = []
    for n = 1:N-1
        l_vs = VariableSelector()
        for i = 1:length(env.contacts)
            β_dim = size(env.contacts[i].obstacle.basis,2)
            add_var!(l_vs, Symbol("β", i), β_dim)
        end

        l_cs = ConstraintSelector()
        for i = 1:length(env.contacts)
            β_dim = size(env.contacts[i].obstacle.basis,2)
            add_ineq!(l_cs, Symbol("β_pos", i), β_dim)
            add_ineq!(l_cs, Symbol("fric_cone", i), 1)
        end
       
        l_options = Dict{String, Any}()
        l_options["num_fosteps"] = 3
        l_options["num_sosteps"] = 2
        l_options["c"] = 1.
        l_options["c_fos"] = 10.
        l_options["c_sos"] = 10.
        
        # l_options = Dict{String, Any}()
        # l_options["num_fosteps"] = 1
        # l_options["num_sosteps"] = 1
        # l_options["c"] = 1.
        # l_options["c_fos"] = 1.
        # l_options["c_sos"] = 1.
        
        push!(lower_vs, l_vs)
        push!(lower_cs, l_cs)
        push!(lower_options, l_options) 
    end

    sim_data = SimData(mechanism,env,
                       x0_cache,xn_cache,envj_cache,
                       Δt,vs,cs,generate_solver_fn,extract_sol,
                       lower_vs,lower_cs,lower_options,N,[],[])

    sim_data
end

function extract_sol_trajopt_direct(sim_data::SimData, xopt::AbstractArray{T}) where T    
    N = sim_data.N
    vs = sim_data.vs
    relax_comp = haskey(vs.vars, :slack1)
    env = sim_data.env

    qtraj = []
    vtraj = []
    utraj = []
    contact_traj = []
    slack_traj = []
    for n = 1:N
        push!(qtraj, vs(xopt, Symbol("q", n)))
        push!(vtraj, vs(xopt, Symbol("v", n)))
        if n < N
            push!(utraj, vs(xopt, Symbol("u", n)))
            if relax_comp
                push!(slack_traj, vs(xopt, Symbol("slack", n)))
            end
            contact_sol = []
            for i = 1:length(env.contacts)
                # TODO also recover the lower problem solutions
                contact_sol = vcat(contact_sol,
                                   vs(xopt, Symbol("c_n", i, "_", n)))
            end
            push!(contact_traj, contact_sol)
        end
    end
    
    # some other usefull vectors
    ttraj = [(i-1)*sim_data.Δt for i = 1:N]
    qv_mat = vcat(hcat(qtraj...),hcat(vtraj...))
    
    qtraj, vtraj, utraj, contact_traj, slack_traj, ttraj, qv_mat
end

function contact_τ_direct!(τ,sim_data::SimData,H,envj::EnvironmentJacobian,
                           dyn_bias,u0,v0,upper_x::AbstractArray{U},n::Int) where U  
                           
    num_contacts = length(sim_data.env.contacts)
    Hi = inv(H)
    env = sim_data.env
    upper_vs = sim_data.vs
    lower_vs = sim_data.lower_vs[n]
    lower_cs = sim_data.lower_cs[n]
    lower_options = sim_data.lower_options[n]

    Qds = []
    rds = []
    for i = 1:num_contacts
        Qd = sim_data.Δt*envj.contact_jacobians[i].J'*Hi*envj.contact_jacobians[i].J
        rd = envj.contact_jacobians[i].J'*(sim_data.Δt*Hi*(dyn_bias - u0) - v0)

        push!(Qds,Qd)
        push!(rds,rd)
    end

    function eval_obj_(x::AbstractArray{L}) where L
        obj = 0.

        for i = 1:num_contacts            
            z = vcat(upper_vs(upper_x, Symbol("c_n", i, "_", n)),lower_vs(x, Symbol("β", i)))
            obj += .5 * z' * Qds[i] * z + rds[i]' * z
        end

        obj
    end

    function eval_cons_(x::AbstractArray{L}) where L
        # TODO in-place, need to accomodate x and upper_x types
        # g = Vector{L}(undef, lower_cs.num_eqs + lower_cs.num_ineqs)
        g = []

        for i = 1:num_contacts
            β = lower_vs(x, Symbol("β", i))
            # g[lower_cs(Symbol("β_pos", i))] .= -β
            # g[lower_cs(Symbol("fric_cone", i))] .= sum(β) - env.contacts[i].obstacle.μ * upper_vs(upper_x, Symbol("c_n", i, "_", n))
            # TODO lucky this is all inequalities or indexing could break
            g = vcat(g, -β)
            g = vcat(g, sum(β) .- env.contacts[i].obstacle.μ * upper_vs(upper_x, Symbol("c_n", i, "_", n)))
        end
        
        g
    end

    fres = DiffResults.HessianResult(zeros(U, lower_vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(U, lower_cs.num_cons), zeros(U, lower_vs.num_vars))
    solver_fn_ = generate_autodiff_solver_fn(eval_obj_,fres,eval_cons_,gres,lower_cs.eqs,lower_cs.ineqs)

    x0 = zeros(lower_vs.num_vars)

    xopt, info = auglag(solver_fn_, lower_cs.num_eqs, lower_cs.num_ineqs, x0, lower_options)
    
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, upper_vs(upper_x, Symbol("c_n", i, "_", n)), lower_vs(xopt, Symbol("β", i)))
    end
end

function generate_solver_fn_trajopt_direct(sim_data::SimData)    
    N = sim_data.N
    Δt = sim_data.Δt
    vs = sim_data.vs
    cs = sim_data.cs
    
    relax_comp = haskey(vs.vars, :slack1)
    num_contacts = length(sim_data.env.contacts)
    num_vel = num_velocities(sim_data.mechanism)
    world = root_body(sim_data.mechanism)
    world_frame = default_frame(world)
    
    function eval_obj(x::AbstractArray{T}) where T
        f = 0.
    
        if relax_comp
            for n = 1:N-1
                slack = vs(x, Symbol("slack", n))
                f += .5 * slack' * slack
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

        for n = 1:N-1
            q0 = vs(x, Symbol("q", n))
            v0 = vs(x, Symbol("v", n))
            u0 = vs(x, Symbol("u", n))
            if relax_comp
                slack = vs(x, Symbol("slack", n))
            end
            qnext = vs(x, Symbol("q", n+1))
            vnext = vs(x, Symbol("v", n+1))
        
            x0 = sim_data.x0_cache[T]
            xn = sim_data.xn_cache[T]
            envj = sim_data.envj_cache[T]
        
            contact_bias = Vector{T}(undef, num_vel)
        
            set_configuration!(x0, q0)
            set_velocity!(x0, v0)
            set_configuration!(xn, qnext)
            set_velocity!(xn, vnext)
        
            H = mass_matrix(x0)
        
            config_derivative = configuration_derivative(xn) # TODO preallocate
            dyn_bias = dynamics_bias(xn) # TODO preallocate
            if (num_contacts > 0)
                contact_jacobian!(envj, xn)
                contact_τ_direct!(contact_bias, sim_data, H, envj, dyn_bias, u0, v0, x, n)
            end

            g[cs(Symbol("kin", n))] .= qnext .- q0 .- Δt .* config_derivative
            g[cs(Symbol("dyn", n))] .= H * (vnext - v0) .- Δt .* (u0 .- dyn_bias .- contact_bias)
        
            for i = 1:num_contacts
                cj = envj.contact_jacobians[i]
                c = cj.contact
                
                # TODO preallocate
                contact_v = cj.contact_rot' * point_jacobian(xn, path(sim_data.mechanism, world, c.body), transform(xn, c.point, world_frame)).J * vnext
                
                c_n = vs(x, Symbol("c_n", i, "_", n))
                                
                g[cs(Symbol("c_n_pos", i, "_", n))] .= -c_n
                g[cs(Symbol("ϕ_pos", i, "_", n))] .= -envj.contact_jacobians[i].ϕ
                if !relax_comp
                    # dist * c_n = 0
                    g[cs(Symbol("ϕ_c_n_comp", i, "_", n))] .= envj.contact_jacobians[i].ϕ .* c_n
                else
                    g[cs(Symbol("ϕ_c_n_comp", i, "_", n))] .= envj.contact_jacobians[i].ϕ .* c_n .- slack' * slack
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
