function get_trajopt_data_indirect(mechanism::Mechanism,env::Environment,Δt::Real,N::Int;
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
                β_dim = size(env.contacts[i].obstacle.basis,2)
                add_var!(vs, Symbol("β", i, "_", n), β_dim)
                add_var!(vs, Symbol("λ", i, "_", n), 1)
                add_var!(vs, Symbol("c_n", i, "_", n), 1)
            end
        end
    end

    cs = ConstraintSelector()

    for n = 1:N-1
        add_eq!(cs, Symbol("kin", n), num_positions(mechanism))
        add_eq!(cs, Symbol("dyn", n), num_velocities(mechanism))
        for i = 1:length(env.contacts)
            β_dim = size(env.contacts[i].obstacle.basis,2)
            add_ineq!(cs, Symbol("β_pos", i, "_", n), β_dim)
            add_ineq!(cs, Symbol("λ_pos", i, "_", n), 1)
            add_ineq!(cs, Symbol("c_n_pos", i, "_", n), 1)
            add_ineq!(cs, Symbol("ϕ_pos", i, "_", n), 1)
            add_ineq!(cs, Symbol("fric_pos", i, "_", n), β_dim)
            add_ineq!(cs, Symbol("cone_pos", i, "_", n), 1)
            if relax_comp
                add_ineq!(cs, Symbol("ϕ_c_n_comp", i, "_", n), 1)
                add_ineq!(cs, Symbol("fric_β_comp", i, "_", n), β_dim)
                add_ineq!(cs, Symbol("cone_λ_comp", i, "_", n), 1)
            else
                add_eq!(cs, Symbol("ϕ_c_n_comp", i, "_", n), 1)
                add_eq!(cs, Symbol("fric_β_comp", i, "_", n), β_dim)
                add_eq!(cs, Symbol("cone_λ_comp", i, "_", n), 1)
            end
        end
    end
    
    x0_cache = StateCache(mechanism)
    xn_cache = StateCache(mechanism)
    envj_cache = EnvironmentJacobianCache(env)

    generate_solver_fn = :generate_solver_fn_trajopt_indirect
    extract_sol = :extract_sol_trajopt_indirect

    sim_data = SimData(mechanism,env,
                       x0_cache,xn_cache,envj_cache,
                       Δt,vs,cs,generate_solver_fn,extract_sol,
                       nothing,nothing,nothing,N,[],[])

    sim_data
end

function extract_sol_trajopt_indirect(sim_data::SimData, xopt::AbstractArray{T}) where T    
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
            for i = 1:length(env.contacts)
                contact_sol = vcat(vs(xopt, Symbol("c_n", i, "_", n)),
                                   vs(xopt, Symbol("β", i, "_", n)),                                 vs(xopt, Symbol("λ", i, "_", n)))
                push!(contact_traj, contact_sol)
            end
        end
    end
    
    # some other usefull vectors
    ttraj = [(i-1)*sim_data.Δt for i = 1:N]
    qv_mat = vcat(hcat(qtraj...),hcat(vtraj...))
    
    qtraj, vtraj, utraj, contact_traj, slack_traj, ttraj, qv_mat
end

function contact_τ_indirect!(τ::AbstractArray{T},sim_data::SimData,envj::EnvironmentJacobian{T},x::AbstractArray{T},n::Int) where T
    # TODO: parallel
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, sim_data.vs(x, Symbol("c_n", i, "_", n)), sim_data.vs(x, Symbol("β", i, "_", n)))
    end
end

function generate_solver_fn_trajopt_indirect(sim_data::SimData)    
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
                contact_τ_indirect!(contact_bias, sim_data, envj, x, n)
            end

            g[cs(Symbol("kin", n))] .= qnext .- q0 .- Δt .* config_derivative
            g[cs(Symbol("dyn", n))] .= H * (vnext - v0) .- Δt .* (u0 .- dyn_bias .- contact_bias)
        
            for i = 1:num_contacts
                cj = envj.contact_jacobians[i]
                c = cj.contact
                
                # TODO preallocate
                contact_v = cj.contact_rot' * point_jacobian(xn, path(sim_data.mechanism, world, c.body), transform(xn, c.point, world_frame)).J * vnext
                
                β = vs(x, Symbol("β", i, "_", n))
                λ = vs(x, Symbol("λ", i, "_", n))
                c_n = vs(x, Symbol("c_n", i, "_", n))
                
                Dtv = c.obstacle.basis' * contact_v
                
                g[cs(Symbol("β_pos", i, "_", n))] .= -β
                g[cs(Symbol("λ_pos", i, "_", n))] .= -λ
                g[cs(Symbol("c_n_pos", i, "_", n))] .= -c_n
                g[cs(Symbol("ϕ_pos", i, "_", n))] .= -envj.contact_jacobians[i].ϕ
                # λe + D'*v >= 0
                g[cs(Symbol("fric_pos", i, "_", n))] .= -(λ .+ Dtv)
                # μ*c_n - sum(β) >= 0
                g[cs(Symbol("cone_pos", i, "_", n))] .= -(c.obstacle.μ .* c_n .- sum(β))
                if !relax_comp
                    # dist * c_n = 0
                    g[cs(Symbol("ϕ_c_n_comp", i, "_", n))] .= envj.contact_jacobians[i].ϕ .* c_n
                    # (λe + Dtv)' * β = 0
                    g[cs(Symbol("fric_β_comp", i, "_", n))] .= (λ .+ Dtv) .* β
                    # (μ * c_n - sum(β)) * λ = 0
                    g[cs(Symbol("cone_λ_comp", i, "_", n))] .= (c.obstacle.μ .* c_n .- sum(β)) .* λ
                else
                    g[cs(Symbol("ϕ_c_n_comp", i, "_", n))] .= envj.contact_jacobians[i].ϕ .* c_n .- slack' * slack
                    g[cs(Symbol("fric_β_comp", i, "_", n))] .= (λ .+ Dtv) .* β .- slack' * slack
                    g[cs(Symbol("cone_λ_comp", i, "_", n))] .= (c.obstacle.μ .* c_n - sum(β)) .* λ .- slack' * slack
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

