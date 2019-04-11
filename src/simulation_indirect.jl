function get_sim_data_indirect(mechanism::Mechanism,env::Environment,Δt::Real;
                               relax_comp=false)
    vs = VariableSelector()
    add_var!(vs, :qnext, num_positions(mechanism))
    add_var!(vs, :vnext, num_velocities(mechanism))
    if relax_comp
        add_var!(vs, :slack, 1)
    end
    for i = 1:length(env.contacts)
        β_dim = size(env.contacts[i].obstacle.basis,2)
        add_var!(vs, Symbol("β", i), β_dim)
        add_var!(vs, Symbol("λ", i), 1)
        add_var!(vs, Symbol("c_n", i), 1)
    end

    cs = ConstraintSelector()
    add_eq!(cs, :kin, num_positions(mechanism))
    add_eq!(cs, :dyn, num_velocities(mechanism))
    for i = 1:length(env.contacts)
        β_dim = size(env.contacts[i].obstacle.basis,2)
        add_ineq!(cs, Symbol("β_pos", i), β_dim)
        add_ineq!(cs, Symbol("λ_pos", i), 1)
        add_ineq!(cs, Symbol("c_n_pos", i), 1)
        add_ineq!(cs, Symbol("ϕ_pos", i), 1)
        add_ineq!(cs, Symbol("fric_pos", i), β_dim)
        add_ineq!(cs, Symbol("cone_pos", i), 1)
        if relax_comp
            add_ineq!(cs, Symbol("ϕ_c_n_comp", i), 1)
            add_ineq!(cs, Symbol("fric_β_comp", i), β_dim)
            add_ineq!(cs, Symbol("cone_λ_comp", i), 1)
        else
            add_eq!(cs, Symbol("ϕ_c_n_comp", i), 1)
            add_eq!(cs, Symbol("fric_β_comp", i), β_dim)
            add_eq!(cs, Symbol("cone_λ_comp", i), 1)
        end
    end
    
    x0_cache = StateCache(mechanism)
    xn_cache = StateCache(mechanism)
    envj_cache = EnvironmentJacobianCache(env)

    generate_solver_fn = :generate_solver_fn_sim_indirect
    extract_sol = :extract_sol_sim_indirect

    sim_data = SimData(mechanism,env,
                       x0_cache,xn_cache,envj_cache,
                       Δt,vs,cs,generate_solver_fn,extract_sol,
                       nothing,nothing,nothing,1,[],[])

    sim_data
end

function extract_sol_sim_indirect(sim_data::SimData, results::AbstractArray{T,2}) where T    
    vs = sim_data.vs
    relax_comp = haskey(vs.vars, :slack)
    env = sim_data.env
    N = size(results,2)

    qtraj = []
    vtraj = []
    utraj = []
    contact_traj = []
    slack_traj = []
    for n = 1:N
        push!(qtraj, vs(results[:,n], Symbol("qnext")))
        push!(vtraj, vs(results[:,n], Symbol("vnext")))
        if relax_comp
            push!(slack_traj, vs(results[:,n], Symbol("slack")))
        end
        for i = 1:length(env.contacts)
            contact_sol = vcat(vs(results[:,n], Symbol("c_n", i)),
                               vs(results[:,n], Symbol("β", i)),
                               vs(results[:,n], Symbol("λ", i)))
            push!(contact_traj, contact_sol)
        end
    end
    
    # some other usefull vectors
    ttraj = [(i-1)*sim_data.Δt for i = 1:N]
    qv_mat = vcat(hcat(qtraj...),hcat(vtraj...))
    
    qtraj, vtraj, utraj, contact_traj, slack_traj, ttraj, qv_mat
end

function contact_τ_indirect!(τ::AbstractArray{T},sim_data::SimData,envj::EnvironmentJacobian{T},x::AbstractArray{T}) where T
    # TODO: parallel
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, sim_data.vs(x, Symbol("c_n", i)), sim_data.vs(x, Symbol("β", i)))
    end
end

function generate_solver_fn_sim_indirect(sim_data,q0,v0,u0)
    x0 = sim_data.x0_cache[Float64]
    Δt = sim_data.Δt
    vs = sim_data.vs
    cs = sim_data.cs
    
    relax_comp = haskey(vs.vars, :slack)
    num_contacts = length(sim_data.env.contacts)
    num_vel = num_velocities(sim_data.mechanism)
    world = root_body(sim_data.mechanism)
    world_frame = default_frame(world)
    
    set_configuration!(x0, q0)
    set_velocity!(x0, v0)
    H = mass_matrix(x0)
    
    function eval_obj(x::AbstractArray{T}) where T
        f = 0.
    
        if relax_comp
            slack = vs(x, :slack)
            f += .5 * slack' * slack
        end
    
        f
    end

    function eval_cons(x::AbstractArray{T}) where T
        xn = sim_data.xn_cache[T]
        envj = sim_data.envj_cache[T]
        
        contact_bias = Vector{T}(undef, num_vel)
        g = Vector{T}(undef, cs.num_eqs + cs.num_ineqs) # TODO preallocate

        qnext = vs(x, :qnext)
        vnext = vs(x, :vnext)
        if relax_comp
            slack = vs(x, :slack)
        end
        
        set_configuration!(xn, qnext)
        set_velocity!(xn, vnext)
        
        config_derivative = configuration_derivative(xn) # TODO preallocate
        dyn_bias = dynamics_bias(xn) # TODO preallocate
        if (num_contacts > 0)
            contact_jacobian!(envj, xn)
            contact_τ_indirect!(contact_bias, sim_data, envj, x)
        end

        g[cs(:kin)] .= qnext .- q0 .- Δt .* config_derivative
        g[cs(:dyn)] .= H * (vnext - v0) .- Δt .* (u0 .- dyn_bias .- contact_bias)
        for i = 1:num_contacts
            cj = envj.contact_jacobians[i]
            c = cj.contact
            
            # TODO preallocate
            contact_v = cj.contact_rot' * point_jacobian(xn, path(sim_data.mechanism, world, c.body), transform(xn, c.point, world_frame)).J * vnext
            
            β = vs(x, Symbol("β", i))
            λ = vs(x, Symbol("λ", i))
            c_n = vs(x, Symbol("c_n", i))
            
            Dtv = c.obstacle.basis' * contact_v
            
            g[cs(Symbol("β_pos", i))] .= -β
            g[cs(Symbol("λ_pos", i))] .= -λ
            g[cs(Symbol("c_n_pos", i))] .= -c_n
            g[cs(Symbol("ϕ_pos", i))] .= -envj.contact_jacobians[i].ϕ
            # λe + D'*v >= 0
            g[cs(Symbol("fric_pos", i))] .= -(λ .+ Dtv)
            # μ*c_n - sum(β) >= 0
            g[cs(Symbol("cone_pos", i))] .= -(c.obstacle.μ .* c_n .- sum(β))
            if !relax_comp
                # dist * c_n = 0
                g[cs(Symbol("ϕ_c_n_comp", i))] .= envj.contact_jacobians[i].ϕ .* c_n
                # (λe + Dtv)' * β = 0
                g[cs(Symbol("fric_β_comp", i))] .= (λ .+ Dtv) .* β
                # (μ * c_n - sum(β)) * λ = 0
                g[cs(Symbol("cone_λ_comp", i))] .= (c.obstacle.μ .* c_n .- sum(β)) .* λ
            else
                g[cs(Symbol("ϕ_c_n_comp", i))] .= envj.contact_jacobians[i].ϕ .* c_n .- slack^2
                g[cs(Symbol("fric_β_comp", i))] .= (λ .+ Dtv) .* β .- slack^2
                g[cs(Symbol("cone_λ_comp", i))] .= (c.obstacle.μ .* c_n - sum(β)) .* λ .- slack^2
            end
        end
                
        g
    end
    
    generate_autodiff_solver_fn(eval_obj,eval_cons,cs.eqs,cs.ineqs,vs.num_vars)
end