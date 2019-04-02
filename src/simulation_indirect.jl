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

    sim_data = SimData(mechanism,env,
                       x0_cache,xn_cache,envj_cache,
                       Δt,vs,cs,generate_solver_fn)

    sim_data
end

function contact_τ!(τ::AbstractArray{T},sim_data::SimData,envj::EnvironmentJacobian{T},x::AbstractArray{T}) where T
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

    function eval_dyn(x::AbstractArray{T}) where T
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
            contact_τ!(contact_bias, sim_data, envj, x)
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
            g[cs(Symbol("cone_pos", i))] .= -(c.obstacle.μ .* c_n - sum(β))
            if !relax_comp
                # dist * c_n = 0
                g[cs(Symbol("ϕ_c_n_comp", i))] .= envj.contact_jacobians[i].ϕ .* c_n
                # (λe + Dtv)' * β = 0
                g[cs(Symbol("fric_β_comp", i))] .= (λ .+ Dtv) .* β
                # (μ * c_n - sum(β)) * λ = 0
                g[cs(Symbol("cone_λ_comp", i))] .= (c.obstacle.μ .* c_n - sum(β)) .* λ
            else
                g[cs(Symbol("ϕ_c_n_comp", i))] .= envj.contact_jacobians[i].ϕ .* c_n .- slack^2
                g[cs(Symbol("fric_β_comp", i))] .= (λ .+ Dtv) .* β .- slack^2
                g[cs(Symbol("cone_λ_comp", i))] .= (c.obstacle.μ .* c_n - sum(β)) .* λ .- slack^2
            end
        end
                
        g
    end

    function eval_f(x::AbstractArray{T}) where T
        f = 0.
    
        if relax_comp
            slack = vs(x, :slack)
            f += .5 * slack' * slack
        end
    
        f
    end
    
    fres = DiffResults.GradientResult(zeros(vs.num_vars))
    fcfg = ForwardDiff.GradientConfig(eval_f, zeros(vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(cs.num_cons), zeros(vs.num_vars))
    gcfg = ForwardDiff.JacobianConfig(eval_dyn, zeros(vs.num_vars))
    function solver_fn(x)        
        ForwardDiff.gradient!(fres, eval_f, x, fcfg)
    
        J = DiffResults.value(fres)
        gJ = DiffResults.gradient(fres)
    
        ForwardDiff.jacobian!(gres, eval_dyn, x, gcfg)
    
        g = DiffResults.value(gres)
        ceq = g[cs.eqs]
        c = g[cs.ineqs]
    
        dgdx = DiffResults.jacobian(gres)
        gceq = dgdx[cs.eqs,:]
        gc = dgdx[cs.ineqs,:]
    
        J, c, ceq, gJ, gc, gceq, false
    end
    
    solver_fn
end