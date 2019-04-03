function get_sim_data_direct(mechanism::Mechanism,env::Environment,Δt::Real)
    vs = VariableSelector()
    add_var!(vs, :qnext, num_positions(mechanism))
    add_var!(vs, :vnext, num_velocities(mechanism))

    cs = ConstraintSelector()
    add_eq!(cs, :kin, num_positions(mechanism))
    add_eq!(cs, :dyn, num_velocities(mechanism))
    add_ineq!(cs, :ϕ, length(sim_data.env.contacts))
    
    x0_cache = StateCache(mechanism)
    xn_cache = StateCache(mechanism)
    envj_cache = EnvironmentJacobianCache(env)

    generate_solver_fn = :generate_solver_fn_sim_direct

    sim_data = SimData(mechanism,env,
                       x0_cache,xn_cache,envj_cache,
                       Δt,vs,cs,generate_solver_fn)

    sim_data
end

function contact_τ!(τ,sim_data,H,envj,dyn_bias,u0,v0,c_ns)
    num_contacts = length(sim_data.env.contacts)
    Hi = inv(H)
    env = sim_data.env

    # TODO this should be done once at the beginning
    vs = VariableSelector()
    for i = 1:length(env.contacts)
        β_dim = size(env.contacts[i].obstacle.basis,2)
        add_var!(vs, Symbol("β", i), β_dim)
    end
    
    cs = ConstraintSelector()
    for i = 1:length(env.contacts)
        β_dim = size(env.contacts[i].obstacle.basis,2)
        add_ineq!(cs, Symbol("β_pos", i), β_dim)
        add_ineq!(cs, Symbol("fric_cone", i), 1)
    end

    Qds = []
    rds = []
    for j = 1:num_contacts
        Qd = sim_data.Δt*envj.contact_jacobians[i].J*Hi*envj.contact_jacobians[i].J'
        rd = envj.contact_jacobians[i].J*(sim_data.Δt*Hi*(dyn_bias - u0) - v0)

        push!(Qds,Qd)
        push!(rds,rd)
    end

    function eval_cons(x::AbstractArray{T}) where T
        g = Vector{T}(undef, cs.num_eqs + cs.num_ineqs)
        
        for i = 1:num_contacts
            β = vs(x, Symbol("β", i))
            g[cs(Symbol("β_pos", i))] .= -β
            g[cs(Symbol("fric_cone", i))] .= sum(β) - env.contacts[i].obstacle.μ * c_ns[i]
        end
    end

    function eval_obj(x::AbstractArray{T}) where T
        obj = 0.
        
        for i = 1:num_contacts
            z = vcat(c_ns[i],vs(x, Symbol("β", i)))
            obj += .5 * z' * Qds[i] * z + rds[i]' * z
        end
        
        obj
    end
    
    fres = DiffResults.GradientResult(zeros(vs.num_vars))
    fcfg = ForwardDiff.GradientConfig(eval_obj, zeros(vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(cs.num_cons), zeros(vs.num_vars))
    gcfg = ForwardDiff.JacobianConfig(eval_cons, zeros(vs.num_vars))
    function solver_fn(x)        
        ForwardDiff.gradient!(fres, eval_obj, x, fcfg)
    
        J = DiffResults.value(fres)
        gJ = DiffResults.gradient(fres)
    
        ForwardDiff.jacobian!(gres, eval_cons, x, gcfg)
    
        g = DiffResults.value(gres)
        ceq = g[cs.eqs]
        c = g[cs.ineqs]
    
        dgdx = DiffResults.jacobian(gres)
        gceq = dgdx[cs.eqs,:]
        gc = dgdx[cs.ineqs,:]
    
        J, c, ceq, gJ, gc, gceq, false
    end
    
    x = auglag(solver_fn)

    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, c_ns[i], vs(x, Symbol("β", i)))
    end
end

function generate_solver_fn_sim_direct(sim_data,q0,v0,u0)
    x0 = sim_data.x0_cache[Float64]
    Δt = sim_data.Δt
    vs = sim_data.vs
    cs = sim_data.cs
    
    num_contacts = length(sim_data.env.contacts)
    num_vel = num_velocities(sim_data.mechanism)

    set_configuration!(x0, q0)
    set_velocity!(x0, v0)
    H = mass_matrix(x0)

    function eval_cons(x::AbstractArray{T}) where T
        xn = sim_data.xn_cache[T]
        envj = sim_data.envj_cache[T]
        
        contact_bias = Vector{T}(undef, num_vel)
        g = Vector{T}(undef, cs.num_eqs + cs.num_ineqs) # TODO preallocate

        qnext = vs(x, :qnext)
        vnext = vs(x, :vnext)
        
        set_configuration!(xn, qnext)
        set_velocity!(xn, vnext)
        
        config_derivative = configuration_derivative(xn) # TODO preallocate
        dyn_bias = dynamics_bias(xn) # TODO preallocate
        if (num_contacts > 0)
            contact_jacobian!(envj, xn)
            contact_τ!(contact_bias, sim_data, envj, qnext, vnext)
        end

        g[cs(:kin)] .= qnext .- q0 .- Δt .* config_derivative
        g[cs(:dyn)] .= H * (vnext - v0) .- Δt .* (u0 .- dyn_bias .- contact_bias)

        g
    end

    function eval_obj(x::AbstractArray{T}) where T
        f = 0.
    
        f
    end
    
    fres = DiffResults.GradientResult(zeros(vs.num_vars))
    fcfg = ForwardDiff.GradientConfig(eval_obj, zeros(vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(cs.num_cons), zeros(vs.num_vars))
    gcfg = ForwardDiff.JacobianConfig(eval_cons, zeros(vs.num_vars))
    function solver_fn(x)        
        ForwardDiff.gradient!(fres, eval_obj, x, fcfg)
    
        J = DiffResults.value(fres)
        gJ = DiffResults.gradient(fres)
    
        ForwardDiff.jacobian!(gres, eval_cons, x, gcfg)
    
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