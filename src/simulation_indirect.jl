function get_sim_data_indirect(mechanism::Mechanism,env::Environment,Δt::Real;
                               relax_comp=false)
    num_q = num_positions(mechanism)
    num_v = num_velocities(mechanism)
    num_contacts = length(env.contacts)

    vs = VarSelector()
    add_var!(vs, :qnext, num_q)
    add_var!(vs, :vnext, num_v)
    if num_contacts > 0
        size_contacts = size(env.contacts[1].obstacle.basis,2) + 2
        add_var!(vs, :contact, num_contacts*size_contacts)
        if relax_comp
            add_var!(vs, :slack, 1)
        end
    end
            
    cs = ConSelector()
    add_eq!(cs, :kin, num_q)
    add_eq!(cs, :dyn, num_v)
    if num_contacts > 0
        add_ineq!(con_selector, :dist, num_contacts)
        add_ineq!(con_selector, :pos, )
        if relax_comp
            add_ineq!(con_selector, :comp, )
        else
            add_eq!(con_selector, :comp, )
        end
    end
    
    x0_cache = StateCache(sim_data.mechanism)
    xnext_cache = StateCache(sim_data.mechanism)
    env_cache = EnvironmentCache(sim_data.env, xnext_cache[ForwardDiff.Dual])

    generate_solver_fn = :generate_solver_fn_sim_indirect

    sim_data = SimData(mechanism,x0_cache,xnext_cache,env_cache,Δt,
                       relax_comp,vs,cs,generate_solver_fn)

    sim_data
end

function indirect_pos_contact_constraints!()
    # TODO
end

function indirect_comp_contact_constraints!()
    # TODO
end

function generate_solver_fn_sim_indirect(sim_data,q0,v0,u0)
    x0 = sim_data.x0_cache[Float64]
    xnext = sim_data.xnext_cache[ForwardDiff.Dual]
    env = sim_data.env_cache
    Δt = sim_data.Δt
    vs = sim_data.vs
    cs = sim_data.cs    
    
    has_slack = haskey(vs.vars, :slack)
    num_contacts = length(env.contact_jacobians)

    config_derivative = configuration(xnext[ForwardDiff.Dual])
    dyn_bias = velocity(xnext[ForwardDiff.Dual])
    contact_bias = zeros(ForwardDiff.Dual, num_v)
    g = Vector{ForwardDiff.Dual}(undef, cs.num_eqs + cs.num_ineqs)
    
    set_configuration!(x0[Float64], q0)
    set_velocity!(x0[Float64], v0)
    H = mass_matrix(x0)
    function eval_dyn(x::AbstractArray{T}) where T
        qnext = vs(x, :qnext)
        vnext = vs(x, :vnext)
        if has_slack
            slack = vs(x, :slack)
        end
        
        set_configuration!(xnext[T], qnext)
        set_velocity!(xnext[T], vnext)
        configuration_derivative!(config_derivative, xnext_cache[T])
        dynamics_bias!(dyn_bias, xnext_cache[T])
        if (num_contacts > 0)
            contact_jacobians!(env, xnext[T])
            contact_τ!(contact_bias, vs(x, :contact), env)
        end

        g[cs(:kin)] .= qnext .- q0 .- Δt .* config_derivative
        g[cs(:dyn)] .= H * (vnext - v0) .- Δt .* (u0 .- dyn_bias .- contact_bias)
        if (num_contacts > 0)
            g[cs(:dist)] .= -contact_distance(env)
            indirect_pos_contact_constraints!(g[cs(:pos)], vs(x, :contact))
            indirect_comp_contact_constraints!(g[cs(:comp)], vs(x, :contact))
        end

        g
    end

    function eval_f(x::AbstractArray{T}) where T
        f = 0.

        if has_slack
            slack = vs(x, :slack)
            f += .5 * slack' * slack
        end

        f
    end

    fres = DiffResults.GradientResult(zeros(vs.num_vars))
    fcfg = ForwardDiff.GradientConfig(eval_f, zeros(vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(cs.num_eqs + cs.num_ineqs), zeros(vs.num_vars))
    gcfg = ForwardDiff.JacobianConfig(eval_dyn, zeros(vs.num_vars))
    function solver_fn(x)
        ForwardDiff.gradient!(fres, eval_f, x, fcfg)
        
        J = DiffResults.value(fres)
        gJ = DiffResults.gradient(fres)

        ForwardDiff.jacobian!(gres, eval_con, x, gcfg)

        g = DiffResults.value(gres)
        ceq = g[eq(cs)]
        c = g[ineq(cs)]

        dgdx = DiffResults.jacobian(gres)
        gceq = dgdx[eq(cs),:]
        gc = dgdx[ineq(cs),:]

        J, c, ceq, gJ, gc, gceq, false
    end

    solver_fn
end