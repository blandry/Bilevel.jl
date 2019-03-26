function get_sim_data_direct(mechanism::Mechanism,env::Environment,Δt::Real;
                             relax_comp=false)
	num_q = num_positions(mechanism)
    num_v = num_velocities(mechanism)
    num_contacts = length(env.contacts)
    if num_contacts > 0
        β_dim = length(contact_basis(env.contacts[1][3]))
    else
        β_dim = 0
    end
    if relax_comp
        num_slack = 1       
    else
        num_slack = 0
    end
    num_xn = num_q+num_v+num_contacts+num_slack

    generate_solver_fn = :generate_solver_fn_sim_direct

    sim_data = SimData(mechanism,env,Δt,num_q,num_v,num_contacts,β_dim,
                       num_slack,num_xn,relax_comp,generate_solver_fn)

    sim_data
end

function generate_solver_fn_sim_direct(sim_data,q0,v0,u0)
    qnext_selector = 1:sim_data.num_q
    vnext_selector = qnext_selector[end] .+ (1:sim_data.num_v)
    last_selector = vnext_selector[end]
    if (sim_data.num_contacts > 0)
        c_n_selector = last_selector .+ (1:sim_data.num_contacts)
        last_selector = c_n_selector[end]
    end
    if (sim_data.num_slack > 0)
        slack_selector = last_selector .+ (1:sim_data.num_slack)
        last_selector = slack_selector[end]
    end

    g_kin_selector = 1:sim_data.num_q
    g_dyn_selector = g_kin_selector[end] .+ (1:sim_data.num_v)
    g_ineq_selector = []
    g_eq_selector = vcat(g_kin_selector,g_dyn_selector)
    if (sim_data.num_contacts > 0)
        g_dist_selector = g_dyn_selector[end] .+ (1:sim_data.num_contacts)
        g_norm_selector = g_dist_selector[end] .+ (1:sim_data.num_contacts)
        g_comp_selector = g_norm_selector[end] .+ (1:sim_data.num_contacts)
        
        # g_ineq_selector = vcat(g_ineq_selector,g_dist_selector,g_norm_selector)
        # g_eq_selector = vcat(g_eq_selector,g_comp_selector)
        
        g_ineq_selector = vcat(g_ineq_selector,g_dist_selector,g_norm_selector,g_comp_selector)
    end

    x0 = MechanismState(sim_data.mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)

    function eval_con(x::AbstractArray{T}) where T
        qnext = x[qnext_selector]
        vnext = x[vnext_selector]
        if (sim_data.num_contacts > 0)
            c_ns = x[c_n_selector] # TODO get this from the dual not as a variable
        end
        if (sim_data.num_slack > 0)
            slack = x[slack_selector]
        end

        xnext = MechanismState{T}(sim_data.mechanism)
        set_configuration!(xnext, qnext)
        set_velocity!(xnext, vnext)

        if (sim_data.num_contacts > 0)
            rel_transforms = Vector{Tuple{Transform3D{T}, Transform3D{T}}}(undef, sim_data.num_contacts) # force transform, point transform
            geo_jacobians = Vector{GeometricJacobian{Matrix{T}}}(undef, sim_data.num_contacts)
            ϕs = Vector{T}(undef, sim_data.num_contacts)
            for i = 1:sim_data.num_contacts
                rel_transforms[i] = (relative_transform(xnext, sim_data.obstacles[i].contact_face.outward_normal.frame, sim_data.world_frame),relative_transform(xnext, sim_data.contact_points[i].frame, sim_data.world_frame))
                geo_jacobians[i] = geometric_jacobian(xnext, sim_data.paths[i])
                ϕs[i] = separation(sim_data.obstacles[i], transform(xnext, sim_data.contact_points[i], sim_data.obstacles[i].contact_face.outward_normal.frame))
            end
        end

        config_derivative = configuration_derivative(xnext)
        dyn_bias = dynamics_bias(xnext)

        if (sim_data.num_contacts > 0)
            contact_force, contact_x, contact_λ, contact_μ = solve_contact_τ(sim_data,H,dyn_bias,rel_transforms,geo_jacobians,v0,u0,c_ns,ip_method=false,in_place=false)
        else
            contact_force = zeros(sim_data.num_v)
        end

        g = zeros(T,sim_data.num_dyn_eq+sim_data.num_dyn_ineq)

        g[g_kin_selector] = qnext .- q0 .- sim_data.Δt .* config_derivative
        g[g_dyn_selector] = H * (vnext - v0) .+ sim_data.Δt*(dyn_bias .- u0 .+ contact_force)

        if (sim_data.num_contacts > 0)
            g[g_dist_selector] = -ϕs
            g[g_norm_selector] = -c_ns

            # g[g_comp_selector] = ϕs .* c_ns
            g[g_comp_selector] = ϕs .* c_ns - slack'*slack
        end

        g
    end

    function eval_f(x::AbstractArray{T}) where T
        f = 0.
        
        if (sim_data.num_slack > 0)
            slack = x[slack_selector]
            f += slack'*slack
        end

        f
    end
    
    fres = DiffResults.GradientResult(zeros(sim_data.num_xn))
    fcfg = ForwardDiff.GradientConfig(eval_f, zeros(sim_data.num_xn))
    gres = DiffResults.JacobianResult(zeros(sim_data.num_dyn_eq+sim_data.num_dyn_ineq), zeros(sim_data.num_xn))
    gcfg = ForwardDiff.JacobianConfig(eval_con, zeros(sim_data.num_xn))
    function update_fn(x)
        ForwardDiff.gradient!(fres, eval_f, x, fcfg)
        
        J = DiffResults.value(fres)
        gJ = DiffResults.gradient(fres)

        ForwardDiff.jacobian!(gres, eval_con, x, gcfg)

        g = DiffResults.value(gres)
        ceq = g[g_eq_selector]
        c = g[g_ineq_selector]

        dgdx = DiffResults.jacobian(gres)
        gceq = dgdx[g_eq_selector,:]
        gc = dgdx[g_ineq_selector,:]

        fail = false

        J, c, ceq, gJ, gc, gceq, fail
    end

    update_fn
end