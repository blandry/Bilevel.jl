function sim_fn_snopt(sim_data,q0,v0,u0)
    qnext_selector = 1:sim_data.num_q
    vnext_selector = qnext_selector[end] .+ (1:sim_data.num_v)
    last_selector = vnext_selector[end]
    if (sim_data.num_slack > 0)
        slack_selector = last_selector .+ (1:sim_data.num_slack)
        last_selector = slack_selector[end]
    end
    if (sim_data.num_contacts > 0 && !sim_data.implicit_contact)
        contact_selector = last_selector .+ (1:sim_data.num_contacts*(2+sim_data.β_dim))
        last_selector = contact_selector[end]
    end

    g_kin_selector = 1:sim_data.num_kin
    g_dyn_selector = g_kin_selector[end] .+ (1:sim_data.num_dyn)
    if (sim_data.num_contacts > 0)
        g_dist_selector = g_dyn_selector[end] .+ (1:sim_data.num_dist)
        if sim_data.implicit_contact
            num_dyn_contact = sim_data.num_v
            num_comp_contact = sim_data.num_contacts*(2+sim_data.β_dim)
            num_pos_contact = sim_data.num_contacts*(1+sim_data.β_dim) + 2*sim_data.num_contacts*(2+sim_data.β_dim)
            contact_x0 = zeros(sim_data.num_contacts*(2+sim_data.β_dim))
            contact_λ0 = zeros(num_dyn_contact+num_comp_contact)
            contact_μ0 = zeros(num_pos_contact)
        else
            g_comp_selector = g_dist_selector[end] .+ (1:sim_data.num_comp)
            g_pos_selector = g_comp_selector[end] .+ (1:sim_data.num_pos)
        end
    end

    x0 = MechanismState(sim_data.mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)

    function eval_con(x::AbstractArray{T}) where T
        qnext = x[qnext_selector]
        vnext = x[vnext_selector]

        if (sim_data.num_slack > 0)
            slack = x[slack_selector]
        end

        xnext = MechanismState{T}(sim_data.mechanism)
        set_configuration!(xnext, qnext)
        set_velocity!(xnext, vnext)

        if (sim_data.num_contacts > 0)
            Dtv = Matrix{T}(undef, sim_data.β_dim,sim_data.num_contacts)
            rel_transforms = Vector{Tuple{Transform3D{T}, Transform3D{T}}}(undef, sim_data.num_contacts) # force transform, point transform
            geo_jacobians = Vector{GeometricJacobian{Matrix{T}}}(undef, sim_data.num_contacts)
            ϕs = Vector{T}(undef, sim_data.num_contacts)
            for i = 1:sim_data.num_contacts
                v = point_velocity(twist_wrt_world(xnext,sim_data.bodies[i]), transform_to_root(xnext, sim_data.contact_points[i].frame) * sim_data.contact_points[i])
                Dtv[:,i] = map(sim_data.Ds[i]) do d
                    dot(transform_to_root(xnext, d.frame) * d, v)
                end
                rel_transforms[i] = (relative_transform(xnext, sim_data.obstacles[i].contact_face.outward_normal.frame, sim_data.world_frame),
                                              relative_transform(xnext, sim_data.contact_points[i].frame, sim_data.world_frame))
                geo_jacobians[i] = geometric_jacobian(xnext, sim_data.paths[i])
                ϕs[i] = separation(sim_data.obstacles[i], transform(xnext, sim_data.contact_points[i], sim_data.obstacles[i].contact_face.outward_normal.frame))
            end
        end

        config_derivative = configuration_derivative(xnext)
        HΔv = H * (vnext - v0)
        bias = u0 - dynamics_bias(xnext)

        if (sim_data.num_contacts > 0)
            if sim_data.implicit_contact
                contact_bias, contact_x0_sol, contact_λ0_sol, contact_μ0_sol, L_sol = solve_implicit_contact_τ(sim_data,ϕs,Dtv,rel_transforms,geo_jacobians,HΔv,bias,contact_x0,contact_λ0,contact_μ0)
            else
                contact_bias = τ_total(x[contact_selector],rel_transforms,geo_jacobians,sim_data)
            end
        else
            contact_bias = zeros(sim_data.num_v)
        end

        g = zeros(T,sim_data.num_dyn_eq+sim_data.num_dyn_ineq)

        # == 0
        g[g_kin_selector] = qnext .- q0 .- sim_data.Δt .* config_derivative # == 0
        g[g_dyn_selector] = HΔv .- sim_data.Δt .* (bias .- contact_bias)

        # <= 0
        if (sim_data.num_contacts > 0)
            g[g_dist_selector] = -ϕs
            if !sim_data.implicit_contact
                g[g_comp_selector] = complementarity_contact_constraints_relaxed(x[contact_selector],slack,ϕs,Dtv,sim_data)
                g[g_pos_selector] = pos_contact_constraints(x[contact_selector],Dtv,sim_data)
            end
        end

        g
    end

    kreg = 0.
    function eval_f(x::AbstractArray{T}) where T
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        f = .5*slack'*slack + kreg*.5*x'*x

        f
    end

    function eval_dfdx(x::AbstractArray{T}) where T
        dfdx = kreg*ones(length(x))
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        dfdx[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack] += slack

        dfdx
    end

    gres = DiffResults.JacobianResult(zeros(sim_data.num_dyn_eq+sim_data.num_dyn_ineq), zeros(sim_data.num_xn))
    gcfg = ForwardDiff.JacobianConfig(eval_con, zeros(sim_data.num_xn))
    function update_fn(x)
        J = eval_f(x)
        gJ = eval_dfdx(x)

        @time ForwardDiff.jacobian!(gres, eval_con, x, gcfg)

        g = DiffResults.value(gres)
        ceq = g[1:sim_data.num_dyn_eq]
        c = g[sim_data.num_dyn_eq+1:sim_data.num_dyn_eq+sim_data.num_dyn_ineq]

        dgdx = DiffResults.jacobian(gres)
        gceq = dgdx[1:sim_data.num_dyn_eq,:]
        gc = dgdx[sim_data.num_dyn_eq+1:sim_data.num_dyn_eq+sim_data.num_dyn_ineq,:]

        fail = false

        J, c, ceq, gJ, gc, gceq, fail
    end

    update_fn
end

function simulate_snopt(sim_data,control!,state0::MechanismState,N)
    # optimization bounds
    x_L = -1e19 * ones(sim_data.num_xn)
    x_U = 1e19 * ones(sim_data.num_xn)
    results = zeros(sim_data.num_xn)
    results[1:sim_data.num_q] = configuration(state0)
    results[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = velocity(state0)

    x_ctrl = MechanismState(sim_data.mechanism)
    u0 = zeros(sim_data.num_v)

    for i in 1:N
        x = results[:,end]
        q0 = x[1:sim_data.num_q]
        v0 = x[sim_data.num_q+1:sim_data.num_q+sim_data.num_v]

        set_configuration!(x_ctrl,q0)
        set_velocity!(x_ctrl,v0)
        setdirty!(x_ctrl)
        control!(u0, (i-1)*sim_data.Δt, x_ctrl)

        update_fn = sim_fn_snopt(sim_data,q0,v0,u0)

        options = Dict{String, Any}()
        options["Derivative option"] = 1
        options["Verify level"] = -1 # -1 = 0ff, 0 = cheap
        options["Major optimality tolerance"] = 1e-3
        if sim_data.implicit_contact
            options["Feasible point"] = true
            options["Major feasibility tolerance"] = 1e-6
        end

        xopt, fopt, info = snopt(update_fn, x, x_L, x_U, options)
        println(info)

        results = hcat(results,xopt)
    end

    results
end
