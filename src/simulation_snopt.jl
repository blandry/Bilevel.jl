function update_constraints_snopt(sim_data,q0,v0,u0)
    x0 = MechanismState(sim_data.mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)

    function evalf(x::AbstractArray{T}) where T
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        f = .5*slack'*slack
        
        f
    end
    
    function evaldfdx(x::AbstractArray{T}) where T
        dfdx = zeros(length(x))
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        dfdx[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack] = slack
        
        dfdx
    end

    function evalg(x::AbstractArray{T}) where T
        qnext = x[1:sim_data.num_q]
        vnext = x[sim_data.num_q+1:sim_data.num_q+sim_data.num_v]
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        xnext = MechanismState{T}(sim_data.mechanism)
        set_configuration!(xnext, qnext)
        set_velocity!(xnext, vnext)
        
        Dtv = Matrix{T}(sim_data.β_dim,sim_data.num_contacts)
        rel_transforms = Vector{Tuple{Transform3D{T}, Transform3D{T}}}(sim_data.num_contacts) # force transform, point transform
        geo_jacobians = Vector{GeometricJacobian{Matrix{T}}}(sim_data.num_contacts)
        ϕs = Vector{T}(sim_data.num_contacts)
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

        config_derivative = configuration_derivative(xnext)
        HΔv = H * (vnext - v0)
        bias = u0 .- dynamics_bias(xnext)
        
        contact_bias = τ_total(x[sim_data.num_q+sim_data.num_v+sim_data.num_slack+1:end],rel_transforms,geo_jacobians,sim_data)

        num_eq = sim_data.num_q + sim_data.num_v
        num_ineq = sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts+sim_data.num_contacts*(1+sim_data.β_dim)
        
        g = zeros(T,num_eq+num_ineq)
        
        g[1:sim_data.num_q] = qnext .- q0 .- sim_data.Δt .* config_derivative # == 0
        g[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = HΔv .- sim_data.Δt .* (bias .- contact_bias) # == 0

        g[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)] =
            complementarity_contact_constraints_relaxed(x[sim_data.num_q+sim_data.num_v+sim_data.num_slack+1:end],slack,ϕs,Dtv,sim_data) # <= 0
        g[sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts] = -ϕs # <= 0
        g[sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts+sim_data.num_contacts*(1+sim_data.β_dim)] =
            pos_contact_constraints(x[sim_data.num_q+sim_data.num_v+sim_data.num_slack+1:end],Dtv,sim_data) # <= 0
        
        g, num_eq, num_ineq
    end

    function evaldgdx(x::AbstractArray{T}) where T
        dgdx = ForwardDiff.jacobian(x̃ -> evalg(x̃)[1], x)
    
        dgdx
    end

    function update_fn(x::AbstractArray{T}) where T
        J = evalf(x)
        g, num_eq, num_ineq = evalg(x)
        ceq = g[1:num_eq]
        c = g[num_eq+1:num_eq+num_ineq]
        gJ = evaldfdx(x)
        dgdx = evaldgdx(x)
        gceq = dgdx[1:num_eq,:] 
        gc = dgdx[num_eq+1:num_eq+num_ineq,:]
        fail = false
        
        J, c, ceq, gJ, gc, gceq, fail
    end
    
    update_fn
end

function update_constraints_implicit_snopt(sim_data,q0,v0,u0,contact_x0,contact_λ0,contact_μ0)
    x0 = MechanismState(sim_data.mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)

    function evalf(x::AbstractArray{T}) where T
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        f = .5*slack'*slack
        
        f
    end
    
    function evaldfdx(x::AbstractArray{T}) where T
        dfdx = zeros(length(x))
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        dfdx[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack] = slack
        
        dfdx
    end

    function evalg(x::AbstractArray{T}) where T
        qnext = x[1:sim_data.num_q]
        vnext = x[sim_data.num_q+1:sim_data.num_q+sim_data.num_v]
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        xnext = MechanismState{T}(sim_data.mechanism)
        set_configuration!(xnext, qnext)
        set_velocity!(xnext, vnext)
        
        Dtv = Matrix{T}(sim_data.β_dim,sim_data.num_contacts)
        rel_transforms = Vector{Tuple{Transform3D{T}, Transform3D{T}}}(sim_data.num_contacts) # force transform, point transform
        geo_jacobians = Vector{GeometricJacobian{Matrix{T}}}(sim_data.num_contacts)
        ϕs = Vector{T}(sim_data.num_contacts)
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

        config_derivative = configuration_derivative(xnext)
        HΔv = H * (vnext - v0)
        bias = u0 .- dynamics_bias(xnext)
        
        contact_bias, contact_x0_sol, contact_λ0_sol, contact_μ0_sol, obj_sol = solve_implicit_contact_τ(sim_data,ϕs,Dtv,rel_transforms,geo_jacobians,HΔv,bias,contact_x0,contact_λ0,contact_μ0)

        if isa(contact_bias, Array{M} where M<:ForwardDiff.Dual)
            cval = map(x̃->x̃.value,contact_bias)
            display(cval)

            # contact_x0_cache .= map(x̃->x̃.value,contact_x0_sol)
            # contact_λ0_cache .= map(x̃->x̃.value,contact_λ0_sol)
            # contact_μ0_cache .= map(x̃->x̃.value,contact_μ0_sol)
        end

        num_eq = sim_data.num_q + sim_data.num_v
        num_ineq = sim_data.num_contacts
        
        g = zeros(T,num_eq+num_ineq)
        
        g[1:sim_data.num_q] = qnext .- q0 .- sim_data.Δt .* config_derivative # == 0
        # g[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = HΔv .- sim_data.Δt .* (bias .- (contact_bias .+ slack)) # == 0
        g[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = HΔv .- sim_data.Δt .* (bias .- contact_bias) # == 0

        g[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts] = -ϕs # <= 0
        
        g, num_eq, num_ineq
    end

    function evaldgdx(x::AbstractArray{T}) where T
        tic()
        dgdx = ForwardDiff.jacobian(x̃ -> evalg(x̃)[1], x)
        toc()
    
        dgdx
    end

    function update_fn(x::AbstractArray{T}) where T
        J = evalf(x)
        g, num_eq, num_ineq = evalg(x)
        ceq = g[1:num_eq]
        c = g[num_eq+1:num_eq+num_ineq]
        gJ = evaldfdx(x)
        dgdx = evaldgdx(x)
        gceq = dgdx[1:num_eq,:] 
        gc = dgdx[num_eq+1:num_eq+num_ineq,:]
        fail = false
        
        J, c, ceq, gJ, gc, gceq, fail
    end
    
    update_fn
end

function simulate_snopt(state0::MechanismState{T, M},
                        env::Environment,
                        Δt::Real,
                        N::Integer,
                        control!;
                        implicit_contact=true) where {T, M}

    sim_data = get_sim_data(state0,env,Δt,implicit_contact)

    # optimization bounds
    x_L = -1e19 * ones(sim_data.num_x)
    x_U = 1e19 * ones(sim_data.num_x)
    if !implicit_contact
        x_L[sim_data.num_q+sim_data.num_v+sim_data.num_slack+1:end] .= 0.
        x_U[sim_data.num_q+sim_data.num_v+sim_data.num_slack+1:end] .= 100.
    end

    if implicit_contact
        results = vcat(configuration(state0),velocity(state0),zeros(sim_data.num_slack))
    else
        results = vcat(configuration(state0),velocity(state0),zeros(sim_data.num_slack),zeros(sim_data.num_contacts*(2+sim_data.β_dim)))
    end

    x_ctrl = MechanismState(sim_data.mechanism)
    u0 = zeros(sim_data.num_v)

    num_dyn = sim_data.num_v
    num_comp = sim_data.num_contacts*(2+sim_data.β_dim)
    num_pos = sim_data.num_contacts*(1+sim_data.β_dim) + 2*sim_data.num_contacts*(2+sim_data.β_dim)
    contact_x0 = zeros(sim_data.num_contacts*(2+sim_data.β_dim))
    contact_λ0 = zeros(num_dyn+num_comp)
    contact_μ0 = zeros(num_pos)
    # global contact_x0_cache = zeros(sim_data.num_contacts*(2+sim_data.β_dim))
    # global contact_λ0_cache = zeros(num_dyn+num_comp)
    # global contact_μ0_cache = zeros(num_pos)

    for i in 1:N
        x = results[:,end]
        q0 = x[1:sim_data.num_q]
        v0 = x[sim_data.num_q+1:sim_data.num_q+sim_data.num_v]

        set_configuration!(x_ctrl,q0)
        set_velocity!(x_ctrl,v0)
        setdirty!(x_ctrl)
        control!(u0, (i-1)*sim_data.Δt, x_ctrl)

        if implicit_contact
            update_fn = update_constraints_implicit_snopt(sim_data,q0,v0,u0,contact_x0,contact_λ0,contact_μ0)
        else
            update_fn = update_constraints_snopt(sim_data,q0,v0,u0)
        end

        options = Dict{String, Any}()
        options["Derivative option"] = 1
        options["Verify level"] = -1 # -1 = 0ff, 0 = cheap

        xopt, fopt, info = snopt(update_fn, x, x_L, x_U, options)
        println(info)

        results = hcat(results,xopt)

        # recover the current implicit contact solution
        # contact_x0 .= contact_x0_cache
        # contact_λ0 .= contact_λ0_cache
        # contact_μ0 .= contact_μ0_cache
    end

    results
end
