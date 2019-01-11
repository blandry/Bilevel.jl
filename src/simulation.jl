struct SimData
    Δt
    mechanism
    num_q
    num_v
    num_contacts
    β_dim
    num_x
    num_slack
    num_h
    num_g
    world
    world_frame
    total_weight
    bodies
    contact_points
    obstacles
    μs
    paths
    Ds
    β_selector
    λ_selector
    c_n_selector
end

function update_constraints(sim_data,q0,v0,u0)
    x0 = MechanismState(sim_data.mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)

    function eval_g(x::AbstractArray{T}, g) where T
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

        g[1:sim_data.num_q] = qnext .- q0 .- sim_data.Δt .* config_derivative # == 0
        g[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = HΔv .- sim_data.Δt .* (bias .- contact_bias) # == 0

        g[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)] =
            complementarity_contact_constraints_relaxed(x[sim_data.num_q+sim_data.num_v+sim_data.num_slack+1:end],slack,ϕs,Dtv,sim_data) # <= 0
        g[sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts] = -ϕs # <= 0
        g[sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts+sim_data.num_contacts*(1+sim_data.β_dim)] =
            pos_contact_constraints(x[sim_data.num_q+sim_data.num_v+sim_data.num_slack+1:end],Dtv,sim_data) # <= 0
    end

    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            # TODO actually figure out the sparsity pattern
            for i = 1:(sim_data.num_h + sim_data.num_g)
                for j = 1:sim_data.num_x
                    rows[(i-1)*sim_data.num_x+j] = i
                    cols[(i-1)*sim_data.num_x+j] = j
                end
            end
        else
            g = zeros(sim_data.num_h + sim_data.num_g)
            tic()
            J = ForwardDiff.jacobian((g̃, x̃) -> eval_g(x̃, g̃), g, x)
            toc()
            values[:] = J'[:]
        end
    end

    eval_g, eval_jac_g
end

function update_constraints_implicit_contact(sim_data,q0,v0,u0,contact_x0,contact_λ0,contact_μ0)
    x0 = MechanismState(sim_data.mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)

    # num_dyn = sim_data.num_v
    # num_comp = sim_data.num_contacts*(2+sim_data.β_dim)
    # num_pos = sim_data.num_contacts*(1+sim_data.β_dim) + 2*sim_data.num_contacts*(2+sim_data.β_dim)

    # aug lag initial guesses
    # these should come from the previous time step
    # contact_x0 = repmat(vcat(zeros(sim_data.β_dim),0.,1.),sim_data.num_contacts)
    # contact_x0 = zeros(sim_data.num_contacts*(2+sim_data.β_dim))
    # contact_λ0 = zeros(num_dyn)
    # contact_λ0 = zeros(num_dyn+num_comp)
    # contact_μ0 = zeros(num_pos)

    function eval_g(x::AbstractArray{T}, g) where T
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

        if isa(contact_bias, Array{T} where T<:ForwardDiff.Dual)
            cval = map(x̃->x̃.value,contact_bias)
            display(cval)

            contact_x0_cache .= map(x̃->x̃.value,contact_x0_sol)
            contact_λ0_cache .= map(x̃->x̃.value,contact_λ0_sol)
            contact_μ0_cache .= map(x̃->x̃.value,contact_μ0_sol)
        end

        g[1:sim_data.num_q] = qnext .- q0 .- sim_data.Δt .* config_derivative # == 0
        # g[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = HΔv .- sim_data.Δt .* (bias .- (contact_bias .+ slack)) # == 0
        g[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = HΔv .- sim_data.Δt .* (bias .- contact_bias) # == 0

        g[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts] = -ϕs # <= 0
    end

    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            # TODO actually figure out the sparsity pattern
            for i = 1:(sim_data.num_h + sim_data.num_g)
                for j = 1:sim_data.num_x
                    rows[(i-1)*sim_data.num_x+j] = i
                    cols[(i-1)*sim_data.num_x+j] = j
                end
            end
        else
            g = zeros(sim_data.num_h + sim_data.num_g)
            tic()
            J = ForwardDiff.jacobian((g̃, x̃) -> eval_g(x̃, g̃), g, x)
            toc()
            values[:] = J'[:]
        end
    end

    eval_g, eval_jac_g
end

function get_sim_data(state0::MechanismState{T, M},
                      env::Environment,
                      Δt::Real,
                      implicit_contact::Bool) where {T, M}

    mechanism = state0.mechanism

    num_q = num_positions(state0)
    num_v = num_velocities(state0)
    num_contacts = length(env.contacts)
    β_dim = length(contact_basis(env.contacts[1][3]))
    # x = [q, v, slack, β1, λ1, c_n1, β2, λ2, c_n2...]
    if implicit_contact
      num_slack = num_v
      num_x = num_q + num_v + num_slack
      num_h = num_q + num_v
      num_g = num_contacts
    else
      num_slack = 1
      num_x = num_q + num_v + num_slack + num_contacts*(2+β_dim)
      num_h = num_q + num_v
      num_g = num_contacts + num_contacts*(2+β_dim) + num_contacts*(1+β_dim)
    end

    # some constants throughout the simulation
    world = root_body(mechanism)
    world_frame = default_frame(world)
    total_weight = mass(mechanism) * norm(mechanism.gravitational_acceleration)
    bodies = []
    contact_points = []
    obstacles = []
    μs = []
    paths = []
    Ds = []
    for (body, contact_point, obstacle) in env.contacts
      push!(bodies, body)
      push!(contact_points, contact_point)
      push!(obstacles, obstacle)
      push!(μs, obstacle.μ)
      push!(paths, path(mechanism, body, world))
      push!(Ds, contact_basis(obstacle))
    end
    β_selector = find(repmat(vcat(ones(β_dim),[0,0]),num_contacts))
    λ_selector = find(repmat(vcat(zeros(β_dim),[1,0]),num_contacts))
    c_n_selector = find(repmat(vcat(zeros(β_dim),[0,1]),num_contacts))

    sim_data = SimData(Δt,mechanism,num_q,num_v,num_contacts,β_dim,num_x,num_slack,num_h,num_g,
                       world,world_frame,total_weight,
                       bodies,contact_points,obstacles,μs,paths,Ds,
                       β_selector,λ_selector,c_n_selector)

    sim_data
end

function simulate(state0::MechanismState{T, M},
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

    g_L = vcat(0. * ones(sim_data.num_h), -1e19 * ones(sim_data.num_g))
    g_U = vcat(0. * ones(sim_data.num_h),    0. * ones(sim_data.num_g))

    if implicit_contact
        results = vcat(configuration(state0),velocity(state0),zeros(sim_data.num_slack))
    else
        results = vcat(configuration(state0),velocity(state0),zeros(sim_data.num_slack),zeros(sim_data.num_contacts*(2+sim_data.β_dim)))
    end

    eval_f = x -> begin
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        .5*slack'*slack
        # sum(abs.(slack))
    end
    eval_grad_f = (x, grad_f) -> begin
        grad_f[:] = 0
        slack = x[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack]
        grad_f[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack] = slack
        # grad_f[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_slack] = sign.(slack)
    end

    x_ctrl = MechanismState(sim_data.mechanism)
    u0 = zeros(sim_data.num_v)

    num_dyn = sim_data.num_v
    num_comp = sim_data.num_contacts*(2+sim_data.β_dim)
    num_pos = sim_data.num_contacts*(1+sim_data.β_dim) + 2*sim_data.num_contacts*(2+sim_data.β_dim)
    contact_x0 = zeros(sim_data.num_contacts*(2+sim_data.β_dim))
    contact_λ0 = zeros(num_dyn+num_comp)
    contact_μ0 = zeros(num_pos)
    global contact_x0_cache = zeros(sim_data.num_contacts*(2+sim_data.β_dim))
    global contact_λ0_cache = zeros(num_dyn+num_comp)
    global contact_μ0_cache = zeros(num_pos)

    for i in 1:N
        x = results[:,end]
        q0 = x[1:sim_data.num_q]
        v0 = x[sim_data.num_q+1:sim_data.num_q+sim_data.num_v]

        set_configuration!(x_ctrl,q0)
        set_velocity!(x_ctrl,v0)
        setdirty!(x_ctrl)
        control!(u0, (i-1)*sim_data.Δt, x_ctrl)

        if implicit_contact
            eval_g, eval_jac_g = update_constraints_implicit_contact(sim_data,q0,v0,u0,contact_x0,contact_λ0,contact_μ0)
        else
            eval_g, eval_jac_g = update_constraints(sim_data,q0,v0,u0)
        end

        prob = createProblem(sim_data.num_x,x_L,x_U,
                             sim_data.num_h+sim_data.num_g,g_L,g_U,
                             sim_data.num_x*(sim_data.num_h+sim_data.num_g),0,
                             eval_f,eval_g,
                             eval_grad_f,eval_jac_g)

        prob.x[:] = results[1:sim_data.num_x,end][:]

        addOption(prob, "hessian_approximation", "limited-memory")
        addOption(prob, "print_level", 1)
        addOption(prob, "tol", 1e-4) # convergence tol default 1e-8
        addOption(prob, "constr_viol_tol", 1e-2) # default 1e-4

        status = solveProblem(prob)
        println(Ipopt.ApplicationReturnStatus[status])

        results = hcat(results,prob.x)

        # recover the current implicit contact solution
        contact_x0 .= contact_x0_cache
        contact_λ0 .= contact_λ0_cache
        contact_μ0 .= contact_μ0_cache
    end

    results
end
