function update_constraints(env,mechanism,Δt,q0,v0,u0)
    0.
end

function update_constraints_implicit_contact(env,mechanism,Δt,q0,v0,u0)    
    x0 = MechanismState(mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)
    
    world = root_body(mechanism)
    world_frame = default_frame(world)
    total_weight = mass(mechanism) * norm(mechanism.gravitational_acceleration)
    
    num_q = length(q0)
    num_v = length(v0)
    num_x = num_q + num_v
    num_contacts = length(env.contacts)
    num_g = num_q + num_v + num_contacts
    
    # some useful variable for inner constraints
    bodies = []
    contact_points = []
    obstacles = []
    contact_bases = []
    μs = []
    paths = []
    Ds = []
    for (body, contact_point, obstacle) in env.contacts
        push!(bodies, body)
        push!(contact_points, contact_point)
        push!(obstacles, obstacle)
        push!(contact_bases, contact_basis(obstacle))
        push!(μs, obstacle.μ)
        push!(paths, path(mechanism, body, world))
        push!(Ds, contact_basis(obstacle))
    end
    β_dim = length(contact_bases[1])
    β_selector = find(repmat(vcat(ones(β_dim),[0,0]),num_contacts))
    λ_selector = find(repmat(vcat(zeros(β_dim),[1,0]),num_contacts))
    c_n_selector = find(repmat(vcat(zeros(β_dim),[0,1]),num_contacts))
    
    rel_transforms = Vector{Tuple{Transform3D, Transform3D}}(num_contacts) # force transform, point transform
    geo_jacobians = Vector{GeometricJacobian}(num_contacts)
    
    function eval_g(x::AbstractArray{T}, g) where T
        qnext = x[1:num_q]
        vnext = x[num_q+1:num_q+num_v]
        xnext = MechanismState{T}(mechanism)
        set_configuration!(xnext, qnext)
        set_velocity!(xnext, vnext)

        ϕs = Vector{T}(num_contacts)
        Dtv = Matrix{T}(β_dim,num_contacts)
        for i = 1:num_contacts
            v = point_velocity(twist_wrt_world(xnext,bodies[i]), transform_to_root(xnext, contact_points[i].frame) * contact_points[i])
            Dtv[:,i] = map(contact_bases[i]) do d
                dot(transform_to_root(xnext, d.frame) * d, v)
            end
            rel_transforms[i] = (relative_transform(xnext, obstacles[i].contact_face.outward_normal.frame, world_frame),
                                 relative_transform(xnext, contact_points[i].frame, world_frame))
            geo_jacobians[i] = geometric_jacobian(xnext, paths[i])
            
            ϕs[i] = separation(obstacles[i], transform(xnext, contact_points[i], obstacles[i].contact_face.outward_normal.frame))
            g[num_v+num_q+i] = ϕs[i] # contact distances > 0
        end
        
        config_derivative = configuration_derivative(xnext)
        HΔv = H * (vnext - v0)
        dyn_bias = dynamics_bias(xnext)
        
        contact_bias = τ_contact_wrenches(env,mechanism,bodies,contact_points,obstacles,Δt,
                                          num_v,num_contacts,β_dim,β_selector,λ_selector,c_n_selector,
                                          contact_bases,μs,ϕs,Ds,Dtv,world_frame,total_weight,
                                          rel_transforms,geo_jacobians,
                                          config_derivative,HΔv,dyn_bias,
                                          q0,v0,u0,qnext,vnext)
        
        bias = dyn_bias + contact_bias
        
        g[1:num_v] = HΔv .- Δt .* (u0 .- bias)
        g[num_v+1:num_v+num_q] = qnext .- q0 .- Δt .* config_derivative
    end
    
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            # TODO actually figure out the sparsity pattern
            for i = 1:num_g
                for j = 1:num_x
                    rows[(i-1)*num_x+j] = i
                    cols[(i-1)*num_x+j] = j
                end
            end
        else
            g = zeros(num_g)
            J = ForwardDiff.jacobian((g̃, x̃) -> eval_g(x̃, g̃), g, x)
            values[:] = J'[:]
        end
    end
    
    eval_g, eval_jac_g
end

function simulate(state0::MechanismState{T, M},
                  env::Environment,
                  Δt::Real,
                  N::Integer) where {T, M}
    0.
end

function simulate_implicit(state0::MechanismState{T, M},
                           env::Environment,
                           Δt::Real,
                           N::Integer) where {T, M}
                           
    mechanism = state0.mechanism
    x0 = vcat(configuration(state0),velocity(state0))
    input_limits = all_effort_bounds(mechanism)
    num_q = num_positions(state0)
    num_v = num_velocities(state0)
    num_contacts = length(env.contacts)
    num_x = num_q + num_v
    num_g = num_q + num_v + num_contacts
    
    results = copy(x0)
    u0 = zeros(num_v)
    
    x_L = -1e19 * ones(num_x)
    x_U = 1e19 * ones(num_x)
    g_L = zeros(num_g)
    g_U = zeros(num_g)
    g_U[num_q+num_v+1:num_g] = 1e19
    
    # null costs for simulation
    eval_f = x -> 0.
    eval_grad_f = (x, grad_f) -> grad_f[:] = zeros(length(x))
    
    for i in 1:N
        x = results[:,end]
        q0 = x[1:num_q]
        v0 = x[num_q+1:num_q+num_v]
        u0 .= zeros(num_v) # for now no controller
               
        eval_g, eval_jac_g = update_constraints_implicit_contact(env,mechanism,Δt,q0,v0,u0)
        
        prob = createProblem(num_x,x_L,x_U,
                             num_g,g_L,g_U,
                             num_x*num_g,0,
                             eval_f,eval_g,
                             eval_grad_f,eval_jac_g)
                             
        prob.x[:] = results[:,end][:]
        
        addOption(prob, "hessian_approximation", "limited-memory")
        addOption(prob, "print_level", 0)              
        
        status = solveProblem(prob)
        println(Ipopt.ApplicationReturnStatus[status])
        
        results = hcat(results,prob.x)
    end
    
    results
end