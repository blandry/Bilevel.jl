struct SimData
    Δt
    mechanism
    x0
    rel_transforms
    geo_jacobians
    num_q
    num_v
    num_contacts
    β_dim
    num_x
    num_h
    num_g
    world
    world_frame
    total_weight
    bodies
    contact_points
    obstacles
    contact_bases
    μs
    paths
    Ds
    β_selector
    λ_selector
    c_n_selector
end

function update_constraints(sim_data,q0,v0,u0)
    set_configuration!(sim_data.x0,q0)
    set_velocity!(sim_data.x0,v0)
    setdirty!(sim_data.x0)
    H = mass_matrix(sim_data.x0)
    
    function eval_g(x::AbstractArray{T}, g) where T
        qnext = x[1:sim_data.num_q]
        vnext = x[sim_data.num_q+1:sim_data.num_q+sim_data.num_v]
        xnext = MechanismState{T}(sim_data.mechanism)
        set_configuration!(xnext, qnext)
        set_velocity!(xnext, vnext)
        
        ϕs = Vector{T}(sim_data.num_contacts)
        Dtv = Matrix{T}(sim_data.β_dim,sim_data.num_contacts)
        for i = 1:sim_data.num_contacts
            v = point_velocity(twist_wrt_world(xnext,sim_data.bodies[i]), transform_to_root(xnext, sim_data.contact_points[i].frame) * sim_data.contact_points[i])
            Dtv[:,i] = map(sim_data.contact_bases[i]) do d
                dot(transform_to_root(xnext, d.frame) * d, v)
            end
            sim_data.rel_transforms[i] = (relative_transform(xnext, sim_data.obstacles[i].contact_face.outward_normal.frame, sim_data.world_frame),
                                          relative_transform(xnext, sim_data.contact_points[i].frame, sim_data.world_frame))
            sim_data.geo_jacobians[i] = geometric_jacobian(xnext, sim_data.paths[i])
            
            ϕs[i] = separation(sim_data.obstacles[i], transform(xnext, sim_data.contact_points[i], sim_data.obstacles[i].contact_face.outward_normal.frame))
        end
        
        config_derivative = configuration_derivative(xnext)
        HΔv = H * (vnext - v0)
        dyn_bias = dynamics_bias(xnext)
        contact_bias = τ_total(x[sim_data.num_q+sim_data.num_v+1:end],
                                 sim_data.num_v,sim_data.num_contacts,sim_data.β_dim,sim_data.bodies,sim_data.contact_points,sim_data.obstacles,sim_data.Ds,
                                 sim_data.world_frame,sim_data.total_weight,sim_data.rel_transforms,sim_data.geo_jacobians)
        bias = dyn_bias + contact_bias
        
        g[1:sim_data.num_q] = qnext .- q0 .- sim_data.Δt .* config_derivative # == 0
        g[sim_data.num_q+1:sim_data.num_q+sim_data.num_v] = HΔv .- sim_data.Δt .* (u0 .- bias) # == 0
                
        g[sim_data.num_q+sim_data.num_v+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)] = 
            f_contact(x[sim_data.num_q+sim_data.num_v+1:end],ϕs,sim_data.μs,Dtv,sim_data.β_selector,sim_data.λ_selector,sim_data.c_n_selector,sim_data.β_dim,sim_data.num_contacts) # <= 0
            
        g[sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts] = -ϕs # <= 0        
        
        g[sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts+1:sim_data.num_q+sim_data.num_v+sim_data.num_contacts*(2+sim_data.β_dim)+sim_data.num_contacts+sim_data.num_contacts*(3+2*sim_data.β_dim)] = 
                g_contact(x[sim_data.num_q+sim_data.num_v+1:end],
                      sim_data.num_contacts,sim_data.β_dim,
                      sim_data.β_selector,sim_data.λ_selector,sim_data.c_n_selector,
                      Dtv,sim_data.μs) # <= 0
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

    mechanism = state0.mechanism
    
    num_q = num_positions(state0)
    num_v = num_velocities(state0)
    num_contacts = length(env.contacts)
    β_dim = length(contact_basis(env.contacts[1][3]))
    # x = [q, v, β1, λ1, c_n1, β2, λ2, c_n2...]
    num_x = num_q + num_v + num_contacts*(2+β_dim)
    num_h = num_q + num_v
    num_g = num_contacts + num_contacts*(5+3*β_dim)
    
    # some constants throughout the simulation
    world = root_body(mechanism)
    world_frame = default_frame(world)
    total_weight = mass(mechanism) * norm(mechanism.gravitational_acceleration)
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
    β_selector = find(repmat(vcat(ones(β_dim),[0,0]),num_contacts))
    λ_selector = find(repmat(vcat(zeros(β_dim),[1,0]),num_contacts))
    c_n_selector = find(repmat(vcat(zeros(β_dim),[0,1]),num_contacts))
    
    x0 = MechanismState(mechanism)
    rel_transforms = Vector{Tuple{Transform3D, Transform3D}}(num_contacts) # force transform, point transform
    geo_jacobians = Vector{GeometricJacobian}(num_contacts)
    
    sim_data = SimData(Δt,mechanism,x0,rel_transforms,geo_jacobians,
                       num_q,num_v,num_contacts,β_dim,num_x,num_h,num_g,
                       world,world_frame,total_weight,
                       bodies,contact_points,obstacles,contact_bases,μs,paths,Ds,
                       β_selector,λ_selector,c_n_selector)

    # optimization bounds
    x_L = -1e19 * ones(num_x)
    x_U = 1e19 * ones(num_x)
    
    g_L = vcat(0. * ones(num_h), -1e19 * ones(num_g))
    g_U = vcat(0. * ones(num_h),  1e-6 * ones(num_g)) 

    # null costs for simulation
    eval_f = x -> 0.
    eval_grad_f = (x, grad_f) -> grad_f[:] = 0.

    results = vcat(configuration(state0),velocity(state0),zeros(2+β_dim))
    
    for i in 1:N
      x = results[:,end]
      q0 = x[1:num_q]
      v0 = x[num_q+1:num_q+num_v]
      u0 = zeros(num_v) # for now no controller
      
      eval_g, eval_jac_g = update_constraints(sim_data,q0,v0,u0)
      
      prob = createProblem(num_x,x_L,x_U,
                           num_h+num_g,g_L,g_U,
                           num_x*(num_h+num_g),0,
                           eval_f,eval_g,
                           eval_grad_f,eval_jac_g)
                           
      prob.x[:] = results[:,end][:]
      
      addOption(prob, "hessian_approximation", "limited-memory")
      # addOption(prob, "print_level", 0)              
      
      status = solveProblem(prob)
      # println(Ipopt.ApplicationReturnStatus[status])
      
      results = hcat(results,prob.x)
      
      g_tmp = zeros(num_h + num_g)
      eval_g(x, g_tmp)
      println(g_tmp)
    end

    results
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

function simulate_implicit(state0::MechanismState{T, M},
                           env::Environment,
                           Δt::Real,
                           N::Integer) where {T, M}
                           
    mechanism = state0.mechanism
    x0 = vcat(configuration(state0),velocity(state0))
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