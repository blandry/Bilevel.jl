RigidBodyDynamics.separation(obs::Obstacle, p::Point3D) = separation(obs.contact_face, p)
contact_normal(obs::Obstacle) = obs.contact_face.outward_normal

function τ_external_wrench(β,λ,c_n,body,contact_point,obstacle,D,world_frame,total_weight,
                           rel_transform,geo_jacobian)
    # compute force in contact frame (obstacle frame)
    n = contact_normal(obstacle)
    v = c_n .* n.v
    for i in eachindex(β)
        v += β[i] .* D[i].v
    end
    contact_force = FreeVector3D(n.frame, total_weight * v)

    # transform from obstacle to world frame   
    c = transform(contact_force, rel_transform[1])
    p = transform(contact_point, rel_transform[2])
    w = Wrench(p, c)

    # convert wrench in world frame to torque in joint coordinates
    torque(geo_jacobian, w)
end

function τ_total(x_sol::AbstractArray{T},β_selector,λ_selector,c_n_selector,num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,
                 world_frame,total_weight,rel_transforms,geo_jacobians) where T     
    β_sol = reshape(x_sol[β_selector],β_dim,num_contacts)
    λ_sol = x_sol[λ_selector]
    c_n_sol = x_sol[c_n_selector]
    
    τ_external_wrenches = zeros(T,num_v)
    for i = 1:num_contacts
        β = β_sol[:,i]
        λ = λ_sol[i]
        c_n = c_n_sol[i]
        τ_external_wrenches += τ_external_wrench(β,λ,c_n,
                                                 bodies[i],contact_points[i],obstacles[i],Ds[i],
                                                 world_frame,total_weight,
                                                 rel_transforms[i],geo_jacobians[i])
    end
    
    τ_external_wrenches
end

function complementarity_contact_constraints(x::AbstractArray{T},ϕs::AbstractArray{M},μs,Dtv,slack_selector,β_selector,λ_selector,c_n_selector,β_dim,num_contacts) where {T,M}
    # dist * c_n = 0
    comp_con = ϕs .* x[c_n_selector]
    
    # (λe + Dtv)' * β = 0
    λ_all = repmat(x[λ_selector]',β_dim,1)
    λpDtv = λ_all .+ Dtv
    β_all = reshape(x[β_selector],β_dim,num_contacts)    
    for i = 1:num_contacts
        comp_con = vcat(comp_con, λpDtv[:,i] .* β_all[:,i])
    end
    
    # (μ * c_n - sum(β)) * λ = 0
    comp_con = vcat(comp_con, (μs .* x[c_n_selector] - sum(β_all,1)[:]) .* x[λ_selector])
    
    comp_con
end

function complementarity_contact_constraints_relaxed(x::AbstractArray{T},ϕs::AbstractArray{M},μs,Dtv,slack_selector,β_selector,λ_selector,c_n_selector,β_dim,num_contacts) where {T,M}
    # dist * c_n = 0
    comp_con = ϕs .* x[c_n_selector] .- x[slack_selector].^2
    
    # (λe + Dtv)' * β = 0
    λ_all = repmat(x[λ_selector]',β_dim,1)
    λpDtv = λ_all .+ Dtv
    β_all = reshape(x[β_selector],β_dim,num_contacts)    
    for i = 1:num_contacts
        comp_con = vcat(comp_con, λpDtv[:,i] .* β_all[:,i] .- x[slack_selector].^2)
    end
    
    # (μ * c_n - sum(β)) * λ = 0
    comp_con = vcat(comp_con, (μs .* x[c_n_selector] - sum(β_all,1)[:]) .* x[λ_selector] .- x[slack_selector].^2)
    
    comp_con
end

function dynamics_contact_constraints(x::AbstractArray{T},β_selector,λ_selector,c_n_selector,HΔv,Δt,bias,num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,
                              world_frame,total_weight,rel_transforms,geo_jacobians) where T    
    # manipulator eq constraint
    τ_contact = τ_total(x,β_selector,λ_selector,c_n_selector,num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,world_frame,total_weight,rel_transforms,geo_jacobians)
    
    dyn_con = HΔv .- Δt .* (bias .- τ_contact)
    
    dyn_con
end

function pos_contact_constraints(x::AbstractArray{T},num_contacts,β_dim,β_selector,λ_selector,c_n_selector,Dtv,μs) where T    
    # β >= 0
    pos_con = -x[β_selector]
    # λ >= 0 
    pos_con = vcat(pos_con, -x[λ_selector])
    # c_n >= 0
    pos_con = vcat(pos_con, -x[c_n_selector])
    
    # λe + D'*v >= 0
    λ_all = repmat(x[λ_selector]',β_dim,1)
    pos_con = vcat(pos_con, reshape(-(λ_all .+ Dtv),β_dim*num_contacts,1))
    
    # μ*c_n - sum(β) >= 0
    pos_con = vcat(pos_con, -(μs.*x[c_n_selector] - reshape(x[β_selector],β_dim,num_contacts)'*ones(β_dim)))

    pos_con
end

function solve_implicit_contact_τ(sim_data,ϕs::AbstractArray{T},Dtv,HΔv,bias,z0;
                                  ip_method=false,α_vect=[1.,1.],c_vect=[2.,4.],I_vect=1e-6*[1.,1.]) where T
                                  
    num_x_contact = 1+sim_data.num_contacts*(sim_data.β_dim+2)
    x0 = vcat(0,z0)

    f = x̃ -> begin
        comp_con = complementarity_contact_constraints(x̃,ϕs,sim_data.μs,Dtv,
                                                       sim_data.slack_selector,sim_data.β_selector,sim_data.λ_selector,sim_data.c_n_selector,
                                                       sim_data.β_dim,sim_data.num_contacts)
        comp_con'*comp_con
    end
    h = x̃ -> dynamics_contact_constraints(x̃,sim_data.β_selector,sim_data.λ_selector,sim_data.c_n_selector,HΔv,sim_data.Δt,bias,sim_data.num_v,sim_data.num_contacts,sim_data.β_dim,sim_data.bodies,sim_data.contact_points,sim_data.obstacles,sim_data.Ds,sim_data.world_frame,sim_data.total_weight,sim_data.rel_transforms,sim_data.geo_jacobians)
    g = x̃ -> pos_contact_constraints(x̃,sim_data.num_contacts,sim_data.β_dim,sim_data.β_selector,sim_data.λ_selector,sim_data.c_n_selector,Dtv,sim_data.μs)
    
    num_h = sim_data.num_v
    num_g = sim_data.num_contacts*(2*sim_data.β_dim+3)

    if ip_method
        x = ip_solve(x0,f,h,g,num_h,num_g)
    else
        x = auglag_solve(x0,f,h,g,num_h,num_g,α_vect,c_vect,I_vect)
    end

    τ = τ_total(x,sim_data.β_selector,sim_data.λ_selector,sim_data.c_n_selector,sim_data.num_v,sim_data.num_contacts,sim_data.β_dim,sim_data.bodies,sim_data.contact_points,sim_data.obstacles,sim_data.Ds,sim_data.world_frame,sim_data.total_weight,sim_data.rel_transforms,sim_data.geo_jacobians)
                   
    return τ, x
end

function solve_implicit_contact_τ(sim_data,q0,v0,u0,z0,qnext::AbstractArray{T},vnext::AbstractArray{T};ip_method=false) where T
    set_configuration!(sim_data.x0,q0)
    set_velocity!(sim_data.x0,v0)
    setdirty!(sim_data.x0)
    H = mass_matrix(sim_data.x0)
    
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
        sim_data.rel_transforms[i] = (relative_transform(xnext, sim_data.obstacles[i].contact_face.outward_normal.frame, sim_data.world_frame),relative_transform(xnext, sim_data.contact_points[i].frame, sim_data.world_frame))
        sim_data.geo_jacobians[i] = geometric_jacobian(xnext, sim_data.paths[i])
        ϕs[i] = separation(sim_data.obstacles[i], transform(xnext, sim_data.contact_points[i], sim_data.obstacles[i].contact_face.outward_normal.frame))
    end

    config_derivative = configuration_derivative(xnext)
    HΔv = H * (vnext - v0)
    bias = u0 .- dynamics_bias(xnext)
    τ, x = solve_implicit_contact_τ(sim_data,ϕs,Dtv,HΔv,bias,z0,ip_method=ip_method)
    
    return τ, x
end