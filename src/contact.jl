global num_steps_default = 5
global α_vect_default = [1.^i for i in 1:num_steps_default]
global c_vect_default = [100.+min.(2.^i,100.) for i in 1:num_steps_default]
global I_vect_default = 1e-16*ones(num_steps_default)

function τ_external_wrench(β,λ,c_n,body,contact_point,obstacle,D,world_frame,total_weight,
                           rel_transform,geo_jacobian)
    # compute force in contact frame (obstacle frame)
    n = contact_normal(obstacle)
    v = c_n .* n.v
    for i in eachindex(β)
        v += β[i] .* Array(D[i].v)
    end

    contact_force = total_weight * v
    
    # convert contact force from surface frame to world frame
    c = (rel_transform[1].mat * vcat(contact_force,1.))[1:3]
    
    # convert contact point from body frame to world frame
    p = transform(contact_point, rel_transform[2])
    
    # wrench in world frame
    w_linear = c
    w_angular = p.v × c
    
    # convert wrench from world frame to torque in joint coordinates
    τ = geo_jacobian.linear' * w_linear + geo_jacobian.angular' * w_angular

    τ
end

function τ_total(x_sol::AbstractArray{T},rel_transforms,geo_jacobians,sim_data) where T    
    β_selector = sim_data.β_selector
    λ_selector = sim_data.λ_selector
    c_n_selector = sim_data.c_n_selector
    num_v = sim_data.num_v
    num_contacts = sim_data.num_contacts
    β_dim = sim_data.β_dim
    bodies = sim_data.bodies
    contact_points = sim_data.contact_points
    obstacles = sim_data.obstacles
    Ds = sim_data.Ds
    world_frame = sim_data.world_frame
    total_weight = sim_data.total_weight

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

function complementarity_contact_constraints(x,ϕs,Dtv,sim_data)
    μs = sim_data.μs
    β_selector = sim_data.β_selector
    λ_selector = sim_data.λ_selector
    c_n_selector = sim_data.c_n_selector
    β_dim = sim_data.β_dim
    num_contacts = sim_data.num_contacts

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

function complementarity_contact_constraints_relaxed(x,slack,ϕs,Dtv,sim_data)
    μs = sim_data.μs
    β_selector = sim_data.β_selector
    λ_selector = sim_data.λ_selector
    c_n_selector = sim_data.c_n_selector
    β_dim = sim_data.β_dim
    num_contacts = sim_data.num_contacts

    # dist * c_n = 0
    comp_con = ϕs .* x[c_n_selector] .- dot(slack,slack)

    # (λe + Dtv)' * β = 0
    λ_all = repmat(x[λ_selector]',β_dim,1)
    λpDtv = λ_all .+ Dtv
    β_all = reshape(x[β_selector],β_dim,num_contacts)
    for i = 1:num_contacts
        comp_con = vcat(comp_con, λpDtv[:,i] .* β_all[:,i] .- dot(slack,slack))
    end

    # (μ * c_n - sum(β)) * λ = 0
    comp_con = vcat(comp_con, (μs .* x[c_n_selector] - sum(β_all,1)[:]) .* x[λ_selector] .- dot(slack,slack))

    comp_con
end

function dynamics_contact_constraints(x,rel_transforms,geo_jacobians,HΔv,bias,sim_data)
    # manipulator eq constraint
    τ_contact = τ_total(x,rel_transforms,geo_jacobians,sim_data)
    dyn_con = HΔv .-  sim_data.Δt .* (bias .- τ_contact)
    
    dyn_con
end

function pos_contact_constraints(x,Dtv,sim_data)
    β_selector = sim_data.β_selector
    λ_selector = sim_data.λ_selector
    c_n_selector = sim_data.c_n_selector
    num_contacts = sim_data.num_contacts
    β_dim = sim_data.β_dim
    μs = sim_data.μs

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

    # upper bounds
    pos_con = vcat(pos_con, x[β_selector] .- 100.)
    pos_con = vcat(pos_con, x[λ_selector] .- 100.)
    pos_con = vcat(pos_con, x[c_n_selector] .- 100.)

    pos_con
end

function solve_implicit_contact_τ(sim_data,ϕs,Dtv,rel_transforms,geo_jacobians,HΔv,bias,z0;
    ip_method=false,α_vect=α_vect_default,c_vect=c_vect_default,I_vect=I_vect_default)
    
    # TODO handle the case where they are note duals
    if isa(ϕs, Array{T} where T<:Real)
        ϕs_value = ϕs
        Dtv_value = Dtv
        rel_transforms_value = rel_transforms
        geo_jacobians_value = geo_jacobians
        HΔv_value = HΔv
        bias_value = bias
    else
        ϕs_value = map(x->x.value,ϕs)
        Dtv_value = map(x->x.value,Dtv)
        rel_transforms_value = map(rel_transforms) do rt
            rt1 = Transform3D(rt[1].from,rt[1].to,Array(map(x->x.value,rt[1].mat)))
            rt2 = Transform3D(rt[2].from,rt[2].to,Array(map(x->x.value,rt[2].mat)))
            (rt1, rt2)
        end
        geo_jacobians_value = map(geo_jacobians) do gj
            GeometricJacobian(gj.body,gj.base,gj.frame,map(x->x.value,gj.angular),map(x->x.value,gj.linear))
        end
        HΔv_value = map(x->x.value,HΔv)
        bias_value = map(x->x.value,bias)    
    end
    
    f = x̃ -> begin
        if isa(x̃,ReverseDiff.TrackedArray)
            comp_con = complementarity_contact_constraints(x̃,ϕs_value,Dtv_value,sim_data)
        else
            comp_con = complementarity_contact_constraints(x̃,ϕs,Dtv,sim_data)
        end
        sum(comp_con) + dot(x̃,x̃)
    end 
    h = x̃ -> begin
        if isa(x̃,ReverseDiff.TrackedArray)
            dynamics_contact_constraints(x̃,rel_transforms_value,geo_jacobians_value,HΔv_value,bias_value,sim_data)
        else
            dynamics_contact_constraints(x̃,rel_transforms,geo_jacobians,HΔv,bias,sim_data)
        end
    end
    g = x̃ -> begin
        if isa(x̃,ReverseDiff.TrackedArray)
            pos_contact_constraints(x̃,Dtv_value,sim_data)
        else
            pos_contact_constraints(x̃,Dtv,sim_data)
        end
    end

    num_h = sim_data.num_v
    num_g = sim_data.num_contacts*(2*sim_data.β_dim+3+sim_data.β_dim+2)

    if ip_method
        x = ip_solve(z0,f,h,g,num_h,num_g)
    else
        x = auglag_solve(z0,f,h,g,num_h,num_g,α_vect,c_vect,I_vect)
    end

    τ = τ_total(x,rel_transforms,geo_jacobians,sim_data)

    return τ, x
end

function solve_implicit_contact_τ(sim_data,q0,v0,u0,z0,qnext::AbstractArray{T},vnext::AbstractArray{T};
    ip_method=false,α_vect=α_vect_default,c_vect=c_vect_default,I_vect=I_vect_default) where T

    x0 = MechanismState(sim_data.mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)

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
    τ, x = solve_implicit_contact_τ(sim_data,ϕs,Dtv,rel_transforms,geo_jacobians,HΔv,bias,z0,ip_method=ip_method,α_vect=α_vect,c_vect=c_vect,I_vect=I_vect)

    return τ, x
end
