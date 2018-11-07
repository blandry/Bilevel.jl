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

function τ_total(x_sol::AbstractArray{T},num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,
                 world_frame,total_weight,rel_transforms,geo_jacobians) where T
                 
    τ_external_wrenches = zeros(T,num_v)
    x_sol = reshape(x_sol,β_dim+2,num_contacts)
    
    for i = 1:num_contacts
        β = x_sol[1:β_dim,i]
        λ = x_sol[β_dim+1,i]
        c_n = x_sol[β_dim+2,i]
        τ_external_wrenches += τ_external_wrench(β,λ,c_n,
                                                 bodies[i],contact_points[i],obstacles[i],Ds[i],
                                                 world_frame,total_weight,
                                                 rel_transforms[i],geo_jacobians[i])
    end
    
    τ_external_wrenches
end

function f_contact(x::AbstractArray{T},ϕs::AbstractArray{M},μs,Dtv,β_selector,λ_selector,c_n_selector,β_dim,num_contacts) where {T,M}
    # complementarity constraints
    
    # dist * c_n = 0
    vx = ϕs .* x[c_n_selector]
    
    # (λe + Dtv)' * β = 0
    λ_all = repmat(x[λ_selector]',β_dim,1)
    λpDtv = λ_all .+ Dtv
    β_all = reshape(x[β_selector],β_dim,num_contacts)    
    for i = 1:num_contacts
        push!(vx, dot(λpDtv[:,i],β_all[:,i]))
    end
    
    # (μ * c_n - sum(β)) * λ = 0
    vx = vcat(vx, (μs .* x[c_n_selector] - sum(β_all,1)[:]) .* x[λ_selector])
    
    return dot(vx,vx)
end

function h_contact(x::AbstractArray{T},HΔv,Δt,u0,dyn_bias,num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,
                   world_frame,total_weight,rel_transforms,geo_jacobians) where T
    # equality constraints
    
    # manipulator eq constraint
    bias = dyn_bias + τ_total(x,num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,
                              world_frame,total_weight,rel_transforms,geo_jacobians)
    
    return HΔv .- Δt .* (u0 .- bias)
end

function g_contact(x::AbstractArray{T},num_contacts,β_dim,β_selector,λ_selector,c_n_selector,Dtv,μs) where T
    # non-negativity constraints
    
    # c_n >= 0
    gx1 = -x[c_n_selector]
    # β >= 0
    gx2 = -x[β_selector]
    # λ >= 0 
    gx3 = -x[λ_selector]
    
    # λe + D'*v >= 0
    λ_all = repmat(x[λ_selector]',β_dim,1)
    gx4 = reshape(-(λ_all .+ Dtv),β_dim*num_contacts,1)
    
    # μ*c_n - sum(β) >= 0
    gx5 = -(μs.*x[c_n_selector] - reshape(x[β_selector],β_dim,num_contacts)'*ones(β_dim))

    return vcat(gx1,gx2,gx3,gx4,gx5)
end

function τ_contact_wrenches(env,mechanism,bodies,contact_points,obstacles,Δt,
                            num_v,num_contacts,β_dim,β_selector,λ_selector,c_n_selector,
                            contact_bases,μs,ϕs,Ds,Dtv,world_frame,total_weight,
                            rel_transforms,geo_jacobians,
                            config_derivative,HΔv,dyn_bias,
                            q0::AbstractArray{M},v0::AbstractArray{M},u0::AbstractArray{M},
                            qnext::AbstractArray{T},vnext::AbstractArray{T}) where {M, T}

    num_x = num_contacts*(β_dim+2)

    # x := [β,λ,c_n,β2,λ2,c_n2,...]    
    x0 = zeros(T,num_x)

    f = x̃ -> f_contact(x̃,ϕs,μs,Dtv,β_selector,λ_selector,c_n_selector,β_dim,num_contacts)
    h = x̃ -> h_contact(x̃,HΔv,Δt,u0,dyn_bias,num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,
                       world_frame,total_weight,rel_transforms,geo_jacobians)
    g = x̃ -> g_contact(x̃,num_contacts,β_dim,β_selector,λ_selector,c_n_selector,Dtv,μs)
    
    num_h = num_v
    num_g = num_contacts*(β_dim+2) + β_dim*num_contacts + num_contacts
    
    # parameters of the augmented lagrangian method
    N = 5
    α_vect = [1.^i for i in 1:N]
    c_vect = [2.^i for i in 1:N]
    I = eye(num_x)
    
    x = augmented_lagrangian_method(x0,f,h,g,num_h,num_g,α_vect,c_vect,I)
    # J = ForwardDiff.jacobian(z̃ -> augmented_lagrangian_method(x0,f,h,g,num_h,num_g,z̃[1:N],z̃[N+1:2*N],I), vcat(α_vect,c_vect))
    # show(STDOUT, "text/plain", J); println("")

    # x = ipopt_solve(x0,f,h,g,num_h,num_g)
    # println(τ_total(x,num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,
    #                world_frame,total_weight,rel_transforms,geo_jacobians))
                   
    return τ_total(x,num_v,num_contacts,β_dim,bodies,contact_points,obstacles,Ds,
                   world_frame,total_weight,rel_transforms,geo_jacobians)
end