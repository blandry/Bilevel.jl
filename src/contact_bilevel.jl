function τ_external_wrench_bilevel(c_n,β_1,β_2,body,contact_point,obstacle,world_frame,total_weight,
                           rel_transform,geo_jacobian;geo_jacobian_surface=Nothing)
                           
    # compute force in contact frame (obstacle frame)
    contact_force = total_weight * [c_n, β_1, β_2]

    # convert contact force from surface frame to world frame
    c = (rel_transform[1].mat * vcat(contact_force,1.))[1:3]

    # convert contact point from body frame to world frame
    p = transform(contact_point, rel_transform[2])

    # wrench in world frame
    w_linear = c
    w_angular = p.v × c

    # convert wrench from world frame to torque in joint coordinates
    τ = geo_jacobian.linear' * w_linear + geo_jacobian.angular' * w_angular

    # # surface reaction torque
    # if !isa(geo_jacobian_surface,Nothing)
    #     surface_contact_force = -contact_force
    #     # world frame
    #     cs = (rel_transform[1].mat * vcat(surface_contact_force,1.))[1:3]
    #     # wrench
    #     ws_linear = cs
    #     ws_angular = p.v × cs
    #     τs = geo_jacobian_surface.linear' * ws_linear + geo_jacobian_surface.angular' * ws_angular
    # 
    #     τ += τs
    # end

    τ
end

function τ_total_bilevel(x_sol::AbstractArray{T},rel_transforms,geo_jacobians,geo_jacobians_surfaces,sim_data) where T
    c_n_selector = sim_data.c_n_selector
    β_1_selector = sim_data.β_1_selector
    β_2_selector = sim_data.β_2_selector
    num_v = sim_data.num_v
    num_contacts = sim_data.num_contacts
    bodies = sim_data.bodies
    contact_points = sim_data.contact_points
    obstacles = sim_data.obstacles
    world_frame = sim_data.world_frame
    total_weight = sim_data.total_weight
    
    c_n_sol = x_sol[c_n_selector]
    β_1_sol = x_sol[β_1_selector]
    β_2_sol = x_sol[β_2_selector]

    τ_external_wrenches = zeros(T,num_v)
    for i = 1:num_contacts
        c_n = c_n_sol[i]
        β_1 = β_1_sol[i]
        β_2 = β_2_sol[i]
        τ_external_wrenches += τ_external_wrench_bilevel(c_n,β_1,β_2,
                                                 bodies[i],contact_points[i],obstacles[i],
                                                 world_frame,total_weight,
                                                 rel_transforms[i],geo_jacobians[i],
                                                 geo_jacobian_surface=geo_jacobians_surfaces[i])
    end

    τ_external_wrenches
end

function bilevel_compression_con(x_sol,sim_data)
    compression_con = -x_sol[sim_data.c_n_selector] 

    compression_con
end

function bilevel_cone_con(x_sol,sim_data)
    cone_con = x_sol[sim_data.β_1_selector].^2 + x_sol[sim_data.β_2_selector].^2 - sim_data.μs.^2 .* x_sol[sim_data.c_n_selector].^2
    
    cone_con
end

function bilevel_comp_con(x_sol,ϕs,contact_vels,sim_data)
    comp_con = ϕs .* x_sol[sim_data.c_n_selector]
    
    comp_con = vcat(comp_con, (x_sol[sim_data.β_1_selector].^2 + x_sol[sim_data.β_2_selector].^2 - sim_data.μs.^2 .* x_sol[sim_data.c_n_selector].^2) .* contact_vels)

    comp_con
end

function bilevel_comp_con_relaxed(x_sol,slack,ϕs,sim_data)
    comp_con = ϕs .* x_sol[sim_data.c_n_selector] .- dot(slack,slack)
    
    comp_con 
end