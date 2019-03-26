struct Contact
    mechanism::Mechanism
    body::RigidBody
    point::Point3D
    obstacle::Obstacle
end

function contact_basis(normal::FreeVector3D, μ::Real, motion_type::Symbol)
    if motion_type == :xz
        return hcat(Rotations.RotY(π/2) * normal, Rotations.RotY(-π/2) * normal)
    elseif motion_type == :xy
        return hcat(Rotations.RotZ(π/2) * normal, Rotations.RotZ(-π/2) * normal)
    elseif motion_type == :yz
        return hcat(Rotations.RotX(π/2) * normal, Rotations.RotX(-π/2) * normal)
    elseif motion_type == :xyz
        R = Rotations.rotation_between([0., 0., 1.], normal.v)
        return hcat([R * RotZ(θ) * [1., 0., 0.] for θ in 0:π/2:3π/2]...)
    else
        throw(ArgumentError("Unrecognized motion type: $motion_type (should be :xy, :xz, :yz, or :xyz)"))
    end
end

mutable struct ContactJacobianCache{T<:Real}
    contact::Contact
    ϕ::T
    rel_transform_force::Transform3D{T}
    rel_transform_point::Transform3D{T}
    geo_jacobian::GeometricJacobian{Matrix{T}}
    geo_jacobian_surface::Union{Nothing,GeometricJacobian{Matrix{T}}}

    function ContactJacobianCache(contact::Contact, state::MechanismState{T}) where T<:Real
        ϕ = separation(contact.obstacle, transform(state, contact.point, contact.obstacle.contact_face.outward_normal.frame))

        world = root_body(contact.mechanism)
        world_frame = default_frame(world)    
        rel_transform_force = relative_transform(state, contact.obstacle.contact_face.outward_normal.frame, world_frame)
        rel_transform_point = relative_transform(state, contact.point.frame, world_frame)
        
        geo_jacobian = geometric_jacobian(state, path(contact.mechanism, contact.body, world))
        if contact.obstacle.is_floating
            geo_jacobian_surface = geometric_jacobian(state, path(contact.mechanism, contact.obstacle.body, world))
        else
            geo_jacobian_surface = nothing
        end
        
        new{T}(contact,ϕ,rel_transform_force,rel_transform_point,geo_jacobian,geo_jacobian_surface)
    end
end

function contact_jacobian!(contact_cache::ContactJacobianCache,state::MechanismState})
    contact = contact_cache.contact
    
    contact_cache.ϕ = separation(contact.obstacle, transform(state, contact.point, contact.obstacle.contact_face.outward_normal.frame))
    
    world = root_body(contact.mechanism)
    world_frame = default_frame(world) 
    contact_cache.rel_transform_force = relative_transform(state, contact.obstacle.contact_face.outward_normal.frame, world_frame)
    contact_cache.rel_transform_point = relative_transform(state, contact.point.frame, world_frame)
    
    geometric_jacobian!(contact_cache.geo_jacobian, state, path(contact.mechanism, contact.body, world))
    if obstacle.is_floating
        geometric_jacobian!(contact_cache.geo_jacobian_surface, state, path(contact.mechanism, contact.obstacle.body, world))
    end
end

function contact_τ!()
    
    # TODO
end