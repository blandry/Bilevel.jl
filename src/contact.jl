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
    F::Matrix{Float64}
    ϕ::T
    contact_rot::Matrix{T}
    contact_trans::Vector{T}
    geo_jacobian::GeometricJacobian{Matrix{T}}
    geo_jacobian_surface::Union{Nothing,GeometricJacobian{Matrix{T}}}
    P::Matrix{T}
    J::Matrix{T}

    function ContactJacobianCache(contact::Contact, state::MechanismState{T}) where T<:Real
        world = root_body(contact.mechanism)
        world_frame = default_frame(world)
        total_weight = mass(contact.mechanism) * norm(contact.mechanism.gravitational_acceleration)
        F = total_weight * hcat(contact.obstacle.normal.v, contact.obstacle.basis)
                             
        ϕ = separation(contact.obstacle, transform(state, contact.point, contact.obstacle.normal.frame))

        contact_rot = relative_transform(state, contact.obstacle.normal.frame, world_frame).mat[1:3,1:3]
        contact_trans = relative_transform(state, contact.point.frame, world_frame).mat[1:3,4]
        
        p = contact.point.v - contact_trans
        P = Matrix{T}([0. -p[3] p[2]; p[3] 0. -p[1]; -p[2] p[1] 0.])
        
        geo_jacobian = geometric_jacobian(state, path(contact.mechanism, contact.body, world))
        J = (geo_jacobian.linear' + geo_jacobian.angular' * P) * contact_rot * F

        if contact.obstacle.is_floating
            geo_jacobian_surface = geometric_jacobian(state, path(contact.mechanism, contact.obstacle.body, world))
            J += -(geo_jacobian_surface.linear' + geo_jacobian_surface.angular' * P) * contact_rot * F
        else
            geo_jacobian_surface = nothing
        end
        
        new{T}(contact,F,ϕ,contact_rot,contact_trans,geo_jacobian,geo_jacobian_surface,P,J)
    end
end

function contact_jacobian!(cj::ContactJacobianCache, state::MechanismState)
    contact = cj.contact
    world = root_body(contact.mechanism)
    world_frame = default_frame(world) 
        
    cj.ϕ = separation(contact.obstacle, transform(state, contact.point, contact.obstacle.normal.frame))
    
    cj.contact_rot = relative_transform(state, contact.obstacle.normal.frame, world_frame).mat[1:3,1:3]
    cj.contact_trans = relative_transform(state, contact.point.frame, world_frame).mat[1:3,4]
    
    p = contact.point.v - cj.contact_trans
    cj.P[1,2] = -p[3]
    cj.P[1,3] = p[2]
    cj.P[2,1] = p[3]
    cj.P[2,3] = -p[1]
    cj.P[3,1] = -p[2]
    cj.P[3,2] = p[1]

    geometric_jacobian!(cj.geo_jacobian, state, path(contact.mechanism, contact.body, world))
    cj.J .= (cj.geo_jacobian.linear' + cj.geo_jacobian.angular' * cj.P) * cj.contact_rot * cj.F
    if contact.obstacle.is_floating
        geometric_jacobian!(cj.geo_jacobian_surface, state, path(contact.mechanism, contact.obstacle.body, world))
        cj.J .+= -(cj.geo_jacobian_surface.linear' + cj.geo_jacobian_surface.angular' * cj.P) * cj.contact_rot * cj.F
    end
end

function contact_τ(cj::ContactJacobianCache, c_n, β)
    cj.J * vcat(c_n, β)
end