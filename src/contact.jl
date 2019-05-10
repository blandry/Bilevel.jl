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

mutable struct ContactJacobian{T}
    contact::Contact
    F::Matrix{Float64}
    ϕ::T
    N::Matrix{T}
    contact_rot::Matrix{T}
    contact_trans::Vector{T}
    geo_jacobian::GeometricJacobian{Matrix{T}}
    geo_jacobian_surface::Union{Nothing,GeometricJacobian{Matrix{T}}}
    P::Matrix{T}
    J::Matrix{T}

    function ContactJacobian{T}(contact::Contact) where T
        total_weight = mass(contact.mechanism) * norm(contact.mechanism.gravitational_acceleration)
        world = root_body(contact.mechanism)
        world_frame = default_frame(world)
        nv = num_velocities(contact.mechanism)
        β_dim = size(contact.obstacle.basis, 2)

        F = total_weight * hcat(contact.obstacle.normal.v, contact.obstacle.basis)

        ϕ = zero(T)
        N = Matrix{T}(undef, 1, nv)
        contact_rot = Matrix{T}(undef, 3, 3)
        contact_trans = Vector{T}(undef, 3)
        geo_jacobian = GeometricJacobian(world_frame, default_frame(contact.body), root_frame(contact.mechanism), Matrix{T}(undef, 3, nv), Matrix{T}(undef, 3, nv))
        geo_jacobian_surface = GeometricJacobian(world_frame, default_frame(contact.obstacle.body), root_frame(contact.mechanism), Matrix{T}(undef, 3, nv), Matrix{T}(undef, 3, nv))
        P = zeros(T, 3, 3)
        J = Matrix{T}(undef, nv, 1 + β_dim)

        new{T}(contact,F,ϕ,N,contact_rot,contact_trans,geo_jacobian,geo_jacobian_surface,P,J)
    end
end

function contact_jacobian!(cj::ContactJacobian, state::MechanismState)
    contact = cj.contact
    world = root_body(contact.mechanism)
    world_frame = default_frame(world)

    cp = transform(state, contact.point, contact.obstacle.normal.frame)

    cj.ϕ = separation(contact.obstacle, cp)
    cj.N .= contact.obstacle.normal.v' * point_jacobian(state, path(contact.mechanism, contact.obstacle.body, contact.body), cp).J

    cj.contact_rot = relative_transform(state, contact.obstacle.normal.frame, world_frame).mat[1:3,1:3]
    cj.contact_trans = relative_transform(state, contact.point.frame, world_frame).mat[1:3,4]

    p = contact.point.v + cj.contact_trans
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

function contact_τ(cj::ContactJacobian, c_n, β)
    cj.J * vcat(c_n, β)
end
