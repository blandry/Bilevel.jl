mutable struct Obstacle
    body::RigidBody
    point::Point3D
    normal::FreeVector3D
    basis::Matrix{Float64}
    μ::Real
    is_floating::Bool

    function Obstacle(body::RigidBody, point::Point3D, normal::FreeVector3D, motion_type::Symbol, μ::Real; is_floating=false)
        @framecheck point.frame normal.frame
        normal_norm = normalize(normal)
        basis = contact_basis(normal_norm, μ, motion_type)
        new(body, point, normal_norm, basis, μ, is_floating)
    end
end

separation(obs::Obstacle, p::Point3D) = dot(p - obs.point, obs.normal)
