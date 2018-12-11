# to avoid using the one in RBD.jl that uses static arrays
mutable struct HalfSpace
    point::Point3D
    outward_normal::FreeVector3D

    function HalfSpace(point::Point3D, outward_normal::FreeVector3D)
        @framecheck point.frame outward_normal.frame
        new(point, normalize(outward_normal))
    end
end

RigidBodyDynamics.separation(halfspace::HalfSpace, p::Point3D) = dot(p - halfspace.point, halfspace.outward_normal)

function contact_basis(contact_face::HalfSpace, μ, motion_type::Symbol)
    a = contact_face.outward_normal.v
    frame = contact_face.outward_normal.frame
    if motion_type == :xz
        return [
            FreeVector3D(frame, Rotations.RotY(π/2) * a),
            FreeVector3D(frame, Rotations.RotY(-π/2) * a)
            ]
    elseif motion_type == :xy
        return [
            FreeVector3D(frame, Rotations.RotZ(π/2) * a),
            FreeVector3D(frame, Rotations.RotZ(-π/2) * a)
            ]
    elseif motion_type == :yz
        return [
            FreeVector3D(frame, Rotations.RotX(π/2) * a),
            FreeVector3D(frame, Rotations.RotX(-π/2) * a)
            ]
    elseif motion_type == :xyz
        R = Rotations.rotation_between([0., 0., 1.], a)
        D = [FreeVector3D(frame, R * RotZ(θ) * [1., 0., 0.])
                for θ in 0:π/2:3π/2]
        for i in 1:length(D)
            @assert isapprox(dot(D[i].v, a), 0, atol=1e-15)
            if i > 1
                @assert isapprox(dot(D[i].v, D[i-1].v), 0, atol=1e-15)
            end
        end
        return D
    else
        throw(ArgumentError("Unrecognized motion type: $motion_type (should be :xy, :xz, :yz, or :xyz)"))
    end
end

struct Obstacle
    contact_face::HalfSpace
    μ
    contact_basis::Vector{FreeVector3D}
end

function Obstacle(contact_face::HalfSpace, μ, motion_type::Symbol)
    basis = contact_basis(contact_face, μ, motion_type)
    Obstacle(contact_face, μ, basis)
end

RigidBodyDynamics.separation(obs::Obstacle, p::Point3D) = separation(obs.contact_face, p)

contact_normal(obs::Obstacle) = obs.contact_face.outward_normal

function planar_obstacle(normal::FreeVector3D, point::Point3D, μ=1.0, motion_type::Symbol=:xyz)
    normal = normalize(normal)
    face = HalfSpace(point, normal)
    Obstacle(face,μ,motion_type)
end

planar_obstacle(frame::CartesianFrame3D, normal::AbstractVector, point::AbstractVector, args...) =
    planar_obstacle(FreeVector3D(frame, normal), Point3D(frame, point), args...)

contact_basis(obs::Obstacle) = obs.contact_basis

struct Environment
    contacts::Array{Tuple{RigidBody, Point3D, Obstacle}}
end

function parse_contacts(mechanism, urdf, obstacles)
    elements = visual_elements(mechanism, URDFVisuals(urdf; tag="collision"))
    point_elements = filter(e -> e.geometry isa HyperSphere && radius(e.geometry) == 0, elements)
    points = map(point_elements) do element
        p = element.transform(origin(element.geometry))
        Point3D(element.frame, p)
    end
    contacts = vec(map(Base.Iterators.product(points, obstacles)) do p
        point, obstacle = p
        body = body_fixed_frame_to_body(mechanism, point.frame)
        (body, point, obstacle)
    end)
    # removes collision with itself
    contacts = filter(c -> c[2].frame != c[3].contact_face.outward_normal.frame, contacts)
    Environment(contacts)
end
