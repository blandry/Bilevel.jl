function valuetype end
function makevalue end

struct Environment
    contacts::Vector{Contact}

    function Environment(mechanism, urdf, obstacles)
        elements = visual_elements(mechanism, URDFVisuals(urdf; tag="collision"))
        point_elements = filter(e -> e.geometry isa GeometryTypes.HyperSphere && GeometryTypes.radius(e.geometry) == 0, elements)
        points = map(point_elements) do element
            p = element.transform(SVector(GeometryTypes.origin(element.geometry)))
            Point3D(element.frame, p)
        end
        contacts = vec(map(Base.Iterators.product(points, obstacles)) do p
            point, obstacle = p
            body = body_fixed_frame_to_body(mechanism, point.frame)
            Contact(mechanism, body, point, obstacle)
        end)
        # removes collision with itself
        contacts = filter(c -> c.point.frame != c.obstacle.normal.frame, contacts)
        new(contacts)
    end
end

struct EnvironmentJacobian{T}
    contact_jacobians::Vector{ContactJacobian{T}}

    function EnvironmentJacobian{T}(env::Environment) where T
        contact_jacobians = [ContactJacobian{T}(contact) for contact in env.contacts]

        new{T}(contact_jacobians)
    end
end

struct EnvironmentJacobianCache <: RigidBodyDynamics.AbstractTypeDict
    env::Environment
    keys::Vector{Tuple{UInt64, Int}}
    values::Vector{EnvironmentJacobian}
end

function EnvironmentJacobianCache(env::Environment)
    EnvironmentJacobianCache(env, [], [])
end

Base.show(io::IO, ::EnvironmentJacobianCache) = print(io, "EnvironmentJacobianCache{…}(…)")

@inline RigidBodyDynamics.valuetype(::Type{EnvironmentJacobianCache}, ::Type{T}) where {T} = EnvironmentJacobian{T}
@inline RigidBodyDynamics.makevalue(envj_c::EnvironmentJacobianCache, ::Type{T}) where {T} = EnvironmentJacobian{T}(envj_c.env)

function contact_jacobian!(envj::EnvironmentJacobian, state::MechanismState)
    # TODO: parallel
    for cj in envj.contact_jacobians
        contact_jacobian!(cj, state)
    end
end
