struct Environment
    contacts::Vector{Contact}
    
    function Environment(mechanism, urdf, obstacles)
        elements = visual_elements(mechanism, URDFVisuals(urdf; tag="collision"))
        point_elements = filter(e -> e.geometry isa GeometryTypes.HyperSphere && GeometryTypes.radius(e.geometry) == 0, elements)
        points = map(point_elements) do element
            p = element.transform(GeometryTypes.origin(element.geometry))
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

struct EnvironmentCache
    contact_jacobians::Vector{ContactJacobianCache}
    
    function EnvironmentCache(env::Environment, state::MechanismState{T}) where T
        contact_jacobians = [ContactJacobianCache(contact, state) for contact in env.contacts] 
        
        new(contact_jacobians)
    end
end

function contact_jacobian!(env_c::EnvironmentCache, state::MechanismState{T}) where T
    # TODO: parallel
    for cj in env_c.contact_jacobians
        contact_jacobian!(cj, state)
    end
end