function contact_basis(contactface::HalfSpace3D{T}, μ, motion_type::Symbol) where T
    a = contactface.outward_normal.v
    frame = contactface.outward_normal.frame
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
        R = Rotations.rotation_between(SVector{3, T}(0, 0, 1), a)
        D = [FreeVector3D(frame, R * Rotations.RotZ(θ) * SVector(1, 0, 0))
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

function contact_distance(q::AbstractVector{T},mechanism::Mechanism,contactpoint::Point3D,contactface::HalfSpace3D) where T
    state = MechanismState{T}(mechanism)
    set_configuration!(state,q)
    setdirty!(state)
    sep = separation(contactface, transform(state,contactpoint,contactface.outward_normal.frame))
end

function contact_velocity(x::AbstractVector{T},mechanism::Mechanism,body::RigidBody,contactpoint::Point3D) where T
    # note: contactpoint should be in root frame
    @framecheck(root_frame(mechanism), contactpoint.frame)
    
    q = x[1:num_positions(mechanism)]
    v = x[num_positions(mechanism)+1:num_positions(mechanism)+num_velocities(mechanism)]
    
    state = MechanismState{T}(mechanism)
    set_configuration!(state, q)
    set_velocity!(state, v)    
    setdirty!(state)
    
    twist = twist_wrt_world(state, body)
    contactv = angular(twist) × contactpoint.v + linear(twist)
end