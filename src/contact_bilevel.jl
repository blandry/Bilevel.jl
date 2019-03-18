struct BilevelContactParams
    H
    dyn_bias
    rel_transforms
    geo_jacobians
end

function solve_contact_τ(sim_data,H,dyn_bias,rel_transforms,geo_jacobians,v0,u0,c_ns;ip_method=false,in_place=false)    
    Hi = inv(H)
    h = sim_data.Δt
    
    Fs = []
    Js = []
    Qds = []
    rds = []
    for j = 1:sim_data.num_contacts
        p = transform(sim_data.contact_points[j], rel_transforms[j][2])
        P = zeros(typeof(p.v[1]),3,3)
        P[1,2] = -p.v[3]
        P[1,3] = p.v[2]
        P[2,1] = p.v[3]
        P[2,3] = -p.v[1]
        P[3,1] = -p.v[2]
        P[3,2] = p.v[1]
        
        F = sim_data.total_weight*hcat(contact_normal(sim_data.obstacles[j]).v,sim_data.Ds[j])
        J = geo_jacobians[j].linear + P'*geo_jacobians[j].angular
        
        Qd = h*J*Hi*J'
        rd = J*(h*Hi*(dyn_bias - u0) - v0)
    
        push!(Fs,F)
        push!(Js,J)
        push!(Qds,Qd)
        push!(rds,rd)
    end
    
    x0 = zeros(sim_data.num_contacts*sim_data.β_dim)
    λ0 = zeros(1) # need to just support no eq. constraints
    μ0 = zeros(sim_data.num_contacts*(1+sim_data.β_dim))
    
    f = x̃ -> begin
        zs = vcat(c_ns',reshape(x̃,sim_data.β_dim,sim_data.num_contacts))
        obj = 0.
        for j = 1:sim_data.num_contacts
            obj += .5*zs[:,j]'*Fs[j]'*Qds[j]*Fs[j]*zs[:,j] + rds[j]'*Fs[j]*zs[:,j]
        end
        return obj
    end
    h = x̃ -> begin
        return [0]
    end
    g = x̃ -> begin
        x = reshape(x̃,sim_data.β_dim,sim_data.num_contacts)
        ineq_cons = []
        for j = 1:sim_data.num_contacts
            ineq_cons = vcat(ineq_cons, sum(x[:,j]) - sim_data.μs[j]*c_ns[j])
        end
        ineq_cons = vcat(ineq_cons, -x̃)
        return ineq_cons
    end

    if ip_method
        (x,λ,μ) = (ip_solve(x0,f,h,g,length(λ0),length(μ0)),λ0,μ0)
    else
        (x,λ,μ) = auglag_solve(x0,λ0,μ0,f,h,g,in_place=in_place,num_fosteps=1,num_sosteps=5)
    end

    τ = 0.
    zs = vcat(c_ns',reshape(x,sim_data.β_dim,sim_data.num_contacts))
    for j = 1:sim_data.num_contacts
        τ += Js[j]'*Fs[j]*zs[:,j]
    end

    return τ, x, λ, μ
end

function compute_bilevel_contact_params(sim_data,q0::AbstractArray{T},v0::AbstractArray{T},u0::AbstractArray{T},qnext::AbstractArray{M},vnext::AbstractArray{M}) where {T,M}
    
    x0 = MechanismState{T}(sim_data.mechanism)
    set_configuration!(x0,q0)
    set_velocity!(x0,v0)
    H = mass_matrix(x0)

    xnext = MechanismState{M}(sim_data.mechanism)
    set_configuration!(xnext, qnext)
    set_velocity!(xnext, vnext)

    rel_transforms = Vector{Tuple{Transform3D{M}, Transform3D{M}}}(undef, sim_data.num_contacts) # force transform, point transform
    geo_jacobians = Vector{GeometricJacobian{Matrix{M}}}(undef, sim_data.num_contacts)
    for i = 1:sim_data.num_contacts
        rel_transforms[i] = (relative_transform(xnext, sim_data.obstacles[i].contact_face.outward_normal.frame, sim_data.world_frame),
                            relative_transform(xnext, sim_data.contact_points[i].frame, sim_data.world_frame))
        geo_jacobians[i] = geometric_jacobian(xnext, sim_data.paths[i])
    end

    dyn_bias = dynamics_bias(xnext)

    contact_params = BilevelContactParams(H,dyn_bias,rel_transforms,geo_jacobians)

    return contact_params
end

function solve_contact_τ(sim_data,q0,v0,u0,qnext,vnext,c_ns;ip_method=false,in_place=false)
    contact_params = compute_bilevel_contact_params(sim_data,q0,v0,u0,qnext,vnext)
                    
    τ, x, λ, μ = solve_contact_τ(sim_data,contact_params.H,contact_params.dyn_bias,
                                 contact_params.rel_transforms,contact_params.geo_jacobians,v0,u0,c_ns,
                                 ip_method=ip_method,in_place=in_place)

    return τ, x, λ, μ
end