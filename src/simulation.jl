struct SimData
    Δt
    mechanism
    num_q
    num_v
    num_contacts
    β_dim
    num_x
    num_slack
    num_h
    num_g
    world
    world_frame
    total_weight
    bodies
    contact_points
    obstacles
    μs
    paths
    Ds
    β_selector
    λ_selector
    c_n_selector
end

function get_sim_data(state0::MechanismState,
                      env::Environment,
                      Δt::Real,
                      implicit_contact::Bool)

    mechanism = state0.mechanism

    num_q = num_positions(state0)
    num_v = num_velocities(state0)
    num_contacts = length(env.contacts)
    β_dim = length(contact_basis(env.contacts[1][3]))
    # x = [q, v, slack, β1, λ1, c_n1, β2, λ2, c_n2...]
    if implicit_contact
      num_slack = num_v
      num_x = num_q + num_v + num_slack
      num_h = num_q + num_v
      num_g = num_contacts
    else
      num_slack = 1
      num_x = num_q + num_v + num_slack + num_contacts*(2+β_dim)
      num_h = num_q + num_v
      num_g = num_contacts + num_contacts*(2+β_dim) + num_contacts*(1+β_dim)
    end

    # some constants throughout the simulation
    world = root_body(mechanism)
    world_frame = default_frame(world)
    total_weight = mass(mechanism) * norm(mechanism.gravitational_acceleration)
    bodies = []
    contact_points = []
    obstacles = []
    μs = []
    paths = []
    Ds = []
    for (body, contact_point, obstacle) in env.contacts
      push!(bodies, body)
      push!(contact_points, contact_point)
      push!(obstacles, obstacle)
      push!(μs, obstacle.μ)
      push!(paths, path(mechanism, body, world))
      push!(Ds, contact_basis(obstacle))
    end
    β_selector = findall(x->x!=0,repeat(vcat(ones(β_dim),[0,0]),num_contacts))
    λ_selector = findall(x->x!=0,repeat(vcat(zeros(β_dim),[1,0]),num_contacts))
    c_n_selector = findall(x->x!=0,repeat(vcat(zeros(β_dim),[0,1]),num_contacts))

    sim_data = SimData(Δt,mechanism,num_q,num_v,num_contacts,β_dim,num_x,num_slack,num_h,num_g,
                       world,world_frame,total_weight,
                       bodies,contact_points,obstacles,μs,paths,Ds,
                       β_selector,λ_selector,c_n_selector)

    sim_data
end
