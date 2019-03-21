mutable struct TrajData
    Δt
    mechanism
    num_q
    num_v
    num_contacts
    β_dim
    world
    world_frame
    total_weight
    bodies
    contact_points
    obstacles
    μs
    paths
    surface_paths
    Ds
    β_selector
    λ_selector
    c_n_selector
    N
    num_slack
    num_xn
    implicit_contact
    num_kin
    num_dyn
    num_comp
    num_dist
    num_pos
    num_dyn_eq
    num_dyn_ineq
    num_x
    num_eq
    num_ineq
    state_eq
    fn_ineq
    fn_obj
    min_τ
    min_v
end

function add_state_eq!(traj_data, con_fn, con_indx; con_len=nothing)
    if isa(con_len,Nothing)
        con_len = traj_data.num_q + traj_data.num_v
    end
    push!(traj_data.state_eq, (con_fn,con_indx,con_len))
    traj_data.num_eq += con_len
end

function add_fn_ineq!(traj_data, con_fn, con_indx)
    push!(traj_data.fn_ineq, (con_fn,con_indx))
    traj_data.num_ineq += 1
end

function add_fn_obj!(traj_data, obj_fn, obj_indx)
    push!(traj_data.fn_obj, (obj_fn, obj_indx))
end

function get_traj_data(mechanism::Mechanism,
                       env::Environment,
                       Δt::Real,
                       N::Int,
                       implicit_contact::Bool;
                       min_τ=true,
                       min_v=false)

    num_q = num_positions(mechanism)
    num_v = num_velocities(mechanism)
    num_contacts = length(env.contacts)
    if num_contacts > 0
        β_dim = length(contact_basis(env.contacts[1][3]))
    else
        β_dim = Int(0)
    end

    # some constants throughout the trajectory
    world = root_body(mechanism)
    world_frame = default_frame(world)
    total_weight = mass(mechanism) * norm(mechanism.gravitational_acceleration)
    bodies = []
    contact_points = []
    obstacles = []
    μs = []
    paths = []
    surface_paths = []
    Ds = []
    for (body, contact_point, obstacle) in env.contacts
      push!(bodies, body)
      push!(contact_points, contact_point)
      push!(obstacles, obstacle)
      push!(μs, obstacle.μ)
      push!(paths, path(mechanism, body, world))
      if obstacle.is_floating
          push!(surface_paths, path(mechanism, obstacle.body, world))
      else
          push!(surface_paths, nothing)
      end
      push!(Ds, contact_basis(obstacle))
    end
    β_selector = findall(x->x!=0,repeat(vcat(ones(β_dim),[0,0]),num_contacts))
    λ_selector = findall(x->x!=0,repeat(vcat(zeros(β_dim),[1,0]),num_contacts))
    c_n_selector = findall(x->x!=0,repeat(vcat(zeros(β_dim),[0,1]),num_contacts))

    if implicit_contact
        num_slack = 0
        num_xn = num_q+num_v+num_v+num_slack
    else
        num_slack = 0
        num_xn = num_q+num_v+num_v+num_slack+num_contacts*(2+β_dim)
    end

    num_kin = num_q
    num_dyn = num_v
    num_comp = num_contacts*(2+β_dim)
    num_dist = num_contacts
    num_pos = num_contacts*(1+β_dim) + 2*num_contacts*(2+β_dim)

    # num_dyn_eq = num_kin+num_dyn
    # num_dyn_ineq = num_comp+num_dist+num_pos
    num_dyn_eq = num_kin+num_dyn+num_comp
    num_dyn_ineq = num_dist+num_pos

    num_x = N*num_xn
    num_eq = (N-1)*num_dyn_eq
    num_ineq = (N-1)*num_dyn_ineq

    state_eq = Vector()
    fn_ineq = Vector()
    fn_obj = Vector()

    traj_data = TrajData(Δt,mechanism,num_q,num_v,num_contacts,β_dim,
                         world,world_frame,total_weight,
                         bodies,contact_points,obstacles,μs,paths,surface_paths,
                         Ds,β_selector,λ_selector,c_n_selector,
                         N,num_slack,num_xn,implicit_contact,
                         num_kin,num_dyn,num_comp,num_dist,num_pos,
                         num_dyn_eq,num_dyn_ineq,num_x,num_eq,num_ineq,state_eq,
                         fn_ineq,fn_obj,min_τ,min_v)

    traj_data
end

mutable struct TrajDataBilevel
    Δt
    mechanism
    num_q
    num_v
    num_contacts
    β_dim
    world
    world_frame
    total_weight
    bodies
    contact_points
    obstacles
    μs
    paths
    surface_paths
    Ds
    N
    num_slack
    num_xn
    num_dyn_eq
    num_dyn_ineq
    num_x
    num_eq
    num_ineq
    state_eq
    fn_ineq
    fn_obj
    min_τ
    min_v
end

function get_traj_data_bilevel(mechanism::Mechanism,
                       env::Environment,
                       Δt::Real,
                       N::Int;
                       min_τ=true,
                       min_v=false)

    num_q = num_positions(mechanism)
    num_v = num_velocities(mechanism)
    num_contacts = length(env.contacts)
    if num_contacts > 0
        β_dim = length(contact_basis(env.contacts[1][3]))
    else
        β_dim = Int(0)
    end

    # some constants throughout the trajectory
    world = root_body(mechanism)
    world_frame = default_frame(world)
    total_weight = mass(mechanism) * norm(mechanism.gravitational_acceleration)
    bodies = []
    contact_points = []
    obstacles = []
    μs = []
    paths = []
    surface_paths = []
    Ds = []
    for (body, contact_point, obstacle) in env.contacts
      push!(bodies, body)
      push!(contact_points, contact_point)
      push!(obstacles, obstacle)
      push!(μs, obstacle.μ)
      push!(paths, path(mechanism, body, world))
      if obstacle.is_floating
          push!(surface_paths, path(mechanism, obstacle.body, world))
      else
          push!(surface_paths, nothing)
      end
      push!(Ds, hcat(map(d->d.v,contact_basis(obstacle))...))
    end

    num_slack = 0
    num_xn = num_q+num_v+num_v+num_contacts+num_slack

    num_dyn_eq = num_q+num_v+num_contacts
    num_dyn_ineq = 2*num_contacts

    num_x = N*num_xn
    num_eq = (N-1)*num_dyn_eq
    num_ineq = (N-1)*num_dyn_ineq

    state_eq = Vector()
    fn_ineq = Vector()
    fn_obj = Vector()

    traj_data = TrajDataBilevel(Δt,mechanism,num_q,num_v,num_contacts,β_dim,
                                world,world_frame,total_weight,
                                bodies,contact_points,obstacles,μs,paths,surface_paths,
                                Ds,N,num_slack,num_xn,num_dyn_eq,num_dyn_ineq,num_x,num_eq,num_ineq,
                                state_eq,fn_ineq,fn_obj,min_τ,min_v)

    traj_data
end
