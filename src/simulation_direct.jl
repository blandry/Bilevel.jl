function get_sim_data_direct(mechanism::Mechanism,env::Environment,Δt::Real;
                             relax_comp=false)
    num_contacts = length(env.contacts)

    vs = VariableSelector()
    add_var!(vs, :qnext, num_positions(mechanism))
    add_var!(vs, :vnext, num_velocities(mechanism))
    if relax_comp
        add_var!(vs, :slack, 1)
    end

    cs = ConstraintSelector()
    add_eq!(cs, :kin, num_positions(mechanism))
    add_eq!(cs, :dyn, num_velocities(mechanism))
    
    x0_cache = StateCache(mechanism)
    xn_cache = StateCache(mechanism)
    envj_cache = EnvironmentJacobianCache(env)

    generate_solver_fn = :generate_solver_fn_sim_direct
    extract_sol = :extract_sol_sim_direct

    normal_vs = VariableSelector()
    for i = 1:length(env.contacts)
        add_var!(normal_vs, Symbol("c_n", i), 1)
    end

    normal_cs = ConstraintSelector()
    for i = 1:length(env.contacts)
        add_ineq!(normal_cs, Symbol("c_n_pos", i), 1)
        add_ineq!(normal_cs, Symbol("ϕ", i), 1)
    end
   
    normal_options = Dict{String, Any}()
    normal_options["num_fosteps"] = 1
    normal_options["num_sosteps"] = 10
    normal_options["c"] = 10
    normal_options["c_fos"] = 10
    normal_options["c_sos"] = 1
    
    fric_vs = VariableSelector()
    for i = 1:length(env.contacts)
        β_dim = size(env.contacts[i].obstacle.basis,2)
        add_var!(fric_vs, Symbol("β", i), β_dim)
    end

    fric_cs = ConstraintSelector()
    for i = 1:length(env.contacts)
        β_dim = size(env.contacts[i].obstacle.basis,2)
        add_ineq!(fric_cs, Symbol("β_pos", i), β_dim)
        add_ineq!(fric_cs, Symbol("fric_cone", i), 1)
    end
   
    fric_options = Dict{String, Any}()
    fric_options["num_fosteps"] = 1
    fric_options["num_sosteps"] = 10
    fric_options["c"] = 10
    fric_options["c_fos"] = 10
    fric_options["c_sos"] = 1
    
    SimData(mechanism,env,
            x0_cache,xn_cache,envj_cache,
            Δt,vs,cs,generate_solver_fn,extract_sol,
            [],[],[],[normal_vs],[normal_cs],[normal_options],[fric_vs],[fric_cs],[fric_options],1,[],[])
end

function extract_sol_sim_direct(sim_data::SimData, results::AbstractArray{T,2}) where T    
    vs = sim_data.vs
    relax_comp = haskey(vs.vars, :slack)
    env = sim_data.env
    N = size(results,2)

    qtraj = []
    vtraj = []
    utraj = []
    contact_traj = []
    slack_traj = []
    for n = 1:N
        push!(qtraj, vs(results[:,n], Symbol("qnext")))
        push!(vtraj, vs(results[:,n], Symbol("vnext")))
        if relax_comp
            push!(slack_traj, vs(results[:,n], Symbol("slack")))
        end
        # TODO compute contact force for comparison purposes
    end
    
    # some other usefull vectors
    ttraj = [(i-1)*sim_data.Δt for i = 1:N]
    qv_mat = vcat(hcat(qtraj...),hcat(vtraj...))

    qtraj, vtraj, utraj, contact_traj, slack_traj, ttraj, qv_mat
end

function contact_normal_τ_direct!(τ,sim_data::SimData,Hi,envj::EnvironmentJacobian,dyn_bias,u0,v0,x_upper::AbstractArray{U},n::Int) where U
    num_contacts = length(sim_data.env.contacts)
    env = sim_data.env
    lower_vs = sim_data.normal_vs[n]
    lower_cs = sim_data.normal_cs[n]
    lower_options = sim_data.normal_options[n]
    h = sim_data.Δt
    
    ϕAs = []
    ϕbs = []
    for i = 1:num_contacts
        J = envj.contact_jacobians[i].J
        ϕ = envj.contact_jacobians[i].ϕ
        N = envj.contact_jacobians[i].N
    
        ϕA = h^2*N*Hi*J
        ϕb = N*(h^2*Hi*(dyn_bias - u0) .- h*v0) .- ϕ 

        push!(ϕAs,ϕA)
        push!(ϕbs,ϕb)
    end
    
    function eval_obj_(x::AbstractArray{L}) where L
        obj = 0.

        for i = 1:num_contacts
            c_n = lower_vs(x, Symbol("c_n", i))
            obj += c_n'*c_n
        end

        obj
    end

    function eval_cons_(x::AbstractArray{L}) where L
        # g = zeros(ForwardDiff.Dual, lower_cs.num_eqs + lower_cs.num_ineqs)
        # g = zeros(Real, lower_cs.num_eqs + lower_cs.num_ineqs)
        g = []

        for i = 1:num_contacts
            c_n = lower_vs(x, Symbol("c_n", i))
            # g[lower_cs(Symbol("c_n_pos", i))] .= -c_n
            # g[lower_cs(Symbol("ϕ", i))] .= ϕAs[i]*vcat(c_n, zeros(4)) + ϕbs[i] # TODO use β_dim
            g = vcat(g, -c_n)
            g = vcat(g, ϕAs[i]*vcat(c_n, zeros(4)) + ϕbs[i])
        end
        
        g
    end

    fres = DiffResults.HessianResult(zeros(U, lower_vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(U, lower_cs.num_cons), zeros(U, lower_vs.num_vars))
    solver_fn_ = generate_autodiff_solver_fn(eval_obj_,fres,eval_cons_,gres,lower_cs.eqs,lower_cs.ineqs)
    # solver_fn_ = generate_autodiff_solver_fn(eval_obj_,eval_cons_,lower_cs.eqs,lower_cs.ineqs)

    x0 = zeros(lower_vs.num_vars)

    xopt, info = auglag(solver_fn_, lower_cs.num_eqs, lower_cs.num_ineqs, x0, lower_options)
    
    # TODO include the total weight here, not in J
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, lower_vs(xopt, Symbol("c_n", i)), zeros(4))
    end

    xopt
end

function contact_friction_τ_direct!(τ,sim_data::SimData,Hi,envj::EnvironmentJacobian,dyn_bias,u0,v0,x_upper::AbstractArray{U},x_normal,n::Int) where U
    num_contacts = length(sim_data.env.contacts)
    env = sim_data.env
    normal_vs = sim_data.normal_vs[n]
    lower_vs = sim_data.fric_vs[n]
    lower_cs = sim_data.fric_cs[n]
    lower_options = sim_data.fric_options[n]
    h = sim_data.Δt
    
    Qds = []
    rds = []
    for i = 1:num_contacts
        J = envj.contact_jacobians[i].J
        ϕ = envj.contact_jacobians[i].ϕ
        N = envj.contact_jacobians[i].N
        
        Qd = h^2*J'*Hi*J
        rd = J'*(h^2*Hi*(dyn_bias - u0) .- h*v0)

        push!(Qds,Qd)
        push!(rds,rd)
    end

    function eval_obj_(x::AbstractArray{L}) where L
        obj = 0.

        for i = 1:num_contacts
            c_n = normal_vs(x_normal, Symbol("c_n", i))
            β = lower_vs(x, Symbol("β", i))
            z = vcat(c_n, β)
            obj += .5*z'*Qds[i]*z + z'*rds[i]
        end

        obj
    end

    function eval_cons_(x::AbstractArray{L}) where L
        # g = zeros(ForwardDiff.Dual, lower_cs.num_eqs + lower_cs.num_ineqs)
        # g = zeros(Real, lower_cs.num_eqs + lower_cs.num_ineqs)
        g = []
        
        for i = 1:num_contacts
            c_n = normal_vs(x_normal, Symbol("c_n", i))
            β = lower_vs(x, Symbol("β", i))
            # g[lower_cs(Symbol("β_pos", i))] .= -β
            # g[lower_cs(Symbol("fric_cone", i))] .= sum(β) .- env.contacts[i].obstacle.μ * c_n
            g = vcat(g, -β)
            g = vcat(g, sum(β) .- env.contacts[i].obstacle.μ * c_n)
        end
        
        g
    end

    fres = DiffResults.HessianResult(zeros(U, lower_vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(U, lower_cs.num_cons), zeros(U, lower_vs.num_vars))
    solver_fn_ = generate_autodiff_solver_fn(eval_obj_,fres,eval_cons_,gres,lower_cs.eqs,lower_cs.ineqs)
    # solver_fn_ = generate_autodiff_solver_fn(eval_obj_,eval_cons_,lower_cs.eqs,lower_cs.ineqs)

    x0 = zeros(lower_vs.num_vars)

    xopt, info = auglag(solver_fn_, lower_cs.num_eqs, lower_cs.num_ineqs, x0, lower_options)
    
    # TODO include the total weight here, not in J
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, normal_vs(x_normal, Symbol("c_n", i)), lower_vs(xopt, Symbol("β", i)))
    end

    xopt
end

function generate_solver_fn_sim_direct(sim_data,q0,v0,u0)
    x0 = sim_data.x0_cache[Float64]
    envj = sim_data.envj_cache[Float64]
    Δt = sim_data.Δt
    vs = sim_data.vs
    cs = sim_data.cs
    
    relax_comp = haskey(vs.vars, :slack)
    num_contacts = length(sim_data.env.contacts)
    num_vel = num_velocities(sim_data.mechanism)

    set_configuration!(x0, q0)
    set_velocity!(x0, v0)
    H = mass_matrix(x0)
    Hi = inv(H)
    
    contact_jacobian!(envj, x0)
    dyn_bias0 = dynamics_bias(x0) # TODO preallocate

    function eval_obj(x::AbstractArray{T}) where T
        f = 0.
    
        if relax_comp
            slack = vs(x, :slack)
            f += .5 * slack' * slack
        end
    
        f
    end

    function eval_cons(x::AbstractArray{T}) where T
        xn = sim_data.xn_cache[T]
        
        normal_bias = Vector{T}(undef, num_vel)
        contact_bias = Vector{T}(undef, num_vel)
        g = Vector{T}(undef, cs.num_eqs + cs.num_ineqs) # TODO preallocate

        qnext = vs(x, :qnext)
        vnext = vs(x, :vnext)
        if relax_comp
            slack = vs(x, :slack)
        end
        
        set_configuration!(xn, qnext)
        set_velocity!(xn, vnext)
        
        if (num_contacts > 0)
            # compute normal forces
            x_normal = contact_normal_τ_direct!(normal_bias, sim_data, Hi, envj, dyn_bias0, u0, v0, x, 1)
                        
            # compute friction forces
            contact_friction_τ_direct!(contact_bias, sim_data, Hi, envj, dyn_bias0, u0, v0, x, x_normal, 1)
        end
        config_derivative = configuration_derivative(xn) # TODO preallocate
        dyn_bias = dynamics_bias(xn) # TODO preallocate

        g[cs(:kin)] .= qnext .- q0 .- Δt .* config_derivative
        g[cs(:dyn)] .= H * (vnext - v0) .- Δt .* (u0 .- dyn_bias .- contact_bias)

        g
    end
        
    generate_autodiff_solver_fn(eval_obj,eval_cons,cs.eqs,cs.ineqs,vs.num_vars)
end

# # usefull to tune the lower solver
# solver_fn_snopt = generate_autodiff_solver_fn(eval_obj_,eval_cons_,lower_cs.eqs,lower_cs.ineqs)
# options_snopt = Dict{String, Any}()
# options_snopt["Derivative option"] = 1
# options_snopt["Verify level"] = -1 # -1 => 0ff, 0 => cheap
# xopt_snopt, info_snopt = snopt(solver_fn_snopt, lower_cs.num_eqs, lower_cs.num_ineqs, x0, options_snopt)
# display(info_snopt)
# τ_snopt = zeros(length(τ))
# τ_snopt .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
#     contact_τ(cj, lower_vs(xopt_snopt, Symbol("c_n", i)), zeros(4))
# end
# display("snopt")
# display(xopt_snopt)
# display(τ_snopt)
# display("auglag")
# display(xopt)
# display(τ)