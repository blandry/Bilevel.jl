function contact_normal_τ_direct!(τ,sim_data::SimData,h,Hi,envj::EnvironmentJacobian,dyn_bias,u0,v0,x_upper::AbstractArray{U},n::Int) where U
    num_contacts = length(sim_data.env.contacts)
    env = sim_data.env
    lower_vs = sim_data.normal_vs[n]
    lower_cs = sim_data.normal_cs[n]
    lower_options = sim_data.normal_options[n]

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
        # TODO use the cs
        g = []

        for i = 1:num_contacts
            c_n = lower_vs(x, Symbol("c_n", i))
            g = vcat(g, -c_n)
            g = vcat(g, ϕAs[i]*vcat(c_n, zeros(4)) + ϕbs[i])
        end

        g
    end

    fres = DiffResults.HessianResult(zeros(U, lower_vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(U, lower_cs.num_cons), zeros(U, lower_vs.num_vars))
    solver_fn_ = generate_autodiff_solver_fn(eval_obj_,fres,eval_cons_,gres,lower_cs.eqs,lower_cs.ineqs)

    x0 = zeros(lower_vs.num_vars)

    xopt, info = auglag(solver_fn_, lower_cs.num_eqs, lower_cs.num_ineqs, x0, lower_options)

    # TODO include the total weight here, not in J
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, lower_vs(xopt, Symbol("c_n", i)), zeros(4))
    end

    xopt
end

function contact_normal_τ_direct_osqp!(τ,sim_data::SimData,h,Hi,envj::EnvironmentJacobian,dyn_bias,u0,v0,x_upper::AbstractArray{U},n::Int) where U
    num_contacts = length(sim_data.env.contacts)
    env = sim_data.env
    lower_vs = sim_data.normal_vs[n]
    lower_cs = sim_data.normal_cs[n]
    lower_options = sim_data.normal_options[n]

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
    
    xopt = Array{U,1}(undef, lower_vs.num_vars)
    P = ones(1,1)
    q = [0.]
    l = [0., -1e19]
    for i = 1:num_contacts
        A = Array{eltype(ϕAs[i]),2}(undef, 2, 1)
        A[1,1] = 1.
        A[2,1] = ϕAs[i][1,1]
        u = Array{eltype(ϕbs[i]),1}(undef, 2)
        u[1] = 1e19
        u[2] = -ϕbs[i][1]
        xopt_i = osqp(P,q,A,l,u,sim_data.normal_osqp_models[n])
        xopt[lower_vs(Symbol("c_n", i))] = xopt_i
    end

    # TODO include the total weight here, not in J
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, lower_vs(xopt, Symbol("c_n", i)), zeros(4))
    end

    xopt
end

function contact_friction_τ_direct!(τ,sim_data::SimData,h,Hi,envj::EnvironmentJacobian,dyn_bias,u0,v0,x_upper::AbstractArray{U},x_normal::AbstractArray{K},n::Int) where {U,K}
    num_contacts = length(sim_data.env.contacts)
    env = sim_data.env
    normal_vs = sim_data.normal_vs[n]
    lower_vs = sim_data.fric_vs[n]
    lower_cs = sim_data.fric_cs[n]
    lower_options = sim_data.fric_options[n]

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
        # TODO use the cs
        g = []

        for i = 1:num_contacts
            c_n = normal_vs(x_normal, Symbol("c_n", i))
            β = lower_vs(x, Symbol("β", i))
            g = vcat(g, -β)
            g = vcat(g, sum(β) .- env.contacts[i].obstacle.μ * c_n)
        end

        g
    end

    fres = DiffResults.HessianResult(zeros(U, lower_vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(U, lower_cs.num_cons), zeros(U, lower_vs.num_vars))
    solver_fn_ = generate_autodiff_solver_fn(eval_obj_,fres,eval_cons_,gres,lower_cs.eqs,lower_cs.ineqs)

    x0 = zeros(lower_vs.num_vars)

    xopt, info = auglag(solver_fn_, lower_cs.num_eqs, lower_cs.num_ineqs, x0, lower_options)

    # TODO include the total weight here, not in J
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, normal_vs(x_normal, Symbol("c_n", i)), lower_vs(xopt, Symbol("β", i)))
    end

    xopt
end

function contact_friction_τ_direct!(τ,sim_data::SimData,h,Hi,envj::EnvironmentJacobian,dyn_bias,u0,v0,x_upper::AbstractArray{U},n::Int) where U
    num_contacts = length(sim_data.env.contacts)
    env = sim_data.env
    upper_vs = sim_data.vs
    lower_vs = sim_data.fric_vs[n]
    lower_cs = sim_data.fric_cs[n]
    lower_options = sim_data.fric_options[n]

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
            c_n = upper_vs(x_upper, Symbol("c_n", i, "_", n))
            β = lower_vs(x, Symbol("β", i))
            z = vcat(c_n, β)
            obj += .5*z'*Qds[i]*z + z'*rds[i]
        end

        obj
    end

    function eval_cons_(x::AbstractArray{L}) where L
        # TODO use the cs
        g = []

        for i = 1:num_contacts
            c_n = upper_vs(x_upper, Symbol("c_n", i, "_", n))
            β = lower_vs(x, Symbol("β", i))
            g = vcat(g, -β)
            g = vcat(g, sum(β) .- env.contacts[i].obstacle.μ * c_n)
        end

        g
    end

    fres = DiffResults.HessianResult(zeros(U, lower_vs.num_vars))
    gres = DiffResults.JacobianResult(zeros(U, lower_cs.num_cons), zeros(U, lower_vs.num_vars))
    solver_fn_ = generate_autodiff_solver_fn(eval_obj_,fres,eval_cons_,gres,lower_cs.eqs,lower_cs.ineqs)

    x0 = zeros(lower_vs.num_vars)

    xopt, info = auglag(solver_fn_, lower_cs.num_eqs, lower_cs.num_ineqs, x0, lower_options)

    # TODO include the total weight here, not in J
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, upper_vs(x_upper, Symbol("c_n", i, "_", n)), lower_vs(xopt, Symbol("β", i)))
    end

    xopt
end

function contact_friction_τ_direct_osqp!(τ,sim_data::SimData,h,Hi,envj::EnvironmentJacobian,dyn_bias,u0,v0,x_upper::AbstractArray{U},x_normal::AbstractArray{K},n::Int) where {U,K}
    num_contacts = length(sim_data.env.contacts)
    env = sim_data.env
    normal_vs = sim_data.normal_vs[n]
    lower_vs = sim_data.fric_vs[n]
    lower_cs = sim_data.fric_cs[n]
    lower_options = sim_data.fric_options[n]

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

    xopt = Array{U,1}(undef, lower_vs.num_vars)
    # TODO use the β selector
    β_max = 1e8
    A = [1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.; 1. 1. 1. 1.]
    l = [0., 0., 0., 0., -4. * β_max]
    for i = 1:num_contacts
        c_n = normal_vs(x_normal, Symbol("c_n", i))[1]
        P = Qds[i][2:end,2:end]
        P = .5 * P' * P
        P[abs.(P) .< 1e-6] .= 0.
        P += I*1e-8
        q = (rds[i][2:end] + .5*Qds[i][1,2:end]*c_n + .5*Qds[i][2:end,1]*c_n)
        u = Array{eltype(x_normal), 1}([β_max, β_max, β_max, β_max, env.contacts[i].obstacle.μ*c_n])
        xopt_i = osqp(P,q,A,l,u,sim_data.fric_osqp_models[n])
        xopt[lower_vs(Symbol("β", i))] = xopt_i
    end

    # TODO include the total weight here, not in J
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, normal_vs(x_normal, Symbol("c_n", i)), lower_vs(xopt, Symbol("β", i)))
    end

    xopt
end

function contact_friction_τ_direct_osqp!(τ,sim_data::SimData,h,Hi,envj::EnvironmentJacobian,dyn_bias,u0,v0,x_upper::AbstractArray{U},n::Int) where U
    num_contacts = length(sim_data.env.contacts)
    env = sim_data.env
    upper_vs = sim_data.vs
    lower_vs = sim_data.fric_vs[n]
    lower_cs = sim_data.fric_cs[n]
    lower_options = sim_data.fric_options[n]

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

    xopt = Array{U,1}(undef, lower_vs.num_vars)
    # TODO use the β selector
    β_max = 1e8
    A = [1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.; 1. 1. 1. 1.]
    l = [0., 0., 0., 0., -4. * β_max]
    for i = 1:num_contacts
        c_n = upper_vs(x_upper, Symbol("c_n", i, "_", n))[1]
        P = Qds[i][2:end,2:end]
        P = .5 * P' * P
        P[abs.(P) .< 1e-6] .= 0.
        P += I*1e-8
        q = (rds[i][2:end] + .5*Qds[i][1,2:end]*c_n + .5*Qds[i][2:end,1]*c_n)
        u = Array{eltype(x_upper), 1}([β_max, β_max, β_max, β_max, env.contacts[i].obstacle.μ*c_n])
        xopt_i = osqp(P,q,A,l,u,sim_data.fric_osqp_models[n])
        xopt[lower_vs(Symbol("β", i))] = xopt_i
    end

    # TODO include the total weight here, not in J
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, upper_vs(x_upper, Symbol("c_n", i, "_", n)), lower_vs(xopt, Symbol("β", i)))
    end

    xopt
end

function contact_τ_indirect!(τ::AbstractArray{T},sim_data::SimData,envj::EnvironmentJacobian{T},x::AbstractArray{T}) where T
    # TODO: parallel
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, sim_data.vs(x, Symbol("c_n", i)), sim_data.vs(x, Symbol("β", i)))
    end
end

function contact_τ_indirect!(τ::AbstractArray{T},sim_data::SimData,envj::EnvironmentJacobian{T},x::AbstractArray{T},n::Int) where T
    # TODO: parallel
    τ .= mapreduce(+, enumerate(envj.contact_jacobians)) do (i,cj)
        contact_τ(cj, sim_data.vs(x, Symbol("c_n", i, "_", n)), sim_data.vs(x, Symbol("β", i, "_", n)))
    end
end
