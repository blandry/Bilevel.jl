using BilevelTrajOpt
using ForwardDiff

# test the gradient
function solve_prob(z)
    function f(x)
        # return x'*A*x + z[1]
        return sum(x)
    end

    function h(x)
        # return [x[1] + x[2] - 1., x[3] - z[2], x[4]*x[1]]
        return [z[1] - x[1], x[1]*x[2], x[1]*.5 + x[3] - z[2]]
    end

    function g(x)
        # return vcat([x[1] - z[4]], z[3] - x[4])
        return vcat(-x[2:3], z[3] - x[4], 10. - x[4], z[4] - x[4], 5. - x[1]*x[4])
    end

    x0 = zeros(4)
    λ0 = zeros(length(h(x0)))
    μ0 = zeros(length(g(x0)))

    # x_sol_ip = ip_solve(x0,f,h,g,length(h(x0)),length(g(x0)))
    # println(x_sol_ip)
    
    x_sol, λ_sol, μ_sol, c_sol = auglag_solve(x0,λ0,μ0,f,h,g)
    # println(x_sol)

    x_sol
end

# z0 = [2.,-1.,0.,1.]
z0 = [6.,4.,5.,17.]

sol = solve_prob(z0)

# autodiff 
J_auto = ForwardDiff.jacobian(solve_prob,z0)

# # numerical
ϵ = sqrt(eps(1.))
J_num = zeros(size(J_auto))
for i = 1:length(z0)
    δ = zeros(length(z0))
    δ[i] = ϵ
    J_num[:,i] = (solve_prob(z0 + δ) - sol) ./ ϵ
end

println("----")
println(sol)
println(J_auto)
println(J_num)