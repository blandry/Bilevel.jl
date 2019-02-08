using Bilevel
using LinearAlgebra

a = rand(4)
A = Matrix(Diagonal(a))

# test the solution
function f(x)
    # return 0.
    # return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    # return x'*A*x 
    return sum(x)
end

function h(x)
    # return [40. - (x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)]
    # return [x[1] + x[2] - 1., x[3] - 1., x[4]*x[1]]
    return [6. .- x[1], x[1]*x[2], x[1]*.5 + x[3] - 4.]
end

function g(x)
    # return [25. - (x[1]*x[2]*x[3]*x[4])]
    # return vcat([25. - (x[1]*x[2]*x[3]*x[4])], 1. - x, x - 5.)
    # return vcat([1. - x[2]],-x)
    # return vcat(-x[2:3],5.-x[4])    
    # return [0.]
    # return vcat([x[1] - .1], 5. - x[4])
    return vcat(-x[2:3],5. - x[4],10. - x[4],17. - x[4],5. - x[1]*x[4])
end

x0 = zeros(4)
λ0 = zeros(length(h(x0)))
μ0 = zeros(length(g(x0)))
c0 = 1.

# x0 = x_sol
# λ0 = λ_sol
# μ0 = μ_sol
# c0 = c_sol

x_sol, λ_sol, μ_sol, c_sol = auglag_solve(x0,λ0,μ0,f,h,g,c0=c0)

num_h = length(λ0)
num_g = length(μ0)
x_sol_ip = ip_solve(x0,f,h,g,num_h,num_g)

# x_sol_known = [1.000, 4.743, 3.821, 1.379]

println(x_sol)
println(x_sol_ip)
# println(x_sol_known)