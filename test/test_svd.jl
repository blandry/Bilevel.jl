using BilevelTrajOpt
using ForwardDiff

z0 = rand(4)*200. .- 100.

function f(a)
    n = Int(sqrt(length(a)))
    A = reshape(a,n,n)
    U,s,V = svd(A)
    return vcat(U[:],s[:],V[:])
end

sol = f(z0)

# autodiff 
J_auto = ForwardDiff.jacobian(f,z0)

# # numerical
ϵ = sqrt(eps(1.))
J_num = zeros(size(J_auto))
for i = 1:length(z0)
    δ = zeros(length(z0))
    δ[i] = ϵ
    J_num[:,i] = (f(z0 + δ) .- sol)/ϵ
end

println(sol)
println(J_auto)
println(J_num)

err = maximum(abs.(J_auto .- J_num))
println(err)
