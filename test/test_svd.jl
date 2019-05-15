using Bilevel
using ForwardDiff
using Bilevel: svd, svd_finite
using LinearAlgebra

A = rand(3,3)
d = svd(A)
# d.S[3] = d.S[2]
A = d.U*Diagonal(d.S)*d.V'
z0 = A[:]
# z0 = rand(4)*200. .- 100.

function f(a)
    n = Int(sqrt(length(a)))
    A = reshape(a,n,n) .* 2.
    if eltype(A) <: ForwardDiff.Dual
        U,s,V = svd_finite(A)
    else
        U,s,V = svd(A)     
    end
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

display(sol)
println("")
display(J_auto)
println("")
display(J_num)
println("")

err = maximum(abs.(J_auto .- J_num))
display(err)
println("")
