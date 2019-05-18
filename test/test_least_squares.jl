using Bilevel
using ForwardDiff
using Bilevel: least_squares
using LinearAlgebra

n = 3
A = rand(n,n)

d = svd(A)
d.S[n] = d.S[n-1]
A = d.U*Diagonal(d.S)*d.V'

b = rand(n)
z0 = vcat(A[:],b)

function f(a)
    A = reshape(a[1:n^2],n,n) .* 2.
    b = a[n^2+1:end]
    x = least_squares(A,b)
    
    x
end

sol = f(z0)

# autodiff
J_auto = ForwardDiff.jacobian(f,z0)

# numerical
ϵ = sqrt(eps(1.))
J_num = zeros(size(J_auto))
for i = 1:length(z0)
    δ = zeros(length(z0))
    δ[i] = ϵ
    J_num[:,i] = (f(z0 + δ) .- sol)/ϵ
end

display(sol)
println("")
println("Auto")
display(J_auto)
println("")
display(maximum(abs.(J_auto)))
println("")
println("")
println("Finite diff")
display(J_num)
println("")
display(maximum(abs.(J_num)))
println("")
println("")

err = maximum(abs.(J_auto .- J_num))
display(err)
println("")