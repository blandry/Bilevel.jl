using ForwardDiff

D = Array{Union{Float64,ForwardDiff.Dual}}(undef, 5)

function f(x)
    D[1] = x[1]
    
    x'*x
end

display(f(ones(5)))
display(D)

display(ForwardDiff.gradient(f,ones(5)))
display(D)