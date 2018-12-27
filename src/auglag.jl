function softmax(x,y;k=1.)
    # safe for large x inputs only
    thresh = 100.
    if x < -thresh
        (k*y + log.(1. + exp.(k*x - k*y)))/k
    elseif x > thresh
        (k*x + log.(1. + exp.(k*y - k*x)))/k
    else
        log.(exp.(k*x) + exp.(k*y))/k
    end
end

function L(x,λ,h,c)
    f(x) - dot(λ,h(x)) + .5*c*dot(h(x),h(x))
end

function auglag_solve(x0,λ0,μ0,f0,h0,g0,x_min,x_max)
    num_steps = 5
    c = 1.
    k = 100.

    num_h0 = length(λ0)
    num_g0 = length(μ0)
    num_x0 = length(x0)

    x = vcat(x0,zeros(num_g0))
    num_x = length(x)
    λ = vcat(λ0,μ0)
    num_h = length(λ)

    h = x̃ -> vcat(h0(x̃[1:num_x0]),g0(x̃[1:num_x0]) + x̃[num_x0+1:num_x0+num_g0].^2)
    f = x̃ -> f0(x̃[1:num_x0])

    for i = 1:num_steps
        hx = h(x)
        ∇h = ForwardDiff.jacobian(h,x)
        ∇f = ForwardDiff.gradient(f,x)
        Hf = ForwardDiff.hessian(f,x)
        gL = ∇f + ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h

        A = vcat(hcat(HL,∇h'),hcat(∇h,zeros(num_h,num_h)))
        δ = (A+eye(num_x+num_h)*1e-12)\(-vcat(gL,hx))
        x += δ[1:num_x]
        λ += δ[num_x+1:num_x+num_h]

        x[1:num_x0] = -softmax.(-x[1:num_x0],-x_max,k=k)
        x[1:num_x0] = softmax.(x[1:num_x0],x_min,k=k)
    end

    x[1:num_x0], λ[1:num_h0], λ[num_h0+1:num_h0+num_g0]
end
