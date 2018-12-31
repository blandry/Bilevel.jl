function softmax(x,y;k=1.)
    # safe for large x inputs only
    thresh = 500.
    if k*x < -thresh
        (k*y + log.(1. + exp.(k*x - k*y)))/k
    elseif k*x > thresh
        (k*x + log.(1. + exp.(k*y - k*x)))/k
    else
        log.(exp.(k*x) + exp.(k*y))/k
    end
end

function L(x,λ,f,h,c)
    f(x) - dot(λ,h(x)) + .5*c*dot(h(x),h(x))
end

function auglag_solve(x0,λ0,μ0,f0,h0,g0,x_min,x_max)
    num_steps = 5
    c = 1.
    # k = 100.

    num_h0 = length(λ0)
    num_g0 = length(μ0)
    num_x0 = length(x0)

    x = vcat(x0,zeros(num_g0))
    num_x = length(x)
    λ = vcat(λ0,μ0)
    num_h = length(λ)

    h = x̃ -> vcat(h0(x̃[1:num_x0]),g0(x̃[1:num_x0]) + x̃[num_x0+1:num_x0+num_g0].*x̃[num_x0+1:num_x0+num_g0])
    f = x̃ -> f0(x̃[1:num_x0]) + 1e-12*.5*dot(x̃[num_x0+1:num_x0+num_g0],x̃[num_x0+1:num_x0+num_g0])

    for i = 1:num_steps
        hx = h(x)

        ∇h = ForwardDiff.jacobian(h,x)
        ∇f = ForwardDiff.gradient(f,x)
        Hf = ForwardDiff.hessian(f,x)

        gL = ∇f - ∇h'*λ + c*∇h'*hx
        # display(gL)
        # display(ForwardDiff.gradient(x̃->L(x̃,λ,f,h,c),x))

        HL = Hf + c*∇h'*∇h
        # display(HL)
        # display(ForwardDiff.hessian(x̃->L(x̃,λ,f,h,c),x))

        τ = 1./c #1e-12
        # A = vcat(hcat(HL,∇h'),hcat(∇h,zeros(num_h,num_h)))
        # δ = (A+τ*eye(num_x+num_h))\(-vcat(gL,hx))
        A = vcat(hcat(HL,∇h'),hcat(∇h,τ*eye(num_h)))
        δ = A\(-vcat(gL,hx))

        x += δ[1:num_x]
        λ += δ[num_x+1:num_x+num_h]

        # x[1:num_x0] = -softmax.(-x[1:num_x0],-x_max,k=k)
        # x[1:num_x0] = softmax.(x[1:num_x0],x_min,k=k)
        x[1:num_x0] = min.(x[1:num_x0],x_max)
        x[1:num_x0] = max.(x[1:num_x0],x_min)

        # c *= 2.
    end

    x[1:num_x0], λ[1:num_h0], λ[num_h0+1:num_h0+num_g0]
end
