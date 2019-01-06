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

function auglag_solve(x0,λ0,μ0,f0,h0,g0;c0=1.)
    num_fosteps = 3
    num_sosteps = 2

    num_h0 = length(λ0)
    num_g0 = length(μ0)
    num_x0 = length(x0)

    x = vcat(copy(x0),zeros(num_g0))
    λ = vcat(copy(λ0),copy(μ0))
    c = c0
    
    num_x = length(x)
    num_h = length(λ)

    h = x̃ -> vcat(h0(x̃[1:num_x0]),
                  g0(x̃[1:num_x0]) + softmax.(x̃[num_x0+1:num_x0+num_g0],0.,k=1.))
    
    f = x̃ -> f0(x̃[1:num_x0])
        
    for i = 1:num_fosteps
        hx = h(x)
        ∇h = ForwardDiff.jacobian(h,x)
        ∇f = ForwardDiff.gradient(f,x)
        Hf = ForwardDiff.hessian(f,x)
        gL = ∇f - ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h

        δx = (HL + (sum(HL.^2)+1e-12)*eye(num_x)) \ (-gL)
        δλ = -c * hx
        
        x += δx
        λ += δλ

        c *= 10.

        display(x)
    end
        
    for i = 1:num_sosteps
        hx = h(x)
        ∇h = ForwardDiff.jacobian(h,x)
        ∇f = ForwardDiff.gradient(f,x)
        Hf = ForwardDiff.hessian(f,x)
        gL = ∇f - ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h     
    
        A = vcat(hcat(HL,∇h'),hcat(∇h,zeros(num_h,num_h)))
        # δxλ = (A + 1e-12*eye(num_x+num_h)) \ (-vcat(gL,hx))
        δxλ = pinv(A) * (-vcat(gL,hx))
        δx = δxλ[1:num_x]
        δλ = δxλ[num_x+1:num_x+num_h]

        x += δx
        λ += δλ

        c *= 1.

        display(x)
    end

    x[1:num_x0], λ[1:num_h0], λ[num_h0+1:num_h0+num_g0], c
end

# δ = Variable(num_x)
# SQP
# sqp_obj = ∇f'*δ + .5*quadform(δ,HL)
# sqp_con = [hx + ∇h*δ == 0.]
# sqp_prob = minimize(sqp_obj, sqp_con)
# solve!(sqp_prob)
# δx = δ.value
# δλ = 0.