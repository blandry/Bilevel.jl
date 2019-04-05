function softmax(x;k=1.)
    thresh = 500.
    if any(k*x .< -thresh)
        (log.(1. .+ exp.(k*x)))/k
    elseif any(k*x .> thresh)
        (k*x + log.(1. .+ exp.(-k*x)))/k
    else
        log.(exp.(k*x) .+ 1.)/k
    end
end

function gsoftmax(x;k=1.)
    thresh = 500.
    if any(k*x .> thresh)
        full(Diagonal(1. .- exp.(-k*x) ./ (1. .+ exp.(-k*x))))
    else
        full(Diagonal(exp.(k*x) ./ (exp.(k*x) .+ 1.)))
    end
end

function L(x,λ,f,h,c)
    f(x) - dot(λ,h(x)) + .5*c*dot(h(x),h(x))
end

function auglag(fun, num_eqs, num_ineqs, x0, options)
    num_fosteps = get(options, "num_fosteps", 1)
    num_sosteps = get(options, "num_sosteps", 3)
    c = get(options, "c", 1.)
    c_fos = get(options, "c_fos", 10.)
    c_sos = get(options, "c_sos", 10.)
    
    num_x = length(x0)
    rtol = eps(1.) * (num_eqs + num_ineqs)
    
    x = vcat(copy(x0), zeros(num_ineqs))
    λ = zeros(num_eqs + num_ineqs)
    
    for i = 1:num_fosteps
        # TODO inplace
        obj, eqs, ineqs, gobj, geqs, gineqs, Hobj = fun(x[1:num_x])
        
        ∇f = vcat(gobj, zeros(num_ineqs))
        Hf = zeros(num_x+num_ineqs, num_x+num_ineqs)
        Hf[1:num_x, 1:num_x] .= Hobj
        hx = vcat(eqs, ineqs + softmax(x[num_x+1:end]))
        ∇h = hcat(vcat(geqs, gineqs), vcat(zeros(num_eqs,num_ineqs),gsoftmax(x[num_x+1:end])))
        
        gL = ∇f - ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h

        # δx = (HL + (sqrt(sum(HL.^2))+rtol)*I) \ (-gL)
        # δx = (pinv(HL) + rtol*I) * (-gL)
        U,S,V = svd(HL)
        tol = rtol*maximum(S) # TODO not smooth
        ksig = 100.
        Sinv = 1. ./ (1. .+ exp.(-ksig*(S .- tol)/tol)) .* (1. ./ S)
        Sinv[isinf.(Sinv)] .= 0.
        HLpinv = V * (Diagonal(Sinv) * U')
        δx = HLpinv * (-gL)
        δλ = -c * hx

        x += δx
        λ += δλ

        c *= c_fos
    end
    
    for i = 1:num_sosteps
        # TODO inplace
        obj, eqs, ineqs, gobj, geqs, gineqs, Hobj = fun(x[1:num_x])

        ∇f = vcat(gobj, zeros(num_ineqs))
        Hf = zeros(num_x+num_ineqs, num_x+num_ineqs)
        Hf[1:num_x, 1:num_x] .= Hobj
        
        hx = vcat(eqs, ineqs + softmax(x[num_x+1:end]))
        ∇h = hcat(vcat(geqs, gineqs), vcat(zeros(num_eqs,num_ineqs),gsoftmax(x[num_x+1:end])))
    
        gL = ∇f - ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h
    
        A = vcat(hcat(HL,∇h'),hcat(∇h,zeros(num_eqs+num_ineqs,num_eqs+num_ineqs)))
        U,S,V = svd(A)
        tol = rtol*maximum(S) # TODO not smooth
        ksig = 100.
        Sinv = 1. ./ (1. .+ exp.(-ksig*(S .- tol)/tol)) .* (1. ./ S)
        Sinv[isinf.(Sinv)] .= 0.
        Apinv = V * (Diagonal(Sinv) * U')
    
        δxλ = Apinv * (-vcat(gL,hx))
    
        δx = δxλ[1:num_x+num_ineqs]
        δλ = δxλ[num_x+num_ineqs+1:num_x+num_ineqs+num_eqs+num_ineqs]
    
        x += δx
        λ += δλ
    
        c *= c_sos
    end
    
    obj, eqs, ineqs, gobj, geqs, gineqs, Hobj = fun(x[1:num_x])
    
    x[1:num_x], "Finished successfully"
end