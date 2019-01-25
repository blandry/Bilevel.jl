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

function auglag_solve(x0,λ0,μ0,f0,h0,g0;c0=1.,in_place=true)
    num_fosteps = 1
    num_sosteps = 9

    num_h0 = length(λ0)
    num_g0 = length(μ0)
    num_x0 = length(x0)

    # note that since this is recursive, the type of x changes after 1 iteration
    # that needs to tbe taken into account
    # TODO need to find the common type across f,h and g
    if in_place
        h0x = h0(x0) # todo useless call
        x = Array{eltype(h0x),1}(undef, num_x0+num_g0)
        x[1:num_x0] = x0[:]
        x[num_x0+1:num_x0+num_g0] .= 0.
        λ = Array{eltype(h0x),1}(undef, num_h0+num_g0)
        λ[1:num_h0] = λ0[:]
        λ[num_h0+1:num_h0+num_g0] = μ0[:]
    else
        x = vcat(copy(x0), zeros(num_g0))
        λ = vcat(copy(λ0), copy(μ0))
    end
    c = c0

    num_x = length(x)
    num_h = length(λ)

    h = x̃ -> vcat(h0(x̃[1:num_x0]),
                  g0(x̃[1:num_x0]) + softmax.(x̃[num_x0+1:num_x0+num_g0],0.,k=1.))
    f = x̃ -> f0(x̃[1:num_x0])

    if in_place
        hres = DiffResults.JacobianResult(h(x), x)
        fres = DiffResults.HessianResult(x)
        hcfg = ForwardDiff.JacobianConfig(h, x)
        fcfg = ForwardDiff.HessianConfig(f, fres, x)
    end

    rtol = eps(1.)*(num_h+num_x)

    for i = 1:num_fosteps
        if in_place
            ForwardDiff.jacobian!(hres, h, x, hcfg)
            hx = DiffResults.value(hres)
            ∇h = DiffResults.jacobian(hres)
            ForwardDiff.hessian!(fres, f, x, fcfg)
            ∇f = DiffResults.gradient(fres)
            Hf = DiffResults.hessian(fres)
        else
            hx = h(x)
            ∇h = ForwardDiff.jacobian(h, x)
            ∇f = ForwardDiff.gradient(f, x)
            Hf = ForwardDiff.hessian(f, x)
        end

        gL = ∇f - ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h

        δx = (HL + (sqrt(sum(HL.^2))+rtol)*I) \ (-gL)
        δλ = -c * hx

        x += δx
        λ += δλ

        c *= 10.
    end

    for i = 1:num_sosteps
        if in_place
            ForwardDiff.jacobian!(hres, h, x, hcfg)
            hx = DiffResults.value(hres)
            ∇h = DiffResults.jacobian(hres)
            ForwardDiff.hessian!(fres, f, x, fcfg)
            ∇f = DiffResults.gradient(fres)
            Hf = DiffResults.hessian(fres)
        else
            hx = h(x)
            ∇h = ForwardDiff.jacobian(h, x)
            ∇f = ForwardDiff.gradient(f, x)
            Hf = ForwardDiff.hessian(f, x)
        end

        gL = ∇f - ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h

        A = vcat(hcat(HL,∇h'),hcat(∇h,zeros(num_h,num_h)))
        U,S,V = svd(A)
        tol = rtol*maximum(S) # TODO not smooth
        ksig = 100.
        Sinv = 1. ./ (1. .+ exp.(-ksig*(S .- tol)/tol)) .* (1. ./ S)
        Sinv[isinf.(Sinv)] .= 0.
        Apinv = V * (Diagonal(Sinv) * U')

        δxλ = Apinv * (-vcat(gL,hx))

        δx = δxλ[1:num_x]
        δλ = δxλ[num_x+1:num_x+num_h]

        x += δx
        λ += δλ

        c *= 1.
    end

    Lsol = 0. #L(x,λ,f,h,c)
    xsol = x[1:num_x0]
    λsol = λ[1:num_h0]
    μsol = λ[num_h0+1:num_h0+num_g0]

    xsol, λsol, μsol, Lsol
end
