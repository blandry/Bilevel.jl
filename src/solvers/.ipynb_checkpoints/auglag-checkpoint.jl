function softmax(x;k=1.)
    thresh = 500.
    g = map(x) do xi
        if k*xi < -thresh
            (log(1. + exp(k*xi)))/k
        elseif k*xi > thresh
            (k*xi + log(1. + exp(-k*xi)))/k
        else
            log(exp(k*xi) + 1.)/k
        end
    end
    g
end

function gsoftmax(x;k=1.)
    thresh = 500.
    g = map(x) do xi
        if k*xi > thresh
            1. / (1. + exp(-k*xi))
        else
            exp(k*xi) / (1. + exp(k*xi))
        end
    end
    Matrix(Diagonal(g))
end

function L(x,λ,f,h,c)
    f(x) - dot(λ,h(x)) + .5*c*dot(h(x),h(x))
end

function auglag(fun, num_eqs, num_ineqs, x0, options)
    num_fosteps = get(options, "num_fosteps", 3)
    num_sosteps = get(options, "num_sosteps", 2)
    c = get(options, "c", 1.)
    c_fos = get(options, "c_fos", 10.)
    c_sos = get(options, "c_sos", 10.)
    ls_method = get(options, "ls_method", :pinv)

    num_x = length(x0)
    rtol = eps(1.) * (num_eqs + num_ineqs)

    x = vcat(copy(x0), zeros(num_ineqs))
    λ = zeros(num_eqs + num_ineqs)

    for i = 1:num_fosteps
        # TODO inplace
        obj, eqs, ineqs, gobj, geqs, gineqs, Hobj = fun(x[1:num_x])

        ∇f = vcat(gobj, zeros(eltype(gobj), num_ineqs))
        Hf = zeros(eltype(Hobj), num_x+num_ineqs, num_x+num_ineqs)
        Hf[1:num_x, 1:num_x] .= Hobj
        hx = vcat(eqs, ineqs + softmax(x[num_x+1:end]))
        ∇h = hcat(vcat(geqs, gineqs), vcat(zeros(eltype(x),num_eqs,num_ineqs),gsoftmax(x[num_x+1:end])))

        gL = ∇f - ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h

        δx = (HL + (sum(HL.^2)+rtol)*I) \ (-gL)
        δλ = -c * hx

        x += δx
        λ += δλ

        c *= c_fos
    end

    for i = 1:num_sosteps
        # TODO inplace
        obj, eqs, ineqs, gobj, geqs, gineqs, Hobj = fun(x[1:num_x])

        ∇f = vcat(gobj, zeros(eltype(gobj),num_ineqs))
        Hf = zeros(eltype(Hobj), num_x+num_ineqs, num_x+num_ineqs)
        Hf[1:num_x, 1:num_x] .= Hobj

        hx = vcat(eqs, ineqs + softmax(x[num_x+1:end]))
        ∇h = hcat(vcat(geqs, gineqs), vcat(zeros(eltype(x),num_eqs,num_ineqs),gsoftmax(x[num_x+1:end])))

        gL = ∇f - ∇h'*λ + c*∇h'*hx
        HL = Hf + c*∇h'*∇h

        A = vcat(hcat(HL,∇h'),hcat(∇h,zeros(eltype(∇h),num_eqs+num_ineqs,num_eqs+num_ineqs)))
        b = -vcat(gL,hx)

        # for numerical stability
        tol = 1e-12
        A[abs.(A) .<= tol] .= 0.
        b[abs.(b) .<= tol] .= 0.

        if (ls_method == :pinv)
            Apinv = pinv(A)
            δxλ = Apinv * b
        elseif (ls_method == :least_squares)
            δxλ = least_squares(A,b)
        else
            U,S,V = svd(A)
            tol = rtol*maximum(S)
            ksig = 100.
            Sinv = 1. ./ (1. .+ exp.(-ksig*(S .- tol)/tol)) .* (1. ./ S)
            Sinv[isinf.(Sinv)] .= 0.
            Apinv = V * (Diagonal(Sinv) * U')
            δxλ = Apinv * b
        end

        δx = δxλ[1:num_x+num_ineqs]
        δλ = δxλ[num_x+num_ineqs+1:num_x+num_ineqs+num_eqs+num_ineqs]

        x += δx
        λ += δλ

        c *= c_sos
    end

    x[1:num_x], "Finished successfully"
end
