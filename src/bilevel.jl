function softmax(x)
    # thresh = 100.
    # x_out = map(x) do xi
    #     if (xi < -thresh)
    #         exp.(xi)
    #     elseif (xi > thresh)
    #         xi + exp.(-xi)
    #     else
    #         log.(1. + exp.(xi))
    #     end
    # end
    # return x_out
    max.(0.,x)
end

function L(x,λ,μ,c,f,h,g)
    hx = h(x)
    mucgx = softmax(μ.+c*g(x))
    f(x) + λ'*hx + .5*c*hx'*hx + 1./(2.*c)*sum(mucgx.*mucgx - μ.*μ)
end

function ∇xL(x,λ,μ,c,f,h,g)
    ReverseDiff.gradient(x̃ -> L(x̃,λ,μ,c,f,h,g),x)
end

function HxL(x,λ,μ,c,f,h,g)
    ReverseDiff.hessian(x̃ -> L(x̃,λ,μ,c,f,h,g),x)
end

function auglag_solve(x,λ,μ,f_obj,h_eq,g_ineq,c_vect)
    num_x = length(x)
    num_h = length(λ)
    num_g = length(μ)
    I = eye(num_x)

    # # scaling of all the inputs
    # # f_scale = sqrt(sum(ForwardDiff.gradient(f_obj,x).^2))
    # f_scale = sqrt(sum(f_obj(x)+1e-12))
    # f_obj_scaled = x̃ -> (1. ./ f_scale) .* f_obj(x̃)
    # # h_scale = sqrt(sum(ForwardDiff.jacobian(h_eq,x).^2))
    # h_scale = sqrt(sum(h_eq(x).^2+1e-12))
    # h_eq_scaled = x̃ -> (1. ./ h_scale) .* h_eq(x̃)
    # # g_scale = sqrt(sum(ForwardDiff.jacobian(g_ineq,x).^2))
    # g_scale = sqrt(sum(g_ineq(x).^2+1e-12))
    # g_ineq_scaled = x̃ -> (1. ./ g_scale) .* g_ineq(x̃)

    for i = 1:length(c_vect)
        gL = ∇xL(x,λ,μ,c_vect[i],f_obj,h_eq,g_ineq)
        HL = HxL(x,λ,μ,c_vect[i],f_obj,h_eq,g_ineq)

        # h = h_eq(x)
        # g = g_ineq(x)
        # ∇h = ForwardDiff.jacobian(h_eq,x)
        # ∇g = ForwardDiff.jacobian(g_ineq,x)
        # N = hcat(∇h',∇g')
        # Hd = N'*(inv(HL)*N)

        # for j = 1:length(μ)
        #     if g[j] < -μ[j] / c_vect[i]
        #         Hd[length(λ)+j,:] .= 0.
        #         Hd[:,length(λ)+j] .= 0.
        #         Hd[length(λ)+j,length(λ)+j] = -1./c_vect[i]
        #     end
        # end

        display(L(x,λ,μ,c_vect[i],f_obj,h_eq,g_ineq))
        x -= (HL + I*1e-12) \ gL
        display(L(x,λ,μ,c_vect[i],f_obj,h_eq,g_ineq))
        display(gL)
        display("---")
        λ = softmax(λ + c_vect[i] * h_eq(x))
        μ = softmax(μ + c_vect[i] * g_ineq(x))
        # d = vcat(λ,μ) - ((Hd+1e-12*eye(length(λ)+length(μ))) \ vcat(h,g))
        # display(d)
        # display(Hd)
        # display(vcat(h,g))
        # λ = max.(0.,d[1:length(λ)])
        # μ = max.(0.,d[length(λ)+1:end])
    end

    # h_final = x̃ -> vcat(h_eq(x̃),μ.*g_ineq(x̃))
    # h = h_final(x)
    # ∇h = ForwardDiff.jacobian(h_final,x)
    # ∇f = ReverseDiff.gradient(f_obj,x)
    # Hf = ReverseDiff.hessian(f_obj,x)
    # gL = ∇f + ∇h'*vcat(λ,μ) + c_vect[end]*∇h'*h
    # HL = Hf + c_vect[end]*∇h'*∇h
    #
    # A = vcat(hcat(HL,∇h'),hcat(∇h,zeros(num_h+num_g,num_h+num_g)))
    # δ = (A+eye(num_x+num_h+num_g)*1e-12)\(-vcat(gL,h))
    # x += δ[1:num_x]

    x, λ, μ
end

function ip_solve(x0::AbstractArray{T},f_obj,h_eq,g_ineq,num_h,num_g) where T
    num_x = length(x0)
    x_L = -1e19 * ones(num_x)
    x_U = 1e19 * ones(num_x)
    g_L = vcat(0. * ones(num_h), -1e19 * ones(num_g))
    g_U = vcat(0. * ones(num_h), 0. * ones(num_g))

    eval_f = x̃ -> f_obj(x̃)
    eval_grad_f = (x̃,grad_f) -> grad_f[:] = ForwardDiff.gradient(eval_f,x̃)[:]

    eval_g = (x̃,g̃) -> g̃[:] = vcat(h_eq(x̃),g_ineq(x̃))[:]
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            for i = 1:num_h+num_g
                for j = 1:num_x
                    rows[(i-1)*num_x+j] = i
                    cols[(i-1)*num_x+j] = j
                end
            end
        else
            J = vcat(ForwardDiff.jacobian(h_eq,x),ForwardDiff.jacobian(g_ineq,x))
            values[:] = J'[:]
        end
    end

    prob = createProblem(num_x,x_L,x_U,
                         num_h+num_g,g_L,g_U,
                         num_x*(num_g+num_h),0,
                         eval_f,eval_g,
                         eval_grad_f,eval_jac_g)
    prob.x[:] = x0[:]

    addOption(prob, "hessian_approximation", "limited-memory")
    addOption(prob, "print_level", 0)
    status = solveProblem(prob)
    println(Ipopt.ApplicationReturnStatus[status])

    prob.x
end
