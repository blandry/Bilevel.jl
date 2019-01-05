function ip_solve(x0::AbstractArray{T},f_obj,h_eq,g_ineq,num_h,num_g) where T
    num_x = length(x0)
    x_L = -1e19*ones(num_x)
    x_U = 1e19*ones(num_x)
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
