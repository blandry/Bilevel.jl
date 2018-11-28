function abse(x,ϵ=1e-12)
    sqrt.(x.*x .+ ϵ)
end

function diffmax(x,y)
    .5*(x .+ y + abse(x .- y))
end

function softmax(x,y)
    log.(exp.(x) .+ exp.(y))
    # x .+ log.(1. .+ exp.(y .- x))
end

function L(x,λ,μ,c,f,h,g)
    hx = h(x)
    # mucgx = max.(0.,μ.+c*g(x))
    mucgx = softmax(0.,μ.+c*g(x))
    f(x) + dot(λ,hx) + .5*c*dot(hx,hx) + 1./(2.*c)*sum(mucgx.*mucgx - μ.*μ)
end

function ∇xL(x,λ,μ,c,f,h,g)
    ForwardDiff.gradient(x̃ -> L(x̃,λ,μ,c,f,h,g),x)
end

function HxL(x,λ,μ,c,f,h,g)
    ForwardDiff.hessian(x̃ -> L(x̃,λ,μ,c,f,h,g),x)
end

function auglag_solve(x0::AbstractArray{T},f_obj,h_eq,g_ineq,num_h,num_g,α_vect,c_vect,I_vect) where T
    λ = zeros(T,num_h)
    μ = zeros(T,num_g)
    I = eye(T,length(x0))
    x = copy(x0)

    for i = 1:length(α_vect)
        gL = ∇xL(x,λ,μ,c_vect[i],f_obj,h_eq,g_ineq)
        HL = HxL(x,λ,μ,c_vect[i],f_obj,h_eq,g_ineq)

        # x -= α_vect[i] .* gL / norm(gL)
        x -= α_vect[i] .* (HL + I_vect[i]*I) \ gL

        λ = λ + c_vect[i] * h_eq(x)
        # μ = max.(0., μ + c_vect[i] * g_ineq(x))
        μ = softmax(0., μ + c_vect[i] * g_ineq(x))
    end

    x
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
