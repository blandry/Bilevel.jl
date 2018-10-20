function simulate(q0, v0, N)
    results = vcat(q0,v0,0.)
    n = length(q0) + length(v0) + 1
    m = n
    
    function eval_f(x)
        return 0.
    end
    function eval_grad_f(x, grad_f)
        grad_f[:] = zeros(n)
    end
    
    for i in 1:N
        x_L = -1000. * ones(n)
        x_U = 1000. * ones(n)
    
        g_L = zeros(m)
        g_U = zeros(m)
        
        q = results[1:length(q0),end]
        v = results[length(q0)+1:length(q0)+length(v0),end]
        eval_g, eval_jac_g = contact_constraints(q,v)
        
        prob = createProblem(n, x_L, x_U, m, g_L, g_U, n*m, 0, eval_f, eval_g, eval_grad_f, eval_jac_g)
        
        prob.x = copy(results[:,end])
    
        addOption(prob, "hessian_approximation", "limited-memory")
        
        status = solveProblem(prob)   
        println(Ipopt.ApplicationReturnStatus[status])

        results = hcat(results,prob.x)
    end
    
    results
end

function simulate_implicit(h, M, G, C, q0, v0, N)
    results = vcat(q0,v0,0.)
    num_q = length(q0)
    num_v = length(v0)
    num_x = num_q + num_v
    num_g = num_x + 1
    
    function eval_f(x)
        return 0.
    end
    
    function eval_grad_f(x, grad_f)
        grad_f[:] = zeros(num_x)
    end
    
    for i in 1:N
        x_L = -1e6 * ones(num_x)
        x_U = 1e6 * ones(num_x)
    
        g_L = zeros(num_g)
        g_U = zeros(num_g)
        g_U[num_g] = 1e16
        
        q0 = results[1:num_q,end]
        v0 = results[num_q+1:num_q+num_v,end]
        λ0 = results[num_q+num_v+1,end]
        eval_g, eval_jac_g = contact_constraints_implicit(h,M,G,C,q0,v0,λ0)
        
        prob = createProblem(num_x, x_L, x_U, num_g, g_L, g_U, 
                             num_x*num_g, 0, eval_f, eval_g, eval_grad_f, eval_jac_g)
        
        prob.x = copy(results[1:num_x,end])
        addOption(prob, "hessian_approximation", "limited-memory")
        
        status = solveProblem(prob)
           
        println(Ipopt.ApplicationReturnStatus[status])
    
        # qnext = prob.x[1:length(q0)]
        # ϕnext = n_c' * qnext
        # λnext = newton_contact_forces(ϕnext,λ0)
        λnext = 0.
        
        results = hcat(results,vcat(prob.x,λnext))
    end
    
    results
end