function simulate(h, M, G, C, q0, v0, N)
    results = vcat(q0,v0,0.)
    num_q = length(q0)
    num_v = length(v0)
    num_x = num_q + num_v + 1
    num_g = num_q + num_v + 1
    
    function eval_f(x)
        return 0.
    end
    function eval_grad_f(x, grad_f)
        grad_f[:] = zeros(num_x)
    end
    
    for i in 1:N
        x_L = -1000. * ones(num_x)
        x_U = 1000. * ones(num_x)
    
        g_L = zeros(num_g)
        g_U = zeros(num_g)
        
        q0 = results[1:num_q,end]
        v0 = results[num_q+1:num_q+num_v,end]
        λ0 = results[num_q+num_v+1,end]
        eval_g, eval_jac_g = contact_constraints(h,M,G,C,q0,v0,λ0)
        
        prob = createProblem(num_x,x_L,x_U,num_g,g_L,g_U,num_x*num_g,0,eval_f,eval_g,eval_grad_f, eval_jac_g)
        
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
    
        qnext = prob.x[1:num_q]
        vnext = prob.x[num_q+1:num_q+num_v]
        λnext = newton_contact_forces(h,M,G,C,q0,v0,λ0,qnext,vnext)
        
        results = hcat(results,vcat(prob.x,λnext))
    end
    
    results
end