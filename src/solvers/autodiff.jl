function generate_autodiff_solver_fn(eval_obj,eval_cons,cs_eqs,cs_ineqs;hessian=true) 
    function solver_fn(x)
        J = eval_obj(x)
        gJ = ForwardDiff.gradient(eval_obj, x)
        if hessian
            HJ = ForwardDiff.hessian(eval_obj, x)
        else
            HJ = zeros(length(x),length(x))
        end

        g = eval_cons(x)
        dgdx = ForwardDiff.jacobian(eval_cons, x)
        
        ceq = g[cs_eqs]
        c = g[cs_ineqs]    
        gceq = dgdx[cs_eqs,:]
        gc = dgdx[cs_ineqs,:]

        J, ceq, c, gJ, gceq, gc, HJ
    end
    
    solver_fn
end

function generate_autodiff_solver_fn(eval_obj,fres,fcfg,eval_cons,gres,gcfg,cs_eqs,cs_ineqs;hessian=true)    
    function solver_fn(x)     
        ForwardDiff.hessian!(fres, eval_obj, x, fcfg)
        J = DiffResults.value(fres)
        gJ = DiffResults.gradient(fres)
        if hessian
            HJ = DiffResults.hessian(fres)
        else
            HJ = zeros(length(x),length(x))
        end
        
        ForwardDiff.jacobian!(gres, eval_cons, x, gcfg)
        g = DiffResults.value(gres)
        dgdx = DiffResults.jacobian(gres)
        
        ceq = g[cs_eqs]
        c = g[cs_ineqs]    
        gceq = dgdx[cs_eqs,:]
        gc = dgdx[cs_ineqs,:]

        J, ceq, c, gJ, gceq, gc, HJ
    end
    
    solver_fn
end

function generate_autodiff_solver_fn(eval_obj,fres,eval_cons,gres,cs_eqs,cs_ineqs;hessian=true)    
    function solver_fn(x)     
        ForwardDiff.hessian!(fres, eval_obj, x)
        J = DiffResults.value(fres)
        gJ = DiffResults.gradient(fres)
        if hessian
            HJ = DiffResults.hessian(fres)
        else
            HJ = zeros(length(x),length(x))
        end
        
        ForwardDiff.jacobian!(gres, eval_cons, x)
        g = DiffResults.value(gres)
        dgdx = DiffResults.jacobian(gres)
        
        ceq = g[cs_eqs]
        c = g[cs_ineqs]    
        gceq = dgdx[cs_eqs,:]
        gc = dgdx[cs_ineqs,:]

        J, ceq, c, gJ, gceq, gc, HJ
    end
    
    solver_fn
end

function generate_autodiff_solver_fn(eval_obj,eval_cons,cs_eqs,cs_ineqs,vs_num_vars;hessian=true)    
    cs_num_cons = length(cs_eqs) + length(cs_ineqs)
    
    fres = DiffResults.HessianResult(zeros(vs_num_vars))
    fcfg = ForwardDiff.HessianConfig(eval_obj, fres, zeros(vs_num_vars))
    gres = DiffResults.JacobianResult(zeros(cs_num_cons), zeros(vs_num_vars))
    gcfg = ForwardDiff.JacobianConfig(eval_cons, zeros(vs_num_vars))
    
    generate_autodiff_solver_fn(eval_obj,fres,fcfg,eval_cons,gres,gcfg,cs_eqs,cs_ineqs,hessian=hessian)
end