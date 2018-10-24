function contact_constraints(h, M, G, C, q0, v0, λ0)
    num_q = length(q0)
    num_v = length(v0)
    num_x = num_q + num_v + 1
    num_g = num_q + num_v + 1  
    
    function eval_g(x, g)
        q = x[1:num_q]
        v = x[num_q+1:num_q+num_v]
        λ = x[num_q+num_v+1]
        
        ϕ = C' * q 
        
        g[1:num_q] = M * (v - v0) .- h * λ * C .- h * M * G
        g[num_q+1:num_q+num_v] = q .- q0 .- h .* v
        g[num_q+num_v+1] = λ + ϕ - sqrt(λ^2 + ϕ^2)
    end
    
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            # for now just assume dense jac
            for i = 1:num_g
                for j = 1:num_x
                    rows[(i-1)*num_x+j] = i
                    cols[(i-1)*num_x+j] = j
                end
            end
        else
            g = zeros(num_q+num_v+1)
            J = ForwardDiff.jacobian((g̃, x̃) -> eval_g(x̃, g̃), g, x)
            values[:] = J'[:]
        end
    end
    
    return eval_g, eval_jac_g
end

function newton_contact_forces(Δt,M,G,C,q0,v0,λ0,qn,vn)
    x = [λ0]
    λ = 1.
    μ = 1.
    c = .01
    
    # manipulator eq in the objective
    # q = -2.*(M*(vn[2] - v0[2])/Δt - M*G[2])
    # f = x -> x' * x + q * x
    # h = x -> (C' * qn) .* x
    
    # complementarity in the objective
    f = x -> ((C' * qn) .* x)'*((C' * qn) .* x)
    h = x -> M*(vn[2] - v0[2]) - Δt*x - Δt*M*G[2]
    
    g = x -> -x
    gp = (x,μ,c) -> max.(g(x), [-μ/c])

    L = (x,λ,μ,c) -> (f(x) + λ*h(x) + .5*c*h(x)'*h(x) + 1./(2.*c)*(max.([0.],μ+c*g(x))'*max.([0.],μ+c*g(x)) - μ^2))[1]
    ∇xL = (x,λ,μ,c) -> ForwardDiff.gradient(x̃ -> L(x̃,λ,μ,c),x)
    HxL = (x,λ,μ,c) -> ForwardDiff.jacobian(x̃ -> ∇xL(x̃,λ,μ,c),x)
    
    N = 6
    α = 1.
    I = 1e-10
    
    for i = 1:N
        gL = ∇xL(x,λ,μ,c)
        HL = HxL(x,λ,μ,c)
        
        dx = (HL + I) \ gL
        x = x - α^i .* dx
        
        λ = λ + c * h(x)[1]
        μ = μ + c * gp(x,μ,c)[1]
        
        c = 2. * c
    end
    
    x[1]
end

function dnewton_contact_forces(h,M,G,C,q0,v0,λ0,qn,vn)
    num_q = length(q0)
    num_v = length(v0)
    
    dλn = ForwardDiff.gradient(x̃ -> newton_contact_forces(h,M,G,C,q0,v0,λ0,x̃[1:num_q],x̃[num_q+1:num_q+num_v]),vcat(qn,vn))

    dλn
end

function contact_forces(h,M,G,C,q0,v0,qn,vn)
    Q = sparse([1],[1],2.)
    q = [-2.*(M*(vn[2] - v0[2])/h - M*G[2])]
    A = sparse([1,2], [1,1], [1.,C'*qn])
    l = [0., 0.]
    u = [1e12, 0.]
    
    m = OSQP.Model()
    OSQP.setup!(m; P=Q, q=q, A=A, l=l, u=u)
    results = OSQP.solve!(m)
    
    results.x, results.y
end

function dcontact_forces(hn,M,G,C,q0,v0,qn,vn,F,Fy)    
    # using the KKT conditions to provide a gradient        
    Q = sparse([1],[1],2.)
    q = [-2.*(M*(vn[2] - v0[2])/hn - M*G[2])]
    
    G = [-1.]
    h = [0.]
    
    A = [qn[2]]
    b = [0.]
    
    K = vcat(hcat(Q,G',A'),
             hcat(diagm(Fy[1])*G,diagm(G*F[1]-h),0.),
             hcat(A,0.,0.))
        
    vdQ = [-F[1],0.,0.]
    vdq = [-1.,0.,0.]
    vdA = [-Fy[2]',0.,-F[1]]
    vdb = [0.,0.,1.]
    vdG = [-Fy[1],-Fy[1]*F[1],0.]
    vdh = [0.,Fy[1],0.]
    
    # dzdQ = (K \ vdQ)[1]
    dzdq = 1.
    # try dzdq = (K \ vdq)[1]
    dzdA = 1.
    # try dzdA = (K \ vdA)[1]
    # dzdb = (K \ vdb)[1]
    # dzdG = (K \ vdG)[1]
    # dzdh = (K \ vdh)[1]
    
    # dQdqn = [0.,0.]
    # dqdqn = [0.,0.]
    dAdqn = [0.,1.]
    # dbdqn = [0.,0.]
    # dGdqn = [0.,0.]
    # dhdqn = [0.,0.]
    
    # dQdvn = [0.,0.]
    dqdvn = [0.,-2.*M/hn]
    # dAdvn = [0.,0.]
    # dbdvn = [0.,0.]
    # dGdvn = [0.,0.]
    # dhdvn = [0.,0.]
    
    # dzdqn = dzdQ*dQdqn + dzdq*dqdqn + dzdA*dAdqn + dzdb*dbdqn + dzdG*dGdqn + dzdh*dhdqn
    # dzdvn = dzdQ*dQdvn + dzdq*dqdvn + dzdA*dAdvn + dzdb*dbdvn + dzdG*dGdvn + dzdh*dhdvn
    
    dzdqn = dzdA*dAdqn
    dzdvn = dzdq*dqdvn
    
    dcontact = hcat(dzdqn,dzdvn)

    dcontact
end

function contact_constraints_implicit(h, M, G, C, q0, v0, λ0)
    num_q = length(q0)
    num_v = length(v0)
    num_x = num_q + num_v
    num_g = num_x + 1

    function eval_g(x, g)
        q = x[1:num_q]
        v = x[num_q+1:end]
        
        # F,Fy = contact_forces(h,M,G,C,q0,v0,q,v)
        F = [newton_contact_forces(h,M,G,C,q0,v0,λ0,q,v)]
        
        g[1:num_v] = M * (v - v0) .- h * F[1] .* C .- h * M .* G
        g[num_v+1:num_v+num_q] = q .- q0 .- h .* v
        g[num_v+num_q+1] = C' * q # ϕ >= 0
    end
    
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            # for now just assume dense jac
            for i = 1:num_g
                for j = 1:num_x
                    rows[(i-1)*num_x+j] = i
                    cols[(i-1)*num_x+j] = j
                end
            end
        else
            # g = zeros(num_g)
            # J = ForwardDiff.jacobian((g̃, x̃) -> eval_g(x̃, g̃), g, x)
            # values[:] = J'[:]
            
            q = x[1:num_q]
            v = x[num_q+1:end]
            # F,Fy = contact_forces(h,M,G,C,q0,v0,q,v) # should be caching this
            # dF = dcontact_forces(h,M,G,C,q0,v0,q,v,F,Fy)
            dF = dnewton_contact_forces(h,M,G,C,q0,v0,λ0,q,v)
            
            values[1] = 0.
            values[2] = 0.
            values[3] = M
            values[4] = 0.
            values[5] = -h * dF[1]
            values[6] = -h * dF[2]
            values[7] = -h * dF[3]
            values[8] = M - h * dF[4]
            values[9] = 1.
            values[10] = 0.
            values[11] = - h
            values[12] = 0.
            values[13] = 0.
            values[14] = 1.
            values[15] = 0.
            values[16] = - h
            values[17] = 0.
            values[18] = 1.
            values[19] = 0.
            values[20] = 0.
        end
    end
    
    return eval_g, eval_jac_g
end