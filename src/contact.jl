function contact_constraints(q0,v0)
    h = 0.05
    n_c = [0, 1]
    mass = 1.0
    G = [0, -9.81]

    n = length(q0)
    
    function eval_g(x, g)
        q = x[1:n]
        v = x[n+1:2*n]
        λ = x[2*n+1]
        
        ϕ = n_c' * q 
        
        g[1:n] = mass * (v - v0) .- h * λ * n_c .- h * mass * G
        g[n+1:2*n] = q .- q0 .- h .* v
        g[2*n+1] = λ + ϕ - sqrt(λ^2 + ϕ^2)
    end
    
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            # for now just assume dense jac
            for i = 1:(2*n+1)
                for j = 1:(2*n+1)
                    rows[(i-1)*(2*n+1)+j] = i
                    cols[(i-1)*(2*n+1)+j] = j
                end
            end
        else
            g = zeros(2*n+1)
            J = ForwardDiff.jacobian((g̃, x̃) -> eval_g(x̃, g̃), g, x)
            values[:] = J'[:]
        end
    end
    
    return eval_g, eval_jac_g
end

function complementarity_constraint(a,b,ϵ=1e-6)
    a + b - sqrt(a^2 + b^2 + ϵ)
end

function augmented_lagrangian(x,μ,λ,ϕ)
    constraint_val = complementarity_constraint(ϕ,x[1])
    .5 * μ * constraint_val^2 + λ * constraint_val
end

function ∇augmented_lagrangian(x,μ,λ,ϕ)
    ForwardDiff.gradient(x̃ -> augmented_lagrangian(x̃,μ,λ,ϕ),x)
end

function Haugmented_lagrangian(x,μ,λ,ϕ)
    ForwardDiff.jacobian(x̃ -> ∇augmented_lagrangian(x̃,μ,λ,ϕ),x)    
end

function newton_contact_forces(ϕ,λ0)
    N = 10
    α = .95
    μ = 1.
    I = 1e-16
    
    x = [0.] # [λ0] # contact force
    λ = 0. # lagrange multiplier
    
    for i = 1:N
        ∇al = ∇augmented_lagrangian(x,μ,λ,ϕ)
        Hal = Haugmented_lagrangian(x,μ,λ,ϕ)
        
        x = x - α^i .* ((Hal + I) \ ∇al)
        λ = λ + μ .* complementarity_constraint(ϕ,x[1])
        μ = μ * 5.
    end
    
    x[1]
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
    # dλ = ForwardDiff.gradient(x̃ -> contact_forces(q0,v0,x̃[1:length(q0)],x̃[length(q0)+1:length(q0)+length(v0)]),vcat(q,v))
    
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
        
        F,Fy = contact_forces(h,M,G,C,q0,v0,q,v)
        
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
            F,Fy = contact_forces(h,M,G,C,q0,v0,q,v) # should be caching this
            dF = dcontact_forces(h,M,G,C,q0,v0,q,v,F,Fy)
            
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