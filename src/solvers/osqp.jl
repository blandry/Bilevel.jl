function osqp_solve(P::AbstractArray{T},q::AbstractArray{U},
              A::AbstractArray{V},l::AbstractArray{W},u::AbstractArray{S}) where {T,U,V,W,S}

     # TODO store and update this
     prob = OSQP.Model()

     if  (T<:ForwardDiff.Dual || U<:ForwardDiff.Dual ||  V<:ForwardDiff.Dual || W<:ForwardDiff.Dual || S<:ForwardDiff.Dual)
         if (T<:ForwardDiff.Dual)
             Pv = map(ForwardDiff.value, P)
             Pp = map(ForwardDiff.partials, P)
         else
             Pv = P
             Pp = zeros(size(P))
         end
         if (U<:ForwardDiff.Dual)
             qv = map(ForwardDiff.value, q)
             qp = map(ForwardDiff.partials, q)
         else
             qv = q
             qp = zeros(size(q))
         end
         if (V<:ForwardDiff.Dual)
             Av = map(ForwardDiff.value, A)
             Ap = map(ForwardDiff.partials, A)
         else
             Av = A
             Ap = zeros(size(A))
         end
         if (W<:ForwardDiff.Dual)
             lv = map(ForwardDiff.value, l)
             lp = map(ForwardDiff.partials, l)
         else
             lv = l
             lp = zeros(size(l))
         end
         if (S<:ForwardDiff.Dual)
             uv = map(ForwardDiff.value, u)
             up = map(ForwardDiff.partials, u)
         else
             uv = u
             up = zeros(size(u))
         end

         m,n = size(Av)

         OSQP.setup!(prob; P=sparse(Pv), q=qv, A=sparse(Av), l=lv, u=uv, alpha=1., verbose=0)
         results = OSQP.solve!(prob)

         xv = results.x
         λ_plus = max.(0., results.y)
         λ_minus = min.(0., results.y)

         Gv = vcat(Av, -Av)
         hv = vcat(uv, -lv)
         λv = vcat(λ_plus, -λ_minus)

         M = vcat(hcat(Pv, Gv'),
         hcat(Diagonal(λv)*Gv, Diagonal(Gv*xv - hv)))

         Apt = Ap[reshape(1:m*n,(m,n))']
         Gp = vcat(Ap, -Ap)
         Gpt = hcat(Apt, -Apt)
         hp = vcat(up, -lp)

         dres = vcat(Pp*xv + qp + Gpt*λv,
         Diagonal(λv)*(Gp*xv - hp))

         xp = (-pinv(Matrix(M)) * dres)[1:n]

         # TODO you have to use the right type here...
         sol = map(T, xv, xp)
     else
         OSQP.setup!(prob; P=sparse(P), q=q, A=sparse(A), l=l, u=u, alpha=1., verbose=0)
         results = OSQP.solve!(prob)

         sol = results.x
     end

     return sol
end
