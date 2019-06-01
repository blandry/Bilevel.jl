function osqp(P::AbstractArray{T},q::AbstractArray{U},
              A::AbstractArray{V},l::AbstractArray{W},u::AbstractArray{S},prob) where {T,U,V,W,S}

#      # TODO store and update this
#      prob = OSQP.Model()

     if  (T<:ForwardDiff.Dual || U<:ForwardDiff.Dual ||  V<:ForwardDiff.Dual || W<:ForwardDiff.Dual || S<:ForwardDiff.Dual)
         if (T<:ForwardDiff.Dual)
             Dtype = eltype(P)
             Ptype = typeof(P[1,1].partials)
         elseif (U<:ForwardDiff.Dual)
             Dtype = eltype(q)
             Ptype = typeof(q[1].partials)
         elseif (V<:ForwardDiff.Dual)
             Dtype = eltype(A)
             Ptype = typeof(A[1,1].partials)
         elseif (W<:ForwardDiff.Dual)
             Dtype = eltype(l)
             Ptype = typeof(l[1].partials)
         else
             Dtype = eltype(u)
             Ptype = typeof(u[1].partials)
         end
              
         if (T<:ForwardDiff.Dual)
             Pv = map(ForwardDiff.value, P)
             Pp = map(ForwardDiff.partials, P)
         else
             Pv = P
             Pp = zeros(Ptype, size(P))
         end
         if (U<:ForwardDiff.Dual)
             qv = map(ForwardDiff.value, q)
             qp = map(ForwardDiff.partials, q)
         else
             qv = q
             qp = zeros(Ptype, size(q))
         end
         if (V<:ForwardDiff.Dual)
             Av = map(ForwardDiff.value, A)
             Ap = map(ForwardDiff.partials, A)
         else
             Av = A
             Ap = zeros(Ptype, size(A))
         end
         if (W<:ForwardDiff.Dual)
             lv = map(ForwardDiff.value, l)
             lp = map(ForwardDiff.partials, l)
         else
             lv = l
             lp = zeros(Ptype, size(l))
         end
         if (S<:ForwardDiff.Dual)
             uv = map(ForwardDiff.value, u)
             up = map(ForwardDiff.partials, u)
         else
             uv = u
             up = zeros(Ptype, size(u))
         end

         m,n = size(Av)

         OSQP.setup!(prob; P=sparse(Pv), q=qv, A=sparse(Av), l=lv, u=uv, verbose=0)
#         OSQP.update!(prob, Px=sparse(Pv), q=qv, Ax=sparse(Av), l=lv, u=uv)
         results = OSQP.solve!(prob)

         xv = results.x
         xv[isnan.(xv)] .= 0.
         xv[isinf.(xv)] .= 0.                                
         λ_plus = max.(0., results.y)
         λ_minus = min.(0., results.y)
         λ_plus[isnan.(λ_plus)] .= 0.
         λ_plus[isinf.(λ_plus)] .= 0.
         λ_minus[isnan.(λ_minus)] .= 0.
         λ_minus[isinf.(λ_minus)] .= 0.
                                            
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
         sol = map(Dtype, xv, xp)
     else
         OSQP.setup!(prob; P=sparse(P), q=q, A=sparse(A), l=l, u=u, verbose=0)
#          OSQP.update!(prob, Px=sparse(Pv), q=qv, Ax=sparse(Av), l=lv, u=uv)
         results = OSQP.solve!(prob)

         sol = results.x
     end

     return sol
end
