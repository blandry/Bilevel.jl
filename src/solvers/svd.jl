function LinearAlgebra.svd(A::Array{T,2}) where T<:ForwardDiff.Dual
    m = size(A,1)
    n = size(A,2)
    @assert m >= n

    Av = map(ForwardDiff.value,A)
    Uv,sv,Vv = svd(Av)

    Ap = map(ForwardDiff.partials,A)
    UvtApVv = Uv'*Ap*Vv
    sp = diag(UvtApVv)

    # if allunique(sv)
        # If all unique, then gradient well defined
        Sv = Matrix(Diagonal(sv))
        svi = 1. ./ sv
        svi[isinf.(svi)] .= 0.
        Svi = Matrix(Diagonal(svi))

        F = zeros((m,n))
        for i = 1:m
            for j = 1:n
                if i!=j
                    d = (sv[j]^2 - sv[i]^2)
                    F[i,j] = 1. / d
                end
            end
        end
        F[isinf.(F)] .= 0.

        VvtApUv = Vv'*Ap[reshape(1:m*n,(m,n))']*Uv

        Up = Uv*(F .* (UvtApVv*Sv + Sv*VvtApUv)) + (I - Uv*Uv')*Ap*Vv*Svi
        Vp = Vv*(F .* (Sv*UvtApVv+VvtApUv*Sv)) + (I - Vv*Vv')*Ap[reshape(1:m*n,(m,n))']*Uv*Svi
    # else
    #     # Otherwise need to least-squares
    #     dU = zeros(size(Uv,1),size(Uv,2),size(A,1),size(A,2))
    #     dV = zeros(size(Vv,1),size(Vv,2),size(A,1),size(A,2))
    #     D = Array{Float64, 2}(undef, 2, 2)
    #     u = Array{Float64, 1}(undef, 2)
    #     for i = 1:m
    #         for j = 1:n
    #             for k = 1:m
    #                 for l = 1:n
    #                     D[1,1] = sv[l]
    #                     D[1,2] = sv[k]
    #                     D[2,1] = sv[k]
    #                     D[2,2] = sv[l]
    #                     u[1] = Uv[i,k]*Vv[j,l]
    #                     u[2] =  -Uv[i,l]*Vv[j,k]
    #                     if (sv[l] == sv[k])
    #                         Ω = pinv(D) * u
    #                     else
    #                         Ω = D \ u
    #                     end
    #                     dU[k,l,i,j] = Ω[1]
    #                     dV[k,l,i,j] = Ω[2]
    #                 end
    #             end
    #         end
    #     end
    #     Up = zeros(typeof(A[1].partials),size(Uv))
    #     Vp = zeros(typeof(A[1].partials),size(Vv))
    #     for k = 1:size(Up,1)
    #         for l = 1:size(Up,2)
    #             for i = 1:m
    #                 for j = 1:n
    #                     Up[k,l] += dU[k,l,i,j] * A[i,j].partials
    #                 end
    #             end
    #         end
    #     end
    #     for k = 1:size(Vp,1)
    #         for l = 1:size(Vp,2)
    #             for i = 1:m
    #                 for j = 1:n
    #                     Vp[k,l] += dV[k,l,i,j] * A[i,j].partials
    #                 end
    #             end
    #         end
    #     end
    # end

    U = map(T,Uv,Up)
    s = map(T,sv,sp)
    V = map(T,Vv,Vp)

    return U, s, V
end
