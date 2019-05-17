function LinearAlgebra.svd(A::Array{T,2}) where T<:ForwardDiff.Dual
    m = size(A,1)
    n = size(A,2)
    @assert m >= n

    Av = map(ForwardDiff.value,A)
    Ap = map(ForwardDiff.partials,A)

    Uv,sv,Vv = svd(Av)

    # If all unique, then gradient well defined
    #
    # tol = eps(1.) * m * maximum(sv)
    # for i = 1:length(sv)
    #     if (sv[i] < tol)
    #         sv[i] = 0.
    #     end
    # end
    #
    # Sv = Matrix(Diagonal(sv))
    # svi = 1. ./ sv
    # svi[isinf.(svi)] .= 0.
    # Svi = Matrix(Diagonal(svi))
    #
    # F = zeros((m,n))
    # for i = 1:m
    #     for j = 1:n
    #         if i!=j
    #             d = (sv[j]^2 - sv[i]^2)
    #             F[i,j] = 1. / d
    #         end
    #     end
    # end
    # F[isinf.(F)] .= 0.
    #
    # UvtApVv = Uv'*Ap*Vv
    # VvtApUv = Vv'*Ap[reshape(1:m*n,(m,n))']*Uv
    #
    # Up = Uv*(F .* (UvtApVv*Sv + Sv*VvtApUv)) + (I - Uv*Uv')*Ap*Vv*Svi
    # sp = diag(UvtApVv)
    # Vp = Vv*(F .* (Sv*UvtApVv+VvtApUv*Sv)) + (I - Vv*Vv')*Ap[reshape(1:m*n,(m,n))']*Uv*Svi

    # Otherwise need to least-squares

    ds = zeros(length(sv),m,n)
    for i = 1:m
        for j = 1:n
            ds[:,i,j] = Uv[i,:] .* Vv[j,:]
        end
    end
    sp = zeros(typeof(A[1].partials),size(sv))
    for k = 1:length(sp)
        for i = 1:m
            for j = 1:n
                sp[k] += ds[k,i,j] * A[i,j].partials
            end
        end
    end

    dU = zeros(size(Uv,1),size(Uv,2),size(A,1),size(A,2))
    dV = zeros(size(Vv,1),size(Vv,2),size(A,1),size(A,2))
    for i = 1:m
        for j = 1:n
            for k = 1:m
                for l = 1:n
                    D = [sv[l] sv[k]; sv[k] sv[l]]
                    u = [Uv[i,k]*Vv[j,l], -Uv[i,l]*Vv[j,k]]
                    Ω = pinv(D) * u
                    dU[k,l,i,j] = Ω[1]
                    dV[k,l,i,j] = Ω[2]
                end
            end
        end
    end
    for i = 1:m
        for j = 1:n
            dU[:,:,i,j] = Uv * dU[:,:,i,j]
            dV[:,:,i,j] = -Vv * dV[:,:,i,j]
        end
    end
    Up = zeros(typeof(A[1].partials),size(Uv))
    Vp = zeros(typeof(A[1].partials),size(Vv))
    for k = 1:size(Up,1)
        for l = 1:size(Up,2)
            for i = 1:m
                for j = 1:n
                    Up[k,l] += dU[k,l,i,j] * A[i,j].partials
                end
            end
        end
    end
    for k = 1:size(Vp,1)
        for l = 1:size(Vp,2)
            for i = 1:m
                for j = 1:n
                    Vp[k,l] += dV[k,l,i,j] * A[i,j].partials
                end
            end
        end
    end

    U = map(T,Uv,Up)
    s = map(T,sv,sp)
    V = map(T,Vv,Vp)

    return U, s, V
end
