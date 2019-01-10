function Base.svd(A::Array{T,2}) where T<:ForwardDiff.Dual
    m, n = size(A)
    num_partials = length(A[1,1].partials)
    Im = eye(m)
    In = eye(n)

    Av = map(x->x.value,A)
    Uv,sv,Vv = svd(Av)
    Sv = diagm(sv)
    Svi = diagm(1. ./ sv)
    Svi[isinf.(Svi)] = 0.

    F = zeros(size(A))
    for i = 1:size(A,1)
        for j = 1:size(A,2)
            if i!=j
                F[i,j] = 1. / (sv[j]^2 - sv[i]^2)
            end
        end
    end
    F[isinf.(F)] = 0. # TODO figure out how to deal with duplicate sv

    Up = Array{eltype(Av),3}(Tuple([size(Uv)...,num_partials]))
    sp = Array{eltype(Av),2}(Tuple([length(sv),num_partials]))
    Vp = Array{eltype(Av),3}(Tuple([size(Vv)...,num_partials]))
    for k = 1:num_partials
        dA = map(a -> a.partials[k],A)

        Upk = Uv*(F .* (Uv'*dA*Vv*Sv + Sv*Vv'*dA'*Uv)) + (Im - Uv*Uv')*dA*Vv*Svi
        spk = diag(Uv'*dA*Vv)
        Vpk = Vv*(F .* (Sv*Uv'*dA*Vv+Vv'*dA'*Uv*Sv)) + (In - Vv*Vv')*dA'*Uv*Svi

        Up[:,:,k] = Upk
        sp[:,k] = spk
        Vp[:,:,k] = Vpk
    end

    U = Array{T,2}(size(Uv))
    for i=1:size(U,1)
        for j=1:size(U,2)
            U[i,j] = T(Uv[i,j], ForwardDiff.Partials(Tuple(Up[i,j,:])))
        end
    end

    S = Array{T,1}(length(sv))
    for i=1:length(sv)
        S[i] = T(sv[i], ForwardDiff.Partials(Tuple(sp[i,:])))
    end

    V = Array{T,2}(size(Vv))
    for i=1:size(V,1)
        for j=1:size(V,2)
            V[i,j] = T(Vv[i,j], ForwardDiff.Partials(Tuple(Vp[i,j,:])))
        end
    end

    return U, S, V
end
