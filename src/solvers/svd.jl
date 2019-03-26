function LinearAlgebra.svd(A::Array{T,2}) where T<:ForwardDiff.Dual
    m = size(A,1)
    n = size(A,2)
    
    Av = map(ForwardDiff.value,A)
    Ap = map(ForwardDiff.partials,A)
    
    Uv,sv,Vv = svd(Av)
    
    Sv = Matrix(Diagonal(sv))
    Svi = Matrix(Diagonal(1. ./ sv))
    Svi[isinf.(Svi)] .= 0.

    F = zeros((m,n))
    for i = 1:m
        for j = 1:n
            if i!=j
                F[i,j] = 1. / (sv[j]^2 - sv[i]^2)
            end
        end
    end
    F[isinf.(F)] .= 0. # TODO better solution for duplicate sv
    
    UvtApVv = Uv'*Ap*Vv
    VvtApUv = Vv'*Ap[reshape(1:m*n,(m,n))']*Uv
    
    Up = Uv*(F .* (UvtApVv*Sv + Sv*VvtApUv)) + (I - Uv*Uv')*Ap*Vv*Svi
    sp = diag(UvtApVv)
    Vp = Vv*(F .* (Sv*UvtApVv+VvtApUv*Sv)) + (I - Vv*Vv')*Ap[reshape(1:m*n,(m,n))']*Uv*Svi
    
    U = map(T,Uv,Up)
    s = map(T,sv,sp)
    V = map(T,Vv,Vp)
    
    return U, s, V
end
