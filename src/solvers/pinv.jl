function LinearAlgebra.pinv(A::Array{T,2}) where T<:ForwardDiff.Dual
    m = size(A,1)
    n = size(A,2)

    Av = map(ForwardDiff.value, A)
    Ap = map(ForwardDiff.partials, A)
    Apt = Ap[reshape(1:m*n,(m,n))'] # taking the transpose

    Pv = pinv(Av)
    Pp = Pv*Pv'*Apt*(I - Av*Pv) + (I - Pv*Av)*Apt*Pv'*Pv - Pv*Ap*Pv

    P = map(T,Pv,Pp)

    return P
end
