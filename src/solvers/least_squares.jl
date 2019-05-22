function least_squares(A::AbstractArray{T}, b::AbstractArray{U}) where {T,U}
    if (T<:ForwardDiff.Dual && U<:ForwardDiff.Dual)
        m = size(A,1)
        n = size(A,2)

        Av = map(ForwardDiff.value, A)
        bv = map(ForwardDiff.value, b)

        P = pinv(Av)
        xv = P * bv

        Ap = map(ForwardDiff.partials, A)
        bp = map(ForwardDiff.partials, b)
        Apt = Ap[reshape(1:m*n,(m,n))'] # taking the transpose
        bpt = reshape(bp,m,1)

        Pp = P*P'*Apt*(I - Av*P) + (I - P*Av)*Apt*P'*P - P*Ap*P
        xp = Pp*bv + P*bpt

        x = map(T,xv,xp)
    elseif (T<:ForwardDiff.Dual)
        m = size(A,1)
        n = size(A,2)

        Av = map(ForwardDiff.value, A)

        P = pinv(Av)
        xv = P * b

        Ap = map(ForwardDiff.partials, A)
        Apt = Ap[reshape(1:m*n,(m,n))'] # taking the transpose

        Pp = P*P'*Apt*(I - Av*P) + (I - P*Av)*Apt*P'*P - P*Ap*P
        xp = Pp*bv

        x = map(T,xv,xp)
    elseif (U<:ForwardDiff.Dual)
        m = size(A,1)
        n = size(A,2)

        bv = map(ForwardDiff.value, b)

        P = pinv(A)
        xv = P * bv

        bp = map(ForwardDiff.partials, b)
        bpt = reshape(bp,m,1)

        xp = P*bpt

        x = map(T,xv,xp)
    else
        x = pinv(A) * b
    end

    # TODO other cases

    x
end
