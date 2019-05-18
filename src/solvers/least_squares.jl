function least_squares(A::AbstractArray{T}, b::AbstractArray{T}) where T
    if (T<:ForwardDiff.Dual)
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
    else
        x = pinv(A) * b
    end

    # TODO other cases
    
    x
end