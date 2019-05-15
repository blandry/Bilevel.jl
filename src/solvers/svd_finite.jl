function svd_finite(A::Array{T,2}) where T<:ForwardDiff.Dual
    m = size(A,1)
    n = size(A,2)
        
    Av = map(ForwardDiff.value,A)   
    num_A = length(Av)
        
    Uv,sv,Vv = svd(Av)
        
    dU = zeros(size(Uv,1),size(Uv,2),num_A)
    ds = zeros(length(sv),num_A)
    dV = zeros(size(Vv,1),size(Vv,2),num_A)
    
    ϵ = sqrt(eps(1.))
    for i = 1:num_A
        δ = zeros(size(Av))
        δ[i] = ϵ
        U,s,V = svd(Av + δ)
        dU[:,:,i] = (U - Uv) ./ ϵ
        ds[:,i] = (s - sv) ./ ϵ
        dV[:,:,i] = (V - Vv) ./ ϵ
    end
    
    Up = zeros(typeof(A[1].partials),size(Uv))
    sp = zeros(typeof(A[1].partials),size(sv))
    Vp = zeros(typeof(A[1].partials),size(Vv))
    for i = 1:size(Up,1)
        for j = 1:size(Up,2)
            Up[i,j] = sum([dU[i,j,k] * A[k].partials for k = 1:length(Av)])
        end
    end
    for i = 1:length(sp)
        sp[i] = sum([ds[i,k] * A[k].partials for k = 1:length(Av)])
    end
    for i = 1:size(Vp,1)
        for j = 1:size(Vp,2)
            Vp[i,j] = sum([dV[i,j,k] * A[k].partials for k = 1:length(Av)])
        end
    end    
    
    U = map(T,Uv,Up)
    s = map(T,sv,sp)
    V = map(T,Vv,Vp)
    
    return U, s, V
end