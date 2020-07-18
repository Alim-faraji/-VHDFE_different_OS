

# Wrap the LDLinv object as a preallocated linear operator
function approxcholOperator(ldli::Laplacians.LDLinv{Tind,Tval},buff::Vector{Tval}) where {Tind,Tval}
    prod = @closure rhs -> Laplacians.LDLsolver!(buff,ldli,rhs)
    return PreallocatedLinearOperator{Tval}(length(ldli.d), length(ldli.d), true, true, prod, nothing, nothing)
end

# Call the Kyrlov cg using the LDLinv preallocated linear operator on an sddm system
function solveApproxChol(sddm::AbstractArray, P; tol::Real=1e-6, maxits=300, maxtime=Inf, verbose=false, pcgIts=Int[], params...)
    a,d = adj(sddm)
    a1 = Laplacians.extendMatrix(a,d)
    la = lap(a1)

    tol_=tol
    maxits_=maxits
    maxtime_=maxtime
    verbose_=verbose
    pcgIts_=pcgIts

    f = function(b;tol=tol_, maxits=maxits_, maxtime=maxtime_, verbose=verbose_, pcgIts=pcgIts_)
        xaug = Krylov.cg(la,[b; -sum(b)] .- Laplacians.mean([b; -sum(b)]), M=P, rtol = tol_, itmax=maxits_ ,verbose=verbose_)[1]
        xaug = xaug .- xaug[end]
        return xaug[1:a.n]
    end

    return f

end
