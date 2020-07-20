

# Wrap the LDLinv object as a preallocated linear operator
function approxcholOperator(ldli::Laplacians.LDLinv{Tind,Tval},buff::Vector{Tval}) where {Tind,Tval}
    prod = @closure rhs -> Laplacians.LDLsolver!(buff,ldli,rhs)
    return PreallocatedLinearOperator{Tval}(length(ldli.d), length(ldli.d), true, true, prod, nothing, nothing)
end

# Returns a function that calls Krylov.cg using the LDLinv preconditioner wrapped as a preallocated linear operator
#  on a SDDM system (grounded case)
function solveApproxChol(sddm::AbstractArray, P; tol::Real=1e-6, maxits=300, verbose=false)
    a,d = adj(sddm)
    a1 = Laplacians.extendMatrix(a,d)
    la = lap(a1)

    tol_=tol
    maxits_=maxits
    verbose_=verbose


    f = function(b;tol=tol_, maxits=maxits_, verbose=verbose_)
        xaug = Krylov.cg(la,[b; -sum(b)] .- Laplacians.mean([b; -sum(b)]), M=P, rtol = tol, itmax=maxits, verbose=verbose)[1]
        xaug = xaug .- xaug[end]
        return xaug[1:a.n]
    end

    return f

end
