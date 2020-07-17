

# Wrap the LDLinv object as a preallocated linear operator
function approxcholOperator(ldli::Laplacians.LDLinv{Tind,Tval},buff::Vector{Tval}) where {Tind,Tval}
    prod = @closure rhs -> Laplacians.LDLsolver!(buff,ldli,rhs)
    return PreallocatedLinearOperator{Tval}(length(ldli.d), length(ldli.d), true, true, prod, nothing, nothing)
end

# Call the Kyrlov cg using the LDLinv preallocated linear operator
function solveApproxChol()
 return nothing
end
