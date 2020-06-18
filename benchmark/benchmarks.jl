using VarianceComponentsHDFE
using Random, BenchmarkTools, Test
using JLD, SparseArrays, LinearAlgebra
using AlgebraicMultigrid, IterativeSolvers
# Make sure LDLFactorizations is version 0.5.0
using LDLFactorizations

use_matlabCMG = false
if use_matlabCMG
    using Laplacians, Pkg
    include(string(Pkg.dir("Laplacians") , "/src/matlabSolvers.jl"))
end

# Entry point for the PkgBenchmarks call.  Can split into different files later.
include("prepare_benchmark_data.jl")
# NOTE: Suite below can assume that the `benchmark/data/...` has been filled

medium_data = load("data/medium_main.jld")
X = medium_data["X_GroundedLaplacian"]
X̃ = medium_data["X_tilde"]
S_xx = medium_data["S_xx"]
X̃_regularized = medium_data["X_tilde_regularized"]

const SUITE = BenchmarkGroup()

# Prepare for direct method benchmarks
m,k = size(X)
luX̃ = lu(X̃)
qrX̃ = qr(X̃)
Rz = zeros(m+k)
# RHS_aug is the first column of the kxk identity augmented by m zeros
RHS_aug = zeros(m+k)
RHS_aug[1] = 1.0

# Direct methods on the augmented system X̃
SUITE["X_tilde inplace direct solve: LU"] = @benchmarkable ldiv!($Rz, $luX̃, $RHS_aug)

# Non-inplace direct methods on the augmented system X̃
SUITE["X_tilde direct solve"] = @benchmarkable \($X̃, $RHS_aug)
SUITE["X_tilde direct solve: LU"] = @benchmarkable \($luX̃, $RHS_aug)
SUITE["X_tilde direct solve: QR"] = @benchmarkable \($qrX̃, $RHS_aug)

# Non-inplace factorizations of the augmented system X̃
SUITE["X_tilde factorization: LU"] = @benchmarkable lu($X̃)
SUITE["X_tilde factorization: QR"] = @benchmarkable qr($X̃)

# Prepare for direct method benchmarks
P = aspreconditioner(ruge_stuben(S_xx))
RHS = SparseMatrixCSC{Float64,Int64}(X[1,:])
z = 0.1.*ones(length(RHS))

# Iterative methods on the original system S_xx
SUITE["S_xx iterative solve: AMG"] = @benchmarkable cg!($z, $S_xx, $RHS, Pl = $P , log=true, maxiter=300)
if use_matlabCMG
    SUITE["S_xx iterative solve: CMG"] = @benchmarkable matlabCmgSolver($S_xx, $RHS; tol=1e-6, maxits=300)
end

# Computation of the preconditioner
SUITE["S_xx precondition: AMG ruge_stuben"] = @benchmarkable aspreconditioner(ruge_stuben($S_xx))

# Prepapre for direct method (augmented system) benchmarks
ldltX̃_reg = ldlt(X̃_regularized)
luX̃_reg = lu(X̃_regularized)
qrX̃_reg = qr(X̃_regularized)
ldlX̃_reg = ldl(X̃_regularized)

# Direct methods on the regularized augmented X̃_regularized
SUITE["X_tilde_reg direct solve: LDLT"] = @benchmarkable \($ldltX̃_reg, $RHS_aug)
SUITE["X_tilde_reg direct solve: LU"] = @benchmarkable \($luX̃_reg, $RHS_aug)
SUITE["X_tilde_reg direct solve: QR"] = @benchmarkable \($qrX̃_reg, $RHS_aug)
SUITE["X_tilde_reg direct solve: LDL"] = @benchmarkable \($ldlX̃_reg, $RHS_aug)

SUITE["X_tilde_reg inplace direct solve: LDL"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $RHS_aug)
SUITE["X_tilde_reg inplace direct solve: LU"] = @benchmarkable ldiv!($Rz, $luX̃_reg, $RHS_aug)

# Non-inplace factorizations of the regularized augmented system X̃
SUITE["X_tilde_reg factorization: LDLT"] = @benchmarkable ldlt($X̃_regularized)
SUITE["X_tilde_reg factorization: LU"] = @benchmarkable lu($X̃_regularized)
SUITE["X_tilde_reg factorization: QR"] = @benchmarkable qr($X̃_regularized)
SUITE["X_tilde_reg factorization: LDL"] = @benchmarkable ldl($X̃_regularized)
