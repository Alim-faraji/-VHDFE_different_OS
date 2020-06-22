using VarianceComponentsHDFE
using Random, BenchmarkTools, Test
using JLD, SparseArrays, LinearAlgebra
using AlgebraicMultigrid, IterativeSolvers
# Make sure LDLFactorizations is version 0.5.0 and that the `multiple-rhs` branch is checked out
using LDLFactorizations

use_matlabCMG = false
if use_matlabCMG
    using Laplacians, Pkg
    include(string(Pkg.dir("Laplacians") , "/src/matlabSolvers.jl"))
end

# Entry point for the PkgBenchmarks call.  Can split into different files later.
include("prepare_benchmark_data.jl")
# NOTE: Suite below can assume that the `benchmark/data/...` has been filled

medium_data = load("benchmark/data/medium_main.jld")
X = medium_data["X_GroundedLaplacian"]
X̃ = medium_data["X_tilde"]
S_xx = medium_data["S_xx"]
X̃_regularized = medium_data["X_tilde_regularized"]
R_p = medium_data["R_p"]

const SUITE = BenchmarkGroup()

# Prepare for direct method benchmarks
m,k = size(X)
luX̃ = lu(X̃)
qrX̃ = qr(X̃)
Rz = zeros(m+k)
# RHS_aug is the first column of the mxm identity augmented by k zeros
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

# TODO: Move around factorizations/preconditioners since they are independent of the RHS

## JLA problem with a single RHS
# JLA_RHS_aug is the first column of the mxm identity augmented by k zeros
JLA_RHS_aug = zeros(m+k)
JLA_RHS_aug[1:m] = R_p[1,:]

# Direct methods on the augmented system X̃
SUITE["JLA: X_tilde inplace direct solve: LU"] = @benchmarkable ldiv!($Rz, $luX̃, $JLA_RHS_aug)

# Non-inplace direct methods on the augmented system X̃
SUITE["JLA: X_tilde direct solve"] = @benchmarkable \($X̃, $JLA_RHS_aug)
SUITE["JLA: X_tilde direct solve: LU"] = @benchmarkable \($luX̃, $JLA_RHS_aug)
SUITE["JLA: X_tilde direct solve: QR"] = @benchmarkable \($qrX̃, $JLA_RHS_aug)

# Prepare for direct method benchmarks
# JLA_RHS is not too sparse, so we benchmark both the sparse and non-sparse representation
JLA_RHS = (R_p*X)[1,:]
JLA_RHS_sparse =  SparseMatrixCSC{Float64,Int64}(sparse(JLA_RHS))
z = 0.1.*ones(length(JLA_RHS))

# Iterative methods on the original system S_xx
SUITE["JLA: S_xx iterative solve: AMG"] = @benchmarkable cg!($z, $S_xx, $JLA_RHS, Pl = $P , log=true, maxiter=300)
SUITE["JLA: S_xx iterative solve: AMG, sparse RHS"] = @benchmarkable cg!($z, $S_xx, $JLA_RHS_sparse, Pl = $P , log=true, maxiter=300)
if use_matlabCMG
    SUITE["JLA: S_xx iterative solve: CMG"] = @benchmarkable matlabCmgSolver($S_xx, $JLA_RHS; tol=1e-6, maxits=300)
    SUITE["JLA: S_xx iterative solve: CMG, sparse RHS"] = @benchmarkable matlabCmgSolver($S_xx, $JLA_RHS_sparse; tol=1e-6, maxits=300)
end

# Direct methods on the regularized augmented X̃_regularized
SUITE["JLA: X_tilde_reg direct solve: LDLT"] = @benchmarkable \($ldltX̃_reg, $JLA_RHS_aug)
SUITE["JLA: X_tilde_reg direct solve: LU"] = @benchmarkable \($luX̃_reg, $JLA_RHS_aug)
SUITE["JLA: X_tilde_reg direct solve: QR"] = @benchmarkable \($qrX̃_reg, $JLA_RHS_aug)
SUITE["JLA: X_tilde_reg direct solve: LDL"] = @benchmarkable \($ldlX̃_reg, $JLA_RHS_aug)

SUITE["JLA: X_tilde_reg inplace direct solve: LDL"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $JLA_RHS_aug)
SUITE["JLA: X_tilde_reg inplace direct solve: LU"] = @benchmarkable ldiv!($Rz, $luX̃_reg, $JLA_RHS_aug)

## LDL Factorization (regularized augmented system) with multiplce right hand sides
JLA_RHS_aug_m = zeros(m+k,2)
JLA_RHS_aug_m[1:m,1:2] = R_p[1:2,:]'
Rz = zeros(size(JLA_RHS_aug_m))
SUITE["JLA: X_tilde_reg inplace direct solve: LDL, multiple RHS: 2"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $JLA_RHS_aug_m)
JLA_RHS_aug_m = zeros(m+k,4)
JLA_RHS_aug_m[1:m,1:4] = R_p[1:4,:]'
Rz = zeros(size(JLA_RHS_aug_m))
SUITE["JLA: X_tilde_reg inplace direct solve: LDL, multiple RHS: 4"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $JLA_RHS_aug_m)
JLA_RHS_aug_m = zeros(m+k,8)
JLA_RHS_aug_m[1:m,1:8] = R_p[1:8,:]'
Rz = zeros(size(JLA_RHS_aug_m))
SUITE["JLA: X_tilde_reg inplace direct solve: LDL, multiple RHS: 8"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $JLA_RHS_aug_m)
JLA_RHS_aug_m = zeros(m+k,16)
JLA_RHS_aug_m[1:m,1:16] = R_p[1:16,:]'
Rz = zeros(size(JLA_RHS_aug_m))
SUITE["JLA: X_tilde_reg inplace direct solve: LDL, multiple RHS: 16"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $JLA_RHS_aug_m)
JLA_RHS_aug_m = zeros(m+k,32)
JLA_RHS_aug_m[1:m,1:32] = R_p[1:32,:]'
Rz = zeros(size(JLA_RHS_aug_m))
SUITE["JLA: X_tilde_reg inplace direct solve: LDL, multiple RHS: 32"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $JLA_RHS_aug_m)
JLA_RHS_aug_m = zeros(m+k,64)
JLA_RHS_aug_m[1:m,1:64] = R_p[1:64,:]'
Rz = zeros(size(JLA_RHS_aug_m))
SUITE["JLA: X_tilde_reg inplace direct solve: LDL, multiple RHS: 64"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $JLA_RHS_aug_m)
JLA_RHS_aug_m = zeros(m+k,200)
JLA_RHS_aug_m[1:m,1:200] = R_p[1:200,:]'
Rz = zeros(size(JLA_RHS_aug_m))
SUITE["JLA: X_tilde_reg inplace direct solve: LDL, multiple RHS: 200"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $JLA_RHS_aug_m)

# TODO JLA problem loop using iterative solver to compare against multiple RHS at the same time

##Medium Data with controls

medium_controls_data = load("benchmark/data/medium_controls_main.jld")
Xcontrols = medium_controls_data["Xcontrols"]
X̃_controls = medium_controls_data["X_tilde"]
S_xx_controls = medium_controls_data["S_xx"]
X̃_regularized_controls = medium_controls_data["X_tilde_regularized"]

# Prepare for direct method benchmarks
m,k = size(Xcontrols)
luX̃ = lu(X̃_controls)
qrX̃ = qr(X̃_controls)
Rz = zeros(m+k)
# RHS_aug is the first column of the kxk identity augmented by m zeros
RHS_aug = zeros(m+k)
RHS_aug[1] = 1.0


# Direct methods on the augmented system X̃
SUITE["X_tilde_controls inplace direct solve: LU"] = @benchmarkable ldiv!($Rz, $luX̃, $RHS_aug)

# Non-inplace direct methods on the augmented system X̃
SUITE["X_tilde_controls direct solve"] = @benchmarkable \($X̃_controls, $RHS_aug)
SUITE["X_tilde_controls direct solve: LU"] = @benchmarkable \($luX̃, $RHS_aug)
SUITE["X_tilde_controls direct solve: QR"] = @benchmarkable \($qrX̃, $RHS_aug)

# Non-inplace factorizations of the augmented system X̃
SUITE["X_tilde_controls factorization: LU"] = @benchmarkable lu($X̃_controls)
SUITE["X_tilde_controls factorization: QR"] = @benchmarkable qr($X̃_controls)

# Prepare for direct method benchmarks
P = aspreconditioner(ruge_stuben(S_xx_controls))
RHS = SparseMatrixCSC{Float64,Int64}(Xcontrols[1,:])
z = 0.1.*ones(length(RHS))

# Iterative methods on the original system S_xx
SUITE["S_xx_controls iterative solve: AMG"] = @benchmarkable cg!($z, $S_xx_controls, $RHS, Pl = $P , log=true, maxiter=300)

# Computation of the preconditioner
SUITE["S_xx_controls precondition: AMG ruge_stuben"] = @benchmarkable aspreconditioner(ruge_stuben($S_xx_controls))

# Prepapre for direct method (augmented system) benchmarks
ldltX̃_reg = ldlt(X̃_regularized_controls)
luX̃_reg = lu(X̃_regularized_controls)
qrX̃_reg = qr(X̃_regularized_controls)
ldlX̃_reg = ldl(X̃_regularized_controls)

# Direct methods on the regularized augmented X̃_regularized
SUITE["X_tilde_reg_controls direct solve: LDLT"] = @benchmarkable \($ldltX̃_reg, $RHS_aug)
SUITE["X_tilde_reg_controls direct solve: LU"] = @benchmarkable \($luX̃_reg, $RHS_aug)
SUITE["X_tilde_reg_controls direct solve: QR"] = @benchmarkable \($qrX̃_reg, $RHS_aug)
SUITE["X_tilde_reg_controls direct solve: LDL"] = @benchmarkable \($ldlX̃_reg, $RHS_aug)

SUITE["X_tilde_reg_controls inplace direct solve: LDL"] = @benchmarkable ldiv!($Rz, $ldlX̃_reg, $RHS_aug)
SUITE["X_tilde_reg_controls inplace direct solve: LU"] = @benchmarkable ldiv!($Rz, $luX̃_reg, $RHS_aug)

# Non-inplace factorizations of the regularized augmented system X̃
SUITE["X_tilde_reg_controls factorization: LDLT"] = @benchmarkable ldlt($X̃_regularized_controls)
SUITE["X_tilde_reg_controls factorization: LU"] = @benchmarkable lu($X̃_regularized_controls)
SUITE["X_tilde_reg_controls factorization: QR"] = @benchmarkable qr($X̃_regularized_controls)
SUITE["X_tilde_reg_controls factorization: LDL"] = @benchmarkable ldl($X̃_regularized_controls)
