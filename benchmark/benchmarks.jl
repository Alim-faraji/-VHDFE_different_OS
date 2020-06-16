using VarianceComponentsHDFE
using Random, BenchmarkTools, Test
using JLD, SparseArrays, LinearAlgebra
using AlgebraicMultigrid, IterativeSolvers

# Entry point for the PkgBenchmarks call.  Can split into different files later.
include("prepare_benchmark_data.jl")
# NOTE: Suite below can assume that the `benchmark/data/...` has been filled

settings_default = Settings()
settings_direct = Settings(lls_algorithm = DirectLLS())
settings_CMG = Settings(lls_algorithm = CMGPreconditionedLLS())

medium_data = load("data/medium_main.jld")
X = medium_data["X_GroundedLaplacian"]
X̃ = medium_data["X_tilde"]
S_xx = medium_data["S_xx"]

const SUITE = BenchmarkGroup()

# Computation of S_xx
# SUITE["Compute S_xx"] = @benchmarkable mul!($S_xx, $X', $X)

# Factorizations of X̃

#SUITE["LDL factorization for regulatized system"]

# Factorizations of X̃
# SUITE["S_xx LU factorization"] = @benchmarkable lu($S_xx)
# SUITE["S_xx QR factorization"] = @benchmarkable qr($S_xx)
# SUITE["S_xx cholesky factorization"] = @benchmarkable cholesky($S_xx)
#SUITE["LDL factorization for regulatized system"]

# Direct Solvers for S_xx
# RHS = X[1,:]
# SUITE["S_xx direct solve"] = @benchmarkable \($S_xx, $RHS)
# SUITE["S_xx LU factored inplace direct solve"] = @benchmarkable ldiv!($z, $luS_xx, $RHS)
# SUITE["S_xx LU factored direct solve"] = @benchmarkable \($luS_xx, $RHS)
# SUITE["S_xx QR factored direct solve"] = @benchmarkable \($qrS_xx, $RHS)
# SUITE["S_xx cholesky factored direct solve"] = @benchmarkable \($cholS_xx, $RHS)


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
S_xx
P = aspreconditioner(ruge_stuben(S_xx))
RHS = SparseMatrixCSC{Float64,Int64}(X[1,:])
z = 0.1.*ones(length(RHS))

# Iterative methods on the original system S_xx
SUITE["S_xx iterative solve: AMG"] = @benchmarkable cg!($z, $S_xx, $RHS, Pl = $P , log=true, maxiter=300)

# Computation of the preconditioner
SUITE["S_xx precondition: AMG ruge_stuben"] = @benchmarkable aspreconditioner(ruge_stuben($S_xx))
