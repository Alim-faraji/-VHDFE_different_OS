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

medium_data = load("benchmark/data/medium_main.jld")
X = medium_data["X_GroundedLaplacian"]
S_xx = medium_data["S_xx"]


const SUITE = BenchmarkGroup()

# Prepare for direct method benchmarks
P = aspreconditioner(ruge_stuben(S_xx))
RHS = SparseMatrixCSC{Float64,Int64}(X[1,:])
z = 0.1.*ones(length(RHS))

# Iterative methods on the original system S_xx
SUITE["S_xx iterative solve: AMG"] = @benchmarkable cg!($z, $S_xx, $RHS, Pl = $P , log=true, maxiter=300)
